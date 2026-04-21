"""
Sidecar: a session-level HTTP server that serves a live-updating plot page
in a browser or VS Code Simple Browser tab.

The sidecar is completely independent of the Jupyter kernel comm — it updates
correctly even when other cells are executing or when joining a job thread.

Usage
-----
    # Once per session
    sidecar = Sidecar(port=8767)
    sidecar.add_pane(LinePane(
        name='transport',
        lines=[LineConfig('R_xx'), LineConfig('R_xy', secondary_y=True)],
        x='B', xlabel='B (T)', ylabel='R (Ω)',
    ))
    sidecar.open()

    # Access from anywhere in the session
    sidecar = Sidecar.instance()

    # Each scan
    sidecar.clear()              # wipe data in all frames, keep layout
    sidecar.clear(frame=0)       # wipe only frame 0
    sidecar.register_job(job)    # wire browser pause/resume/stop to job
    job.start()

    # Context objects can add their own panes to frame 1
    sidecar.add_pane(LinePane(...), frame=1)
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import ceil, sqrt
from typing import Any, Dict, List

import numpy as np

from .job import Control, Job

def _json_response(handler: BaseHTTPRequestHandler, data: Any) -> None:
    payload = json.dumps(data).encode()
    handler.send_response(200)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    handler.wfile.write(payload)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get('Content-Length', 0))
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _nan_to_none_2d(z: list[list[float]]) -> list[list[float | None]]:
    """Replace NaN with None in a 2-D nested list for JSON safety."""
    return [
        [None if (v is not None and math.isnan(v)) else v for v in row]
        for row in z
    ]

"""Pane abstractions"""

@dataclass
class LineConfig:
    """
    Configuration for a single line within a LinePane.

    Parameters
    ----------
    channel : str
        Name of the measurement or coordinate to plot on the y axis.
    secondary_y : bool
        If True, this line is plotted on the secondary (right) y axis.
    color : str or None
        CSS color string. If None, the frontend assigns a color automatically.
    label : str or None
        Display label for the legend. Defaults to channel name.
    """
    channel: str
    secondary_y: bool = False
    color: str | None = None
    label: str | None = None


class Pane(ABC):
    """
    Abstract base for a plot panel in the sidecar.

    Each concrete subclass defines what data it holds, how it serializes
    itself for the frontend, and how it accepts a new data point.

    The frontend dispatches on the `type` field to choose the renderer.
    """

    def __init__(self, name: str):
        self.name  = name
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def pane_type(self) -> str:
        """String type tag used by the frontend renderer dispatcher."""
        ...

    @abstractmethod
    def update(self, idx: np.ndarray | None, values: Dict[str, float]) -> None:
        """Accept a new data point from the observer.
        
        Parameters
        ----------
        idx : ndarray or None
            Scan index for this point. None for non-scan sources.
        values : dict
            Merged coordinate and measurement values.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all accumulated data."""
        ...

    @abstractmethod
    def _serialize_config(self) -> dict:
        """Serialize static configuration (axes labels, line configs, etc.)."""
        ...

    @abstractmethod
    def _serialize_data(self, force_full: bool = False) -> dict:
        """Serialize current data. Called on every /state poll."""
        ...

    def serialize(self, force_full: bool = False) -> dict:
        with self._lock:
            return {
                'type': self.pane_type,
                'name': self.name,
                'config': self._serialize_config(),
                'data': self._serialize_data(force_full=force_full),
            }

class LinePane(Pane):
    """
    A pane that displays one or more lines against a common x axis.
 
    Supports twin (secondary) y axes and optional fading history. When
    max_history > 0, calling clear() rotates the current line into a history
    buffer rather than discarding it. Up to max_history previous lines are
    retained and rendered with decreasing opacity, oldest being most faded.
 
    Parameters
    ----------
    name : str
        Pane identifier, used as the title.
    lines : list of LineConfig
        Lines to display. At least one required.
    x : str
        Name of the coordinate or measurement to use as the x axis.
    xlabel : str
        X axis label (may contain LaTeX in $...$).
    ylabel : str
        Primary y axis label (may contain LaTeX in $...$).
    ylabel2 : str
        Secondary y axis label.
    xscale, yscale, yscale2 : str
        'linear' or 'log'.
    max_points : int
        Maximum number of points to retain per line (oldest discarded).
    max_history : int, default=0
        Number of previous lines to retain as faded history.
        0 means no history — clear() wipes data as before.
    """
 
    def __init__(self,
        name: str,
        lines: List[LineConfig],
        x: str,
        xlabel: str = '',
        ylabel: str = '',
        ylabel2: str = '',
        xscale: str = 'linear',
        yscale: str = 'linear',
        yscale2: str = 'linear',
        max_points: int = 2000,
        max_history: int = 0,
    ):
        super().__init__(name)
        self.lines= lines
        self.x = x
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylabel2 = ylabel2
        self.xscale = xscale
        self.yscale = yscale
        self.yscale2 = yscale2
        self.max_points = max_points
        self.max_history = max_history
 
        # Current line data: channel -> {'x': [], 'y': []}
        self._current: Dict[str, Dict[str, list]] = {
            lc.channel: {'x': [], 'y': []} for lc in lines
        }
        # History: deque of snapshots, index 0 = most recent, -1 = oldest
        # Each snapshot is a dict channel -> {'x': [...], 'y': [...]}
        from collections import deque
        self._history: deque = deque(maxlen=max_history) if max_history > 0 else None
 
    @property
    def pane_type(self) -> str:
        return 'line'
 
    def update(self, idx: np.ndarray | None, values: Dict[str, float]) -> None:
        x_val = values.get(self.x)
        if x_val is None:
            return
        with self._lock:
            for lc in self.lines:
                y_val = values.get(lc.channel)
                if y_val is None:
                    continue
                d = self._current[lc.channel]
                d['x'].append(float(x_val))
                d['y'].append(float(y_val))
                if len(d['x']) > self.max_points:
                    d['x'] = d['x'][-self.max_points:]
                    d['y'] = d['y'][-self.max_points:]
 
    def clear(self) -> None:
        with self._lock:
            if self._history is not None:
                # Snapshot current into history if it has any data
                if any(len(d['x']) > 0 for d in self._current.values()):
                    self._history.appendleft(
                        {ch: {'x': list(d['x']), 'y': list(d['y'])}
                         for ch, d in self._current.items()}
                    )
            for d in self._current.values():
                d['x'].clear()
                d['y'].clear()
 
    def _serialize_config(self) -> dict:
        return {
            'x': self.x,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'ylabel2': self.ylabel2,
            'xscale': self.xscale,
            'yscale': self.yscale,
            'yscale2': self.yscale2,
            'max_history': self.max_history,
            'lines': [
                {
                    'channel': lc.channel,
                    'secondary_y': lc.secondary_y,
                    'color': lc.color,
                    'label': lc.label or lc.channel,
                }
                for lc in self.lines
            ],
        }
 
    def _serialize_data(self, force_full: bool = False) -> dict:
        out: dict = {'current': {ch: dict(d) for ch, d in self._current.items()}}
        if self._history is not None:
            # history[0] = most recent, history[-1] = oldest
            out['history'] = [
                {ch: dict(d) for ch, d in snapshot.items()}
                for snapshot in self._history
            ]
        return out


class HeatmapPane(Pane):
    """
    A pane that displays a 2D colormap filled point-by-point during a scan.

    Pixel placement is always by integer index — physical coordinates are
    used only for tick labels and hover text, avoiding any resampling.

    Data is transmitted as deltas: each poll drains the list of cells written
    since the last poll.  A full z array is included on first serialize,
    after clear(), and when the frontend requests ``?full=1``.

    Parameters
    ----------
    name : str
        Pane title.
    channel : str
        Measurement column whose values become z.
    shape : tuple of (int, int)
        (n_rows, n_cols) — the two scan dimensions this pane covers.
    idx_axes : tuple of (int, int), default (0, 1)
        Which elements of the scan idx to use as (row, col).
        For a 2D product scan this is (0, 1).  For a 3D scan where you want the 
        inner two dims: (1, 2).
    x_coords : array-like or None
        Physical values along the column axis.
    y_coords : array-like or None
        Physical values along the row axis.
    xlabel, ylabel : str
        Axis labels (may contain LaTeX in $...$).
    colorscale : str
        Any named Plotly colorscale (e.g. 'Viridis', 'RdBu', 'Plasma').
    zmin, zmax : float or None
        Fixed color range.  None → auto-track running min/max.
    reversescale : bool
        If True, reverse the colorscale direction.
    """

    def __init__(self,
        name: str,
        channel: str,
        shape: tuple[int, int],
        idx_axes: tuple[int, int] = (0, 1),
        x_coords: np.ndarray | None = None,
        y_coords: np.ndarray | None = None,
        xlabel: str = '',
        ylabel: str = '',
        colorscale: str = 'Viridis',
        zmin: float | None = None,
        zmax: float | None = None,
        reversescale: bool = False,
    ):
        super().__init__(name)
        self.channel = channel
        self.shape = shape
        self.idx_axes = idx_axes
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.colorscale = colorscale
        self.zmin_fixed = zmin
        self.zmax_fixed = zmax
        self.reversescale = reversescale

        self._x_coords = x_coords.tolist() if x_coords is not None else None
        self._y_coords = y_coords.tolist() if y_coords is not None else None

        # Ground truth array — always authoritative
        self._z = np.full(shape, np.nan)

        # Delta tracking
        self._version: int = 0
        self._dirty: list[tuple[int, int, float]] = []
        self._needs_full: bool = True

        # Auto-range tracking
        self._running_min: float = float('inf')
        self._running_max: float = float('-inf')

    @property
    def pane_type(self) -> str:
        return 'heatmap'

    def update(self, idx: np.ndarray | None, values: Dict[str, float]) -> None:
        if idx is None:
            return
        val = values.get(self.channel)
        if val is None:
            return
        r = int(idx[self.idx_axes[0]])
        c = int(idx[self.idx_axes[1]])
        fval = float(val)
        with self._lock:
            self._z[r, c] = fval
            self._version += 1
            self._dirty.append((r, c, fval))
            if fval < self._running_min:
                self._running_min = fval
            if fval > self._running_max:
                self._running_max = fval

    def clear(self) -> None:
        with self._lock:
            self._z[:] = np.nan
            self._version += 1
            self._dirty.clear()
            self._needs_full = True
            self._running_min = float('inf')
            self._running_max = float('-inf')

    def _current_zmin(self) -> float:
        if self.zmin_fixed is not None:
            return self.zmin_fixed
        return self._running_min if math.isfinite(self._running_min) else 0.0

    def _current_zmax(self) -> float:
        if self.zmax_fixed is not None:
            return self.zmax_fixed
        return self._running_max if math.isfinite(self._running_max) else 1.0

    def _serialize_config(self) -> dict:
        return {
            'channel': self.channel,
            'shape': list(self.shape),
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'colorscale': self.colorscale,
            'reversescale': self.reversescale,
            'x_coords': self._x_coords,
            'y_coords': self._y_coords,
            'zmin_fixed': self.zmin_fixed,
            'zmax_fixed': self.zmax_fixed,
        }

    def _serialize_data(self, force_full: bool = False) -> dict:
        out = {
            'version': self._version,
            'updates': list(self._dirty),
            'zmin': self._current_zmin(),
            'zmax': self._current_zmax(),
        }
        if self._needs_full or force_full:
            out['z'] = _nan_to_none_2d(self._z.tolist())
            self._needs_full = False
        self._dirty.clear()
        return out

"""HTML / JS frontend"""

from importlib.resources import files as _res_files
_FRONTEND: str = (
    _res_files(__package__)
    .joinpath('_sidecar_frontend.html')
    .read_text(encoding='utf-8')
)

"""HTTP handler"""

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith('/state'):
            force_full = 'full=1' in self.path
            _json_response(self, self.server._sidecar._get_state(force_full=force_full))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(_FRONTEND.encode())

    def do_POST(self) -> None:
        if self.path == '/control':
            body = _read_body(self)
            self.server._sidecar._handle_control(body.get('action', ''))
            _json_response(self, {'ok': True})
        else:
            self.send_response(404)
            self.end_headers()

"""Sidecar singleton"""

class Sidecar:
    """
    Session-level live plot server.

    One instance per session. Access it from anywhere with Sidecar.instance().

    Panes are organised into numbered frames (default frame=0). Frames are
    rendered top-to-bottom with a thin horizontal rule between them. Frame 0
    is conventionally for runner panes; frame 1+ for context panes (CapContext,
    etc.) that want to plot concurrently.

    Parameters
    ----------
    port : int
        Local port to serve on. Default 8767.
    max_cols : int
        Maximum columns in the auto-computed grid layout, applied per frame.
    """

    _instance: 'Sidecar | None' = None

    def __init__(self,
        port: int = 8767,
        max_cols: int = 4,
    ):
        self._port = port
        self._max_cols = max_cols

        # frames: frame_index -> {pane_name: Pane}  (ordered dicts)
        self._frames: Dict[int, Dict[str, Pane]] = {0: {}}
        self._lock = threading.Lock()

        # Job control
        self._control: Control | None = None
        self._job_status: str = 'none'

        # HTTP server
        self._server = HTTPServer(('localhost', port), _Handler)
        self._server._sidecar = self
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='SidecarServer',
        )
        self._server_thread.start()

        Sidecar._instance = self
        print(f'Sidecar running at http://localhost:{self._port}')

    """Singleton accessor"""

    @classmethod
    def instance(cls) -> 'Sidecar | None':
        """
        Return the current session sidecar, or None if not yet started.

        Contexts, instrument classes, and helper functions should use this
        rather than holding a direct reference to the sidecar object.
        """
        return cls._instance

    """Pane management"""
    
    def add_pane(self, pane: Pane, frame: int = 0) -> 'Sidecar':
        """
        Add a pane to the specified frame.

        Parameters
        ----------
        pane : Pane
        frame : int, default=0
            Frame index. Created automatically if it does not yet exist.

        Returns self for chaining.
        """
        with self._lock:
            if frame not in self._frames:
                self._frames[frame] = {}
            self._frames[frame][pane.name] = pane
        return self

    def remove_pane(self, name: str, frame: int = 0) -> 'Sidecar':
        """Remove a pane by name from the specified frame."""
        with self._lock:
            if frame in self._frames:
                self._frames[frame].pop(name, None)
        return self

    def clear_panes(self, frame: int | None = None) -> 'Sidecar':
        """
        Remove all panes from the specified frame, or from all frames if None.
        Returns self for chaining.
        """
        with self._lock:
            if frame is None:
                for f in self._frames.values():
                    f.clear()
            elif frame in self._frames:
                self._frames[frame].clear()
        return self

    def clear(self, frame: int | None = None) -> 'Sidecar':
        """
        Clear all pane data without removing panes.

        Parameters
        ----------
        frame : int or None
            If None, clears data in all frames. If an int, clears only that frame.

        Returns self for chaining.
        """
        with self._lock:
            if frame is None:
                targets = list(self._frames.values())
            elif frame in self._frames:
                targets = [self._frames[frame]]
            else:
                targets = []
            all_panes = [p for f in targets for p in f.values()]

        # Clear outside the lock so pane locks don't nest inside sidecar lock
        for pane in all_panes:
            pane.clear()
        return self

    """Job control"""

    def register_job(self, job: Job) -> 'Sidecar':
        """
        Register a Job so the browser pause/resume/stop buttons act on it.

        Only one job may be registered at a time. Deregisters automatically
        when the job finishes, stops, or errors. Returns self for chaining.
        """
        if not isinstance(job, Job):
            raise TypeError("job must be a Job instance")

        with self._lock:
            if self._control is not None and self._job_status == 'running':
                raise RuntimeError(
                    "A job is already registered. "
                    "Wait for it to finish or stop it first."
                )
            self._control = job.control
            self._job_status = 'running'

        def _on_finish(j, result):
            with self._lock:
                self._control = None
                self._job_status = 'none'

        def _on_stop(j):
            with self._lock:
                self._job_status = 'stopped'

        def _on_error(j, tb):
            with self._lock:
                self._control = None
                self._job_status = 'none'

        job.on_finish.append(_on_finish)
        job.on_stop.append(_on_stop)
        job.on_error.append(_on_error)
        return self

    def _handle_control(self, action: str) -> None:
        with self._lock:
            ctrl = self._control
            if ctrl is None:
                return
            if action == 'pause':
                ctrl.pause()
                self._job_status = 'paused'
            elif action == 'resume':
                ctrl.resume()
                self._job_status = 'running'
            elif action == 'stop':
                ctrl.stop()
                self._job_status = 'stopped'

    """Observer interface"""

    def __call__(self,
        idx      : Any,
        coords   : Dict[str, float],
        measured : Dict[str, float],
        timestamp: Any | None = None,
    ) -> None:
        """
        Observer-compatible callable. Merges coords and measured then
        dispatches to every pane across all frames, passing idx through
        for panes that need it (e.g. HeatmapPane).
        """
        values = {**coords, **measured}
        with self._lock:
            all_panes = [
                pane
                for frame in self._frames.values()
                for pane in frame.values()
            ]
        for pane in all_panes:
            pane.update(idx, values)

    """State serialization for frontend"""

    def _get_layout(self, n: int) -> dict:
        cols = min(self._max_cols, max(1, ceil(sqrt(n))))
        rows = ceil(n / cols) if cols > 0 else 1
        return {'rows': rows, 'cols': cols}

    def _get_state(self, force_full: bool = False) -> dict:
        with self._lock:
            frames_out = []
            for idx in sorted(self._frames.keys()):
                frame_panes = self._frames[idx]
                if not frame_panes:
                    continue
                frames_out.append({
                    'frame': idx,
                    'layout': self._get_layout(len(frame_panes)),
                    'panes': {
                        name: pane.serialize(force_full=force_full)
                        for name, pane in frame_panes.items()
                    },
                })
            job = {'status': self._job_status}

        return {'frames': frames_out, 'job': job}

    """Opening the browser"""
    
    def open(self) -> 'Sidecar':
        """
        Open the sidecar in a browser or VS Code Simple Browser tab.
        Returns self for chaining.
        """
        url    = f'http://localhost:{self._port}'
        opened = False

        try:
            result = subprocess.run(
                ['code', '--open-url',
                 f'vscode://vscode.simpleBrowser/show?url={url}'],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                print(f'Opened in VS Code Simple Browser: {url}')
                opened = True
        except Exception:
            pass

        if not opened:
            try:
                webbrowser.open(url)
                print(f'Opened in browser: {url}')
            except Exception:
                print(f'Could not open automatically. Navigate to: {url}')

        return self

    def stop_server(self) -> None:
        """Shut down the HTTP server. Call at the end of a session."""
        self._server.shutdown()
        Sidecar._instance = None