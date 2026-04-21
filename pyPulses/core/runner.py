"""
Runner: drives a Scan and Measurement together.

The Runner owns:
- The main loop (iterate, move, measure)
- Timestamp generation
- Result array allocation and population
- start_idx for resuming interrupted scans
- Progress / time-remaining logging
- Threaded execution via Job
- Observer notifications at each point
- Bespoke sidecar integration (separate from the observer list)

Observers are callables with signature:
    f(idx, coords, measured, timestamp) -> Any

The sidecar (when plot=True) is notified after observers via Sidecar.instance()
rather than being stored in _observers, and receives special treatment:
clear(frame=0) is called at the start of each new outermost scan line when
clear_on_new_line=True.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import datetime
import logging
import time
from typing import Callable, Dict, List

import numpy as np

from .scan import ScanBase
from .measurement import Measurement
from .job import checkpoint, Job

if TYPE_CHECKING:
    from .sidecar import Sidecar

def _is_new_line(idx: np.ndarray) -> bool:
    """True when a new outermost line has started."""
    return len(idx) > 1 and bool(idx[-1] == 0)


class Runner:
    """
    Drives a Scan and Measurement together over parameter space.

    Parameters
    ----------
    scan : ScanBase
        Defines the trajectory through parameter space and owns hardware movement.
    measurement : Measurement
        Defines what to measure at each point and fires per-point callbacks.
    retain_return : bool, default=True
        If True, allocates a result array pre-filled with NaN and populates it
        as the scan progresses. Set False for large scans where memory matters.
    timestamp : bool, default=True
        If True, records a datetime for each point.
    start_idx : ndarray, optional
        If provided, points are skipped until an idx matching start_idx is seen.
    min_wait : float, optional
        Floor on the per-step wait time passed to scan.move_to.
    observers : list of Callable, optional
        Called after each point with (idx, coords, measured, timestamp).
    logger : logging.Logger, optional
        Used for progress and time-remaining messages.
    plot : bool, default=False
        If True, notifies the session Sidecar (via Sidecar.instance()) after
        each point. The sidecar is never stored in _observers.
    """

    def __init__(self,
        scan: ScanBase,
        measurement: Measurement,
        retain_return: bool = True,
        timestamp: bool = True,
        start_idx: np.ndarray | None = None,
        min_wait: float | None = None,
        observers: List[Callable] | None = None,
        logger: logging.Logger | None = None,
        plot: bool = False,
    ):
        self.scan = scan
        self.measurement = measurement
        self.retain_return = retain_return
        self.timestamp = timestamp
        self.start_idx = start_idx
        self.min_wait = min_wait
        self._observers = list(observers or [])
        self._logger = logger
        self.plot = plot

        # Set by configure_sidecar
        self._sidecar_clear_on_new_line: bool = False

    """Observer management"""

    def add_observer(self, f: Callable) -> None:
        """Append an observer."""
        self._observers.append(f)

    def remove_observer(self, f: Callable) -> None:
        """Remove an observer by identity."""
        self._observers.remove(f)

    """Threaded execution"""

    def run_threaded(self) -> Job:
        """Run in a background thread, returning a Job with controls."""
        return Job(self.run).start()

    """Main loop"""

    def run(self) -> np.ndarray | None:
        """
        Execute the scan, measuring at each point.

        Returns
        -------
        ndarray or None
            Shape (*scan.dimensions, measurement.num_cols), filled with NaN.
            Only returned if retain_return is True.
        """
        from .sidecar import Sidecar

        dims = self.scan.dimensions
        n_meas = self.measurement.num_cols

        if self.retain_return:
            result = np.full(
                shape=(*dims, n_meas),
                fill_value=np.nan,
                dtype=float,
            )

        reached_start = self.start_idx is None
        points_taken = 0
        start_time = time.time()

        for idx, coords in self.scan._iter():
            checkpoint()

            # Resume logic
            if not reached_start:
                if np.array_equal(idx, self.start_idx):
                    reached_start = True
                else:
                    continue

            # Move hardware
            self.scan.move_to(coords, min_wait=self.min_wait)

            # Timestamp
            now = datetime.datetime.now() if self.timestamp else None

            # Measure
            measured_arr = self.measurement.measure(idx, coords, now)

            # Write into result array
            if self.retain_return:
                result[(*idx,)] = measured_arr

            # Named dict for observers and sidecar
            measured_dict: Dict[str, float] = dict(
                zip(self.measurement.col_names, measured_arr)
            )

            # Regular observers
            for obs in self._observers:
                obs(idx, coords, measured_dict, now)

            # Bespoke sidecar notification
            if self.plot:
                sidecar = Sidecar.instance()
                if sidecar is not None:
                    if self._sidecar_clear_on_new_line and _is_new_line(idx):
                        sidecar.clear(frame=0)
                    sidecar(idx, coords, measured_dict, now)

            points_taken += 1
            self._log_progress(points_taken, start_time)

        if self.retain_return:
            return result

    """Utilities"""

    def as_named(self, result: np.ndarray) -> dict:
        """Return result as a dict of named arrays keyed by column name."""
        return {
            name: result[..., i]
            for i, name in enumerate(self.measurement.col_names)
        }

    """Logging"""

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _log_progress(self, points_taken: int, start_time: float) -> None:
        if self._logger is None:
            return
        npoints = self.scan.npoints
        if npoints == 0:
            return
        pct = points_taken / npoints
        elapsed = time.time() - start_time
        remaining = int(elapsed * (1 / pct - 1)) if pct > 0 else 0
        hrs = remaining // 3600
        remaining -= hrs * 3600
        mns = remaining // 60
        scs = remaining - mns * 60
        self._log(
            f"Taken {points_taken}/{npoints} ({100 * pct:.2f}%); "
            f"Est. {hrs}:{mns:02d}:{scs:02d} remaining."
        )

    """Sidecar configuration"""

    def configure_sidecar(self,
        x: str | None = None,
        twinx: list[tuple[str, str]] = None,
        group_by_unit: bool = False,
        include: list[str] | None = None,
        clear: bool = True,
        clear_on_new_line: bool = True,
        max_history: int = 0,
        frame: int = 0,
    ) -> 'Sidecar':
        """
        Configure a Sidecar's panes from this runner's scan and measurement
        metadata, and enable sidecar plotting on this runner.

        Parameters
        ----------
        sidecar : Sidecar
            The sidecar to configure.
        x : str or None
            Name of the coordinate or measurement to use as the x axis.
            Defaults to the first scan coordinate.
        twinx : list of (str, str) tuples, optional
            Pairs of column names to share a pane. The second name in each
            pair is plotted on the secondary y axis of the first name's pane.
        group_by_unit : bool, default=False
            If True, columns sharing the same unit are grouped into one pane.
        include : list of str or None
            Measurement columns to include. None means all columns.
        clear : bool, default=True
            If True, clears panes in the target frame before adding new ones.
            Set False to add runner panes alongside pre-existing context panes.
        clear_on_new_line : bool, default=True
            If True, clears frame 0 data at the start of each new outermost
            scan line. Useful for 2D sweeps where each line is a fresh trace.
        frame : int, default=0
            Which frame to place the runner's panes in.

        Returns
        -------
        Sidecar
        """
        from .sidecar import LineConfig, LinePane, Sidecar
        sidecar = Sidecar.instance()
        if sidecar is None:
            return

        # Enable plot mode
        self.plot = True
        self._sidecar_clear_on_new_line = clear_on_new_line

        # Determine x axis
        if x is None:
            if not self.scan.coord_names:
                raise ValueError("Scan has no coordinates — cannot determine x axis.")
            x = self.scan.coord_names[0]

        col_names = self.measurement.col_names
        col_long_names = self.measurement.col_long_names
        col_units = self.measurement.col_units

        # Metadata lookup: name -> {long_name, unit}
        meta: dict = {
            name: {'long_name': ln, 'unit': u}
            for name, ln, u in zip(col_names, col_long_names, col_units)
        }
        for ch in self.scan.channels:
            meta[ch.name] = {'long_name': ch.long_name, 'unit': ch.unit}

        def _label(name: str) -> str:
            m = meta.get(name, {})
            base = m.get('long_name') or name
            unit = m.get('unit')
            return f'{base} [{unit}]' if unit else base

        xlabel = _label(x)

        # Build twinx lookups
        twinx = twinx or []
        primary_of: dict[str, str]       = {}
        secondaries: dict[str, list[str]] = {}
        for prim, sec in twinx:
            primary_of[sec] = prim
            secondaries.setdefault(prim, []).append(sec)

        # Names to skip when iterating col_names
        already_placed = set(primary_of.keys()) | {x}

        if include is not None:
            include_set = set(include)
            for prim, secs in secondaries.items():
                if prim in include_set:
                    include_set.update(secs)
            already_placed |= {
                name for name in col_names
                if name not in include_set and name not in set(primary_of.keys())
            }

        if clear:
            sidecar.clear_panes(frame=frame)

        def _make_pane(primary: str) -> LinePane:
            secs = secondaries.get(primary, [])
            lines = [LineConfig(channel=primary)] + \
                        [LineConfig(channel=s, secondary_y=True) for s in secs]
            pane_name = ', '.join([primary] + secs)
            return LinePane(
                name = pane_name,
                lines = lines,
                x = x,
                xlabel = xlabel,
                ylabel = _label(primary),
                ylabel2 = _label(secs[0]) if secs else '',
                max_history = max_history,
            )

        if group_by_unit:
            from collections import defaultdict
            unit_groups: dict = defaultdict(list)
            for name in col_names:
                if name in already_placed:
                    continue
                unit_groups[meta.get(name, {}).get('unit')].append(name)
            for names in unit_groups.values():
                for name in names:
                    sidecar.add_pane(_make_pane(name), frame=frame)
        else:
            for name in col_names:
                if name in already_placed:
                    continue
                sidecar.add_pane(_make_pane(name), frame=frame)

        return sidecar