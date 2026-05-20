"""
Localhost web GUI for the FastFlight2 repetitive signal averager.

Usage
-----
    gui = FastFlight2GUI(my_ff2).start()   # opens at http://localhost:8780
    # ...
    my_ff2.gui.stop()                      # or via the instrument accessor
"""

import asyncio
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

try:
    import websockets
    import websockets.exceptions
except ImportError:
    raise ImportError("websockets package required: pip install websockets")

_FRONTEND_PATH = Path(__file__).parent / 'gui.html'

# Maximum number of points sent to the frontend chart.
# The browser doesn't benefit from more than this, and it keeps WebSocket
# messages small even for long traces.
_MAX_CHART_POINTS = 2000


def _downsample(data: np.ndarray, n_points: int, max_out: int) -> np.ndarray:
    """
    Simple block-max downsample: preserve peaks rather than averaging,
    which is more useful for time-of-flight spectra.
    """
    if n_points <= max_out:
        return data[:n_points]
    block = n_points / max_out
    out = np.empty(max_out, dtype=np.float64)
    for i in range(max_out):
        lo = int(i * block)
        hi = int((i + 1) * block)
        out[i] = float(data[lo:hi].max())
    return out


class FastFlight2GUI:
    """
    Localhost web GUI for the FastFlight2.

    Provides a browser-based interface for:
      - Configuring acquisition protocol (record length, TPP, averages, etc.)
      - Configuring trigger settings
      - Single and continuous spectrum acquisition
      - Live spectrum display with overload/underload indicators

    Parameters
    ----------
    instrument : FastFlight2
    port : int, default 8780
        HTTP port.  WebSocket runs on port+1.
    """

    def __init__(self, instrument, port: int = 8780):
        self._instrument  = instrument
        self._port        = port
        self._ws_port     = port + 1

        # Acquisition state — touched only under _acq_lock or from the
        # acquisition thread itself.
        self._acq_lock      = threading.Lock()
        self._acq_thread    = None
        self._acq_running   = False   # True while the acq thread is alive
        self._acq_mode      = None    # 'single' | 'continuous' | None

        # Pending settings: set freely from any thread; applied by the
        # acquisition thread between spectra.
        self._pending_protocol = None   # FastFlight2Protocol | None
        self._pending_trigger  = None   # dict | None
        self._settings_changed = threading.Event()

        # WebSocket infrastructure
        self._clients : set     = set()
        self._loop              = None
        self._ws_thread         = None
        self._http_server       = None
        self._http_thread       = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> 'FastFlight2GUI':
        """Start HTTP and WebSocket servers. Returns self for chaining."""
        self._loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(
            target=self._run_asyncio, daemon=True, name='FF2GUI-ws'
        )
        self._ws_thread.start()

        _self = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    html = _FRONTEND_PATH.read_text(encoding='utf-8')
                    html = html.replace('__WS_PORT__', str(_self._ws_port))
                    body = html.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as exc:
                    self.send_error(500, str(exc))

            def log_message(self, *args):
                pass

        self._http_server = HTTPServer(('localhost', self._port), _Handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True, name='FF2GUI-http',
        )
        self._http_thread.start()

        self._instrument.gui = self
        print(f"FastFlight2 GUI → http://localhost:{self._port}")
        return self

    def stop(self):
        """Stop all servers and the acquisition thread."""
        self._stop_acquisition()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._http_server:
            threading.Thread(
                target=self._http_server.shutdown, daemon=True
            ).start()
        if getattr(self._instrument, 'gui', None) is self:
            self._instrument.gui = None

    # -----------------------------------------------------------------------
    # asyncio / WebSocket
    # -----------------------------------------------------------------------

    def _run_asyncio(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._ws_handler, 'localhost', self._ws_port):
            await asyncio.Future()

    async def _ws_handler(self, websocket):
        self._clients.add(websocket)
        try:
            await websocket.send(json.dumps(self._state_msg()))
            async for raw in websocket:
                try:
                    await self._dispatch(websocket, json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    async def _dispatch(self, websocket, msg: dict):
        t = msg.get('type')
        if t == 'acquire_single':
            await self._loop.run_in_executor(
                None, self._cmd_acquire, 'single'
            )
        elif t == 'acquire_start':
            await self._loop.run_in_executor(
                None, self._cmd_acquire, 'continuous'
            )
        elif t == 'acquire_stop':
            await self._loop.run_in_executor(
                None, self._stop_acquisition
            )
        elif t == 'apply_settings':
            await self._loop.run_in_executor(
                None, self._cmd_apply_settings, msg
            )
        elif t == 'get_state':
            await websocket.send(json.dumps(self._state_msg()))

    async def _broadcast(self, msg: dict):
        if not self._clients:
            return
        data = json.dumps(msg)
        await asyncio.gather(
            *(ws.send(data) for ws in list(self._clients)),
            return_exceptions=True,
        )

    def _send(self, msg: dict):
        """Thread-safe: schedule a broadcast from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    # -----------------------------------------------------------------------
    # Command handlers (run in executor threads)
    # -----------------------------------------------------------------------

    def _cmd_acquire(self, mode: str):
        """Start an acquisition thread in the given mode."""
        with self._acq_lock:
            if self._acq_running:
                # Stop the existing thread first.
                self._acq_mode = None   # signal it to exit
            # Wait for old thread to finish before starting a new one.
        if self._acq_thread and self._acq_thread.is_alive():
            self._acq_thread.join(timeout=5.0)

        self._acq_mode = mode
        self._acq_running = True
        self._acq_thread = threading.Thread(
            target=self._acquisition_loop,
            daemon=True,
            name=f'FF2GUI-acq-{mode}',
        )
        self._acq_thread.start()
        self._send({'type': 'status', 'acquiring': True, 'mode': mode})

    def _stop_acquisition(self):
        """Signal the acquisition thread to stop and wait for it."""
        self._acq_mode = None
        if self._acq_thread and self._acq_thread.is_alive():
            self._acq_thread.join(timeout=5.0)
        self._send({'type': 'status', 'acquiring': False, 'mode': None})

    def _cmd_apply_settings(self, msg: dict):
        """
        Parse incoming settings message and stage them for the acquisition
        thread to apply between spectra.
        """
        from .fastflight2 import FastFlight2Protocol
 
        ff = self._instrument
 
        # --- Protocol fields ---
        proto = FastFlight2Protocol.from_dict(ff.protocol.to_dict())
        changed = False
 
        if 'record_length' in msg:
            proto.record_length = float(msg['record_length'])
            changed = True
        if 'tpp' in msg:
            proto.time_per_point(float(msg['tpp']))
            changed = True
        if 'records_per_spectrum' in msg:
            proto.records_per_spectrum = int(msg['records_per_spectrum'])
            changed = True
        if 'voltage_offset' in msg:
            proto.voltage_offset = float(msg['voltage_offset'])
            changed = True
        if 'time_offset' in msg:
            proto.time_offset = float(msg['time_offset'])
            changed = True
        if 'precision_enhancer' in msg:
            proto.precision_enhancer = bool(msg['precision_enhancer'])
            changed = True
        if 'correlated_subtraction' in msg:
            proto.correlated_subtraction = bool(msg['correlated_subtraction'])
            changed = True
        if 'compression' in msg:
            proto.compression = int(msg['compression'])
            changed = True
 
        if changed:
            self._pending_protocol = proto
 
        # --- Trigger fields ---
        trigger = {}
        if 'trigger_threshold' in msg:
            trigger['threshold'] = float(msg['trigger_threshold'])
        if 'trigger_rising' in msg:
            trigger['rising'] = bool(msg['trigger_rising'])
        if 'trigger_enable_high' in msg:
            trigger['enable_high'] = bool(msg['trigger_enable_high'])
        if 'external_trigger' in msg:
            trigger['external'] = bool(msg['external_trigger'])
 
        if trigger:
            self._pending_trigger = trigger
 
        if changed or trigger:
            self._settings_changed.set()
 
        # If not currently acquiring, apply settings immediately so the UI
        # gets an accurate echo (including any quantisation of record_length
        # etc.).  If acquiring, leave them staged for the acquisition thread
        # to apply between spectra — but don't echo state yet, because
        # ff.protocol still holds the old values and echoing it would
        # overwrite the user's fields with the previous settings.
        if not self._acq_running:
            self._apply_pending_settings()
            self._send(self._state_msg())


    # -----------------------------------------------------------------------
    # Acquisition loop
    # -----------------------------------------------------------------------

    def _apply_pending_settings(self):
        """
        Apply any staged protocol or trigger changes to the device.
        Called by the acquisition thread between spectra, with no lock held.
        """
        ff = self._instrument

        proto = self._pending_protocol
        if proto is not None:
            self._pending_protocol = None
            ff.protocol = proto
            ff.send_protocol(proto, 0)
            ff.reset_timer()

        trig = self._pending_trigger
        if trig is not None:
            self._pending_trigger = None
            if 'threshold' in trig:
                ff.trigger_threshold(trig['threshold'])
            if 'rising' in trig:
                ff.trigger_rising(trig['rising'])
            if 'enable_high' in trig:
                ff.trigger_enable_high(trig['enable_high'])
            if 'external' in trig:
                ff.external_trigger(trig['external'])

        self._settings_changed.clear()

    def _acquisition_loop(self):
        """
        Main acquisition loop run in its own thread.
 
          1. start_acquisition() — includes clear_buffer()
          2. get_spectrum()      — blocks; no instrument lock held
          3. stop_acquisition()
          4. apply pending settings if any
          5. repeat or exit
 
        The instrument lock (_acq_lock) is held only for the brief
        start/stop moments, not during the blocking get_spectrum() call.
        This allows settings to be staged at any time and applied cleanly
        between spectra.
        """
        ff  = self._instrument
        out = np.zeros(ff.MAX_POINTS, dtype=np.uint32)
        n   = 0
 
        try:
            with self._acq_lock:
                ff.start_acquisition()
 
            while self._acq_mode is not None:
                t0 = time.perf_counter()
 
                # --- Blocking spectrum read (no lock) ---
                try:
                    n = ff.get_spectrum(out)
                except Exception as exc:
                    self._send({'type': 'error', 'message': str(exc)})
                    break
 
                elapsed = time.perf_counter() - t0
 
                # --- Stop, apply settings, restart (brief lock) ---
                with self._acq_lock:
                    ff.stop_acquisition()
                    if self._settings_changed.is_set():
                        self._apply_pending_settings()
                        self._send(self._state_msg())
                    if self._acq_mode is not None:
                        ff.start_acquisition()
 
                # --- Send spectrum to all connected clients ---
                self._send_spectrum(out, n, elapsed)
 
                # Single-shot mode: exit after first spectrum.
                if self._acq_mode == 'single':
                    self._acq_mode = None
 
        finally:
            try:
                with self._acq_lock:
                    ff.stop_acquisition()
            except Exception:
                pass
            self._acq_running = False
            self._send({'type': 'status', 'acquiring': False, 'mode': None})


    def _send_spectrum(self, data: np.ndarray, n_points: int, elapsed: float):
        """
        Downsample and broadcast one spectrum to all connected clients.
        """
        ff      = self._instrument
        p       = ff.protocol
        tpp_ns  = p.time_per_point()
        ds      = _downsample(data, n_points, _MAX_CHART_POINTS)
        n_out   = len(ds)

        # Time axis in nanoseconds for the downsampled points.
        # We keep a uniform grid matching the original bin spacing.
        step = n_points / n_out if n_out > 0 else 1
        t_ns = (np.arange(n_out) * step * tpp_ns).tolist()

        overload  = ff.get_overload()
        msg = {
            'type'               : 'spectrum',
            'n_points'           : n_points,
            'n_chart'            : n_out,
            'tpp_ns'             : tpp_ns,
            't_ns'               : t_ns,
            'data'               : ds.tolist(),
            'elapsed'            : round(elapsed, 3),
            'overload'           : bool(overload & ff.OVERLOAD),
            'underload'          : bool(overload & ff.UNDERLOAD),
            'records_per_spectrum': p.records_per_spectrum,
            'record_length_ns'   : p.record_length,
        }
        self._send(msg)

    # -----------------------------------------------------------------------
    # State snapshot
    # -----------------------------------------------------------------------

    def _state_msg(self) -> dict:
        ff = self._instrument
        p  = ff.protocol
        return {
            'type'                 : 'state',
            'acquiring'            : self._acq_running,
            'mode'                 : self._acq_mode,
            # Protocol
            'record_length'        : p.record_length,
            'tpp'                  : p.time_per_point(),
            'records_per_spectrum' : p.records_per_spectrum,
            'voltage_offset'       : p.voltage_offset,
            'time_offset'          : p.time_offset,
            'precision_enhancer'   : p.precision_enhancer,
            'correlated_subtraction': p.correlated_subtraction,
            'compression'          : p.compression,
            # Trigger
            'trigger_threshold'    : ff._trigger_threshold,
            'trigger_rising'       : ff._trigger_rising,
            'trigger_enable_high'  : ff._trigger_enable_high,
            'external_trigger'     : ff._external_trigger,
        }

    def __repr__(self) -> str:
        status = 'running' if (self._loop and self._loop.is_running()) else 'stopped'
        return f"FastFlight2GUI(port={self._port}, {status})"