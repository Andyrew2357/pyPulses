"""
Localhost web GUI for SRS lock-in amplifiers.

Usage
-----
    gui = SRSLockinGUI(my_sr860).start()  # opens at http://localhost:8760
    # ...
    my_sr860.gui.stop()                   # or via the instrument accessor
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

# Valid option lists for string-valued settings
_STRING_OPTIONS: dict[str, list[str]] = {
    'input_configuration'      : ['A', 'A-B', 'I'],
    'reference_trigger'        : ['sine', 'pos', 'neg'],
    'reference_input_impedance': ['low', 'high'],
    'current_gain'             : ['1MEG', '100MEG'],
    'line_notch_filter'        : ['Out', 'In', '2xIn', 'Both'],
    'low_pass_filter_slope'    : ['6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct'],
}

# sr844 overrides the slope options with an extra '0dB/oct' entry
_SR844_SLOPE_OPTIONS = ['0dB/oct', '6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct']

# Methods that take no arguments (fire-and-forget actions)
_ACTION_METHODS = ['auto_phase', 'auto_gain', 'auto_range']

# All get/set settings, in render order
_ALL_SETTINGS = [
    'reference_phase', 'internal_reference', 'reference_frequency',
    'detection_harmonic', 'reference_trigger', 'reference_input_impedance',
    'sine_output_amplitude', 'sine_output_offset',
    'input_configuration', 'input_shield_grounded', 'input_coupling_DC',
    'input_mode_current', 'input_range', 'current_gain', 'line_notch_filter',
    'input_sensitivity', 'time_constant', 'low_pass_filter_slope',
    'sync_filter_state',
]


class SRSLockinGUI:
    """
    Localhost web GUI for SRS lock-in amplifiers.

    Wraps any SRSLockin subclass and launches a browser-based control panel
    with bidirectional WebSocket communication. Instrument settings, live
    X/Y/R/θ readouts, and auxiliary outputs are all accessible.

    After start(), the instrument's .gui attribute is set to this instance
    for easy lifecycle management:

        my_lock_in.gui.stop()

    Parameters
    ----------
    instrument : SRSLockin
        Any SRSLockin subclass instance (sr830, sr844, sr850, sr860, sr865a).
    port : int, default 8760
        HTTP port. WebSocket runs on port+1.
    poll_interval : float, default 0.5
        Default live-readout polling interval in seconds.
    """

    def __init__(self, instrument, port: int = 8760, poll_interval: float = 0.5):
        self._instrument    = instrument
        self._port          = port
        self._ws_port       = port + 1
        self._poll_interval = poll_interval
        self._polling       = False
        self._state         = {}
        self._capabilities  = {}
        self._clients: set  = set()
        self._loop          = None
        self._thread        = None
        self._http_server   = None
        self._http_thread   = None
        self._inst_lock     = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> 'SRSLockinGUI':
        """
        Start HTTP and WebSocket servers and set instrument.gui = self.
        Returns self for chaining.
        """
        try:
            with self._inst_lock:
                self._state = self._instrument._serialize_state()
        except Exception:
            self._state = {}

        self._capabilities = self._build_capabilities()

        # WebSocket server on its own asyncio loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_asyncio, daemon=True, name='SRSLockinGUI-ws'
        )
        self._thread.start()

        # HTTP server serving the frontend HTML
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
                pass  # suppress HTTP access log noise

        self._http_server = HTTPServer(('localhost', self._port), _Handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True, name='SRSLockinGUI-http',
        )
        self._http_thread.start()

        self._instrument.gui = self
        print(f"SRS Lock-in GUI → http://localhost:{self._port}")
        return self

    def stop(self):
        """Stop all servers and clear instrument.gui."""
        self._polling = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._http_server:
            # Shutdown is blocking; run in a thread to avoid deadlocks
            threading.Thread(
                target=self._http_server.shutdown, daemon=True
            ).start()
        if getattr(self._instrument, 'gui', None) is self:
            self._instrument.gui = None

    # ── Asyncio / WebSocket ────────────────────────────────────────────────

    def _run_asyncio(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._ws_handler, 'localhost', self._ws_port):
            await asyncio.Future()  # run until loop is stopped

    async def _ws_handler(self, websocket):
        self._clients.add(websocket)
        try:
            # Send capabilities and initial state on connect
            await websocket.send(json.dumps(
                {'type': 'capabilities', **self._json_safe(self._capabilities)}
            ))
            await websocket.send(json.dumps({
                'type': 'state',
                'data': self._json_safe(self._state),
            }))

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
        if t == 'set':
            await self._loop.run_in_executor(
                None, self._do_set, websocket, msg['method'], msg['value']
            )
        elif t == 'call':
            await self._loop.run_in_executor(
                None, self._do_call, websocket, msg['method']
            )
        elif t == 'set_aux':
            await self._loop.run_in_executor(
                None, self._do_set_aux, websocket, int(msg['idx']), float(msg['value'])
            )
        elif t == 'poll_start':
            self._poll_interval = float(msg.get('interval', self._poll_interval))
            if not self._polling:
                self._polling = True
                asyncio.ensure_future(self._poll_loop())
        elif t == 'poll_stop':
            self._polling = False
        elif t == 'poll_interval':
            self._poll_interval = float(msg.get('interval', self._poll_interval))

    # ── Instrument dispatch (runs in executor threads) ─────────────────────

    def _do_set(self, websocket, method: str, value):
        try:
            with self._inst_lock:
                getattr(self._instrument, method)(value)
            self._state[method] = value
            asyncio.run_coroutine_threadsafe(
                self._broadcast({'type': 'ack', 'method': method, 'value': value}),
                self._loop,
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps(
                    {'type': 'error', 'method': method, 'message': str(exc)}
                )),
                self._loop,
            )

    def _do_call(self, websocket, method: str):
        try:
            with self._inst_lock:
                getattr(self._instrument, method)()
            asyncio.run_coroutine_threadsafe(
                self._broadcast({'type': 'called', 'method': method}),
                self._loop,
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps(
                    {'type': 'error', 'method': method, 'message': str(exc)}
                )),
                self._loop,
            )

    def _do_set_aux(self, websocket, idx: int, value: float):
        try:
            with self._inst_lock:
                self._instrument.aux_output(idx, value)
            # Update stored aux state
            aux = self._state.get('aux_output', [])
            self._state['aux_output'] = [
                [ch, (value if ch == idx else v)] for ch, v in aux
            ]
            asyncio.run_coroutine_threadsafe(
                self._broadcast({'type': 'ack_aux', 'idx': idx, 'value': value}),
                self._loop,
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps({
                    'type'   : 'error',
                    'method' : f'aux_output[{idx}]',
                    'message': str(exc),
                })),
                self._loop,
            )

    def _poll_once(self) -> tuple[float, float, float, float]:
        """Run in executor: get X, Y, R, θ under the instrument lock."""
        with self._inst_lock:
            x, y = self._instrument.get_xy()
            r, t = self._instrument.get_rt()
        return float(x), float(y), float(r), float(t)

    async def _poll_loop(self):
        while self._polling:
            try:
                x, y, r, t = await self._loop.run_in_executor(None, self._poll_once)
                await self._broadcast({
                    'type': 'reading',
                    'x': x, 'y': y, 'r': r, 't': t,
                    'ts': time.time(),
                })
            except Exception:
                pass
            await asyncio.sleep(self._poll_interval)

    async def _broadcast(self, msg: dict):
        if not self._clients:
            return
        data = json.dumps(self._json_safe(msg))
        await asyncio.gather(
            *(ws.send(data) for ws in list(self._clients)),
            return_exceptions=True,
        )

    # ── Capabilities ───────────────────────────────────────────────────────

    def _build_capabilities(self) -> dict:
        inst = self._instrument
        supported = [m for m in _ALL_SETTINGS if hasattr(inst, m)]
        actions   = [m for m in _ACTION_METHODS if hasattr(inst, m)]

        # String option lists — use model-specific overrides where needed
        options = {m: _STRING_OPTIONS[m] for m in supported if m in _STRING_OPTIONS}
        if inst.__class__.__name__ == 'sr844' and 'low_pass_filter_slope' in supported:
            options['low_pass_filter_slope'] = _SR844_SLOPE_OPTIONS

        # Discrete numeric value arrays (sent so the frontend can render dropdowns)
        discrete: dict[str, list] = {}
        if hasattr(inst, 'sens_vals') and len(inst.sens_vals):
            discrete['input_sensitivity'] = [float(v) for v in inst.sens_vals]
        if hasattr(inst, 'tau_vals') and len(inst.tau_vals):
            discrete['time_constant'] = [float(v) for v in inst.tau_vals]
        if hasattr(inst, 'irng_vals') and len(inst.irng_vals):
            discrete['input_range'] = [float(v) for v in inst.irng_vals]

        return {
            'supported'   : supported,
            'actions'     : actions,
            'options'     : options,
            'discrete'    : discrete,
            'aux_channels': list(inst.out_aux_channels),
            'model'       : inst.__class__.__name__.upper(),
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _json_safe(obj):
        """Recursively convert numpy scalars/arrays to Python natives."""
        if isinstance(obj, dict):
            return {k: SRSLockinGUI._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [SRSLockinGUI._json_safe(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def __repr__(self) -> str:
        status = 'running' if (self._loop and self._loop.is_running()) else 'stopped'
        return (
            f"SRSLockinGUI({self._instrument.__class__.__name__}, "
            f"port={self._port}, {status})"
        )