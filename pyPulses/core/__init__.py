"""
pyPulses.core
=============
Scan construction, measurement orchestration, background job control,
live sidecar plotting, database logging, and tandem hardware sweeps.

Scanning
--------
Scans describe trajectories through parameter space. Leaf scans are
constructed directly; composite scans are built with `+` (concatenation)
and `*` (outer product):

    from pyPulses.core import ScanCut, ScanGrid

    # Linear sweep from start to end
    cut = ScanCut(channels=[Bx], start={'Bx': 0}, end={'Bx': 1}, numpoints=101)

    # Outer-product grid
    grid = ScanGrid(channels=[Vg, Vsd], axes={'Vg': vg_vals, 'Vsd': vsd_vals})

    # Compose: sweep Bx at each grid point
    scan = grid * cut

Available leaf scans:

    ScanCut              — linear interpolation from start to end
    ScanPoints           — explicit list of coordinate dicts
    ScanGrid             — outer product of per-channel 1-D axes
    ScanParallelepiped   — affine grid through a parallelepiped in parameter space
    ScanArbitrary        — user-supplied iterator

Masks
-----
Serializable boolean expressions that filter scan points. Built with
natural Python syntax and attached to any scan node via `space_mask`:

    from pyPulses.core import C, C_const

    circle = (C('Bx')**2 + C('By')**2) < C_const(1.0)**2
    wedge  = (C('V') > 0.1) & (C('V') < 2 * C('B') + 1.0)

    scan = ScanGrid(..., space_mask=circle)

    # Serialize / restore
    d    = circle.to_dict()
    mask = Expr.from_dict(d)

Measurement
-----------
A `Measurement` orchestrates one or more `Query` objects at each scan
point: waits for settling, fires pre-callbacks, collects readings
(threading lazy queries concurrently with eager ones), fires
post-callbacks, and returns a flat array.

    from pyPulses.core import Query, Measurement

    queries = [
        Query(dmm.read, name='R_xx', unit='Ω'),
        Query(lockin.snap, name=['X', 'Y'], unit=['V', 'V'], lazy=True),
    ]
    meas = Measurement(queries, time_per_point=0.3)

Any object satisfying the `QuerySignature` protocol can be used in place
of a `Query` dataclass.

Runner
------
The `Runner` ties a scan and measurement together: iterates the scan,
moves hardware, measures, populates a result array, and notifies observers.

    from pyPulses.core import Runner

    runner = Runner(scan, meas, plot=True)
    runner.configure_sidecar(x='Bx', clear_on_new_line=True)
    result = runner.run()

    # Or in a background thread (auto-registers with the Sidecar)
    job = runner.run_threaded()
    job.join()

Job control
-----------
`Job` runs any callable in a background thread with cooperative
pause/stop. If a `Sidecar` is active, every `Job` is automatically
registered so the browser buttons control it.

    from pyPulses.core import Job, checkpoint

    def my_task():
        for i in range(1000):
            checkpoint()          # honor pause / stop
            do_work(i)

    job = Job(my_task).start()    # auto-registered with Sidecar
    job.pause()
    job.resume()
    job.stop()

`checkpoint()` can be called from anywhere in the call stack — it uses a
`ContextVar` to find the controlling `Job`.

Sidecar (live plotting)
-----------------------
A session-level HTTP server that pushes live-updating Plotly plots to a
browser or VS Code Simple Browser tab. Independent of the Jupyter kernel
comm — updates correctly even when other cells are executing.

    from pyPulses.core import Sidecar, LinePane, LineConfig

    sidecar = Sidecar(port=8767)
    sidecar.add_pane(LinePane(
        name='transport',
        lines=[LineConfig('R_xx'), LineConfig('R_xy', secondary_y=True)],
        x='B', xlabel='B (T)', ylabel='R (Ω)',
    ))
    sidecar.open()

    # Access the singleton from anywhere
    sc = Sidecar.instance()

    # Per-scan lifecycle
    sidecar.clear()                 # wipe data, keep layout
    sidecar.clear(frame=0)          # wipe only frame 0

Database logging
----------------
`DatabaseLogger` streams sweep results into a SQLite database row by row,
compatible with the `ez_data.sqlite_to_xarray` reader.

    from pyPulses.core import DatabaseLogger

    with DatabaseLogger.from_runner(runner, 'experiment.db') as db:
        runner.add_observer(db)
        runner.run()

Tandem sweep
------------
`tandemSweep` moves multiple hardware channels simultaneously while
respecting per-channel step-size and rate constraints derived from each
channel's `SweepConfig`.

    from pyPulses.core import tandemSweep

    tandemSweep(channels=[Bx, By], target={'Bx': 1.0, 'By': 0.5})
"""

# -- Job control -------------------------------------------------------------
from .job import Job, Control, StopRequested, checkpoint

# -- Mask expressions --------------------------------------------------------
from .mask import Expr, Coord, Const, C, C_const

# -- Scan objects ------------------------------------------------------------
from .scan import (
    ScanBase,
    ScanCut,
    ScanPoints,
    ScanGrid,
    ScanParallelepiped,
    ScanArbitrary,
)

# -- Measurement -------------------------------------------------------------
from .measurement import Query, QuerySignature, Measurement

# -- Runner ------------------------------------------------------------------
from .runner import Runner

# -- Sidecar (live plotting) -------------------------------------------------
from .sidecar import Sidecar, Pane, LinePane, LineConfig

# -- Database logging --------------------------------------------------------
from .database import DatabaseLogger, CheckpointMode

# -- Tandem sweep ------------------------------------------------------------
from .tandem_sweep import tandemSweep, SweepResult

__all__ = [
    # Job control
    "Job",
    "Control",
    "StopRequested",
    "checkpoint",
    # Masks
    "Expr",
    "Coord",
    "Const",
    "C",
    "C_const",
    # Scans
    "ScanBase",
    "ScanCut",
    "ScanPoints",
    "ScanGrid",
    "ScanParallelepiped",
    "ScanArbitrary",
    # Measurement
    "Query",
    "QuerySignature",
    "Measurement",
    # Runner
    "Runner",
    # Sidecar
    "Sidecar",
    "Pane",
    "LinePane",
    "LineConfig",
    # Database
    "DatabaseLogger",
    "CheckpointMode",
    # Tandem sweep
    "tandemSweep",
    "SweepResult",
]