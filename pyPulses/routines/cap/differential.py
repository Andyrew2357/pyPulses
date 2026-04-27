"""
Differential capacitance balance procedure.

Nulls compressibility features along a 1D cut in phase space by measuring the 
lock-in response through the excitation gate and complementary gate separately, 
then computing a complex ratio gamma that sets the complementary amplitude and 
phase to cancel the compressibility signal.

This procedure is independent of the main CapFilter / balance machinery and uses 
very little of the CapContext (just the lock-in call and settle time).

Usage
-----
    from pyPulses.routines.cap.differential import differential_balance

    result = differential_balance(
        ctx        = ctx,
        scan       = scan_1d,
        Vex_amp    = lockin.resolve('Vex_amp'),
        Vexp_amp   = lockin.resolve('Vexp_amp'),
        Vexp_phase = lockin.resolve('Vexp_phase'),
    )
"""

from __future__ import annotations

from .context import CapContext
from ...core.scan import ScanBase
from ...core.measurement import Query, Measurement
from ...core.runner import Runner
from ...devices.channel_adapter import ScalarChannel

from dataclasses import dataclass
import logging
import time

import numpy as np


@dataclass
class DifferentialBalanceResult:
    """
    Result of a differential capacitance balance.

    Attributes
    ----------
    status : bool
        True if |gamma| <= max_gamma and the complementary channels were set.
    gamma : complex
        Complex ratio. The complementary gate is set to Vex * gamma.
    Vex : float
        Excitation amplitude read from hardware before the procedure.
    Vexp_amp_set : float | None
        Amplitude actually set on the complementary channel, or None if
        |gamma| exceeded max_gamma.
    Vexp_phase_set : float | None
        Phase (degrees) actually set on the complementary channel, or None.
    Lt : np.ndarray
        Complex lock-in readings from pass 1 (excitation gate), shape (N,).
    Lb : np.ndarray
        Complex lock-in readings from pass 2 (complementary gate), shape (N,).
    Lt_raw : np.ndarray
        Raw runner result array from pass 1, shape (N, 2).
    Lb_raw : np.ndarray
        Raw runner result array from pass 2, shape (N, 2).
    """
    status: bool
    gamma: complex
    Vex: float
    Vexp_amp_set: float | None
    Vexp_phase_set: float | None
    Lt: np.ndarray
    Lb: np.ndarray
    Lt_raw: np.ndarray
    Lb_raw: np.ndarray

    def __str__(self) -> str:
        s = f"DifferentialBalanceResult: {'OK' if self.status else 'FAILED'}\n"
        s += f"  gamma = {self.gamma.real:.5e} + {self.gamma.imag:.5e}i"
        s += f"  (|gamma| = {abs(self.gamma):.5e})\n"
        s += f"  Vex   = {self.Vex:.5e}\n"
        if self.Vexp_amp_set is not None:
            s += f"  Vexp  = {self.Vexp_amp_set:.5e} @ {self.Vexp_phase_set:.2f}°\n"
        else:
            s += f"  Vexp  = NOT SET (|gamma| exceeded max_gamma)\n"
        s += f"  N pts = {len(self.Lt)}\n"
        return s


def _build_measurement(ctx: CapContext) -> Measurement:
    """
    Build a Measurement that collects lock-in X and Y via the context's
    lockin_call. Returns a two-column measurement (LX, LY).
    """
    def _read_lockin():
        mean, _cov = ctx.lockin_call()
        return np.asarray(mean, dtype=float).reshape(2)

    query = Query(
        f=_read_lockin,
        name=['LX', 'LY'],
        long_name=['$L_X$', '$L_Y$'],
        unit=['V', 'V'],
    )
    return Measurement(
        queries=[query],
        time_per_point=ctx.settle_time,
    )


def _result_to_complex(result: np.ndarray) -> np.ndarray:
    """Convert (N, 2) runner result to (N,) complex array."""
    return result[..., 0] + 1j * result[..., 1]


def differential_balance(
    ctx: CapContext,
    scan: ScanBase,
    Vex_amp: ScalarChannel,
    Vexp_amp: ScalarChannel,
    Vexp_phase: ScalarChannel,
    max_gamma: float = 2.0,
    wait: float | None = None,
    logger: logging.Logger | None = None,
    plot: bool = False,
    frame: int = 0,
) -> DifferentialBalanceResult:
    """
    Null compressibility features along a 1D cut by balancing the
    complementary gate against the excitation gate.

    Parameters
    ----------
    ctx : CapContext
        Provides lockin_call and settle_time.
    scan : ScanBase
        Must be one-dimensional (len(scan.dimensions) == 1).
    Vex_amp : ScalarChannel
        Excitation amplitude channel. Read to determine Vex; zeroed during pass 
        2 and restored afterwards.
    Vexp_amp : ScalarChannel
        Complementary amplitude channel. Zeroed during pass 1; set to Vex during 
        pass 2; set to Vex * |gamma| at the end.
    Vexp_phase : ScalarChannel
        Complementary phase channel. Set to 0 during pass 2; set to angle(gamma) 
        in degrees at the end.
    max_gamma : float, default=2.0
        Safety bound on |gamma|. If exceeded, the complementary channels are not 
        set and status is False.
    wait : float, optional
        Settling time after switching gate configuration. Defaults to
        ctx.settle_time.
    logger : Logger, optional

    Returns
    -------
    DifferentialBalanceResult
    """

    # Validate 1D scan
    if len(scan.dimensions) != 1:
        raise ValueError(
            f"differential_balance requires a 1D scan, but got "
            f"scan.dimensions = {scan.dimensions} "
            f"({len(scan.dimensions)}D)."
        )

    if wait is None:
        wait = ctx.settle_time

    measurement = _build_measurement(ctx)
    Vex = float(Vex_amp())

    _log = logger.info if logger else lambda *a, **kw: None
    _warn = logger.warning if logger else lambda *a, **kw: None

    _log(f"Differential balance: Vex = {Vex:.5e}, {scan.npoints} pts")

    # --- Pass 1: excitation gate only ---
    _log("Pass 1: excitation gate (Vexp = 0)")
    Vexp_amp(0.0)
    if wait > 0:
        time.sleep(wait)

    scan.invalidate_cache()
    runner_t = Runner(
        scan=scan,
        measurement=measurement,
        retain_return=True,
        timestamp=False,
        plot=plot,
    )
    if plot:
        runner_t.configure_sidecar(
            x=scan.coord_names[0],
            twinx=[('LX', 'LY')],
            clear=True,
            clear_on_new_line=False,
            max_history=1,
            frame=frame,
        )
    Lt_raw = runner_t.run()

    # --- Pass 2: complementary gate only ---
    _log(f"Pass 2: complementary gate (Vex_amp = 0, Vexp = {Vex:.5e} @ 0°)")
    Vex_amp(0.0)
    Vexp_amp(Vex)
    Vexp_phase(0.0)

    if plot:
        from ...core.sidecar import Sidecar
        sidecar = Sidecar.instance()
        if sidecar is not None:
            sidecar.clear(frame=frame)

    if wait > 0:
        time.sleep(wait)

    scan.invalidate_cache()
    runner_b = Runner(
        scan=scan,
        measurement=measurement,
        retain_return=True,
        timestamp=False,
        plot=plot,
    )
    Lb_raw = runner_b.run()

    # --- Restore excitation ---
    Vex_amp(Vex)
    Vexp_amp(0.0)
    _log(f"Restored Vex_amp = {Vex:.5e}, Vexp_amp = 0")

    # --- Compute gamma ---
    Lt = _result_to_complex(Lt_raw)
    Lb = _result_to_complex(Lb_raw)

    Lt_centered = Lt - np.mean(Lt)
    Lb_centered = Lb - np.mean(Lb)

    numerator = np.sum(np.conj(Lb_centered) * Lt_centered)
    denominator = np.sum(np.abs(Lb_centered) ** 2)

    if np.abs(denominator) < 1e-30:
        _warn("Denominator near zero — complementary gate shows no variation. "
               "Cannot compute gamma.")
        return DifferentialBalanceResult(
            status=False,
            gamma=0j,
            Vex=Vex,
            Vexp_amp_set=None,
            Vexp_phase_set=None,
            Lt=Lt,
            Lb=Lb,
            Lt_raw=Lt_raw,
            Lb_raw=Lb_raw,
        )

    gamma = numerator / denominator

    _log(f"gamma = {gamma.real:.5e} + {gamma.imag:.5e}i  "
         f"(|gamma| = {abs(gamma):.5e})")

    # --- Apply ---
    if abs(gamma) > max_gamma:
        _warn(f"|gamma| = {abs(gamma):.5e} exceeds max_gamma = {max_gamma}. "
              f"Not setting complementary channels.")
        return DifferentialBalanceResult(
            status=False,
            gamma=gamma,
            Vex=Vex,
            Vexp_amp_set=None,
            Vexp_phase_set=None,
            Lt=Lt,
            Lb=Lb,
            Lt_raw=Lt_raw,
            Lb_raw=Lb_raw,
        )

    Vexp_set = Vex * abs(gamma)
    phase_set = float(np.degrees(np.angle(gamma))) % 360

    Vexp_amp(Vexp_set)
    Vexp_phase(phase_set)

    _log(f"Set Vexp_amp = {Vexp_set:.5e}, Vexp_phase = {phase_set:.2f}°")

    return DifferentialBalanceResult(
        status=True,
        gamma=gamma,
        Vex=Vex,
        Vexp_amp_set=float(Vexp_set),
        Vexp_phase_set=float(phase_set),
        Lt=Lt,
        Lb=Lb,
        Lt_raw=Lt_raw,
        Lb_raw=Lb_raw,
    )