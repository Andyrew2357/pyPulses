"""
pyPulses.routines.cap.noise_probe
===================================
cap_noise_probe: a cheap per-bias-point capacitance noise-floor and
non-Gaussianity probe, built for the SNR-tuning framework
(routines/amp_tune) but usable standalone.

Unlike cap_balance, this does not track a moving balance point over a
scan and does not touch ctx.cap_filter/ctx.extrapolator -- it is a
self-contained probe: calibrate locally (cap_balance_three_point or
cap_balance_two_point), optionally refine that estimate
(cap_balance_refine), then take repeated reads and report their
empirical statistics. Deliberately does not use any individual read's
self-reported covariance: the local gain is amp-bias-dependent, so it
must be recalibrated at each probe call, and the noise/non-Gaussianity
statistics are computed as empirical moments across the repeated reads
rather than trusting get_average/get_average_series_correlated's own
reported covariance, which is not always an accurate correction when
sampling faster than the lock-in time constant.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import CapContext

from .initialize import (
    cap_balance_three_point, cap_balance_two_point, cap_balance_refine,
    ThreePointBalanceResult, TwoPointBalanceResult,
)
from ...core.job import checkpoint

from dataclasses import dataclass
from typing import Tuple
import logging
import numpy as np


@dataclass
class CapNoiseProbeResult:
    """
    Result of a cap_noise_probe() call.

    Attributes
    ----------
    status : bool
        False if the local calibration (or refinement) put the balance
        point outside Vstd_max, or the local gain was degenerate. In
        that case std_Cex/std_Closs are a physically meaningful
        worst-case bound (see cap_noise_probe), not a real measurement,
        and mean/kurtosis are not meaningful (0.0).
    mean_Cex, mean_Closs : float
        Empirical mean of the N repeated reads.
    std_Cex, std_Closs : float
        Empirical standard deviation across the N repeated reads -- the
        noise-floor term.
    kurtosis_Cex, kurtosis_Closs : float
        Empirical excess kurtosis across the N repeated reads (0 for a
        Gaussian; positive/leptokurtic indicates heavy tails or
        bimodality, e.g. from slow amplifier gain jumps).
    N : int
        Number of repeated reads (0 if status is False).
    calibration : ThreePointBalanceResult or TwoPointBalanceResult
        The local calibration this probe was built on, for diagnostics.
    """
    status: bool
    mean_Cex: float
    std_Cex: float
    kurtosis_Cex: float
    mean_Closs: float
    std_Closs: float
    kurtosis_Closs: float
    N: int
    calibration: 'ThreePointBalanceResult | TwoPointBalanceResult'

    def __str__(self) -> str:
        s = f"CapNoiseProbeResult: {'OK' if self.status else 'OUT OF RANGE / DEGENERATE GAIN'}\n"
        s += (f"  Cex   = {self.mean_Cex:.5e} +/- {self.std_Cex:.5e}  "
              f"(kurtosis={self.kurtosis_Cex:.3f})\n")
        s += (f"  Closs = {self.mean_Closs:.5e} +/- {self.std_Closs:.5e}  "
              f"(kurtosis={self.kurtosis_Closs:.3f})\n")
        s += f"  N = {self.N}"
        return s


def _excess_kurtosis(x: np.ndarray) -> float:
    """
    Excess kurtosis (0 for a Gaussian), via plain numpy so no new
    dependency is needed for this. Returns 0.0 for a degenerate
    (zero-variance) sample rather than dividing by zero.
    """
    x = x - x.mean()
    m2 = float(np.mean(x**2))
    if m2 <= 0.0:
        return 0.0
    m4 = float(np.mean(x**4))
    return m4 / m2**2 - 3.0


def _invert_gain(
    LX: float, LY: float,
    A_matrix: np.ndarray | None = None,
    A_complex: complex | None = None,
) -> Tuple[float, float] | None:
    """Same dispatch as initialize._invert_gain -- see there for the
    rationale. Duplicated locally rather than imported (a leading
    underscore signals module-private; this ~10-line formula is already
    duplicated between cap_measure.py and cap_balance.py elsewhere in
    this package, so this follows existing convention rather than
    introducing new cross-module coupling for it)."""
    if A_matrix is not None:
        Kc1, Kr1 = A_matrix[0]
        Kc2, Kr2 = A_matrix[1]
        det = Kc1 * Kr2 - Kr1 * Kc2
        if not np.isfinite(det) or det == 0.0:
            return None
        return (
            (Kr2 * LX - Kr1 * LY) / det,
            (-Kc2 * LX + Kc1 * LY) / det,
        )
    else:
        X, Y = A_complex.real, A_complex.imag
        absA2 = X**2 + Y**2
        if not np.isfinite(absA2) or absA2 == 0.0:
            return None
        return (
            (X * LX + Y * LY) / absA2,
            (-Y * LX + X * LY) / absA2,
        )


def _worst_case_result(ctx: 'CapContext', calibration) -> CapNoiseProbeResult:
    worst_case = abs(ctx.Cstd * ctx.Vstd_max / ctx.Vex)
    return CapNoiseProbeResult(
        status=False,
        mean_Cex=0.0, std_Cex=worst_case, kurtosis_Cex=0.0,
        mean_Closs=0.0, std_Closs=worst_case, kurtosis_Closs=0.0,
        N=0, calibration=calibration,
    )


def cap_noise_probe(
    ctx: 'CapContext',
    N: int = 15,
    method: str = 'three_point',
    small_step: Tuple[float, float] = (0.01, 0.01),
    large_step: Tuple[float, float] = (0.94, 0.94),
    dVstd: complex | None = None,
    cal_samples: int = 3,
    cal_wait: float | None = None,
    n_refine: int = 1,
    refine_samples: int = 1,
    refine_wait: float | None = None,
    logger: logging.Logger | None = None,
) -> CapNoiseProbeResult:
    """
    Probe the capacitance noise floor and non-Gaussianity at the
    bridge's *current* gate/sample setting -- no gate sweep, no
    dependence on a real capacitance feature being nearby.

    1. A fresh, locally-valid gain (cap_balance_three_point by default --
       the bridge's usual off-balance calibration method -- or
       cap_balance_two_point) and balance point V0, from a small,
       closed-form (non-iterative) calibration. Moves the bridge to V0.
       Does not touch ctx.cap_filter/ctx.extrapolator, so this cannot
       corrupt whatever production balance-tracking state the context is
       otherwise used for.
    2. Optionally, n_refine cap_balance_refine() correction steps
       sharpening V0 using that same (fixed) gain -- see its docstring.
       Keeps the probe centered on the *true* balance point rather than
       the raw calibration's estimate, so noise is assessed under the
       same conditions ("at balance", real Vex) a real measurement runs
       under.
    3. N independent ctx.lockin_call() reads at V0, each converted to
       (Cex_i, Closs_i) via the same inversion cap_measure() uses --
       the full 2x2 matrix for a three-point calibration (matching
       cap_measure(use_matrix=True)), or the compressed complex gain for
       two-point (matching cap_measure(use_matrix=False)) -- against the
       just-calibrated/refined gain, not whatever ctx.cap_filter happens
       to have cached.
    4. Empirical mean/std/excess-kurtosis of {Cex_i} and {Closs_i}.

    Notes
    -----
    What if the amplifier has ~zero gain? Two reinforcing mechanisms
    prevent this from silently looking like a *good* (small) noise
    floor: (a) with near-zero true gain, the calibration's own V0
    estimate becomes numerically unstable (dividing a residual signal by
    a near-zero gain), almost always landing outside Vstd_max and
    triggering the worst-case-bound fallback below; (b) even in the rare
    case V0 lands in range anyway, step 3 converts every read by
    dividing by that *same* near-zero gain, so ordinary lock-in noise
    maps to enormous apparent capacitance noise -- std_Cex still comes
    out large, not small. `_invert_gain` additionally guards the
    genuinely-zero/non-finite edge case explicitly (a true zero-gain
    reading would otherwise divide by exactly zero).

    Parameters
    ----------
    ctx : CapContext
    N : int, default=15
        Number of repeated reads at the calibrated/refined balance
        point. Must be >= 2 (sample std/kurtosis are undefined for
        N < 2).
    method : {'three_point', 'two_point'}, default='three_point'
        Which local calibration to use. Three-point solves the full
        (possibly asymmetric) 2x2 gain matrix and is the bridge's usual
        off-balance calibration method -- see cap_balance_three_point.
        Two-point is cheaper (2 vs 3 settle+read cycles) but only
        recovers the compressed, symmetric-bridge-assumed complex gain
        -- see cap_balance_two_point.
    small_step, large_step : (float, float)
        Three-point only -- passed through to cap_balance_three_point.
    dVstd : complex, optional
        Two-point only -- calibration step size passed to
        cap_balance_two_point. Defaults to a small fraction (5%) of
        Vstd_range -- comparable to the bridge's own normal calibration
        step sizes, not an arbitrarily large synthetic perturbation.
    cal_samples : int, default=3
        lockin_call averaging per calibration point.
    cal_wait : float, optional
        Settle time passed through to the calibration call's own `wait`.
        None uses that function's own default (3.0 s for three-point,
        1.0 s for two-point -- tuned for real hardware). Override for a
        faster synthetic/no-hardware setup (e.g. in tests).
    n_refine : int, default=1
        Number of cap_balance_refine() correction steps applied to the
        calibration's balance point before taking the N probe reads. 0
        disables refinement (use the raw calibration's V0 as-is).
    refine_samples : int, default=1
        lockin_call averaging per refinement step.
    refine_wait : float, optional
        Settle time passed through to cap_balance_refine's own `wait`.
        None uses its default (ctx.settle_time).
    logger : Logger, optional

    Returns
    -------
    CapNoiseProbeResult
    """
    if N < 2:
        raise ValueError("cap_noise_probe requires N >= 2.")

    cal_kwargs = {} if cal_wait is None else {'wait': cal_wait}
    if method == 'three_point':
        cal = cap_balance_three_point(
            Vstd=ctx.Vstd, Theta=ctx.Theta, lockin_call=ctx.lockin_call,
            Vex=ctx.Vex, Cstd=ctx.Cstd, Vstd_range=ctx.Vstd_range,
            small_step=small_step, large_step=large_step,
            samples=cal_samples, move_to_balance=True,
            Vstd_max=ctx.Vstd_max, logger=logger,
            **cal_kwargs,
        )
    elif method == 'two_point':
        step = dVstd if dVstd is not None else 0.05 * ctx.Vstd_range
        cal = cap_balance_two_point(
            Vstd=ctx.Vstd, Theta=ctx.Theta, lockin_call=ctx.lockin_call,
            Vex=ctx.Vex, Cstd=ctx.Cstd,
            dVstd=step, samples=cal_samples, move_to_balance=True,
            Vstd_max=ctx.Vstd_max, logger=logger, **cal_kwargs,
        )
    else:
        raise ValueError(f"Unknown method '{method}'; use 'three_point' or 'two_point'.")

    if not cal.status:
        # The bridge couldn't even be nulled within Vstd_max at this
        # bias point. Report a physically meaningful worst-case bound
        # (the largest capacitance the bridge could represent at all)
        # rather than an arbitrary sentinel, so this stays on a scale
        # comparable to real evaluations instead of an out-of-band magic
        # number the search would have to reason about differently.
        return _worst_case_result(ctx, cal)

    V0 = cal.V0
    if isinstance(cal, ThreePointBalanceResult):
        A_matrix, A_complex = cal.A_matrix, None
    else:
        A_matrix, A_complex = None, cal.A_complex

    # Fixed-gain sanity check up front, once -- the gain doesn't change
    # across the refinement/probe reads below, so there is no point
    # discovering it is degenerate only after several of them.
    if _invert_gain(0.0, 0.0, A_matrix=A_matrix, A_complex=A_complex) is None:
        return _worst_case_result(ctx, cal)

    if n_refine > 0:
        refine_kwargs = {} if refine_wait is None else {'wait': refine_wait}
        refined = cap_balance_refine(
            ctx, n_refine=n_refine, samples=refine_samples, result=cal,
            logger=logger, **refine_kwargs,
        )
        if not refined.status:
            return _worst_case_result(ctx, cal)
        V0 = refined.V0

    Cex = np.empty(N)
    Closs = np.empty(N)
    for i in range(N):
        checkpoint()
        mean, _ = ctx.lockin_call()  # cov intentionally unused -- see module docstring
        LX, LY = float(mean[0]), float(mean[1])
        dVx, dVy = _invert_gain(LX, LY, A_matrix=A_matrix, A_complex=A_complex)
        Cex[i] = -ctx.Cstd * (V0.real - dVx) / ctx.Vex
        Closs[i] = -ctx.Cstd * (V0.imag - dVy) / ctx.Vex

    result = CapNoiseProbeResult(
        status=True,
        mean_Cex=float(Cex.mean()), std_Cex=float(Cex.std(ddof=1)),
        kurtosis_Cex=_excess_kurtosis(Cex),
        mean_Closs=float(Closs.mean()), std_Closs=float(Closs.std(ddof=1)),
        kurtosis_Closs=_excess_kurtosis(Closs),
        N=N, calibration=cal,
    )
    if logger:
        logger.info(result)
    return result
