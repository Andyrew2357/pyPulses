from __future__ import annotations

from .cap_filter import CapFilter
from .context import CapContext
from .config import CAP_INIT_P_MULT_THREE_POINT, CAP_INIT_P_MULT_TWO_POINT
from ...devices.channel_adapter import ScalarChannel, LockInChannel
from ...core.sidecar import LinePane, LineConfig, Sidecar

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import logging
import time

"""
Result Dataclasses
"""

@dataclass
class ThreePointBalanceResult():
    """
    Result of a three-point AC bridge balance.

    The three-point procedure measures the lock-in signal at three Vstd
    settings that differ in their in-phase and out-of-phase components.
    It solves for the balance point V0 and the effective bridge gain
    matrix K (a 2x2 real matrix mapping Vstd -> lock-in).

    Attributes
    ----------
    status : bool
        True if V0 lies within Vstd_range.
    V0 : complex
        Balance point in Vstd space (V0.real = in-phase, V0.imag = out-of-phase).
    A_complex : complex
        Effective bridge gain as a complex number (X + iY), extracted from K.
        This is the quantity stored in CapFilter.
    A_matrix : (2, 2) ndarray
        Full gain matrix [[Kc1, Kr1], [Kc2, Kr2]] for reference.
    P : (2, 2) ndarray
        Initial covariance for A = [X, Y].
    error : (float, float) or None
        Lock-in (X, Y) reading at the balance point (after moving there).
    Vex : float
        Excitation amplitude used.
    Cstd : float
        Standard capacitance used.
    Cex : float or None
        Estimated capacitance at balance.
    Closs : float or None
        Estimated loss at balance.
    """
    status: bool
    V0: complex
    A_complex: complex
    A_matrix: np.ndarray
    P: np.ndarray
    error: tuple | None
    Vex: float
    Cstd: float
    Cex: float | None = None
    Closs: float | None = None

    def __str__(self) -> str:
        s = f"ThreePointBalanceResult: {'OK' if self.status else 'OUT OF RANGE'}\n"
        s += f"  V0  = {self.V0.real:.5e} + {self.V0.imag:.5e}i  (|V0|={abs(self.V0):.5e})\n"
        s += f"  A   = {self.A_complex.real:.5e} + {self.A_complex.imag:.5e}i\n"
        if self.Cex is not None:
            s += f"  Cex = {self.Cex:.5e}  Closs = {self.Closs:.5e}\n"
        if self.error is not None:
            s += f"  Residual lock-in: X={self.error[0]:.5e}  Y={self.error[1]:.5e}\n"
        return s

@dataclass
class TwoPointBalanceResult():
    """
    Result of a two-point AC bridge balance.

    The two-point procedure takes two lock-in readings separated by a
    known step dVstd and solves for the balance point and gain in one shot.

    Attributes
    ----------
    status : bool
        True if V0 lies within Vstd_range.
    V0 : complex
        Balance point in Vstd space.
    A_complex : complex
        Effective bridge gain A = (Lf - Li) / dVstd.
    P : (2, 2) ndarray
        Initial covariance for A = [X, Y], derived from measurement
        noise and the step size.
    Vi : complex
        Initial Vstd setting.
    Li : complex
        Lock-in reading at Vi.
    Vf : complex
        Final Vstd setting (Vi + dVstd actually set on hardware).
    Lf : complex
        Lock-in reading at Vf.
    Vex : float
    Cstd : float
    Cex : float or None
    Closs : float or None
    """
    status: bool
    V0: complex
    A_complex: complex
    P: np.ndarray
    Vi: complex
    Li: complex
    Li_cov: np.ndarray
    Vf: complex
    Lf: complex
    Lf_cov: np.ndarray
    Vex: float
    Cstd: float
    Cex: float | None = None
    Closs: float | None = None

    def __str__(self) -> str:
        s = f"TwoPointBalanceResult: {'OK' if self.status else 'OUT OF RANGE'}\n"
        s += f"  V0  = {self.V0.real:.5e} + {self.V0.imag:.5e}i  (|V0|={abs(self.V0):.5e})\n"
        s += f"  A   = {self.A_complex.real:.5e} + {self.A_complex.imag:.5e}i\n"
        if self.Cex is not None:
            s += f"  Cex = {self.Cex:.5e}  Closs = {self.Closs:.5e}\n"
        return s
    
@dataclass
class CapInitialFilterConfig():
    """
    Result of cap_initialize_filter, which seeds the CapFilter from a
    prior balance result (either three-point or two-point).

    Attributes
    ----------
    status : bool
        Mirrors the status of the underlying balance result.
    A : (2,) ndarray
        Initial gain estimate [X, Y] stored in the filter.
    P : (2, 2) ndarray
        Initial covariance stored in the filter.
    V0 : complex
        Balance point used to initialize the extrapolator.
    Cex : float or None
    Closs : float or None
    """
    status: bool
    A: np.ndarray
    P: np.ndarray
    V0: complex
    Cex: float | None
    Closs: float | None

    def __str__(self) -> str:
        X, Y = self.A
        s = f"CapInitialFilterConfig: {'OK' if self.status else 'FAILED'}\n"
        s += f"  A   = {X:.5e} + {Y:.5e}i\n"
        s += f"  V0  = {self.V0.real:.5e} + {self.V0.imag:.5e}i\n"
        if self.Cex is not None:
            s += f"  Cex = {self.Cex:.5e}  Closs = {self.Closs:.5e}\n"
        return s
    
"""Internal helpers"""

def _setup_lockin_pane(frame: int = 0) -> LinePane | None:
    """
    Set up a sidecar LinePane for live lock-in monitoring during balance.
    Returns the pane, or None if no sidecar is active.
    """

    sidecar = Sidecar.instance()
    if sidecar is None:
        return None

    sidecar.clear_panes(frame=frame)

    pane = LinePane(
        name='balance lock-in',
        lines=[
            LineConfig('LX'),
            LineConfig('LY', secondary_y=True),
        ],
        x='n',
        xlabel='sample',
        ylabel='$L_X$ (V)',
        ylabel2='$L_Y$ (V)',
    )
    sidecar.add_pane(pane, frame=frame)
    return pane

def _sample_lockin(
    lockin_call: Callable[[], tuple],
    samples: int,
    pane: LinePane | None =None,
    sample_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Call lockin_call `samples` times and return the grand mean and its
    covariance.

    Each call returns (mean_i: ndarray(2,), cov_i: ndarray(2,2)) where
    cov_i is the covariance of that internal average. The grand mean and
    its covariance (assuming independent calls) are:

        mean = sum(mean_i) / N
        cov  = sum(cov_i) / N^2

    Parameters
    ----------
    lockin_call : callable -> (ndarray(2,), ndarray(2,2))
    samples : int

    Returns
    -------
    mean : ndarray(2,)
    cov : ndarray(2,2)
    """
    means = []
    covs = []
    for _ in range(samples):
        m, c = lockin_call()
        means.append(np.asarray(m, dtype=float).reshape(2))
        covs.append(np.asarray(c, dtype=float).reshape(2, 2))
        
        if pane is not None:
            pane.update(None, {
                'n': sample_offset + _,
                'LX': float(m[0]),
                'LY': float(m[1]),
            })

    mean = np.mean(means, axis=0)
    cov = np.sum(covs, axis=0) / (samples ** 2)
    return mean, cov

def _set_Vstd_complex(
    Vstd: Callable,
    Theta: Callable,
    V: complex,
    wait: float = 0.0,
    max_amp: float | None = None,
) -> complex:
    """Set Vstd from a complex value, wait, then read back the actual setting."""
    r  = float(abs(V)) if max_amp is None else float(np.clip(abs(V), 0.0, max_amp))
    th = float(np.degrees(np.angle(V)) % 360)
    Vstd(r)
    Theta(th)
    if wait > 0:
        time.sleep(wait)
    r_set = float(Vstd())
    th_set = float(Theta())
    return complex(r_set * np.cos(np.deg2rad(th_set)),
                   r_set * np.sin(np.deg2rad(th_set)))

"""Three-point balance"""

def cap_balance_three_point(
    Vstd: ScalarChannel,
    Theta: ScalarChannel,
    lockin_call: LockInChannel,
    Vex: float,
    Cstd: float,
    Vstd_range: float,
    small_step: Tuple[float, float] = (0.01, 0.01),
    large_step: Tuple[float, float] = (0.94, 0.94),
    samples: int = 100,
    fudge: float = CAP_INIT_P_MULT_THREE_POINT,
    wait: float = 3.0,
    move_to_balance: bool = True,
    robust_error: bool = False,
    ignore_range_warning: bool = False,
    logger: logging.Logger | None = None,
    plot: bool = False,
    frame: int = 0,
) -> ThreePointBalanceResult:
    """
    Perform a three-point AC bridge balance.

    Measures the lock-in signal at three Vstd settings, solves for the
    balance point V0 and effective bridge gain A, and optionally moves
    the bridge to V0.

    The algorithm follows the Ashoori thesis three-point procedure.
    The kap_bridge sign convention is used throughout:
        V0.real  -> capacitive component  -> Cex = -Cstd * V0.real / Vex
        V0.imag  -> lossy component       -> Closs = -Cstd * V0.imag / Vex

    Parameters
    ----------
    Vstd : callable
        Getter/setter for Vstd magnitude.
    Theta : callable
        Getter/setter for Vstd phase (degrees).
    get_XY : callable -> ([LX, LY], cov)
    Vex : float
        Excitation magnitude.
    Cstd : float
        Standard capacitance.
    Vstd_range : float
        Maximum Vstd magnitude.
    small_step : (float, float)
        (vc, vr) as fractions of Vstd_range for the base measurement point.
    large_step : (float, float)
        (dvc, dvr) as fractions of Vstd_range for the two offset points.
    samples : int
        Number of lock-in samples to average at each point.
    wait : float
        Seconds to wait after setting Vstd before sampling.
    move_to_balance : bool
        If True and status is good, move Vstd to V0 after solving.
    robust_error : bool default=True
        Whether to take samples when finding the residual balance error
    ignore_range_warning : bool
        If True, allow Vstd to exceed 2 V rms during measurement.
    logger : Logger, optional
    plot : bool, default=False
        Whether to plot lock-in readings during the initialization
    frame : int, default=0
        Which frame to use for plotting
    

    Returns
    -------
    ThreePointBalanceResult
    """

    vc, vr   = small_step
    dvc, dvr = large_step

    if not ignore_range_warning:
        hi = max(
            np.sqrt(vr ** 2 + (vc + dvc) ** 2),
            np.sqrt(vc ** 2 + (vr + dvr) ** 2),
        )
        Vhi = Vstd_range * hi
        if Vhi > 1.0:
            msg = (
                f"Balance cancelled: Vstd would reach {Vhi:.5f}. "
                f"Reduce step sizes, or set ignore_range_warning=True."
            )
            if logger:
                logger.warning(msg)
            else:
                print(f"WARNING: {msg}")
            return ThreePointBalanceResult(
                status=False, V0=0j, A_complex=0j,
                A_matrix=np.zeros((2, 2)), P=np.eye(2),
                error=None, Vex=Vex, Cstd=Cstd,
            )
        
    # Set up the pane for plotting, if needed
    pane = _setup_lockin_pane(frame=frame) if (plot and samples > 1) else None
    sample_count = 0    

    # Three measurement points (nominal). Actual values are read back from
    # hardware after setting and used for all subsequent calculations.
    pts_nominal = [
        complex(vc * Vstd_range, vr * Vstd_range),           # base
        complex(vc * Vstd_range, (vr + dvr) * Vstd_range),   # +dVr
        complex((vc + dvc) * Vstd_range, vr * Vstd_range),   # +dVc
    ]

    L      = np.zeros((2, 3))
    Lcov   = np.zeros((2, 2, 3))
    V_set  = []   # actual complex Vstd values read back from hardware

    for n, Vpt in enumerate(pts_nominal):
        V_actual = _set_Vstd_complex(Vstd, Theta, Vpt, wait=wait)
        V_set.append(V_actual)
        mean, cov = _sample_lockin(lockin_call, samples, 
                                   pane=pane, sample_offset=sample_count)
        sample_count += samples
        L[:, n] = mean
        Lcov[:, :, n] = cov
        if logger:
            logger.info(
                f"Three-point measurement {n+1}/3: "
                f"Vbx={V_actual.real:.5e}  Vby={V_actual.imag:.5e}  "
                f"LX={mean[0]:.5e}  LY={mean[1]:.5e}"
            )

    # Derive actual step sizes from hardware read-back
    Vc  = V_set[0].real
    Vr  = V_set[0].imag
    dVr = V_set[1].imag - Vr
    dVc = V_set[2].real - Vc

    # Sensitivity matrix: K maps (dVc, dVr) -> (dLX, dLY)
    Kc1 = (L[0, 2] - L[0, 0]) / dVc   # dLX/dVc
    Kr1 = (L[0, 1] - L[0, 0]) / dVr   # dLX/dVr
    Kc2 = (L[1, 2] - L[1, 0]) / dVc   # dLY/dVc
    Kr2 = (L[1, 1] - L[1, 0]) / dVr   # dLY/dVr

    # Balance point (Ashoori algorithm)
    P_factor = 1.0 / (1.0 - (Kc1 * Kr2) / (Kr1 * Kc2))
    Vr0 = Vr + (P_factor / Kr1) * ((Kc1 / Kc2) * L[1, 0] - L[0, 0])
    Vc0 = Vc + (P_factor / Kc2) * ((Kr2 / Kr1) * L[0, 0] - L[1, 0])
    V0 = complex(Vc0, Vr0)

    # Symmetric gain extraction. For an ideal bridge K = [[X,-Y],[Y,X]],
    # so the best estimates are:
    #   A.real = (Kc1 + Kr2) / 2
    #   A.imag = (Kc2 - Kr1) / 2
    A_real = (Kc1 + Kr2) / 2.0
    A_imag = (Kc2 - Kr1) / 2.0
    A_complex = complex(A_real, A_imag)
    A_matrix = np.array([[Kc1, Kr1], [Kc2, Kr2]])

    # Propagate measurement covariance to P via Jacobian of A w.r.t. measurements.
    # Measurement vector: [L[0,0], L[1,0], L[0,1], L[1,1], L[0,2], L[1,2]]
    # A_real = (Kc1 + Kr2) / 2 = ((L[0,2]-L[0,0])/dVc + (L[1,1]-L[1,0])/dVr) / 2
    # A_imag = (Kc2 - Kr1) / 2 = ((L[1,2]-L[1,0])/dVc - (L[0,1]-L[0,0])/dVr) / 2
    J = np.array([
        # L[0,0]      L[1,0]      L[0,1]       L[1,1]       L[0,2]      L[1,2]
        [-1/(2*dVc),  0,           0,           1/(2*dVr),   1/(2*dVc),  0         ],
        [ 1/(2*dVr), -1/(2*dVc), -1/(2*dVr),   0,           0,          1/(2*dVc) ],
    ])
    C_meas = np.zeros((6, 6))
    C_meas[0:2, 0:2] = Lcov[:, :, 0]
    C_meas[2:4, 2:4] = Lcov[:, :, 1]
    C_meas[4:6, 4:6] = Lcov[:, :, 2]
    P = fudge * J @ C_meas @ J.T

    Cex = -Cstd * Vc0 / Vex
    Closs = -Cstd * Vr0 / Vex
    in_range = abs(V0) <= Vstd_range

    error = None
    if in_range and move_to_balance:
        V0_set = _set_Vstd_complex(Vstd, Theta, V0, wait=wait)
        V0 = V0_set
        Cex = -Cstd * V0.real / Vex
        Closs = -Cstd * V0.imag / Vex

        nerr = samples if robust_error else 1
        mean_err, _ = _sample_lockin(lockin_call, nerr,
                                     pane=pane, sample_offset=sample_count)
        error = (float(mean_err[0]), float(mean_err[1]))
        
        if logger:
            logger.info(
                f"Moved to balance V0={V0.real:.5e}+{V0.imag:.5e}i. "
                f"Residual: LX={error[0]:.5e}  LY={error[1]:.5e}"
            )

    elif not in_range and logger:
        logger.warning(
            f"Balance point |V0|={abs(V0):.5e} exceeds "
            f"Vstd_range={Vstd_range:.5e}. Not moving to balance."
        )

    result = ThreePointBalanceResult(
        status = in_range,
        V0 = V0,
        A_complex = A_complex,
        A_matrix = A_matrix,
        P = P,
        error = error,
        Vex = Vex,
        Cstd = Cstd,
        Cex = Cex if in_range else None,
        Closs = Closs if in_range else None,
    )

    if logger:
        logger.info(result)
    return result

"""Two-point balance"""

def cap_balance_two_point(
    Vstd: Callable[[float | None], float],
    Theta: Callable[[float | None], float],
    lockin_call: Callable[[], tuple],
    Vex: float,
    Cstd: float,
    Vstd_range: float,
    dVstd: complex,
    Vi: complex | None = None,
    samples: int   = 1,
    wait: float = 1.0,
    fudge: float = CAP_INIT_P_MULT_TWO_POINT,
    move_to_balance: bool = True,
    logger: logging.Logger | None = None,
    plot: bool = False,
    frame: int = 0,
) -> TwoPointBalanceResult:
    """
    Perform a two-point AC bridge balance.

    Takes averaged lock-in readings at two Vstd settings separated by
    dVstd, then solves for V0 and A in one shot.

    At each point, `lockin_call` is called `samples` times and the
    results are pooled as mean = sum(mean_i)/N, cov = sum(cov_i)/N^2.
    The covariance of A is then propagated from those measurement
    covariances via the fudge factor.

    Parameters
    ----------
    Vstd : callable
    Theta : callable
    lockin_call : callable -> (ndarray(2,), ndarray(2,2))
    Vex : float
    Cstd : float
    Vstd_range : float
    dVstd : complex
        Vstd step (dVc + i*dVr, amplitude units).
    Vi : complex, optional
        Starting Vstd. If None, reads the current hardware setting.
    samples : int
        Number of lockin_call invocations at each of the two points.
    wait : float
        Seconds to wait after setting Vstd before reading the lock-in.
    fudge : float
        Multiplicative factor when propagating measurement covariance to P.
    move_to_balance : bool
        If True and status is good, move Vstd to V0.
    logger : Logger, optional
    plot : bool, default=False
        Whether to plot lock-in readings during the initialization
    frame : int, default=0
        Which frame to use for plotting

    Returns
    -------
    TwoPointBalanceResult
    """

    pane = _setup_lockin_pane(frame=frame) if (plot and samples > 1) else None
    sample_count = 0

    if Vi is None:
        r0  = float(Vstd())
        th0 = float(Theta())
        Vi = complex(r0 * np.cos(np.deg2rad(th0)), r0 * np.sin(np.deg2rad(th0)))
    else:
        Vi = _set_Vstd_complex(Vstd, Theta, Vi, wait=wait)

    Li_mean, Li_cov = _sample_lockin(lockin_call, samples,
                                     pane=pane, sample_offset=sample_count)
    sample_count += samples
    Li = complex(Li_mean[0], Li_mean[1])

    Vf = _set_Vstd_complex(Vstd, Theta, Vi + dVstd, wait=wait)
    Lf_mean, Lf_cov = _sample_lockin(lockin_call, samples,
                                     pane=pane, sample_offset=sample_count)
    Lf = complex(Lf_mean[0], Lf_mean[1])

    A_c = (Lf - Li) / (Vf - Vi)
    V0 = Vi - Li / A_c

    in_range = abs(V0) <= Vstd_range

    dV_actual = Vf - Vi
    dV_vec = np.array([dV_actual.real, dV_actual.imag])
    dV_mat = np.array([[dV_vec[0], -dV_vec[1]], [dV_vec[1],  dV_vec[0]]])
    P = fudge * dV_mat.T @ (Li_cov + Lf_cov) @ dV_mat / (abs(dV_actual) ** 4)

    Cex = None
    Closs = None
    if in_range:
        Cex = -Cstd * V0.real / Vex
        Closs = -Cstd * V0.imag / Vex

    if in_range and move_to_balance:
        V0_set = _set_Vstd_complex(Vstd, Theta, V0, wait=wait)
        V0 = V0_set
        Cex = -Cstd * V0.real / Vex
        Closs = -Cstd * V0.imag / Vex
        if logger:
            logger.info(
                f"Two-point: moved to V0={V0.real:.5e}+{V0.imag:.5e}i. "
                f"Cex={Cex:.5e}  Closs={Closs:.5e}"
            )
    elif not in_range and logger:
        logger.warning(
            f"Two-point: |V0|={abs(V0):.5e} exceeds "
            f"Vstd_range={Vstd_range:.5e}. Not moving."
        )

    result = TwoPointBalanceResult(
        status = in_range,
        V0 = V0,
        A_complex = A_c,
        P = P,
        Vi = Vi,
        Li = Li,
        Li_cov = Li_cov,
        Vf = Vf,
        Lf = Lf,
        Lf_cov = Lf_cov,
        Vex = Vex,
        Cstd = Cstd,
        Cex = Cex,
        Closs = Closs,
    )

    if logger:
        logger.info(result)
    return result

def cap_initialize_filter(
    ctx: CapContext,
    result: ThreePointBalanceResult | TwoPointBalanceResult,
) -> CapInitialFilterConfig:
    """
    Seed the CapFilter from a prior balance result and reset the extrapolator.

    Creates a CapFilter on `ctx` if one does not exist, sets A and P
    from the balance result, calls `initialize()`, then resets the
    extrapolator and seeds it with V0.

    The caller is responsible for confirming `result.status` is True
    before calling this function.

    Parameters
    ----------
    ctx : CapContext
    result : ThreePointBalanceResult or TwoPointBalanceResult

    Returns
    -------
    CapInitialFilterConfig
    """

    if ctx.cap_filter is None:
        ctx.add_filter(CapFilter(ctx))

    A = np.array([result.A_complex.real, result.A_complex.imag])
    P = result.P.copy()

    ctx.cap_filter.set_init_params(A=A, P=P)
    ctx.cap_filter.initialize()

    # Store K_matrix when available (three-point balance only)
    if isinstance(result, ThreePointBalanceResult):
        ctx.cap_filter.K_matrix = result.A_matrix.copy()
    else:
        ctx.cap_filter.K_matrix = None

    ctx.extrapolator.Vstd_range = ctx.Vstd_range
    ctx.extrapolator.clear()
    ctx.extrapolator.push(result.V0.real, result.V0.imag)

    kind = 'three' if isinstance(result, ThreePointBalanceResult) else 'two'
    ctx.log(
        f"Filter initialized from {kind}-point balance.\n"
        f"  A   = {A[0]:.5e} + {A[1]:.5e}i\n"
        f"  V0  = {result.V0.real:.5e} + {result.V0.imag:.5e}i\n"
        f"  Cex = {result.Cex}  Closs = {result.Closs}"
    )

    return CapInitialFilterConfig(
        status = result.status,
        A = A,
        P = P,
        V0 = result.V0,
        Cex = result.Cex,
        Closs = result.Closs,
    )

"""Context-level initialization helper"""
 
def cap_initialize(
    ctx: CapContext,
    method: str = 'three_point',
 
    # Three-point options
    small_step: Tuple[float, float] = (0.01, 0.01),
    large_step: Tuple[float, float] = (0.94, 0.94),
    ignore_range_warning: bool = False,
    robust_error: bool = False,
 
    # Two-point options
    dVstd: complex | None = None,
    Vi: complex | None = None,
 
    # Shared options
    fudge: float | None = None,
    samples: int   = 5,
    wait: float | None = None,
    move_to_balance: bool  = True,
    plot: bool = False,
    frame: int = 0,
) -> CapInitialFilterConfig:
    """
    Convenience wrapper that runs a full initialization pass from a CapContext.
 
    Pulls hardware accessors (Vstd, Theta, lockin_call), physical parameters 
    (Vex, Cstd, Vstd_range), and timing (settle_time) directly from `ctx`, runs 
    either a three-point or two-point balance, then calls `cap_initialize_filter` 
    to seed the filter and extrapolator.
 
    Parameters
    ----------
    ctx : CapContext
    method : {'three_point', 'two_point'}
        Which balance procedure to use.
    small_step : (float, float)
        Three-point only. (vc, vr) as fractions of Vstd_range.
    large_step : (float, float)
        Three-point only. (dvc, dvr) as fractions of Vstd_range.
    ignore_range_warning : bool
        Three-point only. Skip the amplitude safety check.
    robust_error : bool default=True
        Whether to take samples when finding the residual balance error
    dVstd : complex, optional
        Two-point only. Vstd step. Required if method='two_point'.
    Vi : complex, optional
        Two-point only. Starting Vstd; defaults to current hardware setting.
    fudge : float
        Two-point only. Multiplier on the propagated covariance.
    samples : int
        Number of lockin_call invocations per measurement point. Default 5.
    wait : float, optional
        Wait time after setting Vstd. Defaults to ctx.settle_time.
    move_to_balance : bool
        Whether to move Vstd to V0 after solving.
    plot : bool, default=False
        Whether to plot lock-in readings during the initialization
    frame : int, default=0
        Which frame to use for plotting
 
    Returns
    -------
    CapInitialFilterConfig
    """
    if wait is None:
        wait = ctx.settle_time
 
    if method == 'three_point':
        if fudge is None:
            fudge = CAP_INIT_P_MULT_THREE_POINT
        bal = cap_balance_three_point(
            Vstd = ctx.Vstd,
            Theta = ctx.Theta,
            lockin_call = ctx.lockin_call,
            Vex = ctx.Vex,
            Cstd = ctx.Cstd,
            Vstd_range = ctx.Vstd_range,
            small_step = small_step,
            large_step = large_step,
            samples = samples,
            wait = wait,
            move_to_balance = move_to_balance,
            robust_error = robust_error,
            ignore_range_warning = ignore_range_warning,
            logger = ctx.logger,
            plot = plot,
            frame = frame,
        )
    elif method == 'two_point':
        if fudge is None:
            fudge = CAP_INIT_P_MULT_TWO_POINT
        
        if Vi is None: 
            Vi = 0.01 * ctx.Vstd_range
        if dVstd is None:
            dVstd = 0.93 * ctx.Vstd_range

        bal = cap_balance_two_point(
            Vstd = ctx.Vstd,
            Theta = ctx.Theta,
            lockin_call = ctx.lockin_call,
            Vex = ctx.Vex,
            Cstd = ctx.Cstd,
            Vstd_range = ctx.Vstd_range,
            dVstd = dVstd,
            Vi = Vi,
            samples = samples,
            wait = wait,
            fudge = fudge,
            move_to_balance = move_to_balance,
            logger = ctx.logger,
            plot = plot,
            frame = frame,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'three_point' or 'two_point'.")
 
    return cap_initialize_filter(ctx, bal)