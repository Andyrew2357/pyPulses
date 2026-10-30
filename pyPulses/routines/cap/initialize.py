from __future__ import annotations

from .cap_filter import CapFilter
from .context import CapContext
from .config import (
    CAP_INIT_P_MULT_THREE_POINT, CAP_INIT_P_MULT_TWO_POINT, VSTD_MAX_DEFAULT,
)
from ...devices.channel_adapter import ScalarChannel, LockInChannel
from ...core.job import checkpoint
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
        True if V0 lies within Vstd_max.
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

def three_point_to_dict(r: ThreePointBalanceResult) -> dict:
    """JSON-safe dict of a ThreePointBalanceResult (complex -> [re, im])."""
    return dict(
        status = bool(r.status),
        V0 = [float(r.V0.real), float(r.V0.imag)],
        A_complex = [float(r.A_complex.real), float(r.A_complex.imag)],
        A_matrix = np.asarray(r.A_matrix, dtype=float).tolist(),
        P = np.asarray(r.P, dtype=float).tolist(),
        error = None if r.error is None else [float(e) for e in r.error],
        Vex = float(r.Vex),
        Cstd = float(r.Cstd),
        Cex = None if r.Cex is None else float(r.Cex),
        Closs = None if r.Closs is None else float(r.Closs),
    )

def three_point_from_dict(d: dict) -> ThreePointBalanceResult:
    """
    Inverse of three_point_to_dict. Keys other than status/V0/A_matrix/P/
    Vex/Cstd are optional, so older CapContext 'Crosstalk_kwargs' blocks load.
    """
    def _c(v):
        return complex(v[0], v[1])
    err = d.get('error')
    return ThreePointBalanceResult(
        status = bool(d['status']),
        V0 = _c(d['V0']),
        A_complex = _c(d.get('A_complex', [0.0, 0.0])),
        A_matrix = np.array(d['A_matrix'], dtype=float),
        P = np.array(d['P'], dtype=float),
        error = None if err is None else tuple(float(e) for e in err),
        Vex = float(d['Vex']),
        Cstd = float(d['Cstd']),
        Cex = d.get('Cex'),
        Closs = d.get('Closs'),
    )

@dataclass
class TwoPointBalanceResult():
    """
    Result of a two-point AC bridge balance.

    The two-point procedure takes two lock-in readings separated by a
    known step dVstd and solves for the balance point and gain in one shot.

    Attributes
    ----------
    status : bool
        True if V0 lies within Vstd_max.
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
    Vstd_max: float = VSTD_MAX_DEFAULT,
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
        Scale of the calibration steps (not a limit; see Vstd_max).
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
    robust_error : bool, default=False
        Whether to take `samples` readings (rather than one) for the
        residual balance error.
    Vstd_max : float, default=VSTD_MAX_DEFAULT
        Hard limit on |Vstd| (normally ctx.Vstd_max). Raises ValueError
        before moving if any of the three points would exceed it; a balance
        point beyond it gives status=False and is not moved to.
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

    Vhi = Vstd_range * max(
        np.sqrt(vr ** 2 + (vc + dvc) ** 2),
        np.sqrt(vc ** 2 + (vr + dvr) ** 2),
    )
    if Vhi > Vstd_max:
        raise ValueError(
            f"Three-point balance cancelled: Vstd would reach {Vhi:.5g} > "
            f"Vstd_max={Vstd_max:.5g}. Reduce Vstd_range or the step sizes, "
            f"or raise Vstd_max."
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
    in_range = abs(V0) <= Vstd_max

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
            f"Vstd_max={Vstd_max:.5e}. Not moving to balance."
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
    dVstd: complex,
    Vi: complex | None = None,
    samples: int   = 1,
    wait: float = 1.0,
    fudge: float = CAP_INIT_P_MULT_TWO_POINT,
    move_to_balance: bool = True,
    Vstd_max: float = VSTD_MAX_DEFAULT,
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
    Vstd_max : float, default=VSTD_MAX_DEFAULT
        Hard limit on |Vstd| (normally ctx.Vstd_max). Raises ValueError
        before moving if Vi or Vi + dVstd would exceed it; a balance point
        beyond it gives status=False and is not moved to.
    logger : Logger, optional
    plot : bool, default=False
        Whether to plot lock-in readings during the initialization
    frame : int, default=0
        Which frame to use for plotting

    Returns
    -------
    TwoPointBalanceResult
    """

    if Vi is None:
        r0  = float(Vstd())
        th0 = float(Theta())
        Vi = complex(r0 * np.cos(np.deg2rad(th0)), r0 * np.sin(np.deg2rad(th0)))
        set_Vi = False
    else:
        set_Vi = True

    Vhi = max(abs(Vi), abs(Vi + dVstd))
    if Vhi > Vstd_max:
        raise ValueError(
            f"Two-point balance cancelled: Vstd would reach {Vhi:.5g} > "
            f"Vstd_max={Vstd_max:.5g}. Reduce Vi/dVstd or raise Vstd_max."
        )

    pane = _setup_lockin_pane(frame=frame) if (plot and samples > 1) else None
    sample_count = 0

    if set_Vi:
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

    in_range = abs(V0) <= Vstd_max

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
            f"Vstd_max={Vstd_max:.5e}. Not moving."
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

"""Balance refinement"""

def _invert_gain(
    LX: float, LY: float,
    A_matrix: np.ndarray | None = None,
    A_complex: complex | None = None,
) -> Tuple[float, float] | None:
    """
    Invert a lock-in reading (LX, LY) through a calibrated gain to a
    Vstd-space correction (dVx, dVy), exactly mirroring cap_measure's two
    inversion branches -- the full 2x2 matrix (three-point) or the
    compressed complex gain (two-point). Returns None if the gain is
    degenerate (exactly zero, or non-finite) rather than raising or
    silently producing inf/nan -- this is the direct numerical-safety
    counterpart to "what if the amp has ~zero gain": see
    cap_noise_probe's docstring for the physical discussion.
    """
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


@dataclass
class RefineBalanceResult:
    """
    Result of cap_balance_refine.

    Attributes
    ----------
    status : bool
        True if all n_refine steps completed with V0 staying inside
        Vstd_max and the local gain non-degenerate throughout. False
        means refinement was aborted early -- V0/residual reflect the
        last good step, not what was requested.
    V0 : complex
        Refined balance point.
    residual : (float, float) or None
        Lock-in reading (LX, LY) at V0 after the final refinement step.
    Cex, Closs : float or None
        None if status is False.
    n_refine : int
        Number of refinement steps actually completed (may be less than
        requested if refinement was aborted early).
    """
    status: bool
    V0: complex
    residual: tuple | None
    Cex: float | None
    Closs: float | None
    n_refine: int

    def __str__(self) -> str:
        s = f"RefineBalanceResult: {'OK' if self.status else 'ABORTED EARLY'}\n"
        s += f"  V0 = {self.V0.real:.5e} + {self.V0.imag:.5e}i  ({self.n_refine} step(s))\n"
        if self.Cex is not None:
            s += f"  Cex = {self.Cex:.5e}  Closs = {self.Closs:.5e}\n"
        if self.residual is not None:
            s += f"  Residual: LX={self.residual[0]:.5e}  LY={self.residual[1]:.5e}\n"
        return s


def cap_balance_refine(
    ctx: CapContext,
    n_refine: int = 1,
    samples: int = 1,
    wait: float | None = None,
    result: ThreePointBalanceResult | TwoPointBalanceResult | None = None,
    use_matrix: bool = True,
    logger: logging.Logger | None = None,
) -> RefineBalanceResult:
    """
    Re-null the bridge with `n_refine` fixed-gain Newton steps, using the
    gain already stored on `ctx`.

        cap.cap_balance_refine(ctx)              # one step from where Vstd is
        cap.cap_balance_refine(ctx, n_refine=3)

    Each step: move to the current best V0, read the residual lock-in
    signal there, and correct V0 by inverting the stored gain -- the gain
    itself is *not* re-estimated (contrast cap_balance's Kalman tracking).
    By default the gain is ctx.cap_filter's (K_matrix when present and
    `use_matrix`, as cap_measure does, else A) and the first step starts
    from the current Vstd, so this works right after cap_initialize,
    CapCalibration.apply, or a gate move away from the balance point.
    Vstd is left at the refined V0; the filter and extrapolator are not
    touched.

    Parameters
    ----------
    ctx : CapContext
    n_refine : int, default=1
        Number of correction steps to take.
    samples : int, default=1
        lockin_call averaging per step.
    wait : float, optional
        Seconds to wait after setting Vstd before reading the lock-in.
        Defaults to ctx.settle_time.
    result : ThreePointBalanceResult or TwoPointBalanceResult, optional
        Take the gain (A_matrix for three-point, A_complex for two-point)
        and starting V0 from this result instead of ctx.cap_filter and the
        current Vstd.
    use_matrix : bool, default True
        Use the full 2x2 gain when available, else the compressed complex
        gain.
    logger : Logger, optional
        Defaults to ctx.logger.

    Returns
    -------
    RefineBalanceResult
    """
    if wait is None:
        wait = ctx.settle_time
    if logger is None:
        logger = ctx.logger

    if result is not None:
        V0 = result.V0
        if use_matrix and isinstance(result, ThreePointBalanceResult):
            A_matrix, A_complex = result.A_matrix, None
        else:
            A_matrix, A_complex = None, result.A_complex
    else:
        if ctx.cap_filter is None:
            raise RuntimeError(
                "cap_balance_refine needs a stored gain: initialize the "
                "filter first (cap_initialize, CapCalibration.apply, ...) "
                "or pass result=."
            )
        V0 = ctx.get_Vstd_complex()
        K = ctx.cap_filter.K_matrix
        if use_matrix and K is not None:
            A_matrix, A_complex = K, None
        else:
            X, Y = ctx.cap_filter.get_A()
            A_matrix, A_complex = None, complex(X, Y)

    def _move(V: complex) -> complex:
        V = ctx.set_Vstd_complex(V)     # clamps to Vstd_max
        if wait > 0:
            time.sleep(wait)
        return V

    residual: tuple | None = None
    for i in range(n_refine):
        checkpoint()
        V0 = _move(V0)
        mean, _ = _sample_lockin(ctx.lockin_call, samples)
        LX, LY = float(mean[0]), float(mean[1])
        residual = (LX, LY)

        dV = _invert_gain(LX, LY, A_matrix=A_matrix, A_complex=A_complex)
        if dV is None:
            if logger:
                logger.warning(
                    "cap_balance_refine: degenerate local gain "
                    f"(step {i + 1}/{n_refine}); stopping early."
                )
            return RefineBalanceResult(
                status=False, V0=V0, residual=residual,
                Cex=None, Closs=None, n_refine=i,
            )

        V0_new = V0 - complex(dV[0], dV[1])
        if abs(V0_new) > ctx.Vstd_max:
            if logger:
                logger.warning(
                    f"cap_balance_refine: step {i + 1}/{n_refine} would push "
                    "V0 outside Vstd_max; stopping early."
                )
            return RefineBalanceResult(
                status=False, V0=V0, residual=residual,
                Cex=None, Closs=None, n_refine=i,
            )
        V0 = V0_new

    V0 = _move(V0)
    mean, _ = _sample_lockin(ctx.lockin_call, samples)
    residual = (float(mean[0]), float(mean[1]))

    Cex = -ctx.Cstd * V0.real / ctx.Vex
    Closs = -ctx.Cstd * V0.imag / ctx.Vex

    if logger:
        logger.info(
            f"cap_balance_refine: refined V0={V0.real:.5e}+{V0.imag:.5e}i "
            f"after {n_refine} step(s). "
            f"Residual: LX={residual[0]:.5e}  LY={residual[1]:.5e}"
        )

    return RefineBalanceResult(
        status=True, V0=V0, residual=residual,
        Cex=Cex, Closs=Closs, n_refine=n_refine,
    )


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

    ctx.extrapolator.Vstd_max = ctx.Vstd_max
    ctx.extrapolator.clear()
    ctx.extrapolator.push(result.V0.real, result.V0.imag)
    ctx._Vstd_now = None

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
    (Vex, Cstd, Vstd_range, Vstd_max), and timing (settle_time) directly from
    `ctx`, runs either a three-point or two-point balance, then calls
    `cap_initialize_filter` to seed the filter and extrapolator.

    Raises ValueError (before moving) if the calibration points would exceed
    ctx.Vstd_max, and RuntimeError if the balance point lands beyond
    ctx.Vstd_max; the filter is left untouched in both cases. Vstd_range
    only sets the size of the calibration steps.

    Parameters
    ----------
    ctx : CapContext
    method : {'three_point', 'two_point'}
        Which balance procedure to use.
    small_step : (float, float)
        Three-point only. (vc, vr) as fractions of Vstd_range.
    large_step : (float, float)
        Three-point only. (dvc, dvr) as fractions of Vstd_range.
    robust_error : bool, default=False
        Three-point only. Take `samples` readings (rather than one) for the
        residual balance error.
    dVstd : complex, optional
        Two-point only. Vstd step. Defaults to 0.93 * Vstd_range.
    Vi : complex, optional
        Two-point only. Starting Vstd. Defaults to 0.01 * Vstd_range.
    fudge : float, optional
        Multiplier on the propagated covariance P. Defaults to
        CAP_INIT_P_MULT_THREE_POINT / CAP_INIT_P_MULT_TWO_POINT.
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
            fudge = fudge,
            move_to_balance = move_to_balance,
            robust_error = robust_error,
            Vstd_max = ctx.Vstd_max,
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
            dVstd = dVstd,
            Vi = Vi,
            samples = samples,
            wait = wait,
            fudge = fudge,
            move_to_balance = move_to_balance,
            Vstd_max = ctx.Vstd_max,
            logger = ctx.logger,
            plot = plot,
            frame = frame,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'three_point' or 'two_point'.")

    if not bal.status:
        raise RuntimeError(
            f"{method} balance out of range: |V0|={abs(bal.V0):.3e} > "
            f"Vstd_max={ctx.Vstd_max:.3e}. Filter not initialized."
        )
    return cap_initialize_filter(ctx, bal)