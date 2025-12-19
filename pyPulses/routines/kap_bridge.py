from ..thread_job import _checkpoint
from ..utils.kalman import kalman
from .cap_utils import balanceCapBridgeTwoPoint, TwoPointCapBalance

import logging
from collections import deque
from dataclasses import dataclass
import numpy as np

from typing import Any, Callable, Tuple

def extrap(x: np.ndarray, order: int) -> float:
    coeff = np.polyfit(np.arange(len(x)), x, min(order, len(x) - 1))        
    poly = np.poly1d(coeff)
    return poly(len(x))

def rth_cov_to_xy_cov(Q_rth: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = np.sqrt(v[0]**2 + v[1]**2)
    th = np.arctan2(v[1], v[0])
    c = np.cos(th)
    s = np.sin(th)
    J = np.array([[c, -r * s], [s, r * c]])
    return J @ Q_rth @ J.T

class KFilter():
    def __init__(self, 
                 A: np.ndarray, 
                 P: np.ndarray,
                 support: int, 
                 order: int, 
                 process_noise_coeff: float = 0.01,
                 ):
        self.order = order
        
        # History of balance points in terms of Vstd
        self.x_hist = deque(maxlen = support)
        self.y_hist = deque(maxlen = support)

        """
        The state vector is A = (X, Y); X + iY is the effective small signal
        gain of the capacitance bridge. Given a step (dx, dy) in Vstd, the 
        change in the lock-in reading is
                        (X * dx - Y * dy, Y * dx + X * dy)
        f(x, u) is the predicted evolution of the state (x). In this case, we
        expect minimal variation in the complex gain, so our model for state 
        evolution is f(x) = x. We estimate the process noise by assuming 
        uncorrelated errors in the modulus and phase of the complex gain. If the
        error in phase is dθ and the error in modulus is dR, we can represent
        the covariance in cartesian coordinates by transforming basis under the
        Jacobian. In this case:
            Q = J Q_p J^T, J = ((X/R, -Y), (Y/R, X)), Q_p = diag(dR^2, dθ^2)
        I assume dR = p * R and dθ = p * 2pi. p is the process noise 
        coefficient. This gives me a Q of
                    p^2 ((X^2 + (2pi Y)^2, (1 - (2pi)^2) XY), 
                        ((1 - (2pi)^2) XY, Y^2 + (2pi X)^2))
        h(x, v) is the predicted change in lock-in reading given a step 
        dv = dVstd = (dx, dy). As discussed, this is given by 
                        (X * dx - Y * dy, Y * dx + X * dy)
        The Jacobian with respect to A is H = ((dx, -dy), (dy, dx)). For 
        clarity, I label the state vector as A and the extra arguments for h as
        dv. I also rename f and h to predicted_gain and lockin_response.
        """

        def predicted_gain(A):
            X, Y = A
            Q = process_noise_coeff**2 * rth_cov_to_xy_cov(
                np.diag([ (X**2 + Y**2), (2 * np.pi)**2 ]), A)
            return A, np.eye(2), Q
        
        def lockin_response(A, dv):
            dx, dy = dv
            H = np.array([[dx, -dy], [dy, dx]])
            return H @ A, H
        
        self.kalman = kalman(
            f_F = predicted_gain,
            h_H = lockin_response,
            x = A,
            P = P,
        )

    def append(self, x, y):
        self.x_hist.append(x)
        self.y_hist.append(y)

    def extrapolate(self) -> Tuple[float, float]:
        return extrap(self.x_hist, self.order), extrap(self.y_hist, self.order)

@dataclass
class KapBridgeBalanceResult():
    status  : bool
    Vb      : Tuple[float, float]
    prev_Vb : Tuple[float, float]
    L       : Tuple[float, float]
    prev_L  : Tuple[float, float]
    R       : np.ndarray | None
    prev_R  : np.ndarray | None
    A       : np.ndarray
    P       : np.ndarray | None
    init_L  : Tuple[float, float]
    errt    : float
    itr     : int

@dataclass
class KapBridgeContext():
    Vstd                        : Callable[[float | None], Any]
    Theta                       : Callable[[float | None], Any]
    get_xy                      : Callable[[], Tuple[np.ndarray, np.ndarray]]
    time_const                  : Callable[[float], Any]
    Vstd_range                  : float
    abs_amp_resolution          : float = 0.5 * 1 / 4096    # tailored to AD9854 (unitless)
    phase_resolution            : float = 0.5 * 360 / 16384 # tailored to AD9854 (degrees)
    raw_settle_tc               : float = 5.0
    raw_balance_small_val       : float = 0.01
    raw_balance_step_size       : float = 0.5
    raw_balance_get_xy          : Callable[[], Any] | None = None
    raw_balance_pre_callback    : Callable | None = None
    raw_balance_post_callback   : Callable | None = None
    raw_balance_errmult         : float = 1.0
    max_tries                   : int = 10
    order                       : int = 2
    support                     : int = 3
    process_noise_coeff         : float = 0.01
    store_failed_balances       : bool = True # Usually the failed ones are still close
    erroff                      : float = 0.0
    errmult                     : float = 2.0
    iteration_callback          : Callable = None
    logger                      : logging.Logger = None

    def __post_init__(self):
        self.kfilter: KFilter | None = None
        self.prev_result: KapBridgeBalanceResult | None = None
        self.prev_lockin_reading: float | None = None

    def add_kfilter(self, A: np.ndarray, P: np.ndarray):
        self.kfilter = KFilter(A, P, self.support, self.order, self.process_noise_coeff)
    
    def clear_kfilter(self):
        self.kfilter = None

    def update(self, dL, R, dv):
        self.kfilter.kalman.update(dL, R, dv)
    
    def predict(self):
        return self.kfilter.kalman.predict()
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kfilter.kalman.x.copy(), self.kfilter.kalman.P.copy()

    def append(self, x: float, y: float):
        self.kfilter.append(x, y)

    def extrapolate(self) -> Tuple[float, float]:
        return self.kfilter.extrapolate()

    def log(self, *args):
        if self.logger:
            self.logger.info(*args)

def balanceKapBridge(ctx: KapBridgeContext) -> KapBridgeBalanceResult:
    
    # if we're starting from scratch, perform a raw two-point balance
    if ctx.kfilter is None:        
        Vi_mag = ctx.raw_balance_small_val * ctx.Vstd_range
        ctx.Vstd(Vi_mag)
        ctx.Theta(0.0)

        if ctx.raw_balance_pre_callback:
            ctx.raw_balance_pre_callback(ctx)

        balance = balanceCapBridgeTwoPoint(
            Vstd = ctx.Vstd,
            Theta = ctx.Theta,
            Vout = ctx.raw_balance_get_xy or ctx.get_xy,
            Vstd_range = ctx.Vstd_range,
            dVstd = ctx.Vstd_range * ctx.raw_balance_step_size,
            settle_time = ctx.raw_settle_tc * ctx.time_const(),
            fudge = ctx.raw_balance_errmult,
        )

        if ctx.raw_balance_post_callback:
            ctx.raw_balance_post_callback(ctx)

        if not balance.status:
            raise RuntimeError("Expected balance point falls out of bounds.")

        A = np.array([balance.A.real, balance.A.imag])
        P = balance.P
        ctx.add_kfilter(A, P)
        xb = balance.V0.real
        yb = balance.V0.imag

    # calculate a projected balance point
    else:
        xb, yb = ctx.extrapolate()
    
    ctx.log(f"Projected x = {xb:.5f} V, y = {yb:.5f} V")

    rb = np.sqrt(xb**2 + yb**2)
    thb = np.degrees(np.arctan2(yb, xb)) % 360
    if rb > ctx.Vstd_range:
        new_rb = ctx.Vstd_range
        xb *= new_rb / rb
        yb *= new_rb / rb
        rb = new_rb
        ctx.log(f"Truncated to x = {xb:.5f} V, y = {yb:.5f} V")
    
    # iteratively move to the balance point until error bound is met
    prev_LX = None
    prev_LY = None
    prev_R = np.diag([np.inf, np.inf])
    initial_L = None
    for itr in range(ctx.max_tries):
        _checkpoint()

        ctx.log(
            "==================================================\n"
            f"Set r = {rb:.5f}, th = {thb:.3f}"
        )
        ctx.Vstd(rb)
        ctx.Theta(thb)

        # Have to properly round everything based on the resolution of the AC source
        rb = ctx.Vstd()
        thb = ctx.Theta()
        xb = rb * np.cos(np.deg2rad(thb))
        yb = rb * np.sin(np.deg2rad(thb))

        if ctx.iteration_callback:
            ctx.iteration_callback(itr, ctx)

        L, R = ctx.get_xy()
        LX, LY = L
        if itr == 0:
            initial_L = (LX, LY)
        ctx.prev_lockin_reading = (LX, LY)

        ctx.log(
            f"Lock-in reading: LX = {LX:.5e}, LY = {LY:.5e}\n"
            f"     Covariance: ┏                          ┓\n"
            f"                 ┃ {f"{R[0,0]:.5e}":<12}{f"{R[0,0]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{R[0,0]:.5e}":<12}{f"{R[0,0]:.5e}":>12} ┃\n"
            f"                 ┗                          ┛"
        )

        pA = None
        pP = None
        if not itr == 0:
            # We gain more information about the effective gain of the bridge
            # setup as we take more measurements. Here we update the Kalman
            # filter with this new information.

            dLX = LX - prev_LX
            dLY = LY - prev_LY
            dvx = xb - prev_xb
            dvy = yb - prev_yb
            R_diff = R + prev_R

            dL = np.array([dLX, dLY])
            dv = np.array([dvx, dvy])

            # Since dv is treated as a certain parameter by the Kalman filter,
            # we want to account for the uncertainty in its true value due to 
            # finite resolution by mixing it into the measurement uncertainty.

            A, P = ctx.get_state()
            Amat = np.array([[A[0], -A[1]], [A[1], A[0]]])
            R_dv_rth = np.diag([ctx.abs_amp_resolution**2, 
                             (ctx.phase_resolution * np.pi / 180)**2])
            R_dv = rth_cov_to_xy_cov(R_dv_rth, np.array([xb, yb])) + \
                    rth_cov_to_xy_cov(R_dv_rth, np.array([prev_xb, prev_yb]))
            R_diff_eff = R_diff + Amat @ R_dv @ Amat.T

            ctx.log(
                "Incorporating new measurement into Kalman filter.\n"
                f"dL = [{dLX:.5e}, {dLY:.5e}]\n"
                f"     Covariance: ┏                          ┓\n"
                f"                 ┃ {f"{R_diff_eff[0,0]:.5e}":<12}"
                f"{f"{R_diff_eff[0,0]:.5e}":>12} ┃\n"
                f"                 ┃ {f"{R_diff_eff[0,0]:.5e}":<12}"
                f"{f"{R_diff_eff[0,0]:.5e}":>12} ┃\n"
                f"                 ┗                          ┛\n"
                f"Size of Uncertainties:\n"
                f"              P: {P[0,0]:.5e}, {P[1,1]:.5e}\n"
                f"         R_diff: {R_diff[0,0]:.5e}, {R_diff[1,1]:.5e}\n"
                f"     R_diff_eff: {R_diff_eff[0,0]:.5e}, {R_diff_eff[1,1]:.5e}"
            )

            ctx.update(dL, R_diff_eff, dv)
            pA = A
            pP = P

        ctx.predict()
        A, P = ctx.get_state()
        if pA is not None:
            Amag = np.sqrt(pA[0]*pA[0] + pA[1]*pA[1])
            dAmag = np.sqrt((A[0]-pA[0])**2 + (A[1]-pA[1])**2)
            ctx.log(f"Relative change in gain magnitude: {(dAmag / Amag * 100):.5f} %")

        ctx.log(
            "Predicted Bridge Gain\n"
            f"X = {A[0]:.5e}, Y = {A[1]:.5e}\n"
            f"     Covariance: ┏                          ┓\n"
            f"                 ┃ {f"{P[0,0]:.5e}":<12}{f"{P[0,0]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{P[0,0]:.5e}":<12}{f"{P[0,0]:.5e}":>12} ┃\n"
            f"                 ┗                          ┛"
        )

        # based on the gain, determine how far off we were
        absA2 = A[0]*A[0] + A[1]*A[1]
        dx = (A[0] * LX + A[1] * LY) / absA2
        dy = (A[0] * LY - A[1] * LX) / absA2

        # store the old state
        prev_xb = xb
        prev_yb = yb
        prev_rb = rb

        # move to the new projected balance point
        xb -= dx
        yb -= dy
        rb = np.sqrt(xb**2 + yb**2)
        thb = np.degrees(np.arctan2(yb, xb)) % 360

        ctx.log(
            f"Corrected balance at x = {xb:.5f}, y = {yb:.5f}\n"
            f"(relative change of {100*(rb/prev_rb - 1):.5f} %)"
        )

        # If the lock-in reading was sufficiently small, terminate.
        # By sufficiently small we mean that it is indistinguishible from 0
        # within the uncertainty of the measurement.

        # The threshold is largely set by the first measurement we take, and
        # later we mix it slightly with subsequent measurements.

        LR = np.sqrt(LX * LX + LY * LY)
        L = np.array([[LX], [LY]])
        LRerr = np.sqrt(np.sum(L.T @ R @ L)) / LR

        err_bound = ctx.erroff + ctx.errmult * LRerr
        if itr == 0:
            errt = err_bound
        else:
            errt = 0.95 * errt + 0.05 * err_bound

        ctx.log(f"Error Tolerance: {errt:.5e}\n")

        # If the requested change butts up against the resolutions of the AC
        # source, we terminate to avoid meaningless corrections.
        if abs(rb - prev_rb) < ctx.abs_amp_resolution and \
           abs((thb - np.degrees(np.arctan2(prev_yb, prev_xb))) % 360) < ctx.phase_resolution:
            ctx.log(
                f"Balanced on iteration {itr + 1} due to resolution limit.\n"
                f"  Previous  r = {prev_rb:.5f}, th = {np.degrees(np.arctan2(prev_yb, prev_xb)):.3f}\n"
                f"  Requested r = {rb:.5f}, th = {thb:.3f}\n"
            )

            # append the new balance point ot the historical data
            ctx.append(xb, yb)

            result = KapBridgeBalanceResult(
                status  = True,
                Vb      = (xb, yb),
                prev_Vb = (prev_xb, prev_yb),
                L       = (LX, LY),
                prev_L  = (prev_LX, prev_LY),
                R       = R,
                prev_R  = prev_R,
                A       = A,
                P       = P,
                init_L  = initial_L,
                errt    = errt,
                itr     = itr,
            )
            break

        if LR < errt:
            ctx.log(
                f"Balanced on iteration {itr + 1}.\n"
                f"  Error Tolerance: {errt:.5e}\n"
                f"  Lock-in Reading: {LR:.5e}\n"
            )

            # append the new balance point ot the historical data
            ctx.append(xb, yb)

            result = KapBridgeBalanceResult(
                status  = True,
                Vb      = (xb, yb),
                prev_Vb = (prev_xb, prev_yb),
                L       = (LX, LY),
                prev_L  = (prev_LX, prev_LY),
                R       = R,
                prev_R  = prev_R,
                A       = A,
                P       = P,
                init_L  = initial_L,
                errt    = errt,
                itr     = itr,
            )
            break
        
        # store the previous measurement and uncertainty
        prev_LX = LX
        prev_LY = LY
        prev_R = R

    # if we reach this point, we have not balanced after max_tries
    else:
        ctx.log(f"Failed to balance after {ctx.max_tries} tries.")

        if ctx.store_failed_balances:
            # append the new balance point ot the historical data
            ctx.append(xb, yb)

        result = KapBridgeBalanceResult(
            status  = False,
            Vb      = (xb, yb),
            prev_Vb = (prev_xb, prev_yb),
            L       = (LX, LY),
            prev_L  = (prev_LX, prev_LY),
            R       = R,
            prev_R  = prev_R,
            A       = A,
            P       = P,
            init_L  = initial_L,
            errt    = errt,
            itr     = itr
        )

    ctx.prev_result = result
    return result