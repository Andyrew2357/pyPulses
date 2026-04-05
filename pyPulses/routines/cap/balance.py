from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import CapContext

from dataclasses import dataclass
import numpy as np
import time


@dataclass
class CapBalanceResult:
    """
    Result of a single on-balance capacitance measurement (`cap_balance`).

    The bridge has been iteratively re-balanced. The Kalman filter has
    been updated with all (dL, dv) observations collected during the run,
    and the balance point has been pushed onto the extrapolator history.

    Attributes
    ----------
    status : bool
        True if the balance converged within `max_tries`.
    n_iter : int
        Number of Vstd adjustment iterations performed.
    Vb : (float, float)
        Final balance point (Vbx, Vby).
    L : (float, float)
        Lock-in reading (LX, LY) at the final balance point.
    init_L : (float, float)
        Lock-in reading at the very first iteration (before any correction).
    Cex : float or None
        Capacitance estimated from the final balance point.
    Closs : float or None
        Loss estimated from the final balance point.
    A : (2,) ndarray
        Complex gain [X, Y] at the end of the balance (after all updates).
    P : (2, 2) ndarray
        Covariance at the end of the balance.
    errt : float
        Final error tolerance used for the termination criterion.
    itr : int
        Iteration index at termination.
    """
    status:  bool
    n_iter:  int
    Vb:      tuple
    L:       tuple
    init_L:  tuple
    Cex:     float | None
    Closs:   float | None
    A:       np.ndarray
    P:       np.ndarray
    errt:    float
    itr:     int

    def __str__(self) -> str:
        s = f"CapBalanceResult: {'SUCCESS' if self.status else 'FAILED'} "
        s += f"(n_iter={self.n_iter})\n"
        s += f"  Vb    = ({self.Vb[0]:.5e}, {self.Vb[1]:.5e})\n"
        s += f"  L     = ({self.L[0]:.5e}, {self.L[1]:.5e})\n"
        if self.Cex is not None:
            s += f"  Cex   = {self.Cex:.5e}  Closs = {self.Closs:.5e}\n"
        s += f"  errt  = {self.errt:.5e}\n"
        s += (
            f"  A     = {self.A[0]:.5e} + {self.A[1]:.5e}i\n"
        )
        return s


def cap_balance(ctx: 'CapContext', verbose: bool = False) -> CapBalanceResult:
    """
    Perform a single on-balance capacitance measurement.

    Iteratively moves Vstd toward the balance point, updating the Kalman filter 
    gain estimate with each (dL, dv) observation. The extrapolator history is 
    used to seed the initial Vstd guess; the final balance point is pushed onto 
    the history after a successful balance.

    The caller is responsible for:
      - Calling `cap_initialize()` before the first call.
      - Calling this function at each new phase-space point.

    Parameters
    ----------
    ctx : CapContext
    verbose : bool
        Whether to do verbose logging

    Returns
    -------
    CapBalanceResult
    """
    
    if ctx.cap_filter is None:
        raise RuntimeError(
            "cap_balance called before filter initialization. "
            "Call cap_initialize() first."
        )

    # ───────────────────────────────────────────────────────────────────
    # SETUP: project balance point from extrapolator history
    # ───────────────────────────────────────────────────────────────────

    extrap = ctx.extrapolator.extrapolate()
    if extrap is not None:
        Vbx, Vby = extrap
    else:
        V_now = ctx.get_Vstd_complex()
        Vbx, Vby = V_now.real, V_now.imag

    # Clamp to Vstd_range
    rb  = np.sqrt(Vbx**2 + Vby**2)
    if rb > ctx.Vstd_range:
        scale = ctx.Vstd_range / rb
        Vbx *= scale
        Vby *= scale
        rb = ctx.Vstd_range

    thb = float(np.degrees(np.arctan2(Vby, Vbx)) % 360)

    ctx.log(
        f"cap_balance: projected Vb=({Vbx:.5e}, {Vby:.5e})  "
        f"r={rb:.5e}  th={thb:.3f} deg"
    )

    # ───────────────────────────────────────────────────────────────────
    # BALANCE LOOP
    # ───────────────────────────────────────────────────────────────────

    prev_LX: float | None = None
    prev_LY: float | None = None
    prev_R: np.ndarray = np.diag([np.inf, np.inf])
    prev_rb: float | None = None
    prev_xb: float | None = None
    prev_yb: float | None = None

    initial_L: tuple | None = None
    errt: float = 0.0
    LX = LY = 0.0
    result: CapBalanceResult | None = None

    for itr in range(ctx.max_tries):

        ctx.log(
            f"  Iteration {itr + 1}/{ctx.max_tries}: "
            f"set r={rb:.5e}  th={thb:.3f}"
        )

        # ── Set Vstd ────────────────────────────────────────────────
        rb_set, thb_set = ctx.set_Vstd(rb, thb)
        rb = rb_set
        thb = thb_set
        rad = np.deg2rad(thb)
        xb = rb * np.cos(rad)
        yb = rb * np.sin(rad)

        # ── Iteration_callback ──────────────────────────────────────
        if ctx.iteration_callback is not None:
            ctx.iteration_callback(
                ctx = ctx,
                itr = itr,
                Vb  = (xb, yb),
                L   = (prev_LX, prev_LY),
            )

        # ── Optional rescale ────────────────────────────────────────
        if ctx.sensitivity_channel is not None:
            if itr == 0 and ctx._previous_result is not None:
                L = ctx._previous_result.init_L
            elif itr > 0:
                L = (prev_LX, prev_LY)
            else:
                L = None
            if L is not None:
                level_V = max(abs(L[0]), abs(L[1]))
                ctx.sensitivity_channel(ctx.sensitivity_multiplier * level_V)

        # ── Settle ──────────────────────────────────────────────────
        if ctx.settle_time > 0:
            time.sleep(ctx.settle_time)

        # ── Read lock-in ────────────────────────────────────────────
        mean_L, R = ctx.lockin_call()
        LX, LY = float(mean_L[0]), float(mean_L[1])

        if itr == 0:
            initial_L = (LX, LY)

        ctx.log(
            f"    LX={LX:.5e}  LY={LY:.5e}"
        )

        if verbose:
            ctx.log(
                f"     Covariance: ┏                          ┓\n"
                f"                 ┃ {f"{R[0,0]:.5e}":<12}{f"{R[0,0]:.5e}":>12} ┃\n"
                f"                 ┃ {f"{R[0,0]:.5e}":<12}{f"{R[0,0]:.5e}":>12} ┃\n"
                f"                 ┗                          ┛"
            )

        # ── Kalman filter predict ────────────────────────────────────
        ctx.cap_filter.predict()

        # ── Kalman filter update (from itr 1 onward) ─────────────────
        if itr > 0 and prev_LX is not None:
            dLX = LX - prev_LX
            dLY = LY - prev_LY
            dvx = xb - prev_xb
            dvy = yb - prev_yb
            R_diff = R + prev_R

            # Account for Vstd resolution uncertainty in measurement noise
            A_now = ctx.cap_filter.get_A()
            Amat = np.array([[A_now[0], -A_now[1]], [A_now[1],  A_now[0]]])
            R_dv_rth = np.diag([
                ctx.amp_resolution**2,
                (ctx.phase_resolution * np.pi / 180)**2,
            ])

            def _rth_to_xy(Q_rth, v):
                r = np.sqrt(v[0]**2 + v[1]**2)
                th = np.arctan2(v[1], v[0])
                c, s = np.cos(th), np.sin(th)
                J = np.array([[c, -r * s], [s, r * c]])
                return J @ Q_rth @ J.T

            R_dv = (
                _rth_to_xy(R_dv_rth, np.array([xb, yb]))
                + _rth_to_xy(R_dv_rth, np.array([prev_xb, prev_yb]))
            )
            R_eff = R_diff + Amat @ R_dv @ Amat.T

            ctx.cap_filter.update(
                dL = np.array([dLX, dLY]),
                R = R_eff,
                dv = np.array([dvx, dvy]),
            )

            if verbose:
                P_now = ctx.cap_filter.get_P()
                ctx.log(
                    "Incorporating new measurement into Kalman filter.\n"
                    f"dL = [{dLX:.5e}, {dLY:.5e}]\n"
                    f"     Covariance: ┏                          ┓\n"
                    f"                 ┃ {f"{R_eff[0,0]:.5e}":<12}"
                    f"{f"{R_eff[0,0]:.5e}":>12} ┃\n"
                    f"                 ┃ {f"{R_eff[0,0]:.5e}":<12}"
                    f"{f"{R_eff[0,0]:.5e}":>12} ┃\n"
                    f"                 ┗                          ┛\n"
                    f"Size of Uncertainties:\n"
                    f"              P: {P_now[0,0]:.5e}, {P_now[1,1]:.5e}\n"
                    f"         R_diff: {R_diff[0,0]:.5e}, {R_diff[1,1]:.5e}\n"
                    f"          R_eff: {R_eff[0,0]:.5e}, {R_eff[1,1]:.5e}"
                )

            ctx.log(
                f"    Filter updated: dL=({dLX:.5e}, {dLY:.5e})  "
                f"dv=({dvx:.5e}, {dvy:.5e})"
            )

        A_now = ctx.cap_filter.get_A()
        P_now = ctx.cap_filter.get_P()

        ctx.log(
            f"    A=[{A_now[0]:.5e}, {A_now[1]:.5e}]  "
            f"P_diag=[{P_now[0,0]:.5e}, {P_now[1,1]:.5e}]"
        )

        # ── Compute correction ───────────────────────────────────────
        X, Y = float(A_now[0]), float(A_now[1])
        absA2 = X**2 + Y**2
        dx = (X * LX + Y * LY) / absA2
        dy = (X * LY - Y * LX) / absA2

        # Store previous state for next iteration
        prev_rb = rb
        prev_xb = xb
        prev_yb = yb
        prev_LX = LX
        prev_LY = LY
        prev_R = R

        # Projected next balance point
        xb -= dx
        yb -= dy
        rb = float(np.sqrt(xb**2 + yb**2))
        thb = float(np.degrees(np.arctan2(yb, xb)) % 360)

        ctx.log(
            f"    Correction: dx={dx:.5e}  dy={dy:.5e}  "
            f"new Vb=({xb:.5e}, {yb:.5e})"
        )

        # ── Error bound and termination ──────────────────────────────
        LR = float(np.sqrt(LX**2 + LY**2))
        L_col = np.array([[LX], [LY]])

        if np.any(np.isinf(R)):
            # No covariance available — use a fixed fraction of LR
            LRerr = LR
        else:
            LRerr = float(np.sqrt(np.sum(L_col.T @ R @ L_col))) / max(LR, 1e-30)

        new_bound = ctx.erroff + ctx.errmult * LRerr
        if itr == 0:
            errt = new_bound
        else:
            errt = ctx.errt_alpha * errt + (1 - ctx.errt_alpha) * new_bound

        ctx.log(f"    LR={LR:.5e}  errt={errt:.5e}")

        # Terminate if Vstd change is below hardware resolution
        if itr > 0:
            th_prev = float(np.degrees(np.arctan2(prev_yb, prev_xb)) % 360)
            dr = abs(rb - prev_rb)
            dth = abs((thb - th_prev + 180) % 360 - 180)
            if dr < ctx.amp_resolution and dth < ctx.phase_resolution:
                ctx.log(
                    f"  Balanced (resolution limit) on iteration {itr + 1}."
                )
                ctx.extrapolator.push(xb, yb)
                result = _make_result(
                    status=True, itr=itr, n_iter=itr + 1,
                    xb=xb, yb=yb, LX=LX, LY=LY,
                    initial_L=initial_L, errt=errt,
                    A=ctx.cap_filter.get_A(), P=ctx.cap_filter.get_P(),
                    ctx=ctx,
                )
                break

        # Terminate if lock-in reading is within error bound
        if LR < errt:
            ctx.log(
                f"  Balanced on iteration {itr + 1}.  LR={LR:.5e} < errt={errt:.5e}"
            )
            ctx.extrapolator.push(xb, yb)
            result = _make_result(
                status=True, itr=itr, n_iter=itr + 1,
                xb=xb, yb=yb, LX=LX, LY=LY,
                initial_L=initial_L, errt=errt,
                A=ctx.cap_filter.get_A(), P=ctx.cap_filter.get_P(),
                ctx=ctx,
            )
            break

    else:
        # max_tries exhausted without convergence
        ctx.log(f"  cap_balance: failed after {ctx.max_tries} iterations.")
        result = _make_result(
            status=False, itr=itr, n_iter=itr + 1,
            xb=xb, yb=yb, LX=LX, LY=LY,
            initial_L=initial_L, errt=errt,
            A=ctx.cap_filter.get_A(), P=ctx.cap_filter.get_P(),
            ctx=ctx,
        )

    ctx._previous_result = result
    ctx.log(result)
    return result


def _make_result(
    status: bool,
    itr: int,
    n_iter: int,
    xb: float, yb: float,
    LX: float, LY: float,
    initial_L: tuple,
    errt: float,
    A: np.ndarray,
    P: np.ndarray,
    ctx: 'CapContext',
) -> CapBalanceResult:
    Cex   = ctx.Cstd * xb / ctx.Vex
    Closs = ctx.Cstd * yb / ctx.Vex
    return CapBalanceResult(
        status = status,
        n_iter = n_iter,
        Vb = (xb, yb),
        L = (LX, LY),
        init_L = initial_L if initial_L is not None else (LX, LY),
        Cex = Cex,
        Closs = Closs,
        A = A,
        P = P,
        errt = errt,
        itr = itr,
    )