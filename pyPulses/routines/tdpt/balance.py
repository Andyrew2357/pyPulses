from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import TDPTContext

from .config import (
    INT_ADJ_LAMBDA,
    MAX_RAIL_HI_COUNT,
    RAIL_HI_STREAK_THRESHOLD,
    RAIL_LO_STREAK_THRESHOLD,
    FAILURE_STREAK_THRESHOLD,
    MODERATE_RESET_P_MULT,
)

from dataclasses import dataclass
import numpy as np
import time


@dataclass
class SweepControls:
    X1:       float
    Y1:       float
    Y1_ideal: float
    X2:       float | None = None
    Y2:       float | None = None
    Y2_ideal: float | None = None
    W:        float | None = None
    W_ideal:  float | None = None


@dataclass
class SweepErrors:
    dQ:     float | None = None
    dQ_var: float | None = None
    M:      float | None = None
    M_var:  float | None = None
    I:      float | None = None
    I_var:  float | None = None


@dataclass
class BalanceResult:
    success:   bool
    railed_lo: bool
    railed_hi: bool
    n_iter:    int
    good_count: int

    # Actual hardware values at final sweep
    X1:       float
    Y1:       float
    Y1_ideal: float
    X2:       float | None = None
    Y2:       float | None = None
    Y2_ideal: float | None = None
    W:        float | None = None
    W_ideal:  float | None = None

    # Error measurements at final sweep
    dQ:     float | None = None
    dQ_var: float | None = None
    M:      float | None = None
    M_var:  float | None = None
    I:      float | None = None
    I_var:  float | None = None

    # Filter state at final sweep
    G:    float | None = None
    Cac:  float | None = None
    dMdW: float | None = None

    def __str__(self) -> str:
        s = (
            f"BalanceResult: {'SUCCESS' if self.success else 'FAILED'} "
            f"(n_iter={self.n_iter}, good_count={self.good_count})\n"
        )
        if self.railed_lo:
            s += "  [RAILED LOW]\n"
        if self.railed_hi:
            s += "  [RAILED HIGH]\n"
        s += (
            f"  Controls:  X1={self.X1:.5e}  Y1={self.Y1:.5e}"
            f" (ideal={self.Y1_ideal:.5e})\n"
        )
        if self.W is not None:
            s += (
                f"             X2={self.X2:.5e}  Y2={self.Y2:.5e}"
                f" (ideal={self.Y2_ideal:.5e})\n"
                f"             W={self.W:.5e}"
                f" (ideal={self.W_ideal:.5e})\n"
            )
        s += f"  dQ={self.dQ:.5e} +- {np.sqrt(self.dQ_var):.5e}\n"
        if self.M is not None:
            s += f"  M={self.M:.5e} +- {np.sqrt(self.M_var):.5e}\n"
        if self.I is not None:
            s += f"  I={self.I:.5e} +- {np.sqrt(self.I_var):.5e}\n"
        s += (
            f"  Filter: G={self.G:.5e}  Cac={self.Cac:.5e}"
        )
        if self.dMdW is not None:
            s += f"  dMdW={self.dMdW:.5e}"
        return s


def _try_update_bound(
    bound: tuple[float, float] | None,
    W: float,
    M: float,
    dMdW: float,
    high: bool,
) -> tuple[float, float] | None:
    """
    Update one side of the discharge bracket.

    Parameters
    ----------
    bound : (W, M) or None
        Current bound for this side.
    W : float
        Actual pulse width at which M was measured.
    M : float
        Measured discharge slope error.
    dMdW : float
        Current filter slope estimate, used for slope-informed update.
    high : bool
        True if this is the high side (M > 0), False for low side (M < 0).

    Returns
    -------
    Updated bound or unchanged bound.
    """
    # Wrong sign for this side
    if (M >= 0) != high:
        return bound

    # Naive improvement: new M is closer to zero than existing bound
    if bound is None or (M <= bound[1]) == high:
        return (W, M)

    # Slope-informed improvement: slope suggests this point is closer
    # to the root than the existing bound
    if high == ((dMdW > 0) == (W < bound[0])):
        return (W, M)

    return bound


def TDPT_filter_balance(ctx: TDPTContext, new_point: bool = True) -> BalanceResult:
    """
    Perform a single fully balanced measurement at the current phase
    space point. Updates all Kalman filters and extrapolator histories
    in place. Returns a BalanceResult describing the outcome.

    The caller is responsible for:
      - Setting X1, X2 on the hardware before calling
      - Calling reset_for_polarity_switch() when the excitation sign changes
      - Calling TDPT_map_health_monitor() between points as needed

    Parameters
    ----------
    new_point : bool
        Whether this call represents a new phase space point. If True,
        fires predict_excitation_change() on the filters. Set to False
        when calling repeatedly at the same point (e.g. during testing).
    """

    dis = ctx.is_discharge()
    ing = ctx.is_integrated()
    W_res = ctx.pulse_width_resolution
    W_min = ctx.min_pulse_width
    W_max = ctx.max_pulse_width

    # ═══════════════════════════════════════════════════════
    # PHASE 0: SETUP AND INITIAL PREDICTION
    # ═══════════════════════════════════════════════════════

    X1 = ctx.X1()
    X2 = None
    if dis:
        if ctx.discharge_X_ratio is not None:
            X2 = ctx.X2(-ctx.dis_eta * ctx.discharge_X_ratio * X1)
        else:
            X2 = ctx.X2()

    positive_now    = X1 > 0
    filter_positive = ctx.cap_filter.is_positive_initialized
    if positive_now != filter_positive:
        ctx.reset_for_polarity_switch()

    if new_point:
        ctx.predict_excitation_change()

    Cac = ctx.cap_filter.get_Cac()
    G   = ctx.cap_filter.get_G()
    if dis:
        dMdW = ctx.dis_filter.get_dMdW()

    Y1_ideal = -Cac * X1
    Y1       = ctx.Y1(Y1_ideal)
    W        = None
    W_ideal  = None
    Y2       = None
    Y2_ideal = None

    if dis:
        W_extrap = ctx.extrapolator.extrapolate_W0()
        W_ideal  = W_extrap if W_extrap is not None else ctx.W()
        W        = ctx.W(W_ideal)

        if ctx._next_Y2_ratio_override is not None:
            Y2_ideal                    = ctx._next_Y2_ratio_override * X2
            ctx._next_Y2_ratio_override = None
        elif ing:
            Y2_extrap = ctx.extrapolator.extrapolate_Y2(X2)
            Y2_ideal  = Y2_extrap if Y2_extrap is not None else -Cac * X2
        else:
            Y2_ideal = -Cac * X2
        Y2 = ctx.Y2(Y2_ideal)

    # Discharge balance state
    bracket_lo       = None
    bracket_hi       = None
    illinois_side    = None
    W_prev           = None
    M_prev           = None
    M_var_prev       = None
    dMdW_reliable    = ctx.dis_filter.is_reliable() if dis else False
    newton_exhausted = False
    rail_hi_count    = 0
    railed_lo        = False
    railed_hi        = False

    # Global iteration state
    good_sweep_count = 0
    n_iter           = 0
    success          = False

    # Measurement variables — initialised to None so FINALIZE is safe
    dQ     = None;  dQ_var = None
    M      = None;  M_var  = None
    I      = None;  I_var  = None

    # W/M history for WLS-based strong update at finalize
    W_history     = []
    M_history     = []
    M_var_history = []

    # ═══════════════════════════════════════════════════════
    # MAIN BALANCE LOOP
    # ═══════════════════════════════════════════════════════

    for itr in range(ctx.max_iterations + 1):

        # ── STEP 1: KALMAN PREDICT FOR BALANCE CHANGE ──────

        ctx.predict_balance_change()

        # ── STEP 2: SET CONTROLS AND TAKE SWEEP ────────────

        Y1 = ctx.Y1(Y1)
        if dis:
            W  = ctx.W(W)
            Y2 = ctx.Y2(Y2)

        if ctx.pre_sweep_callback is not None:
            ctx.pre_sweep_callback(
                itr      = itr,
                controls = SweepControls(
                    X1=X1, Y1=Y1, Y1_ideal=Y1_ideal,
                    X2=X2, Y2=Y2, Y2_ideal=Y2_ideal,
                    W=W,   W_ideal=W_ideal,
                ),
            )

        time.sleep(ctx.settle_time)
        high_res = good_sweep_count > 0 or ctx.good_sweep_threshold == 1
        ctx.take_sweep(ctx.high_resolution_sweep_multiplier if high_res else 1)
        n_iter += 1

        ctx.log(
            f"ITERATION {itr + 1} / {ctx.max_iterations + 1}\n"
            f"  Controls:  X1={X1:.5e}  Y1={Y1:.5e} (ideal={Y1_ideal:.5e})\n"
            + (f"  Discharge: X2={X2:.5e}  Y2={Y2:.5e} (ideal={Y2_ideal:.5e})\n"
               f"             W={W:.5e} (ideal={W_ideal:.5e})" if dis else "")
        )

        # ── STEP 3: MEASURE ERROR PARAMETERS ───────────────

        dQ, dQ_var = ctx.capacitance_error()
        if dis:
            M, M_var = ctx.discharge_error()
            if ing:
                I, I_var = ctx.integrated_error()

        if ctx.post_sweep_callback is not None:
            ctx.post_sweep_callback(
                itr      = itr,
                controls = SweepControls(
                    X1=X1, Y1=Y1, Y1_ideal=Y1_ideal,
                    X2=X2, Y2=Y2, Y2_ideal=Y2_ideal,
                    W=W,   W_ideal=W_ideal,
                ),
                errors   = SweepErrors(
                    dQ=dQ, dQ_var=dQ_var,
                    M=M,   M_var=M_var,
                    I=I,   I_var=I_var,
                ),
            )

        ctx.log(
            f"  dQ={dQ:.5e} +- {np.sqrt(dQ_var):.5e}"
            + (f"  M={M:.5e} +- {np.sqrt(M_var):.5e}" if dis else "")
            + (f"  I={I:.5e} +- {np.sqrt(I_var):.5e}" if ing else "")
        )

        # Accumulate (W, M) for WLS strong update at finalize.
        # Only include non-railed measurements — railed points have
        # M values that don't reflect the slope at the balance point.
        if dis and not railed_lo and not railed_hi:
            W_history.append(float(W))
            M_history.append(float(M))
            M_var_history.append(float(M_var))

        # ── STEP 4: UPDATE CAPACITANCE FILTER ──────────────

        ctx.cap_filter.update(dQ, dQ_var)
        Cac = ctx.cap_filter.get_Cac()
        G   = ctx.cap_filter.get_G()

        # ── STEP 5: UPDATE DISCHARGE FILTER ────────────────

        if dis:
            if (W_prev is not None
                    and abs(W - W_prev) > W_res
                    and not railed_lo
                    and not railed_hi):
                dMdW_obs      = (M - M_prev) / (W - W_prev)
                dMdW_R        = (M_var + M_var_prev) / (W - W_prev) ** 2
                ctx.dis_filter.update(dMdW_obs, dMdW_R)
                dMdW          = ctx.dis_filter.get_dMdW()
                dMdW_reliable = ctx.dis_filter.is_reliable()

            W_prev     = W
            M_prev     = M
            M_var_prev = M_var

        ctx.log(
            f"  Filter: G={G:.5e}  Cac={Cac:.5e}"
            + (f"  dMdW={ctx.dis_filter.get_dMdW():.5e}"
               f"  P={ctx.dis_filter.P:.5e}"
               f"  reliable={ctx.dis_filter.is_reliable()}" if dis else "")
        )

        # ── STEP 6: UPDATE BRACKET ──────────────────────────

        if dis:
            bracket_lo = _try_update_bound(bracket_lo, W, M, dMdW, high=False)
            bracket_hi = _try_update_bound(bracket_hi, W, M, dMdW, high=True)

            ctx.log(
                f"  Bracket: lo={bracket_lo}  hi={bracket_hi}"
            )

        # ── STEP 7: CHECK TERMINATION ───────────────────────

        cap_satisfied = abs(dQ) < ctx.cap_balance_threshold * np.sqrt(dQ_var)

        if dis:
            dis_satisfied = abs(M) < ctx.dis_balance_threshold * np.sqrt(M_var)
            int_satisfied = (abs(I) < ctx.int_balance_threshold * np.sqrt(I_var)
                             if ing else True)
            railed_lo = W <= W_min + W_res
            railed_hi = W >= W_max - W_res
        else:
            dis_satisfied = True
            int_satisfied = True
            railed_lo     = False
            railed_hi     = False

        if cap_satisfied and dis_satisfied and int_satisfied:
            good_sweep_count += 1
        else:
            good_sweep_count = 0

        if railed_lo:
            rail_hi_count = 0
        elif railed_hi:
            rail_hi_count   += 1
        else:
            rail_hi_count = 0

        ctx.log(
            f"  Satisfied: cap={cap_satisfied}"
            + (f"  dis={dis_satisfied}  railed_lo={railed_lo}  railed_hi={railed_hi}" if dis else "")
            + f"  good_count={good_sweep_count}"
        )

        if good_sweep_count >= ctx.good_sweep_threshold:
            success = True
            break

        if itr == ctx.max_iterations:
            success = False
            break

        # ── STEP 8: COMPUTE NEXT Y1 ─────────────────────────

        Y1_ideal      = -Cac * X1
        ratio_prev    = Y1 / X1
        ratio_ideal   = Y1_ideal / X1
        ratio_stepped = float(np.clip(ratio_ideal,
                                      ratio_prev - ctx.max_Y1_step,
                                      ratio_prev + ctx.max_Y1_step))
        if abs(ratio_stepped - ratio_ideal) > 1e-15:
            ctx.log(
                f"  Y1 ratio step-clamped: ideal={ratio_ideal:.5e}"
                f"  clamped={ratio_stepped:.5e}"
                f"  max_step={ctx.max_Y1_step:.5e}"
            )
        Y1 = ratio_stepped * X1

        # ── STEP 9: COMPUTE NEXT W ──────────────────────────

        if dis and not railed_lo:

            if bracket_lo is not None and bracket_hi is not None:
                # CASE A: Both brackets — secant + Illinois

                if M < 0 and illinois_side == 'lo':
                    bracket_hi = (bracket_hi[0], bracket_hi[1] / 2)
                elif M > 0 and illinois_side == 'hi':
                    bracket_lo = (bracket_lo[0], bracket_lo[1] / 2)

                illinois_side = 'lo' if M < 0 else 'hi'

                dM      = bracket_hi[1] - bracket_lo[1]
                dW      = bracket_hi[0] - bracket_lo[0]
                W_ideal = bracket_lo[0] - bracket_lo[1] * dW / dM
                W_ideal = float(np.clip(W_ideal, W_min, W_max))
                W       = W_ideal

                if abs(bracket_hi[0] - bracket_lo[0]) < W_res:
                    if M > 0:
                        bracket_lo    = None
                        illinois_side = None
                    else:
                        bracket_hi    = None
                        illinois_side = None

                ctx.log(
                    f"  W next [CASE A: secant]  W={W:.5e}"
                    f"  bracket=({bracket_lo}, {bracket_hi})"
                    f"  illinois_side={illinois_side}"
                )

            elif bracket_lo is not None or bracket_hi is not None:
                # CASE B: One bracket — Newton from that point

                bound   = bracket_lo if bracket_lo is not None else bracket_hi
                W_ideal = bound[0] - bound[1] / dMdW
                W_ideal = float(np.clip(W_ideal, W_min, W_max))
                W       = W_ideal
                illinois_side = None

                if W_ideal >= W_max - W_res:
                    rail_hi_count += 1
                    if rail_hi_count >= MAX_RAIL_HI_COUNT:
                        W = W_ideal
                else:
                    rail_hi_count = 0

                ctx.log(
                    f"  W next [CASE B: Newton/one bracket]  W={W:.5e}"
                    f"  bound=({bound[0]:.5e}, {bound[1]:.5e})"
                    f"  dMdW={dMdW:.5e}  rail_hi_count={rail_hi_count}"
                )

            else:
                # CASE C: No bracket — Newton or probe

                illinois_side = None

                if dMdW_reliable and not newton_exhausted:

                    W_ideal = W - M / dMdW
                    W_ideal = float(np.clip(W_ideal, W_min, W_max))
                    W       = W_ideal

                    if W_ideal >= W_max - W_res:
                        rail_hi_count += 1
                        if rail_hi_count >= MAX_RAIL_HI_COUNT:
                            newton_exhausted = True
                    else:
                        rail_hi_count = 0

                    ctx.log(
                        f"  W next [CASE C: Newton/no bracket]  W={W:.5e}"
                        f"  M={M:.5e}  dMdW={dMdW:.5e}  step={-M/dMdW:.5e}"
                        f"  rail_hi_count={rail_hi_count}"
                    )

                else:
                    newton_exhausted = True
                    W_center = (W_min + W_max) / 2

                    if abs(W_center - W) > W_res:
                        W_ideal = W_center
                        W       = W_ideal
                        rail_hi_count = 0
                        ctx.log(
                            f"  W next [CASE C: probe center]  W={W:.5e}"
                        )
                    else:
                        W_ideal = W_max if W < W_center else W_min
                        W       = W_ideal

                        if W_ideal >= W_max - W_res:
                            rail_hi_count += 1
                            ctx.log(
                                f"  W next [CASE C: probe far side]  W={W:.5e}"
                                f"  rail_hi_count={rail_hi_count}"
                            )
                            if rail_hi_count >= MAX_RAIL_HI_COUNT:
                                success = False
                                if (W_prev is not None
                                        and abs(W - W_prev) > W_res
                                        and not railed_lo
                                        and not railed_hi):
                                    dMdW_obs = (M - M_prev) / (W - W_prev)
                                    if np.sign(dMdW_obs) != np.sign(ctx.dis_filter.dMdW):
                                        ctx.dis_filter.dMdW = abs(ctx.dis_filter.dMdW) * np.sign(dMdW_obs)
                                        ctx.dis_filter._previous_dMdW = ctx.dis_filter.dMdW
                                ctx.dis_filter.reset_uncertainty()
                                ctx.log("  FILTER RESET triggered")
                                break
                        else:
                            rail_hi_count = 0
                            ctx.log(
                                f"  W next [CASE C: probe far side]  W={W:.5e}"
                            )

        # ── STEP 10: COMPUTE NEXT Y2 ─────────────────────────

        if dis:
            if not ing or W <= ctx.dis_amp_adjustment_threshold:
                Y2_ideal      = -Cac * X2
                ratio_prev    = Y2 / X2
                ratio_ideal   = Y2_ideal / X2
                ratio_stepped = float(np.clip(ratio_ideal,
                                              ratio_prev - ctx.max_Y2_step,
                                              ratio_prev + ctx.max_Y2_step))
                if abs(ratio_stepped - ratio_ideal) > 1e-15:
                    ctx.log(
                        f"  Y2 ratio step-clamped: ideal={ratio_ideal:.5e}"
                        f"  clamped={ratio_stepped:.5e}"
                        f"  max_step={ctx.max_Y2_step:.5e}"
                    )
                Y2       = ratio_stepped * X2
                Y2_ideal = ratio_ideal * X2

            else:
                dY2           = -ctx.dis_eta * I / (G * W * INT_ADJ_LAMBDA)
                Y2_ideal      = Y2 + dY2
                ratio_prev    = Y2 / X2
                ratio_ideal   = Y2_ideal / X2
                ratio_stepped = float(np.clip(ratio_ideal,
                                              ratio_prev - ctx.max_Y2_int_step,
                                              ratio_prev + ctx.max_Y2_int_step))
                if abs(ratio_stepped - ratio_ideal) > 1e-15:
                    ctx.log(
                        f"  Y2 int ratio step-clamped: ideal={ratio_ideal:.5e}"
                        f"  clamped={ratio_stepped:.5e}"
                        f"  max_int_step={ctx.max_Y2_int_step:.5e}"
                    )
                Y2       = ratio_stepped * X2
                Y2_ideal = ratio_ideal * X2
                ctx.predict_balance_change()

                # dbal_dev ratio clamp
                ratio_lo, ratio_hi = sorted([
                    -Cac * (1 + ctx.dbal_dev),
                    -Cac * (1 - ctx.dbal_dev),
                ])
                ratio_clamped = float(np.clip(Y2 / X2, ratio_lo, ratio_hi))
                if abs(ratio_clamped - Y2 / X2) > 1e-15:
                    ctx.log(
                        f"  Y2 dbal_dev ratio-clamped: Y2/X2={Y2/X2:.5e}"
                        f"  clamped={ratio_clamped:.5e}"
                        f"  -Cac={-Cac:.5e}  dbal_dev={ctx.dbal_dev:.5e}"
                    )
                Y2 = ratio_clamped * X2

    # ═══════════════════════════════════════════════════════
    # FINALIZE
    # ═══════════════════════════════════════════════════════

    if dis:
        final_railed_lo = W <= W_min + W_res
        final_railed_hi = W >= W_max - W_res
    else:
        final_railed_lo = False
        final_railed_hi = False

    if (success and dis
            and not final_railed_lo and not final_railed_hi):

        # WLS fit over all non-railed (W, M) measurements collected
        # during this balance. Using all points gives a more robust
        # slope estimate than two bracket endpoints alone, and the
        # formal WLS variance naturally down-weights cases where the
        # W range explored was narrow (high R → weak update).
        if len(W_history) >= 2:
            W_arr  = np.array(W_history)
            M_arr  = np.array(M_history)
            Mv_arr = np.array(M_var_history)

            valid  = Mv_arr > 0
            if valid.sum() >= 2:
                W_arr  = W_arr[valid]
                M_arr  = M_arr[valid]
                Mv_arr = Mv_arr[valid]

                w      = 1.0 / Mv_arr
                W_mean = np.sum(w * W_arr) / np.sum(w)
                dW     = W_arr - W_mean
                Sxx    = np.sum(w * dW ** 2)
                Sxy    = np.sum(w * dW * M_arr)

                if Sxx > 0:
                    dMdW_obs = float(Sxy / Sxx)
                    dMdW_R   = float(1.0 / Sxx)
                    ctx.dis_filter.strong_update(dMdW_obs, dMdW_R)
                    ctx.log(
                        f"  Strong update (WLS, {len(W_arr)} points):"
                        f"  dMdW_obs={dMdW_obs:.5e}"
                        f"  R={dMdW_R:.5e}"
                        f"  W_span={W_arr.max()-W_arr.min():.3e}"
                    )

        ctx.extrapolator.push_W0(W)

    if success and ing and not final_railed_hi:
        ctx.extrapolator.push_Y2(Y2, X2)

    if dis:
        if final_railed_hi:
            ctx._rail_hi_streak += 1
            ctx._rail_lo_streak  = 0
        elif final_railed_lo:
            ctx._rail_lo_streak += 1
            ctx._rail_hi_streak  = 0
        else:
            ctx._rail_hi_streak  = 0
            ctx._rail_lo_streak  = 0

    if success:
        ctx._failure_streak    = 0
        ctx._consecutive_good += 1
    else:
        ctx._failure_streak   += 1
        ctx._consecutive_good  = 0

    result = BalanceResult(
        success      = success,
        railed_lo    = final_railed_lo,
        railed_hi    = final_railed_hi,
        n_iter       = n_iter,
        good_count   = good_sweep_count,
        X1           = X1,
        Y1           = Y1,
        Y1_ideal     = Y1_ideal,
        X2           = X2,
        Y2           = Y2,
        Y2_ideal     = Y2_ideal,
        W            = W,
        W_ideal      = W_ideal,
        dQ           = dQ,
        dQ_var       = dQ_var,
        M            = M,
        M_var        = M_var,
        I            = I,
        I_var        = I_var,
        G            = G,
        Cac          = Cac,
        dMdW         = ctx.dis_filter.get_dMdW() if dis else None,
    )

    ctx._previous_result = result
    ctx.log(f"Balance result:\n{result}")
    return result


def TDPT_map_health_monitor(ctx: TDPTContext) -> None:
    """
    Map-level health monitor. Call between phase space points to detect
    and correct persistent failure modes.

    Handles two conditions:
      - Persistent high railing: nudges Y2/X2 ratio via integrated
        balance formula, clamped to dbal_dev fraction of -Cac
      - Persistent non-railed failures: re-probes dMdW directly
    """

    dis = ctx.is_discharge()
    if not dis:
        return

    if ctx._rail_hi_streak >= RAIL_HI_STREAK_THRESHOLD:

        prev = ctx._previous_result
        if (prev is not None
                and prev.I is not None
                and prev.W is not None
                and prev.X2 is not None
                and prev.W > ctx.dis_amp_adjustment_threshold):

            dY2       = (-ctx.dis_eta * prev.I
                         / (prev.G * prev.W * INT_ADJ_LAMBDA))
            new_ratio = (prev.Y2 + dY2) / prev.X2

            # Clamp ratio to dbal_dev fraction of -Cac
            Cac       = ctx.cap_filter.get_Cac()
            ratio_lo  = -Cac * (1 + ctx.dbal_dev)
            ratio_hi  = -Cac * (1 - ctx.dbal_dev)
            ratio_lo, ratio_hi = min(ratio_lo, ratio_hi), max(ratio_lo, ratio_hi)
            new_ratio = float(np.clip(new_ratio, ratio_lo, ratio_hi))

            ctx._next_Y2_ratio_override = new_ratio
            ctx.log(
                f"MAP HEALTH: rail_hi_streak={ctx._rail_hi_streak} >= threshold. "
                f"Y2/X2 override set to {new_ratio:.5e}"
            )

        ctx._rail_hi_streak = 0
        ctx.dis_filter.reset_uncertainty()

    if ctx._rail_lo_streak >= RAIL_LO_STREAK_THRESHOLD:

        W_min      = ctx.min_pulse_width
        W_max      = ctx.max_pulse_width
        W_lo_probe = W_min + 0.2 * (W_max - W_min)
        W_hi_probe = W_min + 0.8 * (W_max - W_min)

        ctx.log(
            f"MAP HEALTH: rail_lo_streak={ctx._rail_lo_streak} >= threshold. "
            f"Re-probing dMdW at W={W_lo_probe:.5e} and W={W_hi_probe:.5e}"
        )

        ctx.predict_balance_change()
        ctx.W(W_lo_probe)
        ctx.take_sweep()
        M_lo, _ = ctx.discharge_error()

        ctx.predict_balance_change()
        ctx.W(W_hi_probe)
        ctx.take_sweep()
        M_hi, _ = ctx.discharge_error()

        dMdW_obs = (M_hi - M_lo) / (W_hi_probe - W_lo_probe)

        # Reinitialize filters from stored init params to clear any
        # state corruption from the railed period, then override
        # dMdW with the freshly probed value.
        ctx.reset_for_polarity_switch(W0=W_hi_probe)
        ctx.dis_filter.hard_reset_dMdW(dMdW_obs)
        ctx.dis_filter.reset_uncertainty(MODERATE_RESET_P_MULT)
        ctx._rail_lo_streak = 0

        ctx.log(
            f"MAP HEALTH: dMdW re-probed as {dMdW_obs:.5e}, "
            f"filters reinitialized, W0 set to {W_hi_probe:.5e}"
        )

    if ctx._failure_streak >= FAILURE_STREAK_THRESHOLD:

        W_min      = ctx.min_pulse_width
        W_max      = ctx.max_pulse_width
        W_lo_probe = W_min + 0.2 * (W_max - W_min)
        W_hi_probe = W_min + 0.8 * (W_max - W_min)

        ctx.log(
            f"MAP HEALTH: failure_streak={ctx._failure_streak} >= threshold. "
            f"Re-probing dMdW at W={W_lo_probe:.5e} and W={W_hi_probe:.5e}"
        )

        ctx.predict_balance_change()
        ctx.W(W_lo_probe)
        ctx.take_sweep()
        M_lo, _ = ctx.discharge_error()

        ctx.predict_balance_change()
        ctx.W(W_hi_probe)
        ctx.take_sweep()
        M_hi, _ = ctx.discharge_error()

        dMdW_obs = (M_hi - M_lo) / (W_hi_probe - W_lo_probe)
        ctx.dis_filter.hard_reset_dMdW(dMdW_obs)
        ctx.dis_filter.reset_uncertainty(MODERATE_RESET_P_MULT)
        ctx._failure_streak = 0

        ctx.log(f"MAP HEALTH: dMdW re-probed as {dMdW_obs:.5e}")