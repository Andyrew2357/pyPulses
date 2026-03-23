from .config import (
    DIS_INIT_BAL_UNC,
    DIS_INIT_EXC_UNC,
    DIS_INIT_P_MULT,
    CAP_INIT_BAL_QG_MULT,
    CAP_INIT_BAL_QC,
    CAP_INIT_BAL_QdC,
    CAP_INIT_EXC_QG_MULT,
    CAP_INIT_EXC_QC,
    CAP_INIT_EXC_QdC,
    CAP_INIT_PG_MULT,
    CAP_INIT_PC,
    CAP_INIT_PdC,
)
from .cap_filter import CapacitanceFilter
from .dis_filter import DischargeFilter
from .context import TDPTContext
from ..linear_balance import lstsqBalance
from ..balance_parameter import balanceKnob

from dataclasses import dataclass
import numpy as np
from typing import Callable, Tuple


@dataclass
class TDPT_initial_filter_metadata():
    X1: float
    X2: float | None


@dataclass
class TDPT_initial_filter_config():
    status: bool
    cap_status: bool
    dis_status: bool
    positive: bool

    cap_err: float
    cap_var: float
    cap_filter_init_parms: dict

    dis_err: float | None
    dis_var: float | None
    dis_filter_init_parms: dict | None

    meta: TDPT_initial_filter_metadata

    def __str__(self) -> str:
        r = (
            f"Status: {'GOOD' if self.status else 'BAD'}\n"
            f"Capacitance Status: {'GOOD' if self.cap_status else 'BAD'} "
            f"({self.cap_err:.5e} +- {np.sqrt(self.cap_var):.5e})\n"
            f"    x = {self.cap_filter_init_parms.get('x')} "
            f"+- {np.sqrt(self.cap_filter_init_parms.get('P'))}\n"
            f"    Qbal =\n{self.cap_filter_init_parms.get('bal_change_Q')}\n"
            f"    Qexc =\n{self.cap_filter_init_parms.get('exc_change_Q')}"
        )
        if self.dis_err is not None:
            r += (
                f"\nDischarge Status: {'GOOD' if self.dis_status else 'BAD'} "
                f"({self.dis_err:.5e} +- {np.sqrt(self.dis_var):.5e})\n"
                f"    dMdW = {self.dis_filter_init_parms.get('dMdW'):.5e} +- "
                f"{np.sqrt(self.dis_filter_init_parms.get('P')):.5e}\n"
                f"    W0 = {self.dis_filter_init_parms.get('W0'):.5e}"
            )
        return r


def TDPT_initialize_filters(
    ctx: TDPTContext,

    cap_guess: float | None = None,
    cap_low_high: Tuple[float, float] = (0.05, 2.0),
    cap_absolute: bool = False,
    cap_error_threshold: float | None = None,

    dis_low_high: Tuple[float, float] = (0.5, 2.0),
    dis_error_threshold: float | None = None,

    X1: float | None = None,
    X2: float | None = None,
    W: float | None = None,

    reguess: bool = True,

    refinements: int | None = None,
    deviation_samples: int = 3,
    settle_time: float | None = None,

    post_measurement_callback: Callable | None = None,

) -> TDPT_initial_filter_config:
    """
    Performs a single initialization pass for the current excitation
    polarity. Sets filter init params and calls the appropriate
    positive/negative initialize method on the filters.

    Must be called after the excitation polarity has been set on the
    hardware. The caller is responsible for calling
    reset_for_polarity_switch() separately as that involves hardware
    setup steps that initialize_filters should not own.

    Parameters
    ----------
    ctx : TDPTContext
    cap_guess : float, optional
        Initial guess for Cac. If None, inferred from current Y1/X1.
    cap_low_high : Tuple[float, float]
        Low and high multipliers for the Y balance knob in lstsqBalance.
    cap_absolute : bool
        Whether cap_low_high offsets are absolute or relative.
    cap_error_threshold : float, optional
        Capacitance balance threshold in units of sqrt(variance).
        Defaults to ctx.cap_balance_threshold.
    dis_low_high : Tuple[float, float]
        Low and high multipliers for the W balance knob in lstsqBalance.
    dis_error_threshold : float, optional
        Discharge balance threshold in units of sqrt(variance).
        Defaults to ctx.dis_balance_threshold.
    X1 : float, optional
        Charge pulse X amplitude to use. If None, uses current hardware.
    X2 : float, optional
        Discharge pulse X amplitude to use. If None, uses current hardware.
    W : float, optional
        Initial discharge pulse width guess. If None, uses current hardware.
    refinements : int, optional
        Number of lstsqBalance refinements. Defaults to ctx.max_iterations.
    deviation_samples : int
        Number of extra measurements at balance point to estimate variance.
    settle_time : float, optional
        Settle time before each measurement. Defaults to ctx.settle_time.
    post_measurement_callback : Callable, optional
        Called after each error measurement in lstsqBalance.
    """

    if settle_time is None:
        settle_time = ctx.settle_time
    if refinements is None:
        refinements = ctx.max_iterations
    if cap_error_threshold is None:
        cap_error_threshold = ctx.cap_balance_threshold
    if dis_error_threshold is None:
        dis_error_threshold = ctx.dis_balance_threshold

    # Set X1, X2, W on hardware and retrieve actual set values
    X1 = ctx.X1(X1) if X1 is not None else ctx.X1()
    X2 = ctx.X2(X2) if X2 is not None else ctx.X2()
    W  = ctx.W(W)   if W  is not None else ctx.W()

    # Infer cap_guess from current hardware state if not provided
    if cap_guess is None:
        cap_guess = -ctx.Y1() / ctx.X1()

    positive = X1 > 0

    # Construct filters if they don't exist yet
    if ctx.cap_filter is None:
        ctx.add_capacitance_filter(CapacitanceFilter(ctx))
    if ctx.is_discharge() and ctx.dis_filter is None:
        ctx.add_discharge_filter(DischargeFilter(ctx))

    ctx.log(
        f"Initializing filters for "
        f"{'positive' if positive else 'negative'} excitation.\n"
        f"X1={X1:.5e}, X2={X2}, W={W}, cap_guess={cap_guess:.5e}"
    )

    # Build balance knobs
    controls = []
    errors   = []

    # Y1 knob — Y2 follows in lockstep to maintain Cac ratio
    def Y1_Y2(v: float | None = None) -> float:
        if v is None:
            return ctx.Y1()
        ctx.Y1(v)
        if ctx.is_discharge() and X2 is not None:
            ctx.Y2(v * (X2 / X1))
        return ctx.Y1()

    controls.append(balanceKnob(
        bounds   = (-ctx.max_pulse_height, ctx.max_pulse_height),
        guess    = -cap_guess * X1,
        l        = cap_low_high[0],
        h        = cap_low_high[1],
        f        = Y1_Y2,
        absolute = cap_absolute,
        logger   = ctx.logger,
        name     = "Y1",
    ))
    errors.append(ctx.capacitance_error)

    # W knob — only if discharge is enabled
    if ctx.is_discharge():
        controls.append(balanceKnob(
            bounds   = (ctx.min_pulse_width, ctx.max_pulse_width),
            guess    = W,
            l        = dis_low_high[0],
            h        = dis_low_high[1],
            f        = ctx.W,
            absolute = False,
            logger   = ctx.logger,
            name     = "W",
        ))
        errors.append(ctx.discharge_error)

    linear_balance = lstsqBalance(
        controls                 = controls,
        error_parms              = errors,
        pre_measurement_callback = lambda *args, **kwargs:
            ctx.averager.take_curve(ctx.high_resolution_sweep_multiplier),
        post_measurement_callback = post_measurement_callback,
        settle_time              = settle_time,
        logger                   = ctx.logger,
    )

    # Run the balance and refinements
    linear_balance.reset()
    linear_balance.balance()
    if reguess:
        linear_balance.set_good(True)
        linear_balance.refine()
        linear_balance.reset()
        linear_balance.balance()

    linear_balance.set_good(True)
    for _ in range(refinements):
        linear_balance.refine()

    # Estimate Cac from the balance point.
    # At balance Y1 = -Cac * X1, so Cac = -Y1_balance / X1.
    Cac = -linear_balance._x0[0] / X1

    # Sample error parameters at the balance point to estimate variance.
    # No filter predict calls here — filters do not exist yet.
    dQ     = linear_balance.error_parms[0].get_error()
    dQ_sq  = dQ * dQ

    if ctx.is_discharge():
        M     = linear_balance.error_parms[1].get_error()
        M_sq  = M * M

    for _ in range(deviation_samples):
        ctx.averager.take_curve()
        dQ_s, _ = linear_balance.error_parms[0]()
        dQ    += dQ_s
        dQ_sq += dQ_s * dQ_s

        if ctx.is_discharge():
            M_s, _ = linear_balance.error_parms[1]()
            M    += M_s
            M_sq += M_s * M_s

    n = deviation_samples + 1
    dQ    /= n
    dQ_sq /= n
    dQ_var = dQ_sq - dQ * dQ if deviation_samples > 0 else dQ_sq

    if ctx.is_discharge():
        M    /= n
        M_sq /= n
        M_var = M_sq - M * M if deviation_samples > 0 else M_sq

    # Construct capacitance filter init params
    G = float(linear_balance._A[0, 0])
    cap_filter_init_parms = {
        'bal_change_Q': np.diag([
            CAP_INIT_BAL_QG_MULT * G,
            CAP_INIT_BAL_QC,
            CAP_INIT_BAL_QdC,
        ]) ** 2,
        'exc_change_Q': np.diag([
            CAP_INIT_EXC_QG_MULT * G,
            CAP_INIT_EXC_QC,
            CAP_INIT_EXC_QdC,
        ]) ** 2,
        'x': np.array([G, Cac, 0.0]),
        'P': np.array([
            CAP_INIT_PG_MULT * G,
            CAP_INIT_PC,
            CAP_INIT_PdC,
        ]) ** 2,
    }

    # Construct discharge filter init params if applicable
    if ctx.is_discharge():
        dMdW = float(linear_balance._A[1, 1])
        dis_filter_init_parms = {
            'bal_change_Q': DIS_INIT_BAL_UNC ** 2,
            'exc_change_Q': DIS_INIT_EXC_UNC ** 2,
            'dMdW':         dMdW,
            'P':            DIS_INIT_P_MULT * dMdW ** 2,
            'W0':           float(linear_balance.controls[1].get_val()),
        }
    else:
        dMdW              = None
        M                 = None
        M_var             = None
        dis_filter_init_parms = None

    # Set init params on filters and initialize for this polarity
    cap_status = abs(dQ) < cap_error_threshold * np.sqrt(dQ_var)
    dis_status = (abs(M) < dis_error_threshold * np.sqrt(M_var)) if ctx.is_discharge() else True

    if positive:
        ctx.cap_filter.set_positive_init_params(
            bal_change_Q = cap_filter_init_parms['bal_change_Q'],
            exc_change_Q = cap_filter_init_parms['exc_change_Q'],
            x            = cap_filter_init_parms['x'],
            P            = cap_filter_init_parms['P'],
        )
        if ctx.is_discharge():
            ctx.dis_filter.set_positive_init_params(
                bal_change_Q = dis_filter_init_parms['bal_change_Q'],
                exc_change_Q = dis_filter_init_parms['exc_change_Q'],
                dMdW         = dis_filter_init_parms['dMdW'],
                P            = dis_filter_init_parms['P'],
            )
        ctx.reinitialize_positive_filter()
    else:
        ctx.cap_filter.set_negative_init_params(
            bal_change_Q = cap_filter_init_parms['bal_change_Q'],
            exc_change_Q = cap_filter_init_parms['exc_change_Q'],
            x            = cap_filter_init_parms['x'],
            P            = cap_filter_init_parms['P'],
        )
        if ctx.is_discharge():
            ctx.dis_filter.set_negative_init_params(
                bal_change_Q = dis_filter_init_parms['bal_change_Q'],
                exc_change_Q = dis_filter_init_parms['exc_change_Q'],
                dMdW         = dis_filter_init_parms['dMdW'],
                P            = dis_filter_init_parms['P'],
            )
        ctx.reinitialize_negative_filter()

    result = TDPT_initial_filter_config(
        status               = cap_status and dis_status,
        cap_status           = cap_status,
        dis_status           = dis_status,
        positive             = positive,
        cap_err              = dQ,
        cap_var              = dQ_var,
        cap_filter_init_parms = cap_filter_init_parms,
        dis_err              = M,
        dis_var              = M_var if ctx.is_discharge() else None,
        dis_filter_init_parms = dis_filter_init_parms,
        meta                 = TDPT_initial_filter_metadata(
            X1 = X1,
            X2 = X2,
        ),
    )

    ctx.log(f"Initialization result:\n{result}")
    return result