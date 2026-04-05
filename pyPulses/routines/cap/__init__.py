"""
pyPulses.routines.cap
=====================
Penetration capacitance measurement routines.

Two measurement modes are supported, both operating on a `CapContext`:

Off-balance (`cap_measure`)
    The bridge is held at the most recent balance point. The lock-in reading
    is converted to (Cex, Closs) using the Kalman filter's cached complex gain.
    Fast and passive — no hardware adjustments, no filter updates.

On-balance (`cap_balance`)
    Iteratively re-balances Vstd at each phase-space point using the Kalman
    filter's gain estimate and the extrapolator's predicted starting point.
    Updates the filter with each (dL, dv) observation.

Typical usage
-------------
    from pyPulses.routines import cap

    # Build context
    ctx = cap.CapContext(
        Vstd        = lockin.resolve('Vstd'),
        Theta       = lockin.resolve('Theta'),
        lockin_call = lockin.resolve('get_average'),
        Vstd_range  = 0.1,
        Vex         = 1.0,
        Cstd        = 1e-12,
    )

    # Optional: attach a serializable rescale callback
    ctx.set_rescale_callback(lockin, 'auto_gain')

    # Initialize (three-point balance seeds A, P, and K_matrix)
    cfg = cap.cap_initialize(ctx, method='three_point')
    # or two-point:
    cfg = cap.cap_initialize(ctx, method='two_point', dVstd=0.05+0j)

    # Off-balance measurement at each phase-space point
    result = cap.cap_measure(ctx)
    result = cap.cap_measure(ctx, use_matrix=True)  # use full K matrix if available

    # On-balance measurement at each phase-space point
    result = cap.cap_balance(ctx)

    # Save / restore state
    ctx.save_state_json('cap_state.json')
    ctx.load_state_json('cap_state.json')

    # Reconstruct fully from config
    ctx2 = cap.CapContext.from_json('cap_state.json')

Result dataclasses
------------------
    cap.CapMeasureResult       — off-balance measurement result
    cap.CapBalanceResult       — on-balance measurement result
    cap.ThreePointBalanceResult — result of three-point initial balance
    cap.TwoPointBalanceResult   — result of two-point initial balance
    cap.CapInitialFilterConfig  — result of cap_initialize / cap_initialize_filter

Context, filter, and extrapolator
----------------------------------
    cap.CapContext
    cap.CapFilter
    cap.CapExtrapolator

Channel protocols (defined in channel_adapter)
----------------------------------------------
    cap.LockInChannel    — protocol for lockin_call: () -> (mean, cov)
    cap.CommandChannel   — protocol for rescale_callback: () -> None

Config constants
----------------
    from pyPulses.routines.cap.config import *
"""

# Config
from .config import (
    PROCESS_NOISE_COEFF,
    CAP_INIT_P_MULT_THREE_POINT,
    CAP_INIT_P_MULT_TWO_POINT,
)

# Filter and extrapolator
from .cap_filter import CapFilter, CapExtrapolator

# Context
from .context import CapContext

# Initialization subroutines
from .initialize import (
    cap_balance_three_point,
    cap_balance_two_point,
    cap_initialize_filter,
    cap_initialize,
    ThreePointBalanceResult,
    TwoPointBalanceResult,
    CapInitialFilterConfig,
)

# Measurement functions
from .measure import cap_measure, CapMeasureResult
from .balance import cap_balance, CapBalanceResult

__all__ = [
    # Config
    'PROCESS_NOISE_COEFF',
    'CAP_INIT_P_MULT_THREE_POINT',
    'CAP_INIT_P_MULT_TWO_POINT',
    # Filter and extrapolator
    'CapFilter',
    'CapExtrapolator',
    # Context
    'CapContext',
    # Initialization
    'cap_balance_three_point',
    'cap_balance_two_point',
    'cap_initialize_filter',
    'cap_initialize',
    'ThreePointBalanceResult',
    'TwoPointBalanceResult',
    'CapInitialFilterConfig',
    # Measurement
    'cap_measure',
    'CapMeasureResult',
    'cap_balance',
    'CapBalanceResult',
]