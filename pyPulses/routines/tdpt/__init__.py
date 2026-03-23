"""
pyPulses.routines.tdpt
======================
Tunneling-Domain Pulsed Tunneling (TDPT) balance routines.

Usage
-----
    from pyPulses.routines import tdpt

    # Context and initialization
    ctx  = tdpt.TDPTContext(...)
    cfg  = tdpt.TDPT_initialize_filters(ctx, ...)

    # Balance loop
    result = tdpt.TDPT_filter_balance(ctx)
    tdpt.TDPT_map_health_monitor(ctx)

    # Result and control dataclasses
    result  # tdpt.BalanceResult
    cfg     # tdpt.TDPT_initial_filter_config

    # Filter classes (rarely needed directly)
    tdpt.CapacitanceFilter
    tdpt.DischargeFilter
    tdpt.DischargeExtrapolator

    # Config constants (importable individually if needed)
    from pyPulses.routines.tdpt.config import *
"""

# Config constants — import * friendly
from .config import (
    HEURISTIC_NOISE_MULT,
    STRONG_UPDATE_R_MULT,
    RELIABILITY_FACTOR,
    LARGE_RESET_P_MULT,
    MODERATE_RESET_P_MULT,
    MAX_RAIL_HI_COUNT,
    INT_ADJ_LAMBDA,
    RAIL_HI_STREAK_THRESHOLD,
    RAIL_LO_STREAK_THRESHOLD,
    FAILURE_STREAK_THRESHOLD,
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

# Filter classes
from .cap_filter import CapacitanceFilter
from .dis_filter import DischargeFilter, DischargeExtrapolator

# Context
from .context import TDPTContext

# Initialization
from .initialize import (
    TDPT_initialize_filters,
    TDPT_initial_filter_config,
    TDPT_initial_filter_metadata,
)

# Balance
from .balance import (
    TDPT_filter_balance,
    TDPT_map_health_monitor,
    BalanceResult,
    SweepControls,
    SweepErrors,
)

__all__ = [
    # Config
    'HEURISTIC_NOISE_MULT',
    'STRONG_UPDATE_R_MULT',
    'RELIABILITY_FACTOR',
    'LARGE_RESET_P_MULT',
    'MODERATE_RESET_P_MULT',
    'MAX_RAIL_HI_COUNT',
    'INT_ADJ_LAMBDA',
    'RAIL_HI_STREAK_THRESHOLD',
    'RAIL_LO_STREAK_THRESHOLD',
    'FAILURE_STREAK_THRESHOLD',
    'DIS_INIT_BAL_UNC',
    'DIS_INIT_EXC_UNC',
    'DIS_INIT_P_MULT',
    'CAP_INIT_BAL_QG_MULT',
    'CAP_INIT_BAL_QC',
    'CAP_INIT_BAL_QdC',
    'CAP_INIT_EXC_QG_MULT',
    'CAP_INIT_EXC_QC',
    'CAP_INIT_EXC_QdC',
    'CAP_INIT_PG_MULT',
    'CAP_INIT_PC',
    'CAP_INIT_PdC',
    # Filters
    'CapacitanceFilter',
    'DischargeFilter',
    'DischargeExtrapolator',
    # Context
    'TDPTContext',
    # Initialization
    'TDPT_initialize_filters',
    'TDPT_initial_filter_config',
    'TDPT_initial_filter_metadata',
    # Balance
    'TDPT_filter_balance',
    'TDPT_map_health_monitor',
    'BalanceResult',
    'SweepControls',
    'SweepErrors',
]