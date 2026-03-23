# Discharge filter — Kalman process noise
HEURISTIC_NOISE_MULT  = 1.0e-3
STRONG_UPDATE_R_MULT  = 1.0e-4
RELIABILITY_FACTOR    = 1.0e-2
LARGE_RESET_P_MULT    = 1.0
MODERATE_RESET_P_MULT = 0.1

# Discharge balance - Catch for repeated railing
MAX_RAIL_HI_COUNT = 3

# Integrated balance
INT_ADJ_LAMBDA        = 1.0

# Map-level health monitor thresholds
RAIL_HI_STREAK_THRESHOLD = 3
RAIL_LO_STREAK_THRESHOLD = 3
FAILURE_STREAK_THRESHOLD = 5

# Filter initialization — discharge
DIS_INIT_BAL_UNC      = 0.0
DIS_INIT_EXC_UNC      = 0.0
DIS_INIT_P_MULT       = 1.0e-4

# Filter initialization — capacitance
CAP_INIT_BAL_QG_MULT  = 1.0e-3
CAP_INIT_BAL_QC       = 1.0e-3
CAP_INIT_BAL_QdC      = 1.0e-4
CAP_INIT_EXC_QG_MULT  = 1.0e-2
CAP_INIT_EXC_QC       = 1.0e-3
CAP_INIT_EXC_QdC      = 1.0e-3
CAP_INIT_PG_MULT      = 0.1
CAP_INIT_PC           = 1.0e-3
CAP_INIT_PdC          = 1.0