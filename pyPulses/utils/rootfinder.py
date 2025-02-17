from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

class RootFinderStatus(Enum):
    READY               = auto()
    NEEDS_EVALUATION    = auto()
    CONVERGED           = auto()
    NO_ROOT_LIKELY      = auto()
    CYCLING             = auto()
    MAX_ITERATIONS      = auto()

@dataclass
class RootFinderState:
    status      : RootFinderStatus
    point       : float             # Point where function evaluation is needed
    root        : Optional[float]   # Final root if found
    iterations  : int               # Number of iterations used
    message     : str               # Descriptive status message
    best_value  : float             # Best function value seen
