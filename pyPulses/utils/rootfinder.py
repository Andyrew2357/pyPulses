"""
These are utility datastructures for root finding algorithms.
"""

from dataclasses import dataclass
from enum import Enum, auto

class RootFinderStatus(Enum):
    READY               = auto()
    NEEDS_EVALUATION    = auto()
    CONVERGED           = auto()
    NO_ROOT_LIKELY      = auto()
    CYCLING             = auto()
    MAX_ITERATIONS      = auto()

    def __str__(self):
        match self:
            case self.READY: 
                return "READY"
            case self.NEEDS_EVALUATION: 
                return "NEEDS_EVALUATION"
            case self.CONVERGED: 
                return "CONVERGED"
            case self.NO_ROOT_LIKELY:
                return "NO_ROOT_LIKELY"
            case self.CYCLING:
                return "CYCLING"
            case self.MAX_ITERATIONS:
                return "MAX_ITERATIONS"

@dataclass
class RootFinderState:
    status      : RootFinderStatus
    point       : float             # Point where function evaluation is needed
    root        : float | None      # Final root if found
    iterations  : int               # Number of iterations used
    message     : str               # Descriptive status message
    best_value  : float             # Best function value seen

    def __str__(self):
        s = ""
        s += f"status       : {str(self.status)}\n"
        s += f"point        : {self.point:.6f}\n"
        s += f"root         : {self.root}\n"
        s += f"iterations   : {self.iterations}\n"
        s += f"message      : {self.message}\n"
        s += f"best_value   : {self.best_value:.2e}\n"
        return s
