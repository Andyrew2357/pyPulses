"""
These are utility datastructures for root finding algorithms.
"""

from dataclasses import dataclass
from enum import Enum, auto

class RootFinderStatus(Enum):
    """
    Enum describing the condition of a root finding process

    Attributes
    ----------
    READY
    NEEDS_EVALUATION
    CONVERGED
    NO_ROOT_LIKELY
    CYCLING
    MAX_ITERATIONS
    """
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
    """
    Describes the state of a root finding process.

    Attributes
    ----------
    status : RootFinderStatus
    point : float
        point where a function evaluation is needed.
    root : float or None
        final root if found.
    iterations : int
        number of completed iterations.
    message : str
        descriptive status message.
    best_value : float
        best function evaluation see (closest to zero).
    """
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
