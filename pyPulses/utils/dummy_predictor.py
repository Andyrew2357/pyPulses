"""
Dummy predictors for future balance points
"""

from typing import Any, Callable

class DummyPred:
    """
    Dummy predictor that always returns the same values for initial and second 
    guess.
    """
    def __init__(self, x0: float, x1: float):
        """
        Parameters
        ----------
        x0, x1 : float
            first and second guesses
        """
        self.x0 = x0
        self.x1 = x1

    def predict0(self, *args, **kwargs) -> float:
        return self.x0
    
    def predict1(self, *args, **kwargs) -> float:
        return self.x1

class DummyPredFunc:
    """
    Dummy predictor that excepts arbitrary functions for its first and second
    guesses.
    """
    def __init__(self, f0: Callable[[Any], float], f1: Callable[[Any], float]):
        """
        Parameters
        ----------
        f0, f1 : Callable
            functions used for first and second guess respectively
        """
        self.predict0 = f0
        self.predict1 = f1
