"""
Dummy predictors for future balance points
"""

from typing import Any, Callable

class DummyPred:
    def __init__(self, x0: float, x1: float):
        self.x0 = x0,
        self.x1 = x1

    def predict0(self, *args, **kwargs) -> float:
        return self.x0
    
    def predict1(self, *args, **kwargs) -> float:
        return self.x1

class DummyPredFunc:
    def __init__(self, f0: Callable[[Any], float], f1: Callable[[Any], float]):
        self.predict0 = f0
        self.predict1 = f1
