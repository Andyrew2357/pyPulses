"""
Predict future balance points by performing a polynomial fit on the last few.
The user must provide functions for default0 and default1.
"""


import numpy as np
from typing import Callable, Tuple

class ExtrapPred1d():
    def __init__(self, support: int, order: int, 
                 default0: Callable[..., float], default1: Callable[..., float]):
        if order >= support:
            raise RuntimeError(
                f"order exceeds the maximum supported ({order} >= {support})."
            ) 
        self.support    = support
        self.order      = order
        self.default0   = default0
        self.default1   = default1

        self.X = np.zeros(support, dtype = float)
        self.Y = np.zeros(support, dtype = float)
        self.seen: int = 0

    def predict0(self, x: float) -> float:
        if self.seen == 0:
            return self.default0(x)
        
        coeff = np.polyfit(self.X[:self.seen], self.Y[:self.seen], 
                           min(self.order, self.seen - 1))
        poly = np.poly1d(coeff)
        return poly(x)
    
    def predict1(self, x: float, p: Tuple[float, float]) -> float:
        return self.default1(x, p, self)
    
    def update(self, x: float, y: float):
        self.X[1:] = self.X[0:-1]
        self.X[0] = x

        self.Y[1:] = self.Y[0:-1]
        self.Y[0] = y

        if self.seen < self.support:
            self.seen += 1

    def reset(self):
        self.seen = 0.
