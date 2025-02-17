
import numpy as np
from typing import Callable

class ExtrapPred1d():
    def __init__(self, support: int, order: int, 
                 default: Callable[..., float]):
        if order >= support:
            raise RuntimeError(
                f"order exceeds the maximum supported ({order} >= {support})."
            ) 
        self.support    = support
        self.order      = order
        self.default    = default

        self.X = np.zeros(support, dtype = float)
        self.Y = np.zeros(support, dtype = float)
        self.seen: int = 0

    def predict(self, x: float, *args) -> float:
        if self.seen == 0:
            return self.default(x, self, *args)
        
        coeff = np.polyfit(self.X[:self.seen], self.Y[:self.seen], 
                           min(self.order, self.seen - 1))
        poly = np.poly1d(coeff)
        return poly(x)
    
    def update(self, x: float, y: float):
        self.X[1:] = self.X[0:-1]
        self.X[0] = x

        self.Y[1:] = self.Y[0:-1]
        self.Y[0] = y

        if self.seen < self.support:
            self.seen += 1

    def reset(self):
        self.seen = 0.
