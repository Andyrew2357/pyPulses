from ..devices.wfatd import wfAverager, wfBalance
from ..utils.getsetter import getSetter

import numpy as np
from typing import Any, Callable, List

class wfBalanceKnob():
    
    def __init__(self,
        l: float, 
        h: float, 
        f: Callable[[float | None], float | None],
        set: Callable[[float], Any] | None = None
    ):
      
        self.l = l
        self.h = h
        
        if set is None:
            self.f = f
        else:
            self.f = getSetter(f, set)

    def __call__(self, *args, **kwargs):
        return self.f(*args **kwargs)
    
    def set_low(self):
        self.f(self.l)

    def set_high(self):
        self.f(self.h)

class wfBalanceResult():
    def __init__(self, 
        knobs: List[wfBalanceKnob], 
        Anorm: np.ndarray, 
        b: np.ndarray,
        qA: np.ndarray,
        qb: np.ndarray,
    ):

        self.qA = qA
        self.qb = qb

        balnorm = -np.inv(Anorm.T @ Anorm) @ Anorm.T @ b
        self.x0 = np.zeros_like(balnorm)
        self.A = np.zeros_like(Anorm)
        self.b = b
        
        self.success = balnorm.min() < 0 or balnorm.max > 1
        for i, x in enumerate(knobs):
            self.A[:,i] = Anorm[:,i] / (x.h - x.l)
            self.qA[:, i] /= (x.h - x.l)
            self.x0[:, i] = x.l * (1 - balnorm[:, i]) + x.h * balnorm[:, i]
            if self.success:
                x(self.x0[i])

def wfBalanceCorrelated(
    knobs: List[wfBalanceKnob],
    balances: List[wfBalance],
    averager: wfAverager
) -> wfBalanceResult:
    
    for x in knobs:
        x.set_low()

    averager.take_curve()
    Y = [B() for B in balances]
    b = np.array([y[0] for y in Y]).reshape(-1, 1)
    qb = np.array([y[1] for y in Y]).reshape(-1, 1)

    A = []
    qA = []
    for x in knobs:
        x.set_high()
        averager.take_curve()
        Y = [B() for B in balances]
        A.append([y[0] for y in Y])
        qA.append([y[0] for y in Y])
        x.set_low()
    A = np.array(A).T
    qA = np.array(qA).T
    return wfBalanceResult(knobs, A, b, qA, qb)
    