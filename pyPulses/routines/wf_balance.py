from ..devices.wfatd import wfAverager, wfBalance
from ..utils.getsetter import getSetter

import time
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

    def __call__(self, val: float | None = None):
        if val is None:
            return self.f()
        if val < self.l or val > self.h:
            raise ValueError(f"Value {val} out of bounds [{self.l}, {self.h}]")
        return self.f(val)
    
    def set_low(self):
        self.f(self.l)

    def set_high(self):
        self.f(self.h)

class wfBalanceResult():
    def __init__(self):
        self.success: bool
        self.x0: np.ndarray
        self.A: np.ndarray
        self.b: np.ndarray
        self.qA: np.ndarray
        self.qb: np.ndarray

class wfBalanceCorrelatedResult(wfBalanceResult):
    def __init__(self, 
        knobs: List[wfBalanceKnob], 
        Anorm: np.ndarray, 
        b: np.ndarray,
        qA: np.ndarray,
        qb: np.ndarray,
    ):

        self.qA = qA
        self.qb = qb

        balnorm = -np.linalg.inv(Anorm.T @ Anorm) @ Anorm.T @ b
        balnorm = balnorm.flatten()
        self.x0 = np.zeros_like(balnorm)
        self.A = np.zeros_like(Anorm)
        self.b = b
        
        self.success = balnorm.min() >= 0 and balnorm.max() <= 1
        for i, x in enumerate(knobs):
            self.A[:,i] = Anorm[:,i] / (x.h - x.l)
            self.qA[:, i] /= (x.h - x.l)**2
            self.x0[i] = x.l * (1 - balnorm[i]) + x.h * balnorm[i]
            if self.success:
                x(self.x0[i])
    
class wfBalanceUncorrelatedResult(wfBalanceResult):
    def __init__(self, 
        knobs: List[wfBalanceKnob], 
        Anorm: np.ndarray, 
        b: np.ndarray,
        qA: np.ndarray,
        qb: np.ndarray,
    ):

        self.qA = qA
        self.qb = qb

        balnorm = -b.flatten() / np.diag(Anorm)
        self.x0 = np.zeros_like(balnorm)
        self.A = np.zeros_like(Anorm)
        self.b = b
        
        self.success = balnorm.min() >= 0 and balnorm.max() <= 1
        for i, x in enumerate(knobs):
            self.A[:,i] = Anorm[:,i] / (x.h - x.l)
            self.qA[:, i] /= (x.h - x.l)**2
            self.x0[i] = x.l * (1 - balnorm[i]) + x.h * balnorm[i]
            if self.success:
                x(self.x0[i])

def balance_against_waveform(
    knobs: List[wfBalanceKnob],
    balances: List[wfBalance],
    averager: wfAverager,
    settle_time: float = 0.0,
    correlated: bool = False,
) -> wfBalanceResult:
    
    for x in knobs:
        x.set_low()
    time.sleep(settle_time)

    averager.take_curve()
    Y = [B() for B in balances]
    b = np.array([y[0] for y in Y]).reshape(-1, 1)
    qb = np.array([y[1] for y in Y]).reshape(-1, 1)
    A = []
    qA = []
    for x in knobs:
        x.set_high()
        time.sleep(settle_time)
        averager.take_curve()
        Y = [B() for B in balances]
        A.append(np.array([y[0] for y in Y]).reshape(-1, 1) - b)
        qA.append(np.array([y[1] for y in Y]).reshape(-1, 1) + qb)
        x.set_low()
    A = np.column_stack(A)
    qA = np.column_stack(qA)

    if correlated:
        return wfBalanceCorrelatedResult(knobs, A, b, qA, qb)
    else: 
        return wfBalanceUncorrelatedResult(knobs, A, b, qA, qb) 
