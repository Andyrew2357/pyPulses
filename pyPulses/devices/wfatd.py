from abc import abstractmethod
from typing import Callable, Tuple
import numpy as np

class wfAverager():
    def __init__(self, scope_call: Callable[[], np.ndarray], dt: float):
        self.curve: np.ndarray = None
        self.scope_call = scope_call
        self.dt = dt

    def take_curve(self):
        self.curve = self.scope_call()

    def get_window(self, ta: float, tb: float) -> Tuple[np.ndarray, np.ndarray]:
        ia = max(0, int(ta // self.dt))
        ib = min(self.curve.size - 1, int(tb // self.dt))
        return self.dt * (ia + np.arange(ib - ia)), self.curve[ia:ib]
    
    def get_masked(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, len(self.curve) * self.dt, self.dt)
        return t[mask], self.curve[mask]
    
class wfBalance():
    def __init__(self, averager: wfAverager):
        self.averager = averager

    @abstractmethod
    def __call__() -> Tuple[float, float]: ...

class wfSlope(wfBalance):
    def __init__(self, 
        averager: wfAverager,
        ta: float,
        tb: float,
    ):
        super().__init__(averager)
        self.ta = ta
        self.tb = tb
        self.m = None
        self.c = None

    def __call__(self) -> Tuple[float, float]:
        (m, c), cov = np.polyfit(*self.averager.get_window(self.ta, self.tb), 1, cov=True)
        self.m = m
        self.c = c
        return m, cov[0, 0]
    
    def get_fit(self) -> Tuple[float, float]:
        return self.m, self.c

class wfSlopeMasked(wfBalance):
    def __init__(self, 
        averager: wfAverager,
        mask: np.ndarray,
    ):
        super().__init__(averager)
        self.mask = mask
        self.m = None
        self.c = None

    def __call__(self) -> Tuple[float, float]:
        (m, c), cov = np.polyfit(*self.averager.get_masked(self.mask), 1, cov=True)
        self.m = m
        self.c = c
        return m, cov[0, 0]
    
    def get_fit(self) -> Tuple[float, float]:
        return self.m, self.c

class wfJump(wfBalance):
    def __init__(self,
        averager: wfAverager,
        t0: float,
        tla: float,
        tlb: float,
        tra: float,
        trb: float,
    ):
        super().__init__(averager)
        self.t0 = t0
        self.tla = tla
        self.tlb = tlb
        self.tra = tra
        self.trb = trb

        self.ml = None
        self.mr = None
        self.cl = None
        self.cr = None

    def __call__(self) -> Tuple[float, float]:
        (ml, cl), covl = np.polyfit(*self.averager.get_window(self.tla, self.tlb), 1, cov=True)
        (mr, cr), covr = np.polyfit(*self.averager.get_window(self.tra, self.trb), 1, cov=True)
        self.ml = ml
        self.mr = mr
        self.cl = cl
        self.cr = cr
        cov = covl + covr
        return (cr - cl) + self.t0 * (mr - ml), cov[1, 1] + self.t0**2 * cov[0, 0]
    
    def get_left_fit(self) -> Tuple[float, float]:
        return self.ml, self.cl
    
    def get_right_fit(self) -> Tuple[float, float]:
        return self.mr, self.cr
    
    def get_left_slope(self) -> float:
        return self.ml
    
    def get_right_slope(self) -> float:
        return self.mr
    
class wfJumpMasked(wfBalance):
    def __init__(self,
        averager: wfAverager,
        mask_l: np.ndarray,
        mask_r: np.ndarray,
    ):
        super().__init__(averager)
        self.mask_l = mask_l
        self.mask_r = mask_r

        self.ml = None
        self.mr = None
        self.cl = None
        self.cr = None

    def __call__(self) -> Tuple[float, float]:
        (ml, cl), covl = np.polyfit(*self.averager.get_masked(self.mask_l), 1, cov=True)
        (mr, cr), covr = np.polyfit(*self.averager.get_masked(self.mask_r), 1, cov=True)
        self.ml = ml
        self.mr = mr
        self.cl = cl
        self.cr = cr
        cov = covl + covr
        return (cr - cl) + self.t0 * (mr - ml), cov[1, 1] + self.t0**2 * cov[0, 0]
    
    def get_left_fit(self) -> Tuple[float, float]:
        return self.ml, self.cl
    
    def get_right_fit(self) -> Tuple[float, float]:
        return self.mr, self.cr
    
    def get_left_slope(self) -> float:
        return self.ml
    
    def get_right_slope(self) -> float:
        return self.mr