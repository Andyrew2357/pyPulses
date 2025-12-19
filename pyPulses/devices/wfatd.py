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
        ia = (ta // self.dt)
        ib = (tb // self.dt)
        return np.arange(ia * self.dt, ib * self.dt, self.dt), self.curve[ia:ib]
    
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

    def __call__(self) -> Tuple[float, float]:
        m, _ = np.polyfit(*self.averager.get_window(self.ta, self.tb), 1)
        return m

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

    def __call__(self) -> Tuple[float, float]:
        ml, cl = np.polyfit(*self.averager.get_window(self.tla, self.tlb), 1)
        mr, cr = np.polyfit(*self.averager.get_window(self.tla, self.tlb), 1)
        return (cr - cl) + self.t0 * (mr - ml)