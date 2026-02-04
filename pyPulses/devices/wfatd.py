from ..utils import curves

from abc import abstractmethod
from typing import Callable, List, Tuple
import numpy as np

class wfBalance():
    def __init__(self, averager: wfAverager):
        self.averager = averager

    @abstractmethod
    def __call__() -> Tuple[float, float]: ...

class wfPostProcess():
    def __init__(self): ...

    @abstractmethod
    def __call__(t: np.ndarray, curve: np.ndarray) -> np.ndarray: ...

class wfAverager():
    def __init__(self, scope_call: Callable[[], np.ndarray], dt: float, N: int):
        self.curve: np.ndarray = None
        self.scope_call = scope_call
        self.dt = dt
        self.t = np.arange(0, len(self.curve) * self.dt, self.dt)
        self.post_processes: List[wfPostProcess] = []

    def take_curve(self):
        self.curve = self.scope_call()
        for process in self.post_processes:
            self.curve = process(self.t, self.curve)

    def get_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.t, self.curve

    def get_window(self, ta: float, tb: float) -> Tuple[np.ndarray, np.ndarray]:
        msk = (self.t >= ta) & (self.t <= tb)
        return self.t[msk], self.curve[msk]
    
    def get_masked(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.t[mask], self.curve[mask]
    
    def add_post_process(self, process: wfPostProcess):
        self.post_processes.append(process)

    def remove_post_process(self, process: wfPostProcess):
        self.post_processes.remove(process)

"""wfBalance Classes; each represents various values against which we can balance"""

class wfFunction(wfBalance):
    def __init__(self, func: Callable[[], Tuple[float, float]]):
        super().__init__(None)
        self.func = func

    def __call__(self) -> Tuple[float, float]:
        return self.func()

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
    
# TODO
class wfIntegral(wfBalance):
    def __init__(self,
        averager: wfAverager,
        ta: float,
        tb: float,
    ):
        super().__init__(averager)
        self.ta = ta
        self.tb = tb

        self.A: float | None = None

    def __call__(self) -> Tuple[float, float]:
        pass

    def get_integral(self) -> float:
        return self.A

# TODO
class wfIntegralMasked(wfBalance):
    def __init__(self,
        averager: wfAverager,
        mask: np.ndarray,
    ):
        super().__init__(averager)
        self.mask = mask

        self.A: float | None = None

    def __call__(self) -> Tuple[float, float]:
        pass

    def get_integral(self) -> float:
        return self.A
    
"""wfPostProcess Classes; each is a post-processing step we can take after the curve is acquired"""

class wfCompensateHighPass(wfPostProcess):
    """
    This process is intended to correct for a high pass filtering effect. Given
    an RC filter like so,

                                 C
                          Q ────┤├─────┬───── Q'
                                       ⌇ R
                                       │
                                       ⏚
    
    the observed signal Q' obeys a differential equation dQ'/dt = -Q'/τ + dQ/dt.
    Integrating this gives us a method to recover Q from the observed Q'.

                    Q(t) = dQ0 + Q'(t) + Int[Q'(t') dt', 0, t]/τ

    The trickiest part of implementing this procedure involves setting a zero
    for the integral. 
    """

    def __init__(self, 
        tau: float, 
        zero: wfSlope | wfSlopeMasked,
        t0: float | None = None,
        t1: float | None = None,
        correct_msk: np.ndarray | None = None,
        ignore_msk: np.ndarray | None = None,
    ):
        self.tau = tau
        self.zero = zero
        self.t0 = t0
        self.t1 = t1
        self.ignore_msk = ignore_msk
        self.correct_msk = correct_msk

    def __call__(self, t: np.ndarray, curve: np.ndarray) -> np.ndarray:
        
        # Mask away artifacts by setting them to NaN
        curve[self.ignore_msk] = np.nan

        # Create a mask for the region we're going to correct
        if self.correct_msk is None:
            correct_msk = (t >= self.t0) & (t <= self.t1)
        else:
            correct_msk = self.correct_msk

        # Determine the 'zero' level
        self.zero()
        zm , zb = self.zero.get_fit()
        zero_lvl = zm * t[correct_msk] + zb

        # Apply the Correction over the desired interval
        correction = curves.integrate_trapz_padded(
            t[correct_msk],
            curve[correct_msk] - zero_lvl,
            mask_nans = True,
        )
        curve[correct_msk] += correction / self.tau

        return curve