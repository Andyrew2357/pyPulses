"""
Abstract classes used for processing data from time domain waveforms for the
purpose of balancing away error parameters using feedback control. `wfatd`
refers to `waveform averaging in the time domain`. 
"""

from .balance_parameter import balanceError
from ..utils import curves

from matplotlib.axes import Axes
from dataclasses import dataclass
from abc import abstractmethod
from typing import Callable, List, Tuple
import numpy as np
import warnings

class wfAveragerView(): ...

@dataclass
class wfCurveData(): ...

class wfPostProcess():
    """Post-processing step applied to time domain data after waveform averaging"""
    def __init__(self): ...

    @abstractmethod
    def __call__(t: np.ndarray, curve: np.ndarray) -> np.ndarray: ...

class wfAverager():
    """Abstract class for taking time domain data."""
    def __init__(self, scope_call: Callable[[], Tuple[np.ndarray, float, float]]):

        self.scope_call = scope_call

        self._unprocessed_curve: np.ndarray = None
        self._t: np.ndarray | None = None
        self._curve: np.ndarray | None = None
        self._dt: float | None = None
        self._t0: float | None = None
        self._N: int | None = None
        
        self._post_processes: List[wfPostProcess] = []
        self._views: List[wfAveragerView] = []
        
        self._supported_balances: List[wfBalance] = []
    
    def add_view(self, view: wfAveragerView):
        self._views.append(view)

    def remove_view(self, view: wfAveragerView):
        if view in self._views:
            self._views.remove(view)

    def add_post_process(self, process: wfPostProcess):
        self._post_processes.append(process)

    def remove_post_process(self, process: wfPostProcess):
        self._post_processes.remove(process)

    def take_curve(self, sweep_multiplier: int = 1):
        for i in range(sweep_multiplier):
            if i == 0:
                self._unprocessed_curve, self._dt, self._t0 = self.scope_call()
            else:
                self._unprocessed_curve += self.scope_call()[0]
        self._unprocessed_curve /= sweep_multiplier
        self._N = self._unprocessed_curve.size
        self._curve = self._unprocessed_curve.copy()
        self._t = self._t0 + self._dt * np.arange(self._N)

        for view in self._views:
            view._new_curve()
        for process in self._post_processes:
            self._curve = process(self._t, self._curve)

        self._unprocessed_curve = self._unprocessed_curve.astype(np.float32)
        self._curve = self._curve.astype(np.float32)
        self._t = self._t.astype(np.float32)

    def get_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._t, self._curve

    def get_unprocessed_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._t, self._unprocessed_curve    

    def get_curve_binary(self) -> bytes:
        return self._curve.tobytes()
    
    def get_unprocessed_curve_binary(self) -> bytes:
        return self._unprocessed_curve.tobytes()
    
    def get_curve_data(self) -> wfCurveData:
        return wfCurveData(
            binary = self.get_curve_binary(),
            unprocessed = None if len(self._post_processes) == 0 else \
                self.get_unprocessed_curve_binary(),
            N = self._N,
            dt = self._dt,
            t0 = self._t0
        )
    
    def plot_curve(self, ax: Axes, *args, **kwargs):
        ax.plot(self._t, self._curve, *args, **kwargs)
    
    def plot_unprocessed_curve(self, ax: Axes, *args, **kwargs):
        ax.plot(self._t, self._unprocessed_curve, *args, **kwargs)
    
    def plot_annotations(self, ax: Axes):
        for balance in self._supported_balances:
            balance.plot_annotations(ax)

@dataclass
class wfCurveData():
    binary: bytes
    unprocessed: bytes | None
    N: int
    dt: float
    t0: float

@dataclass
class wfRange():
    ta: float
    tb: float

    def span(self) -> float:
        return self.tb - self.ta

    def __call__(self, t: np.ndarray, offset) -> np.ndarray:
        return ((self.ta + offset) < t) & (t < (self.tb + offset))

class wfAveragerView():
    def __init__(self, 
        averager: wfAverager, 
        range: wfRange | List[wfRange],
        static_offset: float = 0.0, 
        dynamic_offset: Callable[[], float] = lambda: 0.0,
    ):
        
        self._averager = averager
        if not self in self._averager._views:
            self._averager.add_view(self)
        
        if isinstance(range, wfRange):
            self._ranges = [range]
        else:
            self._ranges = range
        self._static_offset = static_offset
        self._dynamic_offset = dynamic_offset
        self._offset: float | None = None

        # Recorded information so we don't repeat calculations. This isn't
        # really necessary
        self._state = {
            'msk': None,
            't': None,
            'curve': None,
            'tmean': None,
            'tsqm': None,
            'vmean': None,
            'vsqm': None,
            'tvsq': None,
            'lin_fit': None,
            'integral': None,
        }

    def _new_curve(self):
        self._offset = self._static_offset + self._dynamic_offset()
        for k, v in self._state.items():
            self._state[k] = None

    def _msk(self) -> np.ndarray:
        if self._state['msk'] is None:
            for i, rng in enumerate(self._ranges):
                if i == 0:
                    msk = rng(self._averager._t, self._offset)
                else:
                    msk |= rng(self._averager._t, self._offset)
            self._state['msk'] = msk
            if not msk.any():
                warnings.warn("wfAveragerView is disjoint with its wfAverager")

        return self._state['msk']
    
    def _t(self) -> np.ndarray:
        if self._state['t'] is None:
            self._state['t'] = self._averager._t[self._msk()]
        return self._state['t']
    
    def _curve(self) -> np.ndarray:
        if self._state['curve'] is None:
            self._state['curve']  = self._averager._curve[self._msk()]
        return self._state['curve']

    def _tmean(self) -> float:
        if self._state['tmean'] is None:
            self._state['tmean'] = np.nanmean(self._t())
        return self._state['tmean']
    
    def _tsqm(self) -> float:
        if self._state['tsqm'] is None:
            self._state['tsqm'] = np.nanmean(self._t()**2)
        return self._state['tsqm']

    def _vmean(self) -> float:
        if self._state['vmean'] is None:
            self._state['vmean'] = np.nanmean(self._curve())
        return self._state['vmean']
    
    def _vsqm(self) -> float:
        if self._state['vsqm'] is None:
            self._state['vsqm'] = np.nanmean(self._curve()**2)
        return self._state['vsqm']
    
    def _tvsq(self) -> float:
        if self._state['tvsq'] is None:
            self._state['tvsq'] = np.nanmean(self._t()*self._curve())
        return self._state['tvsq']

    def mean(self) -> Tuple[float, float]:
        return self._vmean(), self._vsqm() - self._vmean()**2
    
    def lin_fit(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x = self._tmean()
        y = self._vmean()
        xx = self._tsqm()
        yy = self._vsqm()
        xy = self._tvsq()

        m = (xy - x * y) / (xx - x * x)
        b = y - m * x
        N = self._averager._N
        chisq = (yy + m*m*xx + b*b - 2*m*xy - 2*b*y + 2*m*b*x) / (N - 2)
        m_var = chisq / (xx - x * x)
        b_var = chisq / (1 - x * x / xx)

        return (m, b), (m_var, b_var)
    
    def integral(self) -> Tuple[float, float]:
        if self._state['integral'] is None:
            x = self._t()
            y = self._curve()
            dx = np.diff(x)

            w = np.zeros_like(y)
            w[0] = dx[0] / 2
            w[-1] = dx[-1] / 2
            w[1:-1] = (dx[:-1] + dx[1:]) / 2

            I = np.nansum(w * y)
            var = self._vsqm() - self._vmean()**2
            I_var = var * np.nansum(w**2)

            self._state['integral'] = (I, I_var)

        return self._state['integral']

    def span(self) -> float:
        return sum([rng.span() for rng in self._ranges])

    def plot_lin_fit(self, ax: Axes, color: str):
        (m, c), _ = self.lin_fit()
        ax.plot(self._averager._t, m*self._averager._t + c, color=color, linestyle='dashed')
        for rng in self._ranges:
            ta, tb = rng.ta + self._offset, rng.tb + self._offset
            ax.plot([ta, tb], [m*ta + c, m*tb + c], color=color, alpha=0.4, linewidth=8)

    def __del__(self):
        self._averager.remove_view(self)

"""wfBalance Classes; each represents various values against which we can balance"""

class wfBalance(balanceError):
    def __init__(self):
        super().__init__()
        self._annotate: bool = True

    @abstractmethod
    def __call__(self) -> Tuple[float, float]: ...

    def plot_annotations(self, ax: Axes):
        pass

class wfFunction(balanceError):
    def __init__(self, func: Callable[[], Tuple[float, float]]):
        super().__init__()
        self.func = func

    def __call__(self) -> Tuple[float, float]:
        self.error, self.error_variance = self.func()
        return self.error, self.error_variance

class wfSlope(wfBalance):
    def __init__(self, region: wfAveragerView):
        super().__init__()
        self._region = region
        if not self in self._region._averager._supported_balances:
            self._region._averager._supported_balances.append(self)
        self._plot_color = 'r'

    def __call__(self) -> Tuple[float, float]:
        (m, _), (mvar, _) = self._region.lin_fit()
        self.error = m
        self.var = mvar
        return m, mvar
    
    def plot_annotations(self, ax: Axes):
        if not self._annotate:
            return
        self._region.plot_lin_fit(ax, color=self._plot_color)

    def get_fit(self) -> Tuple[float, float]:
        (m, c), _ = self._region.lin_fit()
        return m, c
    
    def __del__(self):
        if self in self._region._averager._supported_balances:
            self._region._averager._supported_balances.remove(self)
        super().__del__()

class wfJump(wfBalance):
    def __init__(self, left: wfAveragerView, right: wfAveragerView, t0: float):
        super().__init__()
        self._left = left
        self._right = right
        if not self in self._left._averager._supported_balances:
            self._left._averager._supported_balances.append(self)
        if not self in self._right._averager._supported_balances:
            self._right._averager._supported_balances.append(self)
        self._t0 = t0
        self._plot_color = 'g'

    def __call__(self) -> Tuple[float, float]:
        (ml, cl), (mlv, clv) = self._left.lin_fit()
        (mr, cr), (mrv, crv) = self._right.lin_fit()
        self.error = cr - cl + self._t0 * (mr - ml)
        self.var = crv + clv + self._t0 * (mrv + mlv)
        return self.error, self.var
    
    def plot_annotations(self, ax: Axes):
        if not self._annotate:
            return
        self._left.plot_lin_fit(ax, color=self._plot_color)
        self._right.plot_lin_fit(ax, color=self._plot_color)
        ml, cl = self.get_left_fit()
        mr, cr = self.get_right_fit()
        ymin = ml*self._t0 + cl
        ymax = mr*self._t0 + cr
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        ax.vlines(self._t0, ymin=ymin, ymax=ymax, 
                  alpha=0.4, linewidth=8, color=self._plot_color)
    
    def get_left_fit(self) -> Tuple[float, float]:
        (ml, cl), _ = self._left.lin_fit()
        return ml, cl
    
    def get_right_fit(self) -> Tuple[float, float]:
        (mr, cr), _ = self._right.lin_fit()
        return mr, cr

    def get_left_slope(self) -> float:
        return self.get_left_fit()[0]
    
    def get_right_slope(self) -> float:
        return self.get_right_fit()[0]
    
    def __del__(self):
        if self in self._left._averager._supported_balances:
            self._left._averager._supported_balances.remove(self)
        if self in self._right._averager._supported_balances:
            self._right._averager._supported_balances.remove(self)
        super().__del__()
    
class wfIntegral(wfBalance):
    def __init__(self, zero_region: wfAveragerView, int_region: wfAveragerView):
        super().__init__()
        self._zero_region = zero_region
        self._int_region = int_region
        if not self in self._zero_region._averager._supported_balances:
            self._zero_region._averager._supported_balances.append(self)
        if not self in self._int_region._averager._supported_balances:
            self._int_region._averager._supported_balances.append(self)
        self._plot_color = 'b'

    def __call__(self) -> Tuple[float, float]:
        I, Ivar = self._int_region.integral()
        Z, Zvar = self._zero_region.mean()
        R = self._int_region.span()
        self.error = I - R * Z
        self.var = Ivar + R * Zvar
        return self.error, self.var
    
    def plot_annotations(self, ax: Axes):
        pass # TODO implement this nicely

    def __del__(self):
        if self in self._zero_region._averager._supported_balances:
            self._zero_region._averager._supported_balances.remove(self)
        if self in self._int_region._averager._supported_balances:
            self._int_region._averager._supported_balances.remove(self)
        super().__del__()
    
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
    for the integral and identifying the right time constant (Use with caution).
    """

    def __init__(self,
        zero_region: wfAveragerView,
        correction_region: wfAveragerView,
        tau: float,
    ):
        self._zero = zero_region
        self._correction = correction_region
        self._tau = tau

    def __call__(self, _, curve: np.ndarray) -> np.ndarray:
        # Fit the zero level
        (zm, zc), _ = self._zero.lin_fit()
        
        # Get the mask for the correction region
        msk = self._correction._msk()
        tm = self._correction._t()

        # Apply the correction over the desired region
        correction = curves.integrate_trapz_padded(
            tm, curve[msk] - (zm * tm + zc), mask_nans=True
        ) / self._tau

        curve[msk] += correction

        return curve
