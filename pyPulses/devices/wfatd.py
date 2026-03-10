"""
Abstract classes used for processing data from time domain waveforms for the
purpose of balancing away error parameters using feedback control. `wfatd`
refers to `waveform averaging in the time domain`. 
"""

from ..routines.balance_parameter import balanceError
from ..utils import curves

from .abstract_device import abstractDevice
from .channel_adapter import ScopeChannel, OffsetChannel, ErrorChannel
from .registry import (
    register_device_class, 
    format_reference,
    resolve_reference,
    DeferredReference,
    DeviceRegistry,
)

from matplotlib.axes import Axes
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, List, Tuple
import numpy as np
import warnings

class wfPostProcess(abstractDevice):
    """Post-processing step applied to time domain data after waveform averaging"""
    def __init__(self, registry_id: str | None = None, logger: Logger | None = None):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

    def __call__(t: np.ndarray, curve: np.ndarray) -> np.ndarray: ...

@register_device_class("wfAverager")
class wfAverager(abstractDevice):
    """Abstract class for taking time domain data."""
    def __init__(self, 
        scope_call: ScopeChannel | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

        self.scope_call = scope_call

        self._unprocessed_curve: np.ndarray = None
        self._t: np.ndarray | None = None
        self._curve: np.ndarray | None = None
        self._dt: float | None = None
        self._t0: float | None = None
        self._N: int | None = None
        
        self._post_processes: List[wfPostProcess] = []
        self._views: List['wfAveragerView'] = []
        
        self._supported_balances: List['wfBalance'] = []
    
    def add_view(self, view: 'wfAveragerView'):
        self._views.append(view)

    def remove_view(self, view: 'wfAveragerView'):
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
    
    def get_curve_data(self) -> 'wfCurveData':
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

    def _serialize_state(self) -> Dict[str, Any]:
        try:
            scope_ref = format_reference(self.scope_call)
        except:
            scope_ref = None

        post_processes = []
        for pp in self._post_processes:
            try:
                post_processes.append(format_reference(pp))
            except:
                pass  # Skip unregistered post-processes

        return {'scope_call': scope_ref, 'post_processes': post_processes}

    def _deserialize_state(self, state: Dict[str, Any]):
        if 'scope_call' in state and state['scope_call'] is not None:
            self.scope_call = DeferredReference(state['scope_call'])
        if 'post_processes' in state:
            self._post_processes_refs = state['post_processes']
        self._resolve_references()

    def _resolve_references(self):
        if isinstance(self.scope_call, DeferredReference):
            self.scope_call = self.scope_call.unwrap()

        # Resolve post-process references
        if hasattr(self, '_post_processes_refs'):
            self._post_processes = []
            for ref in self._post_processes_refs:
                pp = resolve_reference(ref)
                if pp is not None:
                    self._post_processes.append(pp)
            del self._post_processes_refs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfAverager':
        registry_id = config.pop('registry_id')
        scope_call = config.pop('scope_call', None)
        if scope_call is not None:
            scope_call = DeferredReference(scope_call)
        
        return cls(scope_call=scope_call, registry_id=registry_id)

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
    
    def to_dict(self) -> Dict[str, float]:
        return {'ta': self.ta, 'tb': self.tb}
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'wfRange':
        return cls(ta=d['ta'], tb=d['tb'])

@register_device_class("wfAveragerView")
class wfAveragerView(abstractDevice):
    def __init__(self, 
        averager: wfAverager | DeferredReference, 
        range: wfRange | List[wfRange],
        static_offset: float = 0.0, 
        dynamic_offset: OffsetChannel | DeferredReference | None = None,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_post_init: bool = False,
    ):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)
        
        self._averager = averager
        
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

        if not skip_post_init:
            self._resolve_references()
            self._post_init()

    def _post_init(self):
        if not self in self._averager._views:
            self._averager.add_view(self)

    def _new_curve(self):
        offset = self._dynamic_offset() if callable(self._dynamic_offset) else 0.0
        self._offset += self._static_offset + offset
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

    def _serialize_state(self) -> Dict[str, Any]:
        try:
            dynamic_offset_ref = format_reference(self._dynamic_offset)
        except:
            dynamic_offset_ref = None

        return {
            'averager': format_reference(self._averager),
            'ranges': [rng.to_dict() for rng in self._ranges],
            'static_offset': self._static_offset,
            'dynamic_offset': dynamic_offset_ref,
        }
    
    def _deserialize_state(self, state: Dict[str, Any]):
        if 'averager' in state and state['averager'] is not None:
            self._averager = DeferredReference(state['averager'])
        if 'ranges' in state:
            self._ranges = [wfRange.from_dict(r) for r in state['ranges']]
        if 'static_offset' in state:
            self._static_offset = state['static_offset']
        if 'dynamic_offset' in state and state['dynamic_offset'] is not None:
            self._dynamic_offset = DeferredReference(state['dynamic_offset'])

        self._resolve_references()
        self._post_init()

    def _resolve_references(self):
        if isinstance(self._averager, DeferredReference):
            self._averager = self._averager.unwrap()
        if isinstance(self._dynamic_offset, DeferredReference):
            self._dynamic_offset = self._dynamic_offset.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfAveragerView':
        registry_id = config.pop('registry_id')
        averager = config.pop('averager', None)
        if averager is not None:
            averager = DeferredReference(averager)
        ranges_data = config.pop('ranges', [])
        ranges = [wfRange.from_dict(r) for r in ranges_data]

        static_offset = config.pop('static_offset', 0.0)
        dynamic_offset = config.pop('dynamic_offset', None)
        if dynamic_offset is not None:
            dynamic_offset = DeferredReference(dynamic_offset)
        
        return cls(
            averager=averager,
            ranges=ranges,
            static_offset=static_offset,
            dynamic_offset=dynamic_offset,
            registry_id=registry_id,
            skip_post_init=True,
        )

    def __del__(self):
        if hasattr(self, '_averager') and self._averager:
            self._averager.remove_view(self)

"""wfBalance Classes; each represents various values against which we can balance"""

class wfBalance(balanceError, abstractDevice):
    def __init__(self, registry_id: str | None = None, logger: Logger | None = None):
        balanceError.__init__(self)
        abstractDevice.__init__(self, logger)
        DeviceRegistry.register(self, registry_id=registry_id)
        self._annotate: bool = True

    def __call__(self) -> Tuple[float, float]: ...

    def plot_annotations(self, ax: Axes):
        pass

    def _serialize_state(self) -> Dict[str, Any]:
        return {'annotate': self._annotate}
    
    def _deserialize_state(self, state: Dict[str, Any]):
        if 'annotate' in state:
            self._annotate = state['annotate']

@register_device_class("wfFunction")
class wfFunction(wfBalance):
    def __init__(self, 
        func: ErrorChannel | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,    
    ):
        super().__init__(registry_id=registry_id, logger=logger)
        self.func = func

    def __call__(self) -> Tuple[float, float]:
        self.error, self.error_variance = self.func()
        return self.error, self.error_variance
    
    def _serialize_state(self) -> Dict[str, Any]:
        config = super()._serialize_state()
        try:
            func_ref = format_reference(self.func)
        except:
            func_ref = None
        config.update({'func': func_ref})

    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        if 'func' in state and state['func'] is not None:
            self.func = DeferredReference(state['func'])
        self._resolve_references()

    def _resolve_references(self):
        if isinstance(self.func, DeferredReference):
            self.func = self.func.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfFunction':
        registry_id = config.pop('registry_id')
        func = config.pop('func', None)
        if func is not None:
            func = DeferredReference(func)
        
        return cls(func=func, registry_id=registry_id)
    
@register_device_class("wfSlope")
class wfSlope(wfBalance, abstractDevice):
    def __init__(self, 
        region: wfAveragerView | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_post_init: bool = False,
    ):
        super().__init__(registry_id=registry_id, logger=logger)

        self._region = region
        self._plot_color = 'r'

        if not skip_post_init:
            self._resolve_references()
            self._post_init()

    def _post_init(self):
        if not self in self._region._averager._supported_balances:
            self._region._averager._supported_balances.append(self)

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
    
    def _serialize_state(self) -> Dict[str, Any]:
        config = super()._serialize_state()
        config.update({
            'region': format_reference(self._region) if self._region else None,
            'plot_color': self._plot_color,
        })
        return config
    
    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        if 'region' in state and state['region'] is not None:
            self._region = DeferredReference(state['region'])
        if 'plot_color' in state:
            self._plot_color = state['plot_color']
        
        self._resolve_references()
        self._post_init()

    def _resolve_references(self):
        if isinstance(self._region, DeferredReference):
            self._region = self._region.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfSlope':
        registry_id = config.pop('registry_id')
        region = config.pop('region', None)
        if region is not None:
            region = DeferredReference(region)
        
        return cls(
            region=region,
            registry_id=registry_id,
            skip_post_init=True,
        )
    
    def __del__(self):
        if hasattr(self, '_region') and self._region:
            if self in self._region._averager._supported_balances:
                self._region._averager._supported_balances.remove(self)

@register_device_class("wfJump")
class wfJump(wfBalance, abstractDevice):
    def __init__(self, 
        left: wfAveragerView | DeferredReference, 
        right: wfAveragerView | DeferredReference, 
        t0: float,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_post_init: bool = False,
    ):
        super().__init__(registry_id=registry_id, logger=logger)
        self._left = left
        self._right = right
        self._t0 = t0
        self._plot_color = 'g'

        if not skip_post_init:
            self._resolve_references()
            self._post_init()

    def _post_init(self):
        if not self in self._left._averager._supported_balances:
            self._left._averager._supported_balances.append(self)
        if not self in self._right._averager._supported_balances:
            self._right._averager._supported_balances.append(self)

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
    
    def _serialize_state(self) -> Dict[str, Any]:
        config = super()._serialize_state()
        config.update({
            'left': format_reference(self._left) if self._left else None,
            'right': format_reference(self._right) if self._right else None,
            't0': self._t0,
            'plot_color': self._plot_color,
        })
        return config
    
    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        if 'left' in state and state['left'] is not None:
            self._left = DeferredReference(state['left'])
        if 'right' in state and state['right'] is not None:
            self._right = DeferredReference(state['right'])
        if 't0' in state:
            self._t0 = state['t0']
        if 'plot_color' in state:
            self._plot_color = state['plot_color']
        
        self._resolve_references()
        self._post_init()

    def _resolve_references(self):
        if isinstance(self._left, DeferredReference):
            self._left = self._left.unwrap()
        if isinstance(self._right, DeferredReference):
            self._right = self._right.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfJump':
        registry_id = config.pop('registry_id')
        left = config.pop('left', None)
        if left is not None:
            left = DeferredReference(left)
        right = config.pop('right', None)
        if right is not None:
            right = DeferredReference(right)
        t0 = config.pop('t0', 0.0)
        
        return cls(
            left=left,
            right=right,
            t0=t0,
            registry_id=registry_id,
            skip_post_init=True,
        )
    
    def __del__(self):
        if hasattr(self, '_left') and self._left:
            if self in self._left._averager._supported_balances:
                self._left._averager._supported_balances.remove(self)
        if hasattr(self, '_right') and self._right:
            if self in self._right._averager._supported_balances:
                self._right._averager._supported_balances.remove(self)

@register_device_class("wfIntegral")
class wfIntegral(wfBalance, abstractDevice):
    def __init__(self, 
        zero_region: wfAveragerView | DeferredReference, 
        int_region: wfAveragerView | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_post_init: bool = False,
    ):
        super().__init__(registry_id=registry_id, logger=logger)
        
        self._zero_region = zero_region
        self._int_region = int_region
        self._plot_color = 'b'
        
        if not skip_post_init:
            self._resolve_references()
            self._post_init()

    def _post_init(self):
        if self not in self._zero_region._averager._supported_balances:
            self._zero_region._averager._supported_balances.append(self)
        if self not in self._int_region._averager._supported_balances:
            self._int_region._averager._supported_balances.append(self)
    
    def __call__(self) -> Tuple[float, float]:
        I, Ivar = self._int_region.integral()
        Z, Zvar = self._zero_region.mean()
        R = self._int_region.span()
        self.error = I - R * Z
        self.var = Ivar + R * Zvar
        return self.error, self.var
    
    def plot_annotations(self, ax: Axes):
        pass # TODO implement this nicely

    def _serialize_state(self) -> Dict[str, Any]:
        config = super()._serialize_state()
        config.update({
            'zero_region': format_reference(self._zero_region) if self._zero_region else None,
            'int_region': format_reference(self._int_region) if self._int_region else None,
            'plot_color': self._plot_color,
        })
        return config
    
    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        if 'zero_region' in state and state['zero_region'] is not None:
            self._zero_region = DeferredReference(state['zero_region'])
        if 'int_region' in state and state['int_region'] is not None:
            self._int_region = DeferredReference(state['int_region'])
        if 'plot_color' in state:
            self._plot_color = state['plot_color']
        
        self._resolve_references()
        self._post_init()

    def _resolve_references(self):
        if isinstance(self._zero_region, DeferredReference):
            self._zero_region = self._zero_region.unwrap()
        if isinstance(self._int_region, DeferredReference):
            self._int_region = self._int_region.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfIntegral':
        registry_id = config.pop('registry_id')
        zero_region = config.pop('zero_region', None)
        if zero_region is not None:
            zero_region = DeferredReference(zero_region)
        int_region = config.pop('int_region', None)
        if int_region is not None:
            int_region = DeferredReference(int_region)
        
        return cls(
            zero_region=zero_region,
            int_region=int_region,
            registry_id=registry_id,
            skip_post_init=True,
        )
    
    def __del__(self):
        if hasattr(self, '_zero_region') and self._zero_region:
            if self in self._zero_region._averager._supported_balances:
                self._zero_region._averager._supported_balances.remove(self)
        if hasattr(self, '_int_region') and self._int_region:
            if self in self._int_region._averager._supported_balances:
                self._int_region._averager._supported_balances.remove(self)
    
"""wfPostProcess Classes; each is a post-processing step we can take after the curve is acquired"""

@register_device_class("wfCompensateHighPass")
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
        zero_region: wfAveragerView | DeferredReference,
        correction_region: wfAveragerView | DeferredReference,
        tau: float,
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(registry_id=registry_id, logger=logger)

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

    def _deserialize_state(self, state: Dict[str, Any]):
        if 'zero_region' in state and state['zero_region'] is not None:
            self._zero = DeferredReference(state['zero_region'])
        if 'correction_region' in state and state['correction_region'] is not None:
            self._correction = DeferredReference(state['correction_region'])
        if 'tau' in state:
            self._tau = state['tau']
        
        self._resolve_references()

    def _resolve_references(self):
        if isinstance(self._zero, DeferredReference):
            self._zero = self._zero.unwrap()
        if isinstance(self._correction, DeferredReference):
            self._correction = self._correction.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'wfCompensateHighPass':
        registry_id = config.pop('registry_id')
        zero_region = config.pop('zero_region', None)
        if zero_region is not None:
            zero_region = DeferredReference(zero_region)
        correction_region = config.pop('correction_region', None)
        if correction_region is not None:
            correction_region = DeferredReference(correction_region)
        tau = config.pop('tau', 1.0)
        
        return cls(
            zero_region=zero_region,
            correction_region=correction_region,
            tau=tau,
            registry_id=registry_id,
        )