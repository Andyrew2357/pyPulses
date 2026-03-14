from .abstract_device import abstractDevice
from .channel_adapter import ScalarChannel, CompPair, ScalarChannelAdapter, BoolChannelAdapter
from .registry import (
    register_device_class, 
    format_reference,
    DeferredReference,
    DeviceRegistry,
)

from logging import Logger
from typing import Any, Callable, Dict, List

@register_device_class("pulsePair")
class pulsePair(abstractDevice):
    def __init__(self,
        relay: CompPair | DeferredReference, 
        X: ScalarChannel | DeferredReference,
        Y: ScalarChannel | DeferredReference,
        dependent_pairs: List[pulsePair | DeferredReference] | None = None,
        logical_low: float = -0.5,
        logical_high: float = 2.5,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_post_init: bool = False,
    ):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

        self._relay = relay
        self._X = X
        self._Y = Y
        if dependent_pairs is not None:
            self._dependent_pairs = dependent_pairs
        else:
            self._dependent_pairs = []

        # relative jump locations compared to internal DTG timing
        self._pol: bool | None = None
        self._xtuning = {
            True:  {'dT0': 0.0, 'dT1': 0.0},
            False: {'dT0': 0.0, 'dT1': 0.0},
        }
        self._ytuning = {
            True:  {'dT0': 0.0, 'dT1': 0.0},
            False: {'dT0': 0.0, 'dT1': 0.0},
        }
        self.logical_low = logical_low
        self.logical_high = logical_high
        if not skip_post_init:
            self._post_init()

    def _post_init(self):
        if hasattr(self._relay, '_post_init'):
            self._relay._post_init()
        self._relay.Xlow(self.logical_low)
        self._relay.Ylow(self.logical_low)
        self._relay.Xhigh(self.logical_high)
        self._relay.Yhigh(self.logical_high)
        self._pol = self._relay.xpolarity()
        self._relay.ypolarity(not self._pol)

    def add_dependent_pair(self, pair: 'pulsePair'):
        """Add a pulsePair that depends on this one's polarity."""
        self._dependent_pairs.append(pair)

    def get_polarity(self) -> bool:
        return self._pol

    def enable(self, on: bool | None = None) -> bool | None:
        return self._relay.enable(on)
    
    def recalibrate_amplitudes(self):
        """Recalibrate X and Y channels if they support it."""
        if hasattr(self._X, 'recalibrate'):
            self._X.recalibrate()
        if hasattr(self._Y, 'recalibrate'):
            self._Y.recalibrate()

    def T0(self, t: float | None = None) -> float | None:
        dT0X = self._xtuning[self._pol]['dT0']
        if t is None:
            return self._relay.ldelay() + dT0X
        self._relay.ldelay(t - dT0X)

    def dT0(self, dt: float | None = None) -> float | None:
        dT0X = self._xtuning[self._pol]['dT0']
        dT0Y = self._ytuning[self._pol]['dT0']
        if dt is None:
            return self._relay.ldoff() + dT0Y - dT0X
        self._relay.ldoff(dt + dT0X - dT0Y)

    def T1(self, t: float | None = None) -> float | None:
        dT1X = self._xtuning[self._pol]['dT1']
        if t is None:
            return self._relay.tdelay() + dT1X
        self._relay.tdelay(t - dT1X)

    def dT1(self, dt: float | None = None) -> float | None:
        dT1X = self._xtuning[self._pol]['dT1']
        dT1Y = self._ytuning[self._pol]['dT1']
        if dt is None:
            return self._relay.tdoff() + dT1Y - dT1X
        self._relay.tdoff(dt + dT1X - dT1Y)

    def W(self, w: float | None = None) -> float | None:
        dT0X = self._xtuning[self._pol]['dT0']
        dT1X = self._xtuning[self._pol]['dT1']
        if w is None:
            return self._relay.width() + dT1X - dT0X
        self._relay.width(w + dT0X - dT1X)

    def X(self, v: float | None = None) -> float | None:
        eta = 1 if self._pol else -1
        if v is None:
            return eta * self._X()
        if eta * v < 0:
            self._switch_polarity()
        self._X(abs(v))

    def Y(self, v: float | None = None) -> float | None:
        """
        In terms of polarity, Y is subordinate to X, so we do not allow it to
        switch polarity on its own. Instead, if Y is asked to switch polarity,
        we set it to 0 instead. The only exception is if X is already 0, in
        which case, we allow Y to take the lead in switching polarity.
        """
        eta = -1 if self._pol else 1
        if v is None:
            return eta * self._Y()
        if eta * v < 0:
            if self._X() != 0.0:
                self._Y(0.0)
                return
            else:
                self._switch_polarity()
        self._Y(abs(v))

    def _switch_polarity(self):
        # Save the abstract timing settings
        T0 = self.T0()
        T1 = self.T1()
        dT0 = self.dT0()
        dT1 = self.dT1()

        # Invert the polarity
        self._pol = not self._pol
        self._relay.xpolarity(self._pol)
        self._relay.ypolarity(not self._pol)

        # Maintain the abstract timing settings
        self.T0(T0)
        self.T1(T1)
        self.dT0(dT0)
        self.dT1(dT1)

        # Notify dependent pairs to recalibrate their amplitudes
        for pair in self._dependent_pairs:
            # Handle DeferredReference if not yet resolved
            if isinstance(pair, DeferredReference):
                pair = pair.unwrap()
            if hasattr(pair, 'recalibrate_amplitudes'):
                pair.recalibrate_amplitudes()

    def _serialize_state(self) -> Dict[str, Any]:
        # format_reference can raise; capture per-item so one bad ref
        # doesn't prevent serializing the rest of the state.
        try:
            Xref = format_reference(self._X)
        except Exception:
            Xref = None
        try:
            Yref = format_reference(self._Y)
        except Exception:
            Yref = None
        try:
            Rref = format_reference(self._relay)
        except Exception:
            Rref = None

        # Serialize dependent pairs as references
        dependent_refs = []
        for pair in self._dependent_pairs:
            try:
                dependent_refs.append(format_reference(pair))
            except Exception:
                pass  # Skip pairs that can't be serialized

        def _safe_call(fn):
            try:
                return fn()
            except Exception:
                return None
        levels = [_safe_call(self.X), _safe_call(self.Y)]

        config = {
            'levels': levels,
            'timing': {'T0': self.T0(), 'dT0': self.dT0(),
                       'T1': self.T1(), 'dT1': self.dT1()},
            'xtuning': {str(k): v for k, v in self._xtuning.items()},
            'ytuning': {str(k): v for k, v in self._ytuning.items()},
            'CHX': Xref,
            'CHY': Yref,
            'relay': Rref,
            'dependent_pairs': dependent_refs,
            'logical_low': float(self.logical_low),
            'logical_high': float(self.logical_high),
        }
        return config

    def _deserialize_state(self, state: Dict[str, Any]):

        if 'relay' in state:
            self._relay = DeferredReference(state['relay'])
        if 'CHX' in state:
            self._X = DeferredReference(state['CHX'])
        if 'CHY' in state:
            self._Y = DeferredReference(state['CHY'])
        if 'dependent_pairs' in state:
            self._dependent_pairs = [
                DeferredReference(ref) for ref in state['dependent_pairs']
            ]
        self._resolve_references()

        if 'logical_low' in state:
            self.logical_low = state['logical_low']
        if 'logical_high' in state:
            self.logical_high = state['logical_high']

        if 'xtuning' in state:
            self._xtuning = {k == 'True': v for k, v in state['xtuning'].items()}
        if 'ytuning' in state:
            self._ytuning = {k == 'True': v for k, v in state['ytuning'].items()}

        self._post_init()
        if 'timing' in state:
            self.T0(state['timing']['T0'])
            self.dT0(state['timing']['dT0'])
            self.T1(state['timing']['T1'])
            self.dT1(state['timing']['dT1'])
        if 'levels' in state:
            self.X(state['levels'][0])
            self.Y(state['levels'][1])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'pulsePair':
        relay = config.pop('relay')
        if relay is not None:
            relay = DeferredReference(relay)
        X = config.pop('CHX')
        if X is not None:
            X = DeferredReference(X)
        Y = config.pop('CHY')
        if Y is not None:
            Y = DeferredReference(Y)

        dependent_pairs = config.pop('dependent_pairs', [])
        dependent_pairs = [DeferredReference(ref) for ref in dependent_pairs]
        
        logical_low = config.pop('logical_low')
        logical_high = config.pop('logical_high')
        registry_id = config.pop('registry_id')

        instance = cls(
            relay=relay,
            X=X,
            Y=Y,
            dependent_pairs=dependent_pairs,
            logical_low=logical_low,
            logical_high=logical_high,
            registry_id=registry_id,
            skip_post_init=True # We only do a post_init after _deserialize_state
        )
        return instance

    def _resolve_references(self):
        if isinstance(self._relay, DeferredReference):
            self._relay = self._relay.unwrap()
        if isinstance(self._X, DeferredReference):
            self._X = self._X.unwrap()
        if isinstance(self._Y, DeferredReference):
            self._Y = self._Y.unwrap()

        # Resolve dependent pair references
        resolved = []
        for pair in self._dependent_pairs:
            if isinstance(pair, DeferredReference):
                resolved.append(pair.unwrap())
            else:
                resolved.append(pair)
        self._dependent_pairs = resolved

    def resolve(self, accessor: str) -> 'pulsePair_channel':
        if accessor == 'X':
            return pulsePair_channel(self, 'X', self.X)
        if accessor == 'Y':
            return pulsePair_channel(self, 'Y', self.Y)
        if accessor == 'T0':
            return pulsePair_channel(self, 'T0', self.T0)
        if accessor == 'T1':
            return pulsePair_channel(self, 'T1', self.T1)
        if accessor == 'W':
            return pulsePair_channel(self, 'W', self.W)
        if accessor == 'pol':
            return BoolChannelAdapter(self, 'pol', self.get_polarity)
        raise ValueError(f"Unknown accessor: {accessor}")

class pulsePair_channel(ScalarChannelAdapter):
    def __init__(self, parent: pulsePair, accessor: str, f: Callable):
        super().__init__(parent, accessor)
        self._func = f

    def get_output(self) -> float:
        return self._func
    
    def set_output(self, value):
        self._func(value)