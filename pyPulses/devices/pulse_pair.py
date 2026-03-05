from .abstract_device import abstractDevice
from .dtg_comp_pair import dtgCompPair
from typing import Any, Callable
import numpy as np

class pulsePair(abstractDevice):
    def __init__(self,
        relay: dtgCompPair, 
        X: Callable[[float], Any] | Callable[[], float],
        Y: Callable[[float], Any] | Callable[[], float],
        logical_low: float = -0.5,
        logical_high: float = 2.5,
        attenuation: float = 0.0
    ):
        super().__init__()
        self._relay = relay
        self._X = X
        self._Y = Y
        self._attenuation = attenuation

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
        self._post_init()

    def _post_init(self):
        self._relay._post_init()
        self._relay.Xlow(self.logical_low)
        self._relay.Ylow(self.logical_low)
        self._relay.Xhigh(self.logical_high)
        self._relay.Yhigh(self.logical_high)
        self._pol = self._relay.xpolarity()
        self._relay.ypolarity(not self._pol)

    def enable(self, on: bool | None = None) -> bool | None:
        return self._relay.enable(on)

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
            return eta * self._X() * self._attenFrac()
        if eta * v < 0:
            self._switch_polarity()

        self._X(abs(v) / self._attenFrac())

    def Y(self, v: float | None = None) -> float | None:
        """
        In terms of polarity, Y is subordinate to X, so we do not allow it to
        switch polarity on its own. Instead, if Y is asked to switch polarity,
        we set it to 0 instead. The only exception is if X is already 0, in
        which case, we allow Y to take the lead in switching polarity.
        """
        eta = -1 if self._pol else 1
        if v is None:
            return eta * self._Y() * self._attenFrac()
        if eta * v < 0:
            if self._X() != 0.0:
                self._Y(0.0)
            else:
                self._switch_polarity()
        self._Y(abs(v) / self._attenFrac())

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

    def _serialize_state(self) -> dict:
        return {
            'levels': [self.X(), self.Y()],
            'timing': {'T0': self.T0(), 'dT0': self.dT0(),
                       'T1': self.T1(), 'dT1': self.dT1()},
            'xtuning': self._xtuning,
            'ytuning': self._ytuning,
        }

    def _deserialize_state(self, state: dict):
        self._xtuning = state['xtuning']
        self._ytuning = state['ytuning']
        self.T0(state['timing']['T0'])
        self.dT0(state['timing']['dT0'])
        self.T1(state['timing']['T1'])
        self.dT1(state['timing']['dT1'])
        self.X(state['levels'][0])
        self.Y(state['levels'][1])

    def _attenFrac(self) -> float:
        if self._attenuation > 0.0:
            raise RuntimeError("Attenuator should not be positive!")
        return 10**(self._attenuation / 20.0)

    # TODO Come up with a robust way of tuning this.
    # def set_tuning(self, pol: bool, tuning_parm: str, offset: float):
    #     self._tuning[pol][tuning_parm] = offset

    # def tune_from_curve(self, pol: bool, tuning_parm: str, wf: wfAverager, ta: float, tb: float):
    #     pass
    