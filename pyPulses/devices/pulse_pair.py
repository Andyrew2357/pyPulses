from .abstract_device import abstractDevice
from .dtg_diff_pair import dtgDifferentialPair
from .wfatd import wfAverager
from typing import Any, Callable
import numpy as np

# TODO NEED TO MODIFY THIS BECAUSE OF WEIRDNESS DUE TO LHOLD AND THOLD TYPE STUFF
class pulsePair(abstractDevice):
    def __init__(self,
        timing: dtgDifferentialPair, 
        X: Callable[[float], Any] | Callable[[], float],
        Y: Callable[[float], Any] | Callable[[], float],
        logical_low: float = 0.0,
        logical_high: float = 2.0,
    ):
        super().__init__()
        self._timing = timing
        self._X = X
        self._Y = Y

        # relative jump locations compared to internal DTG timing
        # Generically, because the respective behavior of rises and falls should
        # be consistent, we expect dT0X = dT1Y and dT1X = dT0Y. We also expect
        # that under a change in polarity dT0X <-> dT0Y and dT1X <-> dT1Y
        self._tuning = {
            True:  {'dT0X': 0.0, 'dT0Y': 0.0, 'dT1X': 0.0, 'dT1Y': 0.0},
            False: {'dT0X': 0.0, 'dT0Y': 0.0, 'dT1X': 0.0, 'dT1Y': 0.0},
        }

        self.logical_low = logical_low
        self.logical_high = logical_high

    def enable(self, on: bool | None = None) -> bool | None:
        if on is None:
            return self._timing.enable
        if on:
            self._timing.Xlow = self.logical_low
            self._timing.Xhigh = self.logical_high
            self._timing.Ylow = self.logical_low
            self._timing.Yhigh = self.logical_high
        self._timing.enable(on)

    def T0(self, t: float | None = None) -> float | None:
        dT0X = self._tuning[self._timing.polarity]['dT0X']
        if t is None:
            return self._timing.ldelay + dT0X
        self._timing.ldelay = t - dT0X

    def dT0(self, dt: float | None = None) -> float | None:
        dT0X = self._tuning[self._timing.polarity]['dT0X']
        dT0Y = self._tuning[self._timing.polarity]['dT0Y']
        if dt is None:
            return self._timing.ldoff - dT0X + dT0Y
        self._timing.ldoff = dt + dT0X - dT0Y

    def T1(self, t: float | None = None) -> float | None:
        dT1X = self._tuning[self._timing.polarity]['dT1X']
        if t is None:
            return self._timing.tdelay + dT1X
        self._timing.tdelay = t - dT1X

    def dT1(self, dt: float | None = None) -> float | None:
        dT1X = self._tuning[self._timing.polarity]['dT1X']
        dT1Y = self._tuning[self._timing.polarity]['dT1Y']
        if dt is None:
            return self._timing.tdoff - dT1X + dT1Y
        self._timing.tdoff = dt + dT1X - dT1Y

    def W(self, w: float | None = None) -> float | None:
        if w is None:
            return self.T1() - self.T0()
        self.T1(self.T0() + w)

    def X(self, v: float | None = None) -> float | None:
        pol = self._timing.polarity
        eta = 1 if pol else -1
        if v is None:
            return eta * self._X()
        
        if eta * v < 0:
            self._switch_polarity()
        self._X(abs(v))

    def Y(self, v: float | None = None) -> float | None:
        pol = self._timing.polarity
        eta = -1 if pol else 1
        if v is None:
            return eta * self._Y()
        
        if eta * v < 0:
            self._switch_polarity()
        self._Y(abs(v))

    def _switch_polarity(self):
        # Get the abstract timing settings
        T0 = self.T0()
        dT0 = self.dT0()
        T1 = self.T1()
        dT1 = self.dT1()

        # Invert the polarity
        self._timing.polarity = not self._timing.polarity

        # Set up all the timings to reflect the new polarity
        self.T0(T0)
        self.dT0(dT0)
        self.T1(T1)
        self.dT1(dT1)

    def _serialize_state(self) -> dict:
        return {
            'levels': [self.X(), self.Y()],
            'timing': {'T0': self.T0(), 'dT0': self.dT0(),
                       'T1': self.T1(), 'dT1': self.dT1()},
            'tuning': self._tuning
        }
    
    def _deserialize_state(self, state: dict):
        self._tuning = state['tuning']
        self.T0(state['timing']['T0'])
        self.dT0(state['timing']['dT0'])
        self.T1(state['timing']['T1'])
        self.dT1(state['timing']['dT1'])
        self.X(state['levels'][0])
        self.Y(state['levels'][1])

    def set_tuning(self, pol: bool, tuning_parm: str, offset: float):
        self._tuning[pol][tuning_parm] = offset

    def tune_from_curve(self, pol: bool, tuning_parm: str, wf: wfAverager, ta: float, tb: float):
        pass