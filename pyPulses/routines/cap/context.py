from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .balance import CapBalanceResult

from .cap_filter import CapFilter, CapExtrapolator

from ...devices.channel_adapter import ScalarChannel, LockInChannel, CommandChannel
from ...devices.registry import resolve_reference, format_reference

from dataclasses import dataclass
from typing import Any, Callable, Dict
import numpy as np
import logging
import json

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class CapContext():
    """
    Unified context for penetration capacitance measurements.

    Supports two measurement modes:

    Off-balance (`cap_measure`)
        The bridge is held at the most recent balance point. The lock-in reading
        is converted to (Cex, Closs) using the cached complex gain A from the
        Kalman filter. The filter is not updated.

    On-balance (`cap_balance`)
        The bridge is iteratively rebalanced at each phase-space point. The
        Kalman filter is updated with each (dL, dv) observation, and the
        extrapolator history is used to see the next balance.

    Initialization (`cap_initialize_three_point` or `cap_initialize_two_point`)
    must be called before either mode to establish the initial balance point,
    gain estimate, and covariance.

    Hardware
    --------
    Vstd : ScalarChannel
        Amplitude channel for the reference
    Theta : ScalarChannel
        Phase channel for the reference
    lockin_call : LockInChannel
    Vstd_range: float
        Maximum allowed amplitude on Vstd
    Vex : float, default=1.0
        Applied excitation on the sample side (constant throughout measurement)
    Cstd: float, default=1.0
        Value of the reference capacitor
    sensitivity_channel : ScalarChannel
        If provided, the balance procedure will attempt to set the input
        sensitivity of the lock-in on each iteration to get better precision

    """

    # Hardware
    Vstd: ScalarChannel
    Theta: ScalarChannel
    lockin_call: LockInChannel
    Vstd_range: float
    sensitivity_channel: ScalarChannel | None = None
    sensitivity_multiplier: float = 4.0
    Vex: float = 1.0
    Cstd: float = 1.0

    # Hardware Resolution Limits
    amp_resolution: float = 0.5 * 1/4096 # tailored to AD9854 (unitless)
    phase_resolution: float = 0.5 * 360/16384 # tailored to AD9854 (degrees)

    # Balance Loop Parameters
    max_tries: int = 10
    errmult: float = 0.075
    erroff: float = 0.0
    errt_alpha: float = 0.95
    settle_time: float = 0.0

    # Extrapolator Parameters
    extrap_support: int = 7
    extrap_order: int = 2

    # Callbacks
    iteration_callback: Callable | None = None

    # Logging
    logger: logging.Logger | None = None

    def __post_init__(self):
        self.cap_filter: CapFilter | None = None
        self.extrapolator = CapExtrapolator(
            support = self.extrap_support,
            order = self.extrap_order,
            Vstd_range = self.Vstd_range,
        )

        self._previous_result: CapBalanceResult | None = None
        self._Vstd_now: float | None = None

    """Logging"""

    def log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)

    """Hardware Accessors"""

    def set_Vstd(self, r: float, theta_deg: float):
        """
        Set Vstd magnitude and phase, clamping magnitude to Vstd_range.
        Returns the actual (r, theta) set on hardware.
        """
        r = float(np.clip(r, 0.0, self.Vstd_range))
        self.Vstd(r)
        self.Theta(theta_deg % 360)
        r_set = self.Vstd()
        theta_set = self.Theta()
        return float(r_set), float(theta_set)

    def get_Vstd_complex(self) -> complex:
        """Return current Vstd as a complex number (X + iY)."""
        r = float(self.Vstd())
        th = float(self.Theta())
        rad = np.deg2rad(th)
        self._Vstd_now = complex(r * np.cos(rad), r * np.sin(rad))
        return self._Vstd_now

    def set_Vstd_complex(self, V: complex) -> complex:
        """
        Set Vstd from a complex value. Clamps magnitude to Vstd_range.
        Returns the complex Vstd actually set on hardware.
        """
        r = float(np.clip(abs(V), 0.0, self.Vstd_range))
        th = float(np.degrees(np.angle(V)) % 360)
        self.Vstd(r)
        self.Theta(th)
        return self.get_Vstd_complex()
    
    """Filter management"""
    
    def add_filter(self, f: CapFilter):
        self.cap_filter = f
        f.ctx = self

    """Serialization"""

    @classmethod
    def from_json(cls, path: str) -> 'CapContext':
        with open(path, 'r') as f:
            conf = json.load(f)
        return cls.from_config(conf)
 
    @classmethod
    def from_config(cls, conf: Dict[str, Any]) -> 'CapContext':
        device_refs = conf.get('device_refs', {})
 
        Vstd = resolve_reference(device_refs['Vstd'])
        Theta = resolve_reference(device_refs['Theta'])
        lockin_call = resolve_reference(device_refs['lockin_call'])
        sensitivity_channel = resolve_reference(device_refs['sensitivity_channel']) \
            if device_refs.get('sensitivity_channel') else None
 
        kwargs = conf.get('CapContext_kwargs', {})
        obj = cls(
            Vstd = Vstd,
            Theta = Theta,
            lockin_call = lockin_call,
            sensitivity_channel = sensitivity_channel,
            **kwargs,
        )
 
        obj._deserialize_state(conf)
        return obj
 
    def load_state_json(self, path: str):
        with open(path, 'r') as f:
            state = json.load(f)
        self._deserialize_state(state)
 
    def save_state_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._serialize_state(), f, indent=2, cls=_NumpyEncoder)
 
    def _serialize_state(self) -> dict:
        device_refs: dict = {
            'Vstd':        format_reference(self.Vstd),
            'Theta':       format_reference(self.Theta),
            'lockin_call': format_reference(self.lockin_call),
        }
        if self.sensitivity_channel is not None:
            device_refs['sensitivity_channel'] = format_reference(self.sensitivity_channel)
 
        result: dict = {
            'CapContext_kwargs': dict(
                Vstd_range = float(self.Vstd_range),
                sensitivity_multiplier = self.sensitivity_multiplier,
                Vex = float(self.Vex),
                Cstd = float(self.Cstd),
                amp_resolution = float(self.amp_resolution),
                phase_resolution = float(self.phase_resolution),
                max_tries = int(self.max_tries),
                errmult = float(self.errmult),
                erroff = float(self.erroff),
                errt_alpha = float(self.errt_alpha),
                settle_time = float(self.settle_time),
                extrap_support = int(self.extrap_support),
                extrap_order = int(self.extrap_order),
            ),
            'device_refs': device_refs,
        }
 
        if self.cap_filter is not None:
            result.update(self.cap_filter._serialize_state())
        result.update(self.extrapolator._serialize_state())
 
        return result
 
    def _deserialize_state(self, state: dict):
        for key, value in state.get('CapContext_kwargs', {}).items():
            if hasattr(self, key):
                setattr(self, key, value)
 
        if state.get('CapFilter_kwargs') is not None:
            if self.cap_filter is None:
                self.add_filter(CapFilter(self))
            self.cap_filter._deserialize_state(state['CapFilter_kwargs'])
 
        if state.get('CapExtrapolator_kwargs') is not None:
            self.extrapolator._deserialize_state(state['CapExtrapolator_kwargs'])