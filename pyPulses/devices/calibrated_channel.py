from .abstract_device import abstractDevice
from .attenuator import Attenuator
from .channel_adapter import BoolChannel, ScalarChannel
from .calibration import CalibrationModel
from .registry import (
    register_device_class, 
    format_reference, 
    resolve_reference, 
    DeferredReference,
    DeviceRegistry,
)

import numpy as np
from logging import Logger
from typing import Any, Dict, List, Tuple
    
"""
Calibrated Channel
"""

class AttenuatorMode:
    """How aggressively to adjust the attenuator."""
    PASSIVE = 'passive'
    AGRESSIVE = 'aggressive'

class AttenuatorPreference:
    """Which attenuation to prefer when multiple settings work."""
    CLOSEST = 'closest'
    SMALLEST = 'smallest'
    LARGEST = 'largest'

@register_device_class("CalibratedChannel")
class CalibratedChannel(abstractDevice):
    """
    A channel with calibration, optional attenuation, and crosstalk compensation.
    """

    def __init__(self,
        hardware: ScalarChannel | DeferredReference,
        calibration: CalibrationModel | None,
        control_min: float,
        control_max: float,
        output_min: float,
        output_max: float,
        attenuator: Attenuator | DeferredReference | None = None,
        attenuator_active: bool = True,
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):

        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

        self._hw = hardware
        self._calibration = calibration
        self._attenuator = attenuator
        self._attenuator_active = attenuator_active

        # control and output rails
        self._c_min = control_min
        self._c_max = control_max
        self._o_min = output_min
        self._o_max = output_max

        # attenuator settings
        self._attenuator_mode = AttenuatorMode.PASSIVE
        self._attenuator_preference = AttenuatorPreference.CLOSEST
        self._attenuator_locked = False

        # crosstalk compensation
        self._crosstalk: List[Tuple[str, float]] = []

        # cached values
        self._cached_control: float | None = None

    @property
    def output_min(self) -> float:
        """Minimum achievable output (at current attenuation)"""
        atten = self._get_attenuation(lazy=True)
        return self._o_min * atten
    
    @property
    def output_max(self) -> float:
        """Maximum achievable output (at current attenuation)"""
        atten = self._get_attenuation(lazy=True)
        return self._o_max * atten
    
    def set_calibration(self, calibration: CalibrationModel):
        self._calibration = calibration
        self._o_min, self._o_max = calibration.output_bounds(
            self._c_min, self._c_max
        )

    def add_crosstalk(self, other: ScalarChannel | str, coeff: float):
        """Register crosstalk from another channel."""
        if isinstance(other, str):
            ref = other
        else:
            ref = format_reference(other)
        self._crosstalk.append((ref, coeff))

    def clear_crosstalk(self) -> None:
        """Remove all crosstalk registrations."""

        self._crosstalk.clear()

    """
    Core Output Interface
    """

    def get_output(self, lazy: bool = True) -> float:
        """
        Get the current output, including crosstalk correction.

        Parameters
        ----------
        lazy : bool
            If true, we use cached control values when available
        """

        c = self._get_control(lazy=lazy)
        o = self._calibration.forward(c) * self._get_attenuation()
        for other, coeff in self._crosstalk:
            o += coeff * resolve_reference(other).get_output()

        return o
    
    def set_output(self, o: float):
        """
        Set the desired output. 
        Adjusts the attenuator (if available and not locked) and control to 
        achieve the requested amplitude.

        Note: Crosstalk is NOT compensated when setting (by design, to avoid
        complexity and additional hardware queries).
        """

        if np.isnan(o):
            raise ValueError(f"Cannot set amplitude to NaN")
        
        if o == 0:
            self._hw.set_output(0.0)
            self._cached_control = (0.0)
            if self._attenuator and self._attenuator_active \
                and not self._attenuator_locked:
                self._attenuator.set_higher(0.0)
            return
        
        # Check polarity
        if (self._o_min > 0 and o < 0) or (self._o_max < 0 and o > 0):
            raise ValueError(
                f"Cannot achieve output value {o} with unipolar output "
                f"[{self._o_min}, {self._o_max}]"
            )
        
        # Determine attenuation
        if self._attenuator is not None:
            atten = self._pick_attenuation(o)
            target_raw = o / atten
        else:
            atten = 1.0
            target_raw = o

        # Clamp to achievable range
        if target_raw < self._o_min:
            self.warn(f"Requested output {o} is below allowed range {self._o_min}; Clamping...")
            target_raw = self._o_min
        elif target_raw > self._o_max:
            self.warn(f"Requested output {o} is above allowed range {self._o_max}; Clamping...")
            target_raw = self._o_max

        # Compute required control setting
        control = self._calibration.inverse(target_raw, self._c_min, self._c_max)

        # Set hardware
        self._hw.set_output(control)
        self._cached_control = control

    def __call__(self, value: float | None = None) -> float | None:
        if value is None:
            return self.get_output()
        return self.set_output(value)

    """
    Attenuator Handling
    """

    def _get_attenuation(self, lazy: bool = True) -> float:
        """Get current attenuation factor."""
        
        if self._attenuator is None:
            return 1.0
        return self._attenuator.get()
    
    def _pick_attenuation(self, target_output: float) -> float:
        """Choose appropriate attenuation for target output."""

        if not self._attenuator_active or self._attenuator_locked:
            return self._get_attenuation(lazy=True)
        
        current_atten = self._get_attenuation(lazy=True)

        # In passive mode, only change if we can't reach target
        if self._attenuator_mode == AttenuatorMode.PASSIVE:
            achievable_min = self._o_min * current_atten
            achievable_max = self._o_max * current_atten
            if achievable_min <= target_output <= achievable_max:
                return current_atten
            
        # Need to pick a new attenuation
        abs_target = abs(target_output)

        if self._attenuator_preference == AttenuatorPreference.SMALLEST:
            # Least attenuation that can still reach target
            max_raw = max(abs(self._o_min), abs(self._o_max))
            ideal = abs_target / max_raw
            actual = self._attenuator.set_lower(ideal)
        
        elif self._attenuator_preference == AttenuatorPreference.LARGEST:
            # Most attenuation that can still reach target
            min_raw = min(abs(self._o_min), abs(self._o_max))
            if min_raw > 0:
                ideal = abs_target / min_raw
            else:
                ideal = abs_target / max(abs(self._o_min), abs(self._o_max))
            actual = self._attenuator.set_higher(ideal)
        
        else:  # CLOSEST
            # Geometric mean of output range
            geo_mean = np.sqrt(abs(self._o_min * self._o_max))
            if geo_mean > 0:
                ideal = abs_target / geo_mean
            else:
                ideal = abs_target / max(abs(self._o_min), abs(self._o_max))
            actual = self._attenuator.set_closest(ideal)
        
        return actual
    
    """
    Control Access
    """

    def _get_control(self, lazy: bool = True) -> float:
        """Get current control."""
        if lazy and self._cached_control is not None:
            return self._cached_control
        
        self._cached_control = self._hw.get_output()
        return self._cached_control
    
    def invalidate_cache(self) -> None:
        """Force next query to read from hardware."""
        self._cached_control = None

    """
    Serialization support
    """

    def _serialize_state(self) -> dict:
        config = {
            'hardware': format_reference(self._hw),
            'attenuator': format_reference(self._attenuator),
            'calibration': self._calibration.to_dict(),
            'crosstalk': self._crosstalk,
            'c_min': self._c_min,
            'c_max': self._c_max,
            'o_min': self._o_min,
            'o_max': self._o_max,
            'attenuator_active': self._attenuator_active,
            'attenuator_mode': self._attenuator_mode,
            'attenuator_preference': self._attenuator_preference,
            'attenuator_locked': self._attenuator_locked,
        }
        
        return config
    
    def _deserialize_state(self, config: Dict[str, Any]):
        
        if 'calibration' in config:
            self._calibration = CalibrationModel.from_config(config['calibration'])
        if 'hardware' in config:
            self._hw = DeferredReference(config['hardware'])
        if 'attenuator' in config:
            self._attenuator = config['attenuator']
            if self._attenuator is not None:
                self._attenuator = DeferredReference(self._attenuator)
        if 'crosstalk' in config:
            self._crosstalk = config['crosstalk']

        if 'c_min' in config:
            self._c_min = config['c_min']
        if 'c_max' in config:
            self._c_max = config['c_max']
        if 'o_min' in config:
            self._o_min = config['o_min']
        if 'o_max' in config:
            self._o_max = config['o_max']
        if 'attenuator_active' in config:
            self._attenuator_active = config['attenuator_active']
        if 'attenuator_mode' in config:
            self._attenuator_mode = config['attenuator_mode']
        if 'attenuator_preference' in config:
            self._attenuator_preference = config['attenuator_preference']
        if 'attenuator_locked' in config:
            self._attenuator_locked = config['attenuator_locked']

        self._cached_control = None

        self._resolve_references()

    def _resolve_references(self):
        if isinstance(self._hw, DeferredReference):
            self._hw = self._hw.unwrap()
        if isinstance(self._attenuator, DeferredReference):
            self._attenuator = self._attenuator.unwrap()

    @classmethod
    def from_config(cls, config) -> 'CalibratedChannel':

        hardware = config.pop("hardware")
        if hardware is not None:
            hardware = DeferredReference(hardware)
        calibration = CalibrationModel.from_config(config.pop('calibration'))
        control_min = config.pop('c_min')
        control_max = config.pop('c_max')
        output_min = config.pop('o_min')
        output_max = config.pop('o_max')
        attenuator = config.pop('attenuator')
        if attenuator is not None:
            attenuator = DeferredReference(attenuator)
        attenuator_active = config.pop('attenuator_active')
        registry_id = config.pop('registry_id')

        instance = cls(
            hardware=hardware,
            calibration=calibration,
            control_min=control_min,
            control_max=control_max,
            output_min=output_min,
            output_max=output_max,
            attenuator=attenuator,
            attenuator_active=attenuator_active,
            registry_id=registry_id,
        )

        return instance