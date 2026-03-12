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
        self._cached_attenuation: float | None = None

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
                self._cached_attenuation = self._attenuator.set_higher(0.0)
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
        
        if lazy and self._cached_attenuation is not None:
            return self._cached_attenuation
        
        self._cached_attenuation = self._attenuator.get()
        return self._cached_attenuation
    
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
        
        self._cached_attenuation = actual
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
        self._cached_attenuation = None

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
        self._cached_attenuation = None

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
    
@register_device_class("PolarityCalibratedChannel")
class PolarityCalibratedChannel(abstractDevice):
    """
    A channel whose calibration depends on an external polarity signal.
    
    Delegates to two CalibratedChannel instances (one per polarity) and
    selects between them based on the polarity source.
    
    This is useful for hardware like pulse shapers where the DC control
    always outputs positive values, but the effective output polarity
    is determined by relay timing.
    """
    
    def __init__(self,
        polarity_source: BoolChannel | DeferredReference,
        pos_channel: CalibratedChannel | DeferredReference,
        neg_channel: CalibratedChannel | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)
        
        self._polarity_source = polarity_source
        self._pos = pos_channel
        self._neg = neg_channel
    
    @property
    def polarity(self) -> bool:
        """Current polarity state."""
        return self._polarity_source()
    
    @property
    def _active(self) -> CalibratedChannel:
        """The currently active calibrated channel based on polarity."""
        return self._pos if self.polarity else self._neg
    
    @property
    def _inactive(self) -> CalibratedChannel:
        """The currently inactive calibrated channel."""
        return self._neg if self.polarity else self._pos
    
    def _sync_caches(self):
        """
        Synchronize cached values from active channel to inactive channel.
        
        Since both channels share the same hardware, their cached control
        and attenuation values must stay in sync.
        """
        self._inactive._cached_control = self._active._cached_control
        self._inactive._cached_attenuation = self._active._cached_attenuation
    
    # Forward ScalarChannel interface to active channel
    
    def get_output(self, lazy: bool = True) -> float:
        """
        Get output from active channel, with sign based on polarity.
        
        Positive polarity returns positive values, negative polarity
        returns negative values.
        """
        magnitude = self._active.get_output(lazy=lazy)
        self._sync_caches()
        return magnitude if self.polarity else -magnitude
    
    def set_output(self, o: float):
        """
        Set output on active channel.
        
        The polarity of the requested value should match the current
        hardware polarity. If not, the magnitude is used and a warning
        is issued (polarity switching should be done at a higher level).
        """
        if (o >= 0) != self.polarity:
            self.warn(
                f"Requested output {o} has opposite sign to current polarity "
                f"({'positive' if self.polarity else 'negative'}). "
                f"Using magnitude {abs(o)}."
            )
        self._active.set_output(abs(o))
        self._sync_caches()
    
    def __call__(self, v: float | None = None) -> float | None:
        if v is None:
            return self.get_output()
        self.set_output(v)
    
    # Expose properties from active channel
    
    @property
    def output_min(self) -> float:
        """Minimum achievable output (signed based on polarity)."""
        if self.polarity:
            return self._active.output_min
        else:
            return -self._active.output_max
    
    @property
    def output_max(self) -> float:
        """Maximum achievable output (signed based on polarity)."""
        if self.polarity:
            return self._active.output_max
        else:
            return -self._active.output_min
    
    def invalidate_cache(self) -> None:
        """Invalidate cache on both channels."""
        self._pos.invalidate_cache()
        self._neg.invalidate_cache()
    
    # Attenuator access
    
    @property
    def attenuator_locked(self) -> bool:
        return self._active._attenuator_locked
    
    @attenuator_locked.setter
    def attenuator_locked(self, value: bool):
        # Lock both to keep them in sync
        self._pos._attenuator_locked = value
        self._neg._attenuator_locked = value
    
    @property
    def attenuator_mode(self) -> AttenuatorMode:
        return self._active._attenuator_mode
    
    @attenuator_mode.setter
    def attenuator_mode(self, value: AttenuatorMode):
        self._pos._attenuator_mode = value
        self._neg._attenuator_mode = value
    
    @property
    def attenuator_preference(self) -> AttenuatorPreference:
        return self._active._attenuator_preference
    
    @attenuator_preference.setter
    def attenuator_preference(self, value: AttenuatorPreference):
        self._pos._attenuator_preference = value
        self._neg._attenuator_preference = value
    
    # Crosstalk management
    
    def add_crosstalk(self, 
        other: ScalarChannel | str, 
        coeff: float, 
        polarity: bool | None = None
    ):
        """
        Add crosstalk from another channel.
        
        Parameters
        ----------
        other : ScalarChannel | str
            The channel causing crosstalk, or a reference string.
        coeff : float
            Crosstalk coefficient.
        polarity : bool | None
            If None, add to both polarities. Otherwise add only to the
            specified polarity's channel.
        """
        if polarity is None:
            self._pos.add_crosstalk(other, coeff)
            self._neg.add_crosstalk(other, coeff)
        elif polarity:
            self._pos.add_crosstalk(other, coeff)
        else:
            self._neg.add_crosstalk(other, coeff)
    
    def clear_crosstalk(self, polarity: bool | None = None):
        """Clear crosstalk registrations."""
        if polarity is None:
            self._pos.clear_crosstalk()
            self._neg.clear_crosstalk()
        elif polarity:
            self._pos.clear_crosstalk()
        else:
            self._neg.clear_crosstalk()
    
    # Access to underlying channels for direct manipulation
    
    @property
    def pos_channel(self) -> CalibratedChannel:
        return self._pos
    
    @property
    def neg_channel(self) -> CalibratedChannel:
        return self._neg
    
    def get_channel(self, polarity: bool) -> CalibratedChannel:
        """Get the calibrated channel for a specific polarity."""
        return self._pos if polarity else self._neg
    
    # Calibration access
    
    def get_calibration(self, polarity: bool | None = None) -> CalibrationModel:
        """
        Get calibration for specified polarity, or active polarity if None.
        """
        if polarity is None:
            polarity = self.polarity
        return self._pos._calibration if polarity else self._neg._calibration
    
    def set_calibration(self, calibration: CalibrationModel, polarity: bool):
        """Set calibration for specified polarity."""
        channel = self._pos if polarity else self._neg
        channel._calibration = calibration
        # Recompute output bounds
        channel._o_min, channel._o_max = calibration.output_bounds(
            channel._c_min, channel._c_max
        )
    
    # Serialization
    
    def _serialize_state(self) -> Dict[str, Any]:
        return {
            'polarity_source': format_reference(self._polarity_source),
            'pos_channel': format_reference(self._pos),
            'neg_channel': format_reference(self._neg),
        }
    
    def _deserialize_state(self, state: Dict[str, Any]):
        if 'polarity_source' in state:
            self._polarity_source = DeferredReference(state['polarity_source'])
        if 'pos_channel' in state:
            self._pos = DeferredReference(state['pos_channel'])
        if 'neg_channel' in state:
            self._neg = DeferredReference(state['neg_channel'])
        
        self._resolve_references()
    
    def _resolve_references(self):
        if isinstance(self._polarity_source, DeferredReference):
            self._polarity_source = self._polarity_source.unwrap()
        if isinstance(self._pos, DeferredReference):
            self._pos = self._pos.unwrap()
        if isinstance(self._neg, DeferredReference):
            self._neg = self._neg.unwrap()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PolarityCalibratedChannel':
        registry_id = config.pop('registry_id')
        
        polarity_source = config.pop('polarity_source', None)
        if polarity_source is not None:
            polarity_source = DeferredReference(polarity_source)
        
        pos_channel = config.pop('pos_channel', None)
        if pos_channel is not None:
            pos_channel = DeferredReference(pos_channel)
        
        neg_channel = config.pop('neg_channel', None)
        if neg_channel is not None:
            neg_channel = DeferredReference(neg_channel)
        
        return cls(
            polarity_source=polarity_source,
            pos_channel=pos_channel,
            neg_channel=neg_channel,
            registry_id=registry_id,
        )