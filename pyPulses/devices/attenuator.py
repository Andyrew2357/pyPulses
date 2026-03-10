from .abstract_device import abstractDevice
from .registry import register_hardware_class, HardwareRegistry

from typing import Any, Dict

class Attenuator(abstractDevice):
    """Protocol for attenuator hardware interface"""

    def get(self) -> float:
        """Get current attenuation as a fraction"""
        ...

    def get_min(self) -> float:
        """Minimum attenuation (least attenuation, closes to 1.0)"""
        ...

    def get_max(self) -> float:
        """Maximum attenuation (most attenuation, closes to 0.0)"""
        ...

    def set_closest(self, f: float) -> float:
        """Set to the closest available attenuation, return actual value."""
        ...

    def set_lower(self, f: float) -> float:
        """Set to the largest attenuation <= fraction, return actual value."""
        ...

    def set_higher(self, f: float) -> float:
        """Set to the least attenuation >= fraction, return actual value."""
        ...

@register_hardware_class("FixedAttenuator")
class FixedAttenuator(Attenuator):
    def __init__(self, f: float, registry_id: str | None = None, **kwargs):
        super().__init__()

        self.f = f
        HardwareRegistry.register(self, registry_id)

    def get(self) -> float:
        return self.f
    
    def get_min(self) -> float:
        return self.f
    
    def get_max(self) -> float:
        return self.f
    
    def set_closest(self, f: float) -> float:
        return self.f
    
    def set_lower(self, f: float) -> float:
        return self.f
    
    def set_higher(self, f: float) -> float:
        return self.f
    
    def _serialize_state(self) -> Dict[str, Any]:
        config = super()._serialize_state()
        config['atten_frac'] = self.f
        return config
    
    def _deserialize_state(self, state):
        super()._deserialize_state(state)
        if 'atten_frac' in state:
            self.f = state['atten_frac']

    @classmethod
    def from_config(cls, config):
        f = config.pop('atten_frac')
        registry_id = config.pop('registry_id')

        return cls(f=f, registry_id=registry_id, **config)