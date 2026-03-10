"""
Base class for all devices with logging and serialization support.
"""

import json
from typing import Dict, Any


class abstractDevice:
    """
    Base class for all devices.
    
    Provides:
    - Logging via optional logger
    - Serialization interface (_serialize_state, _deserialize_state)
    - from_config classmethod interface for deserialization
    
    Subclasses should:
    - Call super().__init__(logger) 
    - Implement _serialize_state() returning all config needed to reconstruct
    - Implement from_config() to construct from serialized config
    """
    
    # Set by registry on registration
    _hardware_registry_id_: str | None = None
    _device_registry_id_: str | None = None
    
    # Set by @register_hardware_class / @register_device_class decorators
    _registry_class_tag_: str | None = None
    
    def __init__(self, logger=None):
        self.logger = logger

    """
    -------------------------------------------------------------------------
    Logging
    -------------------------------------------------------------------------
    """ 

    def debug(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def warn(self, msg):
        if self.logger:
            self.logger.warning(msg)

    def error(self, msg):
        if self.logger:
            self.logger.error(msg)

    """
    -------------------------------------------------------------------------
    Serialization
    -------------------------------------------------------------------------
    """

    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize device state to a dictionary.
        
        Subclasses should override this to include all parameters needed
        to reconstruct the device and restore its state.
        
        Returns
        -------
        dict
            Configuration that can be passed to from_config().
        """
        return {}
    
    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """
        Restore device state from a dictionary.
        
        Called when a device already exists and we want to restore
        saved settings without recreating it.
        
        Parameters
        ----------
        state : dict
            Previously serialized state.
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "abstractDevice":
        """
        Construct a device from serialized configuration.
        
        Subclasses must implement this to handle their specific
        construction requirements.
        
        Parameters
        ----------
        config : dict
            Must contain 'registry_id' plus class-specific parameters.
        
        Returns
        -------
        abstractDevice
            The constructed device instance.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config() for deserialization"
        )

    """
    -------------------------------------------------------------------------
    Convenience methods for file-based save/load
    -------------------------------------------------------------------------
    """
    
    def save_state_json(self, path: str) -> None:
        """Save device state to a JSON file."""
        try:
            state = self._serialize_state()
        except Exception:
            state = {}
            registry_id = self._hardware_registry_id_ or self._device_registry_id_
            print(f"State serialization failed for {registry_id or 'UNREGISTERED_DEVICE'}")

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state_json(self, path: str) -> None:
        """Load device state from a JSON file."""
        with open(path, 'r') as f:
            state = json.load(f)

        try:
            self._deserialize_state(state)
        except Exception:
            registry_id = self._hardware_registry_id_ or self._device_registry_id_
            print(f"State deserialization failed for {registry_id or 'UNREGISTERED_DEVICE'}")

    """
    -------------------------------------------------------------------------
    Cleanup
    -------------------------------------------------------------------------
    """
    
    def __del__(self):
        if self.logger:
            while self.logger.hasHandlers():
                self.logger.removeHandler(self.logger.handlers[0])