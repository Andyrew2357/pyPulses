"""
Global variables that must be accessed and modified by various device classes.
"""

from typing import Dict

class DeviceRegistry:
    """
    This is so that we don't create multiple instances of an object 
    corresponding to the same instrument. This is most likely to arise if 
    we're using the instrument inside of another instrument object (eg. DC box
    in a pulseGenerator) but still want to access it for other uses (eg. could
    have other channels we want to use for a different purpose).
    
    We make sure to add objects to the registry for all classes that control a
    single instrument and to check the registry whenever we're using that
    instrument as part of a compound device object.
    """
    
    _instance = None
    _active_devices: Dict[str, object] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_device(cls, instrument_name: str):
        """Get a connection if it exists."""
        return cls._active_devices.get(instrument_name)
    
    @classmethod
    def register_device(cls, instrument_name: str, device: object):
        """Register a new device."""
        cls._active_devices[instrument_name] = device

    @classmethod
    def unregister_device(cls, instrument_name: str):
        """Unregister a device."""
        if instrument_name in cls._active_devices:
            cls._active_devices[instrument_name].device.close()
            del cls._active_devices[instrument_name]

    @classmethod
    def get_all_devices(cls) -> Dict[str, object]:
        """Return a copy of _active_devices."""
        return cls._active_devices.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered devices."""
        for dev in cls._active_devices:
            cls._active_devices[dev].device.close()
        cls._active_devices.clear()
