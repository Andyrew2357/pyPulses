"""
Global variables that must be accessed and modified by various device classes.

This is mostly so that we don't create multiple instances of an object 
corresponding to the same instrument. This is most likely to arise if we're 
using the instrument inside of another instrument object but still want to 
access it for other uses (eg. DCbox with channels that get used in various 
contexts).

We make sure to add objects to the registry for all classes that control a 
single instrument and to check the registry whenever we're using that instrument
as part of a compound device object.
"""

from .abstract_device import abstractDevice

from typing import Dict

class DeviceRegistry:
    """
    Provides a registry of physical instruments connected to the computer. VISA
    resources are referenced by their resource id and Zurich Instruments
    resources are referenced by their device name (eg. dev1234). Other 
    unconventional instruments will have alternative identifiers (eg. 
    FastFlight-2 gets registered under 'FASTFLIGHT2').
    """
    
    _instance = None
    _active_devices: Dict[str, object] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_device(cls, instrument_name: str) -> abstractDevice | None:
        """
        Get a connection if it exists.

        Parameters
        ----------
        instrument_name : str, optional
            unique identifier for the physical instrument.

        Returns
        -------
        device : abstractDevice or None
            control class associated with the physical instrument.
        """

        if not instrument_name in cls._active_devices:
            return None
        return cls._active_devices.get(instrument_name)
    
    @classmethod
    def register_device(cls, instrument_name: str, device: abstractDevice):
        """
        Register a new device.
        
        Parameters
        ----------
        instrument_name: str, optional
            unique identifier for the physical instrument.
        device : abstractDevice
            control class associated with the physical instrument.
        """

        if hasattr(device, '_registry_name_'):
            device._registry_name_ = instrument_name
        cls._active_devices[instrument_name] = device

    @classmethod
    def unregister_device(cls, instrument_name: str):
        """
        Unregister a device.
        
        Parameters
        ----------
        instrument_name : str, optional
            unique identifier for the physical instrument.
        """
        if instrument_name in cls._active_devices:
            del cls._active_devices[instrument_name]

    @classmethod
    def get_all_devices(cls) -> Dict[str, object]:
        """
        Return a copy of `_active_devices`.
        
        Returns
        -------
        _active_devices : dict
            maps the identifiers for physical devices to their 
            control classes.
        """
        return cls._active_devices.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered devices. This is an important cleanup step if we
        want other programs to access resources.
        """
        for dev in list(cls._active_devices.keys()):
            del cls._active_devices[dev]
        cls._active_devices.clear()
