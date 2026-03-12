from .pyvisa_device import pyvisaDevice
from .registry import register_hardware_class
from .channel_adapter import ScalarChannelAdapter

from logging import Logger

@register_hardware_class("keithley2000")
class keithley2000(pyvisaDevice):
    """Class representation of the Keithley 2000 digital multimeter."""

    DEFAULT_PYVISA_CONFIG = {
        'output_buffer_size': 512,
        'gpib_eos_mode': False,
        'gpib_eos_char': ord('\n'),
        'gpib_eoi_mode': True,
        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.05,
    }
    
    def __init__(self,
        resource_name: str, 
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_connect: bool = False,
        **kwargs,              
    ):
        """
        Parameters
        ----------
        resource_name : str
            VISA resource name.
        registry_id : str, optional
            Name to register this instance under in the HardwareRegistry
        logger : Logger, optional
            logger used by abstractDevice.
        **kwargs
        """

        super().__init__(resource_name, registry_id, logger, skip_connect, **kwargs)
    
    def get_V(self) -> float:
        """
        Query the voltage.
        
        Returns
        -------
        V : float
        """
        """Query the measured voltage."""
        self.write(":CONF:VOLT:DC")
        return float(self.query(":READ?"))
    
    def resolve(self, accessor: str) -> 'keithley2000_channel':
        if accessor == 'V':
            return keithley2000_channel(self)
        raise ValueError(f"keithley2000 Cannot resolve accessor: {accessor}")

class keithley2000_channel(ScalarChannelAdapter):
    def __init__(self, parent: keithley2000):
        super().__init__(parent, 'V')

    def set_output(self, value: float | None = None):
        raise RuntimeError('keithley2000 Cannot set an output.')
    
    def get_output(self):
        return self._parent.get_V()