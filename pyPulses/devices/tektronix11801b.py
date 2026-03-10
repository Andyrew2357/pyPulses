from .pyvisa_device import pyvisaDevice
from .registry import register_hardware_class

import numpy as np
from logging import Logger
from typing import Tuple

@register_hardware_class("tektronix11801b")
class tektronix11801b(pyvisaDevice):
    """Class representation of the tektronix 11801B digital sampling oscilloscope."""

    DEFAULT_PYVISA_CONFIG = {
        'timeout': 10,
        'input_buffer_size': 131072,
        'gpib_eos_char': ord('\n'),
        'gpib_eoi_mode': True,
        'gpib_eos_mode': False,
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

    def get_trace(self) -> Tuple[float, np.ndarray, np.ndarray]:
        
        data = map(float, self.query("CURVe?").split(','))
        spec = data[0]
        data = np.array(data[1:])
        dt = int(self.query("WFMP? XIN")[14:])
        time = dt * np.arange(data.size)
        return spec, time, data
