"""This class is an interface to the HP34401A digital multimeter"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
from typing import Optional

class hp34401a(pyvisaDevice):
    def __init__(self, logger = None, instrument_id: Optional[str] = None):
        
        self.config = {
            "resource_name" : "",
            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

    def get_V(self) -> float:
        """Query the voltage."""
        return float(self.device.query(":MEAS:VOLT:DC?").strip())
