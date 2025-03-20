"""
This class is an interface for communicating with the Keithley 2000 multimeter
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from typing import Optional
from math import ceil
import numpy as np
import time

class keithley2400(pyvisaDevice):
    def __init__(self, logger: Optional[str] = None, 
                 instrument_id: Optional[str] = None):
        
        self.config = {
            "resource_name" : "GPIB0::24::INSTR",

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
        """Query the measured voltage."""
        self.device.write(":CONF:VOLT:DC")
        return float(self.device.query(":READ?"))


    def get_I(self) -> float:
        """Query the measured current."""
        self.device.write(":CONF:VOLT:DC")
        return float(self.device.query(":READ?"))
