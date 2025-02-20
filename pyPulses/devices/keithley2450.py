"""
This class is an interface for communicating with the Keithley 2450 SMU.
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from typing import Optional
from math import ceil
import numpy as np
import time

class keithley2450(pyvisaDevice):
    def __init__(self, logger: Optional[str] = None, 
                 max_step: Optional[float] = 0.05, 
                 wait: Optional[float] = 0.1, 
                 instrument_id: Optional[str] = None):
        
        self.config = {
            "resource_name" : "GPIB0::24::INSTR",
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        # Set Output Buffer Size to 512 bytes
        # self.device.set_buffer(pyvisa.constants.VI_WRITE_BUF, 512)
        
        # Set EOS Character Code to LF (Line Feed, ASCII 10)
        self.device.set_visa_attribute(
            pyvisa.constants.VI_ATTR_TERMCHAR, ord('\n')
        )

        # Set EOI Mode to 'on'
        self.device.send_end = True

        # Set EOS Mode to 'none'
        self.device.set_visa_attribute(
            pyvisa.constants.VI_ATTR_TERMCHAR_EN, False
        )

        self.max_step = max_step
        self.wait = wait

    def sweep_V(self, V, max_step = None, wait = None):
        """Sweep smoothly to a new voltage."""

        if not max_step:
            max_step = self.max_step
        if not wait:
            wait = self.wait

        start = self.get_V()
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            self.set_V(v, chatty = False)
            time.sleep(wait)
        
        self.info(f"Keithley2450: Swept voltage to {V} V.")

    def set_V(self, V: float, chatty = True):
        """Set voltage source to V."""
        self.device.write(f"SOUR:VOLT {V}")
        if chatty:
            self.info(f"Keithley2450: Set voltage source to {V} V.")

    def get_V(self) -> float:
        """Query the measured voltage."""
        source_mode = self.device.query("SOUR:FUNC?").strip()
        if source_mode == 'VOLT':
            return float(self.device.query("READ? \"defbuffer1\", SOUR"))
        else:
            return float(self.device.query("MEAS:VOLT?"))

    def set_I(self, I: float):
        """Set current source to I."""
        self.device.write(f"SOUR:CURR {I}")
        self.info(f"Keithley2450: Set current source to {I} A.")

    def get_I(self) -> float:
        """Query the measured current."""
        source_mode = self.device.query("SOUR:FUNC?").strip()
        if source_mode == 'CURR':
            return float(self.device.query("READ? \"defbuffer1\", SOUR"))
        else:
            return float(self.device.query("MEAS:CURR?"))

    def set_compliance(self, val: float):
        """
        Set the compliance by adding protections.
        If the instrument is acting as a voltage source, this limits the current
        and visa versa.
        """
        source = self.device.query("SOUR:FUNC?").strip()
        sense = 'ILIM' if source == 'VOLT' else 'VLIM'
        self.device.write(f"SOUR:{source}:{sense}:LEV {val}")
        self.info(f"Keithley2450: Set {sense} compliance to {val}.")

    def get_compliance(self) -> float:
        """
        Query the true compliance value.
        """
        source = self.device.query("SOUR:FUNC?").strip()
        sense = 'ILIM' if source == 'VOLT' else 'VLIM'
        return float(self.device.query(f"SOUR:{source}:{sense}:LEV?"))
    
    def set_source_volt(self, volt: bool):
        """Set the source to voltage or current."""
        self.device.write(f"SOUR:FUNC {'VOLT' if volt else 'CURR'}")
        self.info(f"Keithley2450: Set source to {'volt' if volt else 'curr'}.")

    def is_source_volt(self) -> bool:
        """Return True if the source setting is voltage."""
        return self.device.query("SOUR:FUNC?").strip() == 'VOLT'

    def set_output_on(self, on: bool):
        """Set the output on or off."""
        self.device.write(f"OUTP:STAT {'ON' if on else 'OFF'}")
        self.info(f"Keithley2450: Set output {'on' if on else 'off'}.")

    def is_output_on(self) -> bool:
        """Return True if the output is on."""
        return int(self.device.query("OUTP:STAT?")) == 1
    
    def set_source_V_range(self, V: float):
        """Set the source voltage range."""
        self.device.write(f"SOUR:VOLT:RANG {V}")
        self.info(f"Keithley2450: Set source voltage range to {V} V.")

    def get_source_V_range(self) -> float:
        """Query the source voltage range."""
        return float(self.device.query("SOUR:VOLT:RANG?"))
