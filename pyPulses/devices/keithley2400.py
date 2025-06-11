"""
This class is an interface for communicating with the Keithley 2400 SMU.
"""

from .pyvisa_device import pyvisaDevice
from typing import Optional
from math import ceil
import numpy as np
import time

class keithley2400(pyvisaDevice):
    def __init__(self, logger = None, max_step: float = 0.05, 
                 wait: float = 0.1, instrument_id: str = None):
        
        self.pyvisa_config = {
            "resource_name" : "GPIB0::24::INSTR",

            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

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
            time.sleep(wait)
            self.set_V(v, chatty = False)
        
        self.info(f"Keithley2400: Swept voltage to {V} V.")

    def set_V(self, V: float, chatty = True):
        """Set voltage source to V."""
        self.device.write("SOUR:FUNC VOLT")
        self.device.write("SOUR:VOLT:MODE FIXED")
        self.device.write(f"SOUR:VOLT:LEV {V}")
        if chatty:
            self.info(f"Keithley2400: Set voltage source to {V} V.")

    def get_V(self) -> float:
        """Query the measured voltage."""
        self.device.write("SENS:VOLT:RANG:AUTO ON")
        self.device.write("FORM:ELEM VOLT")
        return float(self.device.query("READ?"))

    def set_I(self, I: float):
        """Set current source to I."""
        self.device.write("SOUR:FUNC CURR")
        self.device.write("SOUR:CURR:MODE FIXED")
        self.device.write(f"SOUR:CURR:LEV {I}")
        self.info(f"Keithley2400: Set current source to {I} A.")

    def get_I(self) -> float:
        """Query the measured current."""
        self.device.write("SENS:CURR:RANG:AUTO ON")
        self.device.write("FORM:ELEM CURR")
        return float(self.device.query("READ?"))

    def set_compliance(self, val: float):
        """
        Set the compliance by adding protections.
        If the instrument is acting as a voltage source, this limits the current
        and visa versa.
        """
        source = self.device.query("SOUR:FUNC?").strip()
        sense = 'CURR' if source == 'VOLT' else 'VOLT'
        self.device.write(f"SOUR:{source}:RANG:AUTO ON")
        self.device.write(f"{sense}:PROT {val}")
        self.info(f"Keithley2400: Set {sense} compliance to {val}.")

    def get_compliance(self) -> float:
        """
        Query the true compliance value.
        This is the minimum of the measurement range and compliance value.
        """
        source = self.device.query("SOUR:FUNC?").strip()
        sense = 'CURR' if source == 'VOLT' else 'VOLT'
        range_val = float(self.device.query(f"{sense}:RANGE?"))
        prot = float(self.device.query(f"{sense}:PROT?"))
        return min(range_val, prot)
    
    def set_source_volt(self, volt: bool):
        """Set the source to voltage or current."""
        self.device.write(f"SOUR:FUNC {'VOLT' if volt else 'CURR'}")
        self.info(f"Keithley2400: Set source to {'volt' if volt else 'curr'}.")

    def is_source_volt(self) -> bool:
        """Return True if the source setting is voltage."""
        return self.device.query("SOUR:FUNC?").strip() == 'VOLT'

    def set_output_on(self, on: bool):
        """Set the output on or off."""
        self.device.write(f"OUTP:STAT {'ON' if on else 'OFF'}")
        self.info(f"Keithley2400: Set output {'on' if on else 'off'}.")

    def is_output_on(self) -> bool:
        """Return True if the output is on."""
        return int(self.device.query("OUTP:STAT?")) == 1
    
    def get_resistance(self) -> float:
        """Measure resistance."""
        self.device.write("SENS:RES:MODE MAN")
        self.device.write("SENSE:RES:RANG:AUTO ON")
        self.device.write("FORM:ELEM RES")
        return float(self.device.query("READ?"))
    
    def set_source_V_range(self, V: float):
        """Set the source voltage range."""
        self.device.write(f"SOUR:VOLT:RANG {V}")
        self.info(f"Keithley2400: Set source voltage range to {V} V.")

    def get_source_V_range(self) -> float:
        """Query the source voltage range."""
        return float(self.device.query("SOUR:VOLT:RANG?"))
