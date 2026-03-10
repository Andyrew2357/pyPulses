from .pyvisa_device import pyvisaDevice
from .registry import register_hardware_class

import numpy as np
import time
from math import ceil
from logging import Logger

@register_hardware_class("keithley2400")
class keithley2400(pyvisaDevice):
    """Class representation of the Keithley 2400 SMU"""

    max_step = 0.05
    wait = 0.1

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

    def sweep_V(self, V, max_step = None, wait = None):
        """
        Sweep DC voltage smoothly to the target.
        
        Parameters
        ----------
        V : float
            target voltage.
        max_step : float, default=None
            maximum step between voltages while sweeping.
        wait : float, default=None
            wait time between steps while sweeping.
        """

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
        """
        Set the DC output voltage.
        
        Parameters
        ----------
        V : float
            target voltage.
        chatty : bool, default=True
            whether to log the change in channel settings.
        """
        self.write("SOUR:FUNC VOLT")
        self.write("SOUR:VOLT:MODE FIXED")
        self.write(f"SOUR:VOLT:LEV {V}")
        if chatty:
            self.info(f"Keithley2400: Set voltage source to {V} V.")

    def get_V(self) -> float:
        """
        Query the measured voltage.
        
        Returns
        -------
        V : float
        """
        self.write("SENS:VOLT:RANG:AUTO ON")
        self.write("FORM:ELEM VOLT")
        return float(self.query("READ?"))

    def set_I(self, I: float):
        """
        Set the DC output current in A.
        
        Parameters
        ----------
        I : float
        """
        self.write("SOUR:FUNC CURR")
        self.write("SOUR:CURR:MODE FIXED")
        self.write(f"SOUR:CURR:LEV {I}")
        self.info(f"Keithley2400: Set current source to {I} A.")

    def get_I(self) -> float:
        """
        Query the measured current.
        
        Returns 
        -------
        I : float
        """
        self.write("SENS:CURR:RANG:AUTO ON")
        self.write("FORM:ELEM CURR")
        return float(self.query("READ?"))

    def set_compliance(self, val: float):
        """
        Set the compliance by adding protections.
        If the instrument is acting as a voltage source, this limits the current
        and visa versa.

        Parameters
        ----------
        val : float
        """
        source = self.query("SOUR:FUNC?").strip()
        sense = 'CURR' if source == 'VOLT' else 'VOLT'
        self.write(f"SOUR:{source}:RANG:AUTO ON")
        self.write(f"{sense}:PROT {val}")
        self.info(f"Keithley2400: Set {sense} compliance to {val}.")

    def get_compliance(self) -> float:
        """
        Query the true compliance value.
        This is the minimum of the measurement range and compliance value.

        Returns
        -------
        compliance_val : float
        """
        source = self.query("SOUR:FUNC?").strip()
        sense = 'CURR' if source == 'VOLT' else 'VOLT'
        range_val = float(self.query(f"{sense}:RANGE?"))
        prot = float(self.query(f"{sense}:PROT?"))
        return min(range_val, prot)
    
    def set_source_volt(self, volt: bool):
        """
        Set the source to voltage or current.
        
        Parameters
        ----------
        volt : bool
            true = voltage source, false = current source.
        """
        self.write(f"SOUR:FUNC {'VOLT' if volt else 'CURR'}")
        self.info(f"Keithley2400: Set source to {'volt' if volt else 'curr'}.")

    def is_source_volt(self) -> bool:
        """Return True if the source setting is voltage."""
        return self.query("SOUR:FUNC?").strip() == 'VOLT'

    def set_output_on(self, on: bool):
        """
        Set the output on or off.
        
        Parameters
        ----------
        on : bool
            true = enabled, false = disabled.
        """
        self.write(f"OUTP:STAT {'ON' if on else 'OFF'}")
        self.info(f"Keithley2400: Set output {'on' if on else 'off'}.")

    def is_output_on(self) -> bool:
        """Return True if the output is on."""
        return int(self.query("OUTP:STAT?")) == 1
    
    def get_resistance(self) -> float:
        """
        Measure resistance (V/I) in Ohms.
        
        Returns
        -------
        R : float
        """
        self.write("SENS:RES:MODE MAN")
        self.write("SENSE:RES:RANG:AUTO ON")
        self.write("FORM:ELEM RES")
        return float(self.query("READ?"))
    
    def set_source_V_range(self, V: float):
        """
        Set the source voltage range.
        
        Parameters
        ----------
        V : float
        """
        self.write(f"SOUR:VOLT:RANG {V}")
        self.info(f"Keithley2400: Set source voltage range to {V} V.")

    def get_source_V_range(self) -> float:
        """
        Query the source voltage range.
        
        Returns
        -------
        Vrange : float
        """
        return float(self.query("SOUR:VOLT:RANG?"))
