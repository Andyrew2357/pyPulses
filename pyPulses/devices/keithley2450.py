"""
This class is an interface for communicating with the Keithley 2450 SMU.
"""

from .pyvisa_device import pyvisaDevice
from math import ceil
import numpy as np
import time

class keithley2450(pyvisaDevice):
    """Class interface for controlling the Keithley 2450."""
    def __init__(self, logger = None, max_step: float = 0.05, 
                 wait: float = 0.1, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        max_step : float, default=0.05
            maximum voltage step to take when sweeping.
        wait : float, default=0.1
            time to wait between setting voltages while sweeping.
        instrument_id : str, optional
            VISA resource name.
        """        
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
        
        self.info(f"Keithley2450: Swept voltage to {V} V.")

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
        self.device.write(f"SOUR:VOLT {V}")
        if chatty:
            self.info(f"Keithley2450: Set voltage source to {V} V.")

    def get_V(self) -> float:
        """
        Query the measured voltage.
        
        Returns
        -------
        V : float
        """
        source_mode = self.device.query("SOUR:FUNC?").strip()
        if source_mode == 'VOLT':
            return float(self.device.query("READ? \"defbuffer1\", SOUR"))
        else:
            return float(self.device.query("MEAS:VOLT?"))

    def set_I(self, I: float):
        """
        Set the DC output current in A.
        
        Parameters
        ----------
        I : float
        """
        self.device.write(f"SOUR:CURR {I}")
        self.info(f"Keithley2450: Set current source to {I} A.")

    def get_I(self) -> float:
        """
        Query the measured current.
        
        Returns 
        -------
        I : float
        """
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

        Parameters
        ----------
        val : float
        """
        source = self.device.query("SOUR:FUNC?").strip()
        sense = 'ILIM' if source == 'VOLT' else 'VLIM'
        self.device.write(f"SOUR:{source}:{sense}:LEV {val}")
        self.info(f"Keithley2450: Set {sense} compliance to {val}.")

    def get_compliance(self) -> float:
        """
        Query the true compliance value.
        This is the minimum of the measurement range and compliance value.

        Returns
        -------
        compliance_val : float
        """
        source = self.device.query("SOUR:FUNC?").strip()
        sense = 'ILIM' if source == 'VOLT' else 'VLIM'
        return float(self.device.query(f"SOUR:{source}:{sense}:LEV?"))
    
    def set_source_volt(self, volt: bool):
        """
        Set the source to voltage or current.
        
        Parameters
        ----------
        volt : bool
            true = voltage source, false = current source.
        """
        self.device.write(f"SOUR:FUNC {'VOLT' if volt else 'CURR'}")
        self.info(f"Keithley2450: Set source to {'volt' if volt else 'curr'}.")

    def is_source_volt(self) -> bool:
        """Return True if the source setting is voltage."""
        return self.device.query("SOUR:FUNC?").strip() == 'VOLT'

    def set_output_on(self, on: bool):
        """
        Set the output on or off.
        
        Parameters
        ----------
        on : bool
            true = enabled, false = disabled.
        """
        self.device.write(f"OUTP:STAT {'ON' if on else 'OFF'}")
        self.info(f"Keithley2450: Set output {'on' if on else 'off'}.")

    def is_output_on(self) -> bool:
        """Return True if the output is on."""
        return int(self.device.query("OUTP:STAT?")) == 1
    
    def set_source_V_range(self, V: float):
        """
        Set the source voltage range.
        
        Parameters
        ----------
        V : float
        """
        self.device.write(f"SOUR:VOLT:RANG {V}")
        self.info(f"Keithley2450: Set source voltage range to {V} V.")

    def get_source_V_range(self) -> float:
        """
        Query the source voltage range.
        
        Returns
        -------
        Vrange : float
        """
        return float(self.device.query("SOUR:VOLT:RANG?"))
