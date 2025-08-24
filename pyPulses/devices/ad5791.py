"""
This class is an interface for communicating with the AD5791 DC box. The
instrument in question has an Arduino Uno connected to the Analog Devices DAC
that takes serial bus input to set 20-bit unipolar DC outputs on 8 channels.
"""

from .pyvisa_device import pyvisaDevice
import pyvisa.constants
import numpy as np
from math import ceil
import time
import json
import os

class ad5791(pyvisaDevice):
    """Class interface for communicating with the AD5791 DC box."""
    def __init__(self, logger = None, max_step: float = 0.05, 
                 wait: float = 0.1, calibration_json: str = None,
                 instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        max_step : float, default=0.05
            maximum voltage step to take when sweeping.
        wait : float, default=0.1
            time to wait between setting voltages while sweeping.
        calibration_json : str, optional
            path to a file containing DAC calibration data.
        instrument_id : str, optional
            VISA resource name.
        """

        # configurations for pyvisa resource manager
        self.pyvisa_config = {
            "resource_name"     : "ASRL5::INSTR",
            "baud_rate"         : 115200,
            'write_termination' : '\n',
            'read_termination'  : '\n',

            'max_retries'       : 1,
            'min_interval'      : 0.05,
            "timeout"           : 3000
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        # maximum bounds on channel values
        self.max_V      = 10.
        self.min_V      = -10.

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

        self.load_calibration(calibration_json)

        # arduino reset delay after connect
        time.sleep(2.5)

    # Output enable

    def output(self, ch: int, on: bool = None) -> bool | None:
        """
        Set or query the output state on the desired channel.

        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        on : bool, Optional

        Returns
        -------
        bool or None
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if on is None:
            return int(self.query(f"OUTP{ch}?"))
        
        self.write(f"OUTP{ch} {int(on)}")
        self.info(f"{'En' if on else 'Dis'}abled channel {ch} output; "
                   "setting to a maximally clean zero...")
        self.set_V(ch, 0.0)

    def output_all(self, on: bool):
        """
        Enable or disable all outputs.
        """

        for ch in range(8):
            self.output(ch, on)

    # Raw uncalibrated voltages

    def sweep_raw_V(self, ch: int, V: float, 
                    max_step: float = None, wait: float = None):
        """
        Sweep DC value of a given channel smoothly to the target. This is the 
        uncalibrated value inferred from +-10V rails.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        max_step : float, default=None
            maximum step between voltages while sweeping.
        wait : float, default=None
            wait time between steps while sweeping.
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5791 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        if not max_step:
            max_step = self.max_step
        if not wait:
            wait = self.wait

        start = self.get_raw_V(ch)
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            time.sleep(wait)
            self.set_raw_V(ch, v, chatty = False)
        
        self.info(f"Swept channel {ch} to {V} V (raw).")

    def get_raw_V(self, ch: int):
        """
        Get the raw DC value on a given channel. This is the uncalibrated value
        inferred from +-10V rails.

        Parameters
        ----------
        ch : int
            target channel.

        Returns
        -------
        V : float
            voltage on the target channel Note: This is not a true query. It 
            simply returns what is saved on the computer. There is currently 
            no way to ask the arduino directly.
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return None

        return float(self.query(f"VOLT{ch}?"))       

    def set_raw_V(self, ch: int, V: float, chatty: bool = True):
        """
        Set the raw DC value of a given channel. This is the uncalibrated value
        inferred from +-10V rails.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        chatty : bool, default=True
            whether to log the change in channel settings.
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5791 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        self.write(f"VOLT{ch} {V}")
        self.info(f"Set channel {ch} to {V} V (raw).")

    # Calibrated voltages

    def _raw_to_cal(self, ch: int, V: float) -> float:
        """
        Return the calibrated voltage corresponding to a raw, uncalibrated 
        voltage.
        """

        a, b = self.calibration[ch]
        return a*V + b
    
    def _cal_to_raw(self, ch: int, V: float) -> float:
        """
        Return the raw, uncalibrated voltage corresponding to a calibrated
        voltage.
        """

        a, b = self.calibration[ch]
        return (V - b) / a

    def sweep_V(self, ch: int, V: float, 
                max_step: float = None, wait: float = None):
        """
        Sweep DC value of a given channel smoothly to the target.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        max_step : float, default=None
            maximum step between voltages while sweeping.
        wait : float, default=None
            wait time between steps while sweeping.
        """

        self.sweep_raw_V(ch, self._cal_to_raw(ch, V), max_step, wait)
        self.info(f"Swept channel {ch} to calibrated value {V} V.")

    def get_V(self, ch: int):
        """
        Get the DC value on a given channel.

        Parameters
        ----------
        ch : int
            target channel.

        Returns
        -------
        V : float
            voltage on the target channel Note: This is not a true query. It 
            simply returns what is saved on the computer. There is currently 
            no way to ask the arduino directly.
        """

        return self._raw_to_cal(ch, self.get_raw_V(ch))
    
    def set_V(self, ch: int, V: float, chatty: bool = True):
        """
        Set the DC value of a given channel.

        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        chatty : bool, default=True
            whether to log the change in channel settings.
        """

        self.set_raw_V(ch, self._cal_to_raw(V), chatty = False)
        if chatty:
            self.info(f"Set channel {ch} to calibrated value {V} V.")

    def load_calibration(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                r'ad5791_cal.json'
            )
        with open(path, 'r') as f:
            self.calibration = json.load(f)