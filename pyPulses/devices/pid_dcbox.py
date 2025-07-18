"""
This class is an interface for communicating with the PID DC box.
"""

from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from typing import Tuple
from math import ceil
import numpy as np
import time

class PIDbox(pyvisaDevice):
    """Class interface for communicating with the PID DC box."""
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
        # configurations for pyvisa resource manager
        self.pyvisa_config = {
            "resource_name" : "ASRL6::INSTR",
            "baud_rate"     : 9600,
            "data_bits"     : 8,
            "parity"        : pyvisa.constants.Parity.none,
            "stop_bits"     : pyvisa.constants.StopBits.one,
            "flow_control"  : pyvisa.constants.VI_ASRL_FLOW_NONE,
            "write_buffer_size" : 512
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        # maximum bounds on channel values
        self.hard_max_V = 10.
        self.hard_min_V = -10.
        self.max_V = {i: self.hard_max_V for i in range(4)}
        self.min_V = {i: self.hard_min_V for i in range(4)}

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

    def get_V(self, ch: int) -> float:
        """
        Get the DC value on a given channel.

        Parameters
        ----------
        ch : int
            target channel

        Returns
        -------
        V : float
        """
        self._select_channel(ch)
        return float(self.device.query("SOURce:VOLTage?\n"))

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
        self._select_channel(ch)
        
        if V > self.max_V[ch] or V < self.min_V[ch]:
            Vt = min(self.max_V[ch], max(self.min_V[ch], V))
            self.warn(
                f"{V} on PID DC box channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        self.device.write(f"SOURce:VOLTage {V}\n")
        if chatty:
            self.info(f"Set source on channel {ch} to {V} V.")

    def get_PID(self, ch: int) -> float:
        """
        Get the target point for the PID loop.

        Parameters
        ----------
        ch : int

        Returns
        -------
        float
        """
        self._select_channel(ch)
        return float(self.device.query("PID:SET?\n"))

    def set_PID(self, ch: int, V: float):
        """
        Set the target level for the PID loop.
        
        Parameters
        ----------
        ch : int
        V : float
        """
        self._select_channel(ch)
        
        if V > self.max_V[ch] or V < self.min_V[ch]:
            Vt = min(self.max_V[ch], max(self.min_V[ch], V))
            self.warn(
                f"{V} on PID DC box channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        self.device.write(f"PID:SET {V}\n")
        self.info(f"Set PID loop on channel {ch} to {V} V.")

    def get_PID_status(self, ch: int) -> bool:
        """
        Check whether PID loop is enabled on selected channel.
        
        Parameters
        ----------
        ch : int

        Returns
        -------
        bool
        """
        self._select_channel(ch)
        return self.device.query("PID:STAT?\n") != 'PID control disabled\r\n'

    def set_PID_status(self, ch: int, on: bool):
        """
        Enable or disable the PID loop.
        
        Parameters
        ----------
        ch : int
        on : bool
        """
        self._select_channel(ch)
        self.device.write(f"PID:{'ON' if on else 'OFF'}\n")
        self.info(f"{'En' if on else 'Dis'}abled PID loop on channel {ch}.")

    def P(self, P: float = None) -> float | None:
        """
        Query or set proportional coefficient for PID loop.
        
        Parameters
        ----------
        P : float or None

        Returns
        -------
        float or None
        """
        if P is None:
            return float(self.device.query("PID:P?\n"))
        else:
            self.device.write(f"PID:P {P}\n")

    def I(self, I: float = None) -> float | None:
        """
        Query or set integral coefficient for PID loop.
        
        Parameters
        ----------
        I : float or None

        Returns
        -------
        float or None
        """
        if I is None:
            return float(self.device.query("PID:I?\n"))
        else:
            self.device.write(f"PID:I {I}\n")

    def D(self, D: float = None) -> float | None:
        """
        Query or set derivative coefficient for PID loop.
        
        Parameters
        ----------
        D : float or None

        Returns
        -------
        float or None
        """
        if D is None:
            return float(self.device.query("PID:D?\n"))
        else:
            self.device.write(f"PID:D {D}\n")

    def get_ADC(self) -> float:
        """
        Get the current ADC reading.
        
        Returns
        -------
        float
        """
        return float(self.device.query("ADC?\n"))

    def set_channel_lim(self, ch: int, lim: Tuple[float | None, float | None]):
        """
        Manually set the voltage limits for a channel.
        
        Parameters
        ----------
        ch : int
        lim : tuple of float or none
            (low, high). If either is None, the class default extreme is applied
        """
        self._select_channel(ch)
        
        vl, vh = lim

        if vh is None:
            vh = self.hard_max_V
        if vl is None:
            vl = self.hard_min_V

        vh = min(self.hard_max_V, vh)
        vl = max(self.hard_min_V, vl)
        self.max_V[ch] = vh
        self.min_V[ch] = vl

        self.device.write(f"LIMIT:LO {vl}\n")
        self.device.write(f"LIMIT:HI {vl}\n")
        
        self.info(f"Set hard channel {ch} limits to [{vl}, {vh}] V")

    def sweep_V(self, ch: int, V: float, 
                max_step: float = None, wait: float = None):
        """
        Sweep the voltage on selected channel.
        
        Parameters
        ----------
        ch : int
        V : float
        max_step : float, default=None
            maximum voltage step; uses `self.max_step` if None.
        wait : float, default=None
            wait time between steps; uses `self.wait` if None.
        """
        
        self._select_channel(ch)
        if V > self.max_V[ch] or V < self.min_V[ch]:
            Vt = min(self.max_V[ch], max(self.min_V[ch], V))
            self.warn(
                f"{V} on channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        if not max_step:
            max_step = self.max_step
        if not wait:
            wait = self.wait

        start = self.get_V(ch)
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            time.sleep(wait)
            self.set_V(ch, v, chatty = False)
        
        self.info(f"Swept channel {ch} from {start} to {V} V.")

    def _select_channel(self, ch: int):
        if ch < 0 or ch > 3:
            self.error(f"PID DC box does not have a channel {ch}.")
            return
        self.device.write(f"INSTrument:SELect {ch}\n")
