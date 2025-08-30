"""
This class is an interface to the PCM1704-based 24-bit DC box. The box uses an
NI USB-6051 board to communicate, which we programmatically control with a C++
subroutine. 

Technically, we could get away with just using nidaqmx in python, but I switched 
over to this when I saw what I *thought* were inconsistencies in the set value 
due to timing, and I determined that I needed microsecond level delays, which 
python can't handle without rounding up to milliseconds, introducing huge
overhead. It turns out these were actually just due to me hitting the multimeter 
too quickly. Regardless, I've tested it to be very marginally faster this way,
so I'll stick with this
"""

from ..utils.curves import MonotonicPiecewiseLinear

from ._registry import DeviceRegistry
from .abstract_device import abstractDevice

try:
    from .subroutines.pcm1704_driver import PCM1704Driver # type: ignore
except ImportError:
    class PCM1704Driver():
        def __init__(*args, **kwargs):
            raise RuntimeError(
                "pcm1704_driver extension not built. "
                "Please install NI-DAQmx and rebuild pyPulses."
            )

import os
import time
import json
import threading
from math import ceil
import numpy as np

class pcm1704(abstractDevice):
    """Class interface for controlling the PCM1704-based 24-bit DC box."""
    
    v_fullscale = 12.0
    bits_max = 0xFFFFFF
    bits_half_max = 0x7FFFFF
    n_ch = 8

    def __init__(self, logger = None, max_step: float = 0.05, 
                 wait: float = 0.1, calibration_json: str = 'Darjeeling',
                 dev_name: str = 'Dev2', change_delay_us: int = 0):
        """
        Parameters
        ----------
        logger : Logger, optional
        max_step : float, default=0.05
            maximum voltage step to take when sweeping.
        wait : float, default=0.1
            time to wait between setting voltages while sweeping.
        calibration_json : str, default='Darjeeling'
            path to a file containing DAC calibration data. There are also two
            presets, 'Lipton' and 'Darjeeling' corresponding to the two
            instances of this instrument in our lab.
        dev_name : str, default='Dev2'
        change_delay_us : int, default=2
            microsecond delays while bit-banging.
        """
        
        super().__init__(logger)
        DeviceRegistry.register_device(dev_name, self)

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

        self.load_calibration(calibration_json)

        self._lock = threading.Lock()
        self.ch_bits = [0.0] * self.n_ch
        self.driver = PCM1704Driver(dev_name, change_delay_us)

    def _close(self):
        """Close the C++ driver (destructor will clean up DAQmx task)"""
        if hasattr(self, 'driver'):
            del self.driver

    def __del__(self):
        self._close()
        super().__del__()

    def _raw_voltage_to_bits(self, rawV: float) -> int:
        if abs(rawV) > self.v_fullscale:
            rawV = self.v_fullscale * (1 if rawV > 0 else -1)
        bits = int(((rawV * self.bits_half_max) / self.v_fullscale) + \
                  (0.5 if rawV >= 0 else -0.5 ))
        return bits if bits >= 0 else (self.bits_max + bits + 1)
    
    def _bits_to_raw_voltage(self, bits: int) -> float:
        if bits < self.bits_half_max + 1:
            return bits * self.v_fullscale / self.bits_half_max
        else:
            return (bits - self.bits_max - 1) \
                    * self.v_fullscale      \
                    / self.bits_half_max
        
    def _set_bits(self, ch: int, bits: int) -> float:
        v = max(0, min(bits, self.bits_max))
        self.ch_bits[ch] = v
        with self._lock:
            self.driver.set_bits(ch, bits)

    def _get_bits(self, ch: int) -> int:
        return self.ch_bits[ch]
    
    def set_raw_V(self, ch: int, V: float, chatty: bool = True):
        """
        Set the raw DC value of a given channel. This is the uncalibrated value
        inferred from +-12V rails.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        chatty : bool, default=True
            whether to log the change in channel settings.
        """

        minv = self.calibration['raw_min'][ch]
        maxv = self.calibration['raw_max'][ch]
        if V > maxv or V < minv:
            Vt = min(maxv, max(minv, V))
            self.warn(
                f"{V} on PCM1704 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        bits = self._raw_voltage_to_bits(V)
        self._set_bits(ch, bits)
        if chatty:
            self.info(f"Set channel {ch} to raw value {V} V (0x{bits:X})")

    def get_raw_V(self, ch: int) -> float:
        """
        Get the raw DC value on a given channel. This is the uncalibrated value
        inferred from +-12V rails.

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

        return self._bits_to_raw_voltage(self.ch_bits[ch])
    
    def sweep_raw_V(self, ch: int, V: float, 
                    max_step: float = None, wait: float = None):
        """
        Sweep DC value of a given channel smoothly to the target. This is the 
        uncalibrated value inferred from +-12V rails.
        
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
        
        minv = self.calibration['raw_min'][ch]
        maxv = self.calibration['raw_max'][ch]
        if V > maxv or V < minv:
            Vt = min(maxv, max(minv, V))
            self.warn(
                f"{V} on PCM1704 channel {ch} is out of range; truncating to {Vt}."
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
    
    # Calibrated voltages

    def _raw_to_cal(self, ch: int, V: float) -> float:
        """
        Return the calibrated voltage corresponding to a raw, uncalibrated 
        voltage.
        """

        return self.calibration['pwl_fit'][ch](V)
    
    def _cal_to_raw(self, ch: int, V: float) -> float:
        """
        Return the raw, uncalibrated voltage corresponding to a calibrated
        voltage.
        """

        return self.calibration['pwl_fit'][ch].inverse(V)
    
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

        self.set_raw_V(ch, self._cal_to_raw(ch, V), chatty = False)
        if chatty:
            self.info(f"Set channel {ch} to calibrated value {V} V.")

    def load_calibration(self, path: str = None):
        if path in ['Lipton', 'Darjeeling']:
            mypath = os.path.dirname(os.path.abspath(__file__))
            try:
                with open(os.path.join(mypath, r'pcm1704_cal.json'), 'r') as f:
                    self.calibration = json.load(f)[path]
            except:
                self.warn("No calibration file found.")
                self.calibration = {
                    'raw_min': {ch: -self.v_fullscale for ch in range(self.n_ch)},
                    'raw_max': {ch: self.v_fullscale for ch in range(self.n_ch)},
                    'pwl_fit': {ch: {
                        'x_breaks': [-self.v_fullscale, self.v_fullscale],
                        'y_breaks': [-self.v_fullscale, self.v_fullscale],
                    } 
                    for ch in range(self.n_ch)}
                }

        else:
            with open(path, 'r') as f:
                self.calibration = json.load(f)

        self.calibration['raw_min'] = {int(k): v for k, v 
                    in self.calibration['raw_min'].items()}
        self.calibration['raw_max'] = {int(k): v for k, v 
                    in self.calibration['raw_max'].items()}
        self.calibration['pwl_fit'] = {
            int(k): MonotonicPiecewiseLinear(p['x_breaks'], p['y_breaks'])
            for k, p in self.calibration['pwl_fit'].items()
        }
        