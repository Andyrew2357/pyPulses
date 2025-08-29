"""
This class is an interface to the PCM1704 based 24-bit DC box. The box uses an
NI USB-6051 board to communicate, which we programmatically control with nidaqmx
"""

from ..utils.curves import MonotonicPiecewiseLinear

from ._registry import DeviceRegistry
from .abstract_device import abstractDevice

import os
import time
import json
import threading
from math import ceil
import numpy as np
import nidaqmx
from nidaqmx.constants import LineGrouping

class pcm1704(abstractDevice):
    """Class interface for controlling the PCM1704 based 24-bit DC box."""
    
    # Bit masks
    SCLK  = 1 << 0
    SDATA = 1 << 1
    AD2   = 1 << 2
    AD1   = 1 << 3
    AD0   = 1 << 4
    WCE1  = 1 << 5
    WCE0  = 1 << 6
    
    full_input = 0x7F
    v_fullscale = 12.0
    bits_max = 0xFFFFFF
    bits_half_max = 0x7FFFFF
    n_ch = 8

    def __init__(self, logger = None, max_step: float = 0.05, 
                 wait: float = 0.1, calibration_json: str = 'Darjeeling',
                 dev_name: str = 'Dev2', change_delay: float = 2e-6):
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
        change_delay : float
            USB-6051 spam protection to avoid communication errors.
        """
        
        super().__init__(logger)
        DeviceRegistry.register_device(dev_name, self)

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

        self.load_calibration(calibration_json)

        self.change_delay = change_delay
        self._lock = threading.Lock()
        self.open(dev_name)

    def open(self, dev_name):
        """Open all the relevant USB-6051 lines."""

        self.task_out = nidaqmx.Task()
        self.task_out.do_channels.add_do_chan(f'{dev_name}/port0/line0:7', 
                            line_grouping = LineGrouping.CHAN_FOR_ALL_LINES)
        # Port B = digital input (for test/check)
        self.task_in = nidaqmx.Task()
        self.task_in.di_channels.add_di_chan(f'{dev_name}/port1/line0:7',
                            line_grouping = LineGrouping.CHAN_FOR_ALL_LINES)
        
        self.current_output = 0
        self.ch_bits = [0.0] * self.n_ch

    def close(self):
        """Close all of the USB-6051 lines."""

        self.task_out.close()
        self.task_in.close()

    def __del__(self):
        self.close()
        super().__del__()

    def test(self):
        self._write_port_A(self.full_input, self.full_input)
        inputs = self.task_in.read()
        return inputs

    def _write_port_A(self, value: int, mask: int = 0xFF):
        self.current_output = (self.current_output & ~mask) | (value & mask)
        self.task_out.write(self.current_output, auto_start = True)
        time.sleep(self.change_delay)

    def _pulse(self, bitmask: int):
        self._write_port_A(self.current_output | bitmask, bitmask)
        time.sleep(self.change_delay)
        self._write_port_A(self.current_output & ~bitmask, bitmask)
        time.sleep(self.change_delay)

    def _select_channel(self, ch: int):
        assert 0 <= ch < self.n_ch, f"illegal channel: {ch}"
        addr = (self.AD0 if ch & 1 else 0) \
             | (self.AD1 if ch & 2 else 0) \
             | (self.AD2 if ch & 4 else 0)
        self._write_port_A(addr, self.AD0 | self.AD1 | self.AD2)

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
            self._select_channel(ch)

            for i in range(24):
                if i == 1:
                    self._write_port_A(self.WCE0, self.WCE0 | self.WCE1)
                bitval = (v >> (23 - i)) & 1 if i != 23 else 0
                self._write_port_A(self.SDATA if bitval else 0, self.SDATA)
                self._pulse(self.SCLK)

            # latch
            self._write_port_A(self.WCE1, self.WCE0 | self.WCE1)
            self._pulse(self.SCLK)
            self._pulse(self.SCLK)

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
