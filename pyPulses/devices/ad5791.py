"""
This class is an interface for communicating with the AD5791 DC box. The
instrument in question has an Arduino Uno connected to the Analog Devices DAC
that takes serial bus input to set 20-bit unipolar DC outputs on 8 channels.
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
import pyvisa.constants
import numpy as np
from math import ceil
from typing import Optional
import time

class ad5791(pyvisaDevice):
    def __init__(self, logger = None, max_step: Optional[float] = 0.05, 
                 wait: Optional[float] = 0.1, instrument_id: Optional[str] = None):

        # configurations for pyvisa resource manager
        self.config = {
            "resource_name" : "FILL THIS IN CORRECTLY",
            "baud_rate"     : 115200,
            "data_bits"     : 8,
            "parity"        : pyvisa.constants.Parity.none,
            "stop_bits"     : pyvisa.constants.StopBits.one,
            "flow_control"  : pyvisa.constants.VI_ASRL_FLOW_NONE
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        # self.device.set_buffer(pyvisa.constants.VI_WRITE_BUF, 512)

        # maximum bounds on channel values
        self.max_V      = 10.
        self.min_V      = -10.

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

    def sweep_V(self, ch, V, max_step = None, wait = None):
        """Sweep smoothly to the DC value on a given channel."""

        if ch not in self.channel_map:
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

        start = self.get_V(ch)
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            self.set_V(ch, v, chatty = False)
            time.sleep(wait)
        
        self.info(f"Channel Settings: {self.V}")

    def get_V(self, ch):
        """
        Get the DC value on a given channel.
        Note: Unlike for the AD5791, this is a true query
        """
        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return None
        
        # Clear command
        clear_cmd = bytes([255, 254, 251, ch, 0, 0])
        self.device.write_raw(clear_cmd)
        time.sleep(0.02)
        self.device.clear()

        # Read command
        read_cmd = bytes([255, 254, ch, 144, 0, 0])
        self.device.write_raw(read_cmd)
        time.sleep(0.02)

        # Read response
        self.device.write_raw(read_cmd)
        time.sleep(0.02)
        response = self.device.read_bytes(6)
        time.sleep(0.02)
        self.device.clear()

        # Parse response bytes
        mid_byte = response[4]
        lo_byte = response[5]
        hi_byte_tmp = response[3]

        hi_byte_tmp_dac = hi_byte_tmp // 16
        hi_byte = hi_byte_tmp -16 * hi_byte_tmp_dac

        tmp = lo_byte + mid_byte * 256 + hi_byte * 65536

        # Convert to voltage
        if 0 <= tmp <= 2**19:
            voltage = 10.7 * tmp / (2**19 - 1)
        elif 2**19 < tmp <= 2**20:
            voltage = (tmp - 2**20) * 10.7 / 2**19
        else:
            self.error("AD5791: Invalid voltage read from Arduino.")
            return None

        return voltage

    def set_V(self, ch, V, chatty = True):
        """Set the DC value on a given channel."""

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5791 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        # Calculate 20-bit decimal equivalent
        if V >= 0:
            dec20 = round((2**19 - 1) * V / 10.7)
        else:
            dec20 = round(2**20 - abs(V) / 10.7 * 2**19)

        try:
            # Convert to 20-bit binary
            # Using numpy's binary_repr to ensure 20-bit representation
            bin20 = np.binary_repr(dec20, width=20)
            
            # Split into one 4 bit and two 8-bit parts; convert back to decimal
            # First 4 bits (MSB)
            d1 = int(bin20[:4], 2) + 16 # we add 16 here for some reason
            # Second 8 bits
            d2 = int(bin20[4:12], 2)
            # Third 8 bits (LSB)
            d3 = int(bin20[12:], 2)
            
            # Create command sequence
            command = bytes([255, 254, 253, ch, d1, d2, d3])
            
            # Write to instrument using PyVISA
            self.device.write_raw(command)
            self.device.clear()
                
            self.V[ch] = V
            if chatty:
                self.info(f"Channel Settings: {self.V}")
                
        except Exception as e:
            self.error(f"Error when writing to AD5791: {e}")
            pass
