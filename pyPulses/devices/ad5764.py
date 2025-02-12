# This class is an interface for communicating with the AD5764 DC box. The
# instrument in question has an Arduino Uno connected to the Analog Devices DAC
# that takes serial bus input to set 20-bit unipolar DC outputs on 8 channels.

from pyvisa_device import pyvisaDevice
import pyvisa.constants
import numpy as np
from math import ceil
import time

class ad5764(pyvisaDevice):
    def __init__(self, logger = None):

        # configurations for pyvisa resource manager
        self.config = {
            "resource_name" : "ASRLCOM3::INSTR",
            "baud_rate"     : 115200,
            "data_bits"     : 8,
            "parity"        : pyvisa.constants.Parity.none,
            "stop_bits"     : pyvisa.constants.StopBits.one,
            "flow_control"  : pyvisa.constants.VI_ASRL_FLOW_NONE
        }

        super().__init__(self.config, logger)

        self.device.set_buffer(pyvisa.constants.VI_WRITE_BUF, 512)

        # maximum bounds on channel values
        self.max_V      = 10.
        self.min_V      = 0.

        # sweep parameters
        self.max_step   = 0.05
        self.wait       = 0.1

        self.V = [0] * 8

        # mapping for controlling the instrument channels via serial bus
        self.channel_map = {
            0: (19, 0, 1, 0),
            1: (18, 0, 1, 0),
            2: (17, 0, 1, 0),
            3: (16, 0, 1, 0),
            4: (0, 19, 0, 1),
            5: (0, 18, 0, 1),
            6: (0, 17, 0, 1),
            7: (0, 16, 0, 1)
        }

    def sweep_V(self, ch, V, max_step = None, wait = None):
        """Sweep smoothly to the DC value on a given channel."""

        if ch not in self.channel_map:
            self.error(f"AD5764 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5764 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        if not max_step:
            max_step = self.max_step
        if not wait:
            wait = self.wait

        start = self.V[ch]
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            self.set_V(ch, v)
            time.sleep(wait)

    def set_V(self, ch, V):
        """Set the DC value on a given channel."""

        if ch not in self.channel_map:
            self.error(f"AD5764 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5764 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        n1, n2, m1, m2 = self.channel_map[ch]
        # Calculate 16-bit decimal equivalent
        if V >= 0:
            dec16 = round((2**15 - 1) * V / 10)
        else:
            dec16 = round(2**16 - abs(V) / 10 * 2**15)

        try:
            # Convert to 16-bit binary
            # Using numpy's binary_repr to ensure 16-bit representation
            bin16 = np.binary_repr(dec16, width=16)
            
            # Split into two 8-bit parts and convert back to decimal
            # First 8 bits (MSB)
            d1 = int(bin16[:8], 2)
            # Second 8 bits (LSB)
            d2 = int(bin16[8:], 2)
            
            # Create command sequence
            command = bytes([255, 254, 253, n1, d1*m1, d2*m1, n2, d1*m2, d2*m2])
            
            # Write to instrument using PyVISA
            self.device.write_raw(command)
            
            # Clear the read buffer
            try:
                self.device.read_raw()
            except pyvisa.errors.VisaIOError:
                pass  # No data available to read
                
            self.V[ch] = V
            self.info(f"Channel Settings: {self.V}")
                
        except Exception as e:
            self.error(f"Error when writing to AD5764: {e}")
            pass
