"""
This class is an interface for communicating with the AD5764 DC box. The
instrument in question has an Arduino Uno connected to the Analog Devices DAC
that takes serial bus input to set 16-bit unipolar DC outputs on 8 channels.
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
import pyvisa.constants
import numpy as np
from math import ceil
from typing import Optional
import time

class ad5764(pyvisaDevice):
    def __init__(self, logger = None, max_step: Optional[float] = 0.05, 
                 wait: Optional[float] = 0.1, instrument_id: Optional[str] = None):

        # configurations for pyvisa resource manager
        self.config = {
            "resource_name" : "ASRL5::INSTR",
            "baud_rate"     : 115200,
            "data_bits"     : 8,
            "parity"        : pyvisa.constants.Parity.none,
            "stop_bits"     : pyvisa.constants.StopBits.one,
            "flow_control"  : pyvisa.constants.VI_ASRL_FLOW_NONE,
            
            "write_buffer_size" : 512
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        # maximum bounds on channel values
        self.max_V      = 10.
        self.min_V      = -10.

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

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
        wait = max(wait, 0.05) # anything lower than this is error prone

        start = self.V[ch]
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            time.sleep(wait)
            self.set_V(ch, v, chatty = False)
        
        self.info(f"Channel Settings: {self.V}")

    def get_V(self, ch):    
        """
        Get the DC value on a given channel.
        Note: This is not a true query. It simply returns what is saved on the
        computer. There is currently no way to ask the arduino how you set it.
        """
        if ch not in self.channel_map:
            self.error(f"AD5764 does not have a channel {ch}.")
            return None
        
        return self.V[ch]

    def set_V(self, ch, V, chatty = True):
        """Set the DC value on a given channel."""
        V = round(V, 10)
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

        dec16 = max(0, min(dec16, 2**16 - 1))

        # Convert to 16-bit binary
        # Using numpy's binary_repr to ensure 16-bit representation
        bin16 = np.binary_repr(dec16, width=16)
        # Using format
        # bin16 = format(int(dec16), '016b')
        
        # Split into two 8-bit parts and convert back to decimal
        # First 8 bits (MSB)
        d1 = int(bin16[:8], 2)
        # Second 8 bits (LSB)
        d2 = int(bin16[8:], 2)
        
        # Create command sequence
        command = bytes([255, 254, 253, n1, d1*m1, d2*m1, n2, d1*m2, d2*m2])

        try:
            # Write to instrument using PyVISA
            self.device.write_raw(command)
            
            # Clear the read buffer
            try:
                self.device.read_raw()
            except pyvisa.errors.VisaIOError:
                pass  # No data available to read
            self.device.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
                
            self.V[ch] = float(V)
            if chatty:
                self.info(f"Channel Settings: {self.V}")
                
        except Exception as e:
            self.error(f"Error when writing to AD5764: {e}")
            self.error(f"Attempting to refresh the connection.")
            
            # Attempt to refresh the connection
            self.refresh()

            # Write to instrument using PyVISA
            self.device.write_raw(command)
            
            # Clear the read buffer
            try:
                self.device.read_raw()
            except pyvisa.errors.VisaIOError:
                pass  # No data available to read
            self.device.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
                
            self.V[ch] = float(V)
            if chatty:
                self.info(f"Channel Settings: {self.V}")
            

    def broken_true_query(self, ch):
        return

        getter_map = {
            0: (147, 0),
            1: (146, 0),
            2: (145, 0),
            3: (144, 0),
            4: (0, 147),
            5: (0, 146),
            6: (0, 145),
            7: (0, 144)
        }

        nc1, nc2 = getter_map[ch]
        ask_cmd = bytes([255, 254, 253, nc1, 0, 0, nc2, 0, 0])
        self.device.write_raw(ask_cmd)
        # Clear the read buffer
        try:
            self.device.read_raw()
        except pyvisa.errors.VisaIOError:
            pass  # No data available to read

        read_cmd = bytes([255, 254, 253, 0, 0, 0, 0, 0, 0])
        self.device.write_raw(read_cmd)
        out = self.device.read_raw(6)
        print(out)

        if ch >= 4:
            high_byte = out[4]
            low_byte = out[5]
        else:
            high_byte = out[1]
            low_byte = out[2]
            
        tmp = low_byte + high_byte * 256
        if 0 <= tmp <= 2**15:
            v = 10 * tmp / (2**15 - 1)
        elif 2**15 < tmp <= 2**16:
            v = 10 * (tmp - 2**16) / 2**15
        else:
            self.error("AD5791: Invalid voltage read from Arduino.")
            return None

        return v
