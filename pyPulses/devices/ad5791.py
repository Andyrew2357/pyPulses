# Note: This does not currently work. I believe there is something wrong with
# the box itself (output 7 appears to be shorted to the +15 V input).

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

class ad5791(pyvisaDevice):
    """Class interface for communicating with the AD5791 DC box."""
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
            "resource_name" : "ASRL4::INSTR",
            "baud_rate"     : 115200,
            "data_bits"     : 8,
            "parity"        : pyvisa.constants.Parity.none,
            "stop_bits"     : pyvisa.constants.StopBits.one,
            "flow_control"  : pyvisa.constants.VI_ASRL_FLOW_NONE,

            "write_buffer_size" : 512
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        # maximum bounds on channel values
        self.max_V      = 10.
        self.min_V      = -10.

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

        self.V = [0] * 8

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

        start = self.get_V(ch)
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            time.sleep(wait)
            self.set_V(ch, v, chatty = False)
        
        self.info(f"Channel Settings: {self.V}")

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

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return None
        
        # Clear command
        clear_cmd = bytes([255, 254, 251, ch, 0, 0])
        self.device.write_raw(clear_cmd)
        time.sleep(0.02)
        
        # Clear the read buffer
        try:
            self.device.read_raw()
        except pyvisa.errors.VisaIOError:
            pass  # No data available to read

        # Read command
        read_cmd = bytes([255, 254, ch, 144, 0, 0])
        self.device.write_raw(read_cmd)
        time.sleep(0.02)

        # Read response
        self.device.write_raw(read_cmd)
        time.sleep(0.02)
        response = self.device.read_bytes(6)
        time.sleep(0.02)
        
        # Clear the read buffer
        try:
            self.device.read_raw()
        except pyvisa.errors.VisaIOError:
            pass  # No data available to read

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

        self.V[ch] = float(voltage)
        return voltage

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
            
            # Clear the read buffer
            try:
                self.device.read_raw()
            except pyvisa.errors.VisaIOError:
                print("error handling used")
                pass  # No data available to read
                
            self.V[ch] = float(V)
            if chatty:
                self.info(f"Channel Settings: {self.V}")
                
        except Exception as e:
            self.error(f"Error when writing to AD5791: {e}")
            pass
