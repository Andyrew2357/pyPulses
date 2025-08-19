# Note: This does not currently work. I believe there is something wrong with
# the box itself (output 7 appears to be shorted to the +15 V input).
# TODO FIGURE OUT IF WE HAVE A WORKING VERSION OF THIS INSTRUMENT AND TEST IF 
# THIS CONTROLLER WORKS...

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
            "stop_bits"     : pyvisa.constants.StopBits.two,
            "flow_control"  : pyvisa.constants.VI_ASRL_FLOW_NONE,
            "write_buffer_size" : 512,

            'max_retries': 1,
            'min_interval': 0.05
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        # maximum bounds on channel values
        self.max_V      = 10.
        self.min_V      = -10.

        # sweep parameters
        self.max_step   = max_step
        self.wait       = wait

        # arduino reset delay after connect
        time.sleep(2.5) # attempt to mirror the original C++ implementation

        # perform true queries during initialization to maintain consistency
        # with the Arduino. These are slow, so we prefer to do them only when
        # the class is initialized.
        self.V = [0] * 8
        self._true_query_state()

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
        return self.V[ch]        

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
            tmp = round((2**19 - 1) * V / 10.7)
        else:
            tmp = round(2**20 - abs(V) / 10.7 * 2**19)

        # Convert to 20-bit binary
        hiByte_tmp = tmp // 65536
        loByte_tmp = tmp - 65536 * hiByte_tmp
        midByte = loByte_tmp // 256
        loByte = loByte_tmp - 256 * midByte
        hiByte = hiByte_tmp + 16
        
        # Create command sequence
        command = bytes([255, 254, 253, ch, hiByte, midByte, loByte])
        
        try:            
            # Write to instrument using PyVISA
            # weirdly, the old C++ appears to send this twice, waiting in 
            # between. I am just replicating that behavior here
            self.write_raw(command)
            time.sleep(0.015)
            self.write_raw(command)
            time.sleep(0.03)
            
            # Clear the read buffer
            while True:
                try:
                    self.device.read_raw()
                except pyvisa.errors.VisaIOError:
                    break  # No data available to read
                
            self.V[ch] = float(V)
            if chatty:
                self.info(f"Channel Settings: {self.V}")
                
        except Exception as e:
            self.error(f"Error when writing to AD5791: {e}")
            pass

    def _true_query(self, ch: int):
        """Query the actual DAC voltage from the Arduino."""
        
        if ch not in range(8):
            self.error("Channel must be between 0 and 7.")
            return None

        # Clear packet
        clear_cmd = bytes([255, 254, 251, ch, 0, 0])
        self.write_raw(clear_cmd)
        time.sleep(0.02)
        # Drain buffer
        while True:
            try:
                self.device.read_raw()
            except pyvisa.errors.VisaIOError:
                break

        # Read packet
        read_cmd = bytes([255, 254, ch, 144, 0, 0])
        self.write_raw(read_cmd)
        time.sleep(0.02)
        self.write_raw(read_cmd)
        time.sleep(0.02)

        # Read six ASCII integers
        buff = []
        for _ in range(6):
            try:
                line = self.read().decode(errors="ignore").strip()
            except pyvisa.errors.VisaIOError:
                self.error("Timeout while reading from Arduino.")
                return None
            if not line:
                continue
            try:
                buff.append(int(line))
            except ValueError:
                self.error(f"Malformed integer from Arduino: {line!r}")
                return None
        if len(buff) < 6:
            self.error(f"Incomplete response: {buff}")
            return None

        # Parse into 20-bit value
        midByte = buff[4]
        loByte = buff[5]
        hiByte_tmp = buff[3]
        hiByte_tmp_dac = hiByte_tmp // 16
        hiByte = hiByte_tmp - 16 * hiByte_tmp_dac
        tmp = loByte + midByte * 256 + hiByte * 65536

        if 0 <= tmp <= 2**19:
            v = 10.7 * tmp / (2**19 - 1)
        elif 2**19 < tmp <= 2**20:
            v = (tmp - 2**20) * 10.7 / 2**19
        else:
            self.error("Invalid voltage read from Arduino.")
            return None

        self.V[ch] = v
        return v

    def _true_query_state(self):
        """Query the actual DAC voltages from the Arduino."""

        for ch in range(8):
            v = self._true_query(ch)
            if v is not None:
                self.V[ch] = v
            else:
                self.error(f"Failed to query channel {ch} voltage.")
