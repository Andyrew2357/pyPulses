from .pyvisa_device import pyvisaDevice
from .channel_adapter import ScalarChannelAdapter
from .registry import register_hardware_class

import pyvisa.constants
import numpy as np
import time
from math import ceil
from logging import Logger
from typing import Any, Dict, Tuple

@register_hardware_class("ad5764")
class ad5764(pyvisaDevice):
    """
    Class representation of the Arduino-AD5764 DC box.
    
    The instrument in question has an Arduino Uno connected to the Analog Devices 
    DAC that takes serial bus input to set 16-bit bipolar DC outputs on 8 channels.
    """

    DEFAULT_PYVISA_CONFIG = {
        'baud_rate': 115200,
        'data_bits': 8,
        'parity': pyvisa.constants.Parity.none,
        'stop_bits': pyvisa.constants.StopBits.two,
        'flow_control': pyvisa.constants.VI_ASRL_FLOW_NONE,
        'write_buffer_size': 512,
        'max_retries': 1,
        'min_interval': 0.05
    }

    hard_max_V = 10.
    hard_min_V = -10.

    def __init__(self, 
        resource_name: str, 
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_connect: bool = False,
        **kwargs,
    ):
        
        """
        Parameters
        ----------
        resource_name : str
            VISA resource name.
        registry_id : str, optional
            Name to register this instance under in the HardwareRegistry
        logger : Logger, optional
            logger used by abstractDevice.
        **kwargs
        """

        super().__init__(resource_name, registry_id, logger, skip_connect, **kwargs)

        # maximum bounds on channel values
        self.max_V = {i: self.hard_max_V for i in range(8)}
        self.min_V = {i: self.hard_min_V for i in range(8)}

        # sweep parameters
        self.max_step = 0.05
        self.wait = 0.1
        
        # mapping for controlling the instrument channels via serial bus
        self.channel_map = {
            0: (19, 0),
            1: (18, 0),
            2: (17, 0),
            3: (16, 0),
            4: (0, 19),
            5: (0, 18),
            6: (0, 17),
            7: (0, 16),
        }

        self.V = [None] * 8

        if not skip_connect:
            time.sleep(3.0) # wait a little; otherwise the first query may fail

    def _serailize_state(self) -> Dict[str, Any]:
        """
        Serialize all state needed to reconstruct and restore this device.
        """

        config = super()._serialize_state()
        config.update({
            'max_V': self.max_V,
            'min_V': self.min_V,
            'max_step': self.max_step,
            'step_wait': self.wait,
        })

    def _deserialize_state(self, state: Dict[str, Any]):
        """
        Restore device state from serialized config.
        
        Called when device already exists and we want to apply saved settings.
        """

        super()._deserialize_state(state)
        if 'max_V' in state:
            self.max_V = state['max_V']
        if 'min_V' in state:
            self.min_V = state['max_V']
        if 'max_step' in state:
            self.max_step = state['max_step']
        if 'step_wait' in state:
            self.wait = state['step_wait']

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ad5764':
        """
        Construct from serialized config.
        
        Parameters
        ----------
        config : dict
            Output from _serialize_state(), plus 'registry_id'.
        """

        # Extract required fields
        registry_id = config.pop('registry_id')
        resource_name = config.pop('resource_name')
        
        # Construct instance
        instance = cls(
            resource_name=resource_name,
            registry_id=registry_id,
            skip_connect=False,
            **config  # Remaining kwargs go to pyvisaDevice
        )        
        instance._deserialize_state(config)

        return instance

    def resolve(self, accessor: str) -> 'ad5764_channel':
        try:
            assert accessor.startswith('ch')
            ch = int(accessor[2:])
            assert 0 <= ch <= 7
        except:
            return None
        return ad5764_channel(self, accessor, ch)

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

        if ch not in self.channel_map:
            self.error(f"AD5764 does not have a channel {ch}.")
            return
        
        if V > self.max_V[ch] or V < self.min_V[ch]:
            Vt = min(self.max_V[ch], max(self.min_V[ch], V))
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

    def get_V(self, ch: int):    
        """
        Get the DC value on a given channel.

        Parameters
        ----------
        ch : int
            target channel

        Returns
        -------
        V : float
            voltage on the target channel Note: This is not a true query. It 
            simply returns what is saved on the computer. There is currently 
            no way to ask the arduino directly.
        """

        if ch not in self.channel_map:
            self.error(f"AD5764 does not have a channel {ch}.")
            return None
        
        if self.V[ch] is None:
            return self._true_query(ch)
        
        return self.V[ch]

    def set_V(self, ch, V, chatty = True):
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

        V = round(V, 10)
        if ch not in self.channel_map:
            self.error(f"AD5764 does not have a channel {ch}.")
            return
        
        if V > self.max_V[ch] or V < self.min_V[ch]:
            Vt = min(self.max_V[ch], max(self.min_V[ch], V))
            self.warn(
                f"{V} on AD5764 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        n1, n2 = self.channel_map[ch]
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
        command = bytes([255, 254, 253, n1, d1, d2, n2, d1, d2])

        try:
            # Write to instrument using PyVISA
            self.write_raw(command)
            
            # Clear the read buffer
            try:
                self.read_raw()
            except pyvisa.errors.VisaIOError:
                pass  # No data available to read
            self.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
                
            self.V[ch] = float(V)
            if chatty:
                self.info(f"Channel Settings: {self.V}")
                
        except Exception as e:
            self.error(f"Error when writing to AD5764: {e}")
            self.error(f"Attempting to refresh the connection.")
            
            # Attempt to refresh the connection
            self.refresh()

            # Write to instrument using PyVISA
            self.write_raw(command)
            
            # Clear the read buffer
            try:
                self.read_raw()
            except pyvisa.errors.VisaIOError:
                pass  # No data available to read
            self.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
                
            self.V[ch] = float(V)
            if chatty:
                self.info(f"Channel Settings: {self.V}")

    def set_channel_lim(self, ch: int, lim: Tuple[float | None, float | None]):
        """
        Manually set the voltage limits for a channel
        
        Parameters
        ----------
        ch : int
            target channel.
        lim : tuple of float or None
            channel voltage limits (low, high). If either 'low' or 'high' is 
            None, the class' extreme will be used.
        """

        vl, vh = lim

        if vh is None:
            vh = self.hard_max_V
        if vl is None:
            vl = self.hard_min_V

        vh = min(self.hard_max_V, vh)
        vl = max(self.hard_min_V, vl)
        self.max_V[ch] = float(vh)
        self.min_V[ch] = float(vl)
        
        self.info(f"Set hard channel {ch} limits to [{vl}, {vh}] V")

    def _true_query(self, ch: int):
        """Query the actual DAC voltage from the Arduino."""

        if ch not in range(8):
            self.error("Channel must be between 0 and 7.")
            return None

        # Map channel -> (chanCode1, chanCode2) exactly as in C++
        getter_map = {
            0: (147, 0),
            1: (146, 0),
            2: (145, 0),
            3: (144, 0),
            4: (0, 147),
            5: (0, 146),
            6: (0, 145),
            7: (0, 144),
        }
        nc1, nc2 = getter_map[ch]

        # First query: ask
        bufferAsk = bytes([255, 254, 253, nc1, 0, 0, nc2, 0, 0])
        self.write_raw(bufferAsk)
        time.sleep(0.01)

        # Clear any response
        while True:
            try:
                self.device.read_raw()
            except pyvisa.errors.VisaIOError:
                break

        # Second query: read
        bufferRead = bytes([255, 254, 253, 0, 0, 0, 0, 0, 0])
        self.write_raw(bufferRead)
        time.sleep(0.02)

        # Read 6 integers like readIntData() did
        buff = []
        for _ in range(6):
            try:
                line = self.read().strip()
            except pyvisa.errors.VisaIOError:
                self.error("Timeout while reading from Arduino.")
                return None
            
            if not line:
                self.error("Got empty line from Arduino.")
                return None
            
            try: 
                val = int(line)
            except ValueError:
                self.error(f"Malformed integer from Arduino: {line!r}")
                return None
            
            buff.append(val)

        # Parse according to chanCode1/chanCode2
        if nc1 == 0 and nc2 != 0:
            hiByte = buff[4]
            loByte = buff[5]
        elif nc1 != 0 and nc2 == 0:
            hiByte = buff[1]
            loByte = buff[2]
        else:
            self.error("Unexpected channel coding.")
            return None

        tmp = loByte + hiByte * 256

        # Convert back to voltage
        if 0 <= tmp <= 2**15:
            v = 10.0 * tmp / (2**15 - 1)
        elif 2**15 < tmp <= 2**16:
            v = (tmp - 2**16) * 10.0 / 2**15
        else:
            self.error("Invalid voltage read from Arduino.")
            return None

        self.V[ch] = v
        return v

    def _true_query_state(self):
        """Query the actual DAC voltages from the Arduino."""
                
        for ch in range(8):
            v = self._true_query(ch)
            if v is None:
                self.error(f"Failed to query channel {ch} voltage.")

class ad5764_channel(ScalarChannelAdapter):
    def __init__(self, parent: ad5764, accessor: str, ch: int):
        super().__init__(parent, accessor)
        self.ch = ch

    def get_output(self) -> float:
        return self._parent.get_V(self.ch)

    def set_output(self, value: float):
        self._parent.set_V(self.ch, value, chatty=False)