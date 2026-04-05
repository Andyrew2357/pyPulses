from .pyvisa_device import pyvisaDevice
from .channel_adapter import ScalarChannelAdapter
from .registry import register_hardware_class

import pyvisa.constants
import time
from math import floor
from logging import Logger
from typing import Any, Dict

@register_hardware_class("ad9854")
class ad9854(pyvisaDevice):
    """
    Class representation of the Arduino-AD5984 AC box.
    
    The instrument in question has an Arduino Uno connected to two Analog 
    Devices 9854 evaluation boards that takes serial bus input to set AC outputs 
    on 2 channels, each of which consist of an X and Y output 90 degrees out of 
    phase. The channel phases can be controlled independently. The amplitude range
    is 0 to 120 mV with 12 bit resolution, the phase resolution is 14 bits, and 
    the frequency range is up to ~ 50 MHz with 48 bit resolution.
    """

    DEFAULT_PYVISA_CONFIG = {
        'baud_rate': 19200,
        'data_bits': 8,
        'parity': pyvisa.constants.Parity.none,
        'stop_bits': pyvisa.constants.StopBits.two,
        'flow_control': pyvisa.constants.VI_ASRL_FLOW_NONE,
        'write_buffer_size': 512,
        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.05
    }

    # Device parameters
    vmax = 0.120   # Maximum amplitude [V]
    refclk = 25    # Reference clock [MHz]
    sysclk = 4 * refclk * 1e6  # System clock [Hz]
    Nfreq = 48     # Frequency resolution bits
    Nphase = 14    # Phase resolution bits
    Namp = 12      # Amplitude resolution bits

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

        self.amplitudes = {
            (1, 'X'): 0,
            (1, 'Y'): 0,
            (2, 'X'): 0,
            (2, 'Y'): 0
        }
        self.phase = 0
        self.freq = 0

        if not skip_connect:
            self.reset_arduino()
            time.sleep(2.0)
            self.master_reset()
            time.sleep(1.0)
            self.configure_control_register()

    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize all state needed to reconstruct and restore this device.
        """

        config = super()._serialize_state()
        sanitized_amplitudes = {
            f"{chan}{chip}": val
            for (chip, chan), val in self.amplitudes.items()
        }
        config.update({
            'freq': self.freq,
            'phase': self.phase,
            'amplitudes': sanitized_amplitudes,
        })
        
        return config

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """
        Restore device state from serialized config.
        
        Called when device already exists and we want to apply saved settings.
        """

        super()._deserialize_state(state)
        
        if 'freq' in state:
            self.set_frequency(state['freq'])
        if 'phase' in state:
            self.set_phase(state['phase'])
        if 'amplitudes' in state:
            for k, v in state['amplitudes'].items():
                chip = int(k[1])
                chan = k[0]
                self.set_amplitude(chip, chan, v)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ad9854':
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
    
    def resolve(self, accessor: str) -> ScalarChannelAdapter:
        if accessor == 'X1':
            return ad9854_amplitude_channel(self, 'X1', 1, 'X')
        if accessor == 'Y1':
            return ad9854_amplitude_channel(self, 'Y1', 1, 'Y')
        if accessor == 'X2':
            return ad9854_amplitude_channel(self, 'X2', 2, 'X')
        if accessor == 'Y2':
            return ad9854_amplitude_channel(self, 'Y2', 2, 'Y')
        if accessor == 'X1_unitless':
            return ad9854_amplitude_unitless_channel(self, 'X1_unitless', 1, 'X')
        if accessor == 'Y1_unitless':
            return ad9854_amplitude_unitless_channel(self, 'Y1_unitless', 1, 'Y')
        if accessor == 'X2_unitless':
            return ad9854_amplitude_unitless_channel(self, 'X2_unitless', 2, 'X')
        if accessor == 'Y2_unitless':
            return ad9854_amplitude_unitless_channel(self, 'Y2_unitless', 2, 'Y')
        if accessor == 'phase':
            return ad9854_phase_channel(self)
        if accessor == 'freq':
            return ad9854_frequency_channel(self)
        raise ValueError(f"Unknown accessor: {accessor}")

    def set_frequency(self, f: float):
        """
        Set output frequency for both chips simultaneously.
        
        Parameters
        ----------
        f : float
            frequency in Hz.
        """

        if f > self.sysclk/2:
            self.error(f"Frequency ({f} Hz) exceeds maximum {self.sysclk/2} Hz.")
            return
        
        ftw = int((f * 2**self.Nfreq) / self.sysclk)
        bytes_ = [(ftw >> (8*i)) & 0xFF for i in range(5, -1, -1)]
        command = bytes([255, 254, 253, 12, 2] + bytes_)
        self._write_command(command)
        time.sleep(0.2)
        self.freq = f

        self.info(f"Set frequency to {f} Hz")
        return f
    
    def get_frequency(self) -> float:
        """
        Returns
        -------
        frequency : float
        """
        return self.freq

    def set_phase(self, phase: float) -> float:
        """
        Set phase offset for specified chip.
        
        Parameters
        ----------
        phase : float
            phase offset in degrees.
        """

        chip = 2        
        phase = phase % 360 # Wrap to [0, 360)
        ptw = int(floor(phase / 360 * 2**self.Nphase))
        if ptw == (2**self.Nphase - 1):
            ptw = 0
        pw1, pw2 = (ptw >> 8) & 0xFF, ptw & 0xFF
        command = bytes([255, 254, 253, chip, 0, pw1, pw2, 0, 0, 0, 0])
        self._write_command(command)
        time.sleep(0.2)
        self.phase = ptw * 360 / 2**self.Nphase

        self.info(f"Set phase offset to {phase} degrees.")
        return self.phase

    def get_phase(self) -> float:
        """
        Returns
        -------
        phase : float
            phase offset in degrees.
        """
        return self.phase
    
    def set_amplitude_unitless(self, chip: int, channel: str, A: float) -> float:
        if not chip in [1, 2]:
            self.error(f"Chip ({chip}) must be either 1 or 2.")
            return

        if channel == 'X':
            chnum = 8
        elif channel == 'Y':
            chnum = 9
        else:
            self.error(f"Channel ({channel}) must be either 'X' or 'Y'.")
            return
        
        amplitude = max(0, min(A, 1.0))
        vtw = int(amplitude * (2**self.Namp - 1))
        vw1, vw2 = (vtw >> 8) & 0xFF, vtw & 0xFF
        command = bytes([255, 254, 253, chip, chnum, vw1, vw2, 0, 0, 0, 0])
        self._write_command(command)
        time.sleep(0.2)
        self.amplitudes[(chip, channel)] = amplitude

        self.info(f"Set {channel}{chip} amplitude to {amplitude} V rms.")
        return amplitude
    
    def get_amplitude_unitless(self, chip: int, channel: str) -> float | None:
        if chip not in [1, 2] or channel not in ['X', 'Y']:
            self.error(f"chip = {chip}, channel = {channel} is invalid.")
            return
        
        return self.amplitudes[(chip, channel)]

    def set_amplitude(self, chip: int, channel: str, Vrms: float) -> float:
        """
        Set amplitude for specified chip and channel.
        
        Parameters
        ----------
        chip : int
            chip number (1 or 2).
        channel : str
            channel name ('X' or 'Y').
        Vrms : float
            RMS voltage.
        """

        amplitude = self.set_amplitude_unitless(chip, channel, Vrms / self.vmax)
        return self.vmax * amplitude

    def get_amplitude(self, chip: int, channel: str) -> float | None:
        """
        Parameters
        ----------
        chip : int
            chip number (1 or 2).
        channel : str
            channel name ('X' or 'Y').

        Returns
        -------
        Vrms : float
            RMS voltage
        """
        if chip not in [1, 2] or channel not in ['X', 'Y']:
            self.error(f"chip = {chip}, channel = {channel} is invalid.")
            return
        
        return self.vmax * self.amplitudes[(chip, channel)]

    def master_reset(self):
        """
        Perform master reset of the system. This has to do with setting the 
        reference clock.
        """
        # this has something to do with setting the reference clock.
        # won't think about it too much until we have to build a new box...
        self._write_command(bytes([255, 254, 253, 12, 55, 1, 2, 3, 4, 5, 6]),
                            speedy = False)
        self.info("Perfomed master reset of AC box.")
        time.sleep(0.5)

    def configure_control_register(self):
        """Configure device control registers."""
        self._write_command(bytes([255, 254, 253, 12, 7, 16, 68, 0, 32, 0, 0]),
                            speedy = False)
        self.info("Configured AC box control register.")
        time.sleep(1.0)

    def _write_command(self, command: bytes, speedy: bool = True):
        """Internal method to handle command writing with error recovery."""
        try:
            self.write_raw(command)
            if not speedy: 
                self._flush_buffers()
        except Exception as e:
            self.error(f"Write error: {e}, attempting reconnect...")
            self.refresh()
            self.write_raw(command)
            if not speedy:
                self._flush_buffers()

    def _flush_buffers(self):
        """Clear communication buffers."""
        # try:
        #     self.read_raw()
        # except pyvisa.errors.VisaIOError:
        #     pass
        # self.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
        pass

    def reset_arduino(self):
        try:
            self.device.dtr = False
            time.sleep(0.1)
            self.device.dtr = True
            time.sleep(2.0)
            self.info("Arduino reset via DTR toggle.")
        except Exception as e:
            self.warn(f"Failed to reset Arduino via DTR: {e}")

class ad9854_amplitude_channel(ScalarChannelAdapter): 
    def __init__(self, parent: ad9854, accessor: str, chip: int, chan: str):
        super().__init__(parent, accessor)
        self.chip = chip
        self.chan = chan

    def get_output(self) -> float:
        return self._parent.get_amplitude(self.chip, self.chan)

    def set_output(self, value: float):
        self._parent.set_amplitude(self.chip, self.chan, value)

class ad9854_amplitude_unitless_channel(ScalarChannelAdapter):
    """
    ScalarChannelAdapter for ad9854 amplitude in unitless [0, 1] DAC units.
 
    Useful for numerical conditioning when the amplitude range is known
    and absolute voltage calibration is handled separately.
    """
    def __init__(self, parent: 'ad9854', accessor: str, chip: int, chan: str):
        super().__init__(parent, accessor)
        self.chip = chip
        self.chan = chan
 
    def get_output(self) -> float:
        return self._parent.get_amplitude_unitless(self.chip, self.chan)
 
    def set_output(self, value: float):
        self._parent.set_amplitude_unitless(self.chip, self.chan, value)

class ad9854_phase_channel(ScalarChannelAdapter):
    def __init__(self, parent: ad9854):
        super().__init__(parent, "phase")

    def get_output(self) -> float:
        return self._parent.get_phase()
    
    def set_output(self, value: float):
        self._parent.set_phase(value)

class ad9854_frequency_channel(ScalarChannelAdapter):
    def __init__(self, parent: ad9854):
        super().__init__(parent, "freq")

    def get_output(self) -> float:
        return self._parent.get_frequency()
    
    def set_output(self, value: float):
        self._parent.set_frequency(value)