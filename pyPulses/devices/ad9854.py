"""
This class is an interface for communicating with the AD5984 AC box. The
instrument in question has an Arduino Uno connected to two Analog Devices 9854
evaluation boards that takes serial bus input to set AC outputs on 2 channels,
each of which consist of an X and Y output 90 degrees out of phase. The channel
phases can be controlled independently. The amplitude range is 0 to 120 mV with
12 bit resolution, the phase resolution is 14 bits, and the frequency range is
up to ~ 50 MHz with 48 bit resolution. 
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from math import modf
from typing import Optional

class ad9854(pyvisaDevice):
    def __init__(self, logger=None, instrument_id: Optional[str] = None):
        self.config = {
            "resource_name": "ASRL3::INSTR",
            "baud_rate": 19200,
            "data_bits": 8,
            "parity": pyvisa.constants.Parity.none,
            "stop_bits": pyvisa.constants.StopBits.one,
            "flow_control": pyvisa.constants.VI_ASRL_FLOW_NONE,
            "write_buffer_size": 512
        }
        
        if instrument_id:
            self.config["resource_name"] = instrument_id
            
        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)
        
        # Device parameters
        self.vmax = 0.120   # Maximum amplitude [V]
        self.refclk = 25    # Reference clock [MHz]
        self.sysclk = 4 * self.refclk * 1e6  # System clock [Hz]
        self.Nfreq = 48     # Frequency resolution bits
        self.Nphase = 14    # Phase resolution bits
        self.Namp = 12      # Amplitude resolution bits

        self.amplitudes = {
            (1, 'X'): 0,
            (1, 'Y'): 0,
            (2, 'X'): 0,
            (2, 'Y'): 0
        }
        self.phase = 0
        self.freq = 0

    def set_frequency(self, f: float):
        """Set output frequency for both chips simultaneously."""
        if f > self.sysclk/2:
            self.error(f"Frequency ({f} Hz) exceeds maximum {self.sysclk/2} Hz.")
            return
        
        ftw = int((f * 2**self.Nfreq) / self.sysclk)
        bytes_ = [(ftw >> (8*i)) & 0xFF for i in range(5, -1, -1)]
        command = bytes([255, 254, 253, 12, 2] + bytes_)
        self._write_command(command)

        self.info(f"Set frequency to {f} Hz")
        return f
    
    def get_frequency(self) -> float:
        return self.freq

    def set_phase(self, chip: int, phase: float):
        """Set phase offset for specified chip (1 or 2)."""
        if not chip in [1, 2]:
            self.error(f"Chip ({chip}) must be either 1 or 2.")
            return
        
        phase = modf(phase/360)[0] * 360  # Wrap to [0, 360)
        ptw = int((phase / 360) * 2**self.Nphase)
        pw1, pw2 = (ptw >> 8) & 0xFF, ptw & 0xFF
        command = bytes([255, 254, 253, chip, 0, pw1, pw2, 0, 0, 0, 0])
        self._write_command(command)

        self.info(f"Set phase offset to {phase} degrees.")
        return phase

    def get_phase(self) -> float:
        return self.phase

    def set_amplitude(self, chip: int, channel: str, Vrms: float):
        """Set amplitude for specified chip (1/2) and channel (X/Y)."""
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
        
        amplitude = max(0, min(Vrms, self.vmax))
        vtw = int((amplitude / self.vmax) * (2**self.Namp - 1))
        vw1, vw2 = (vtw >> 8) & 0xFF, vtw & 0xFF
        command = bytes([255, 254, 253, chip, chnum, vw1, vw2, 0, 0, 0, 0])
        self._write_command(command)

        self.info(f"Set {channel}{chip} amplitude to {amplitude} V rms.")
        return amplitude

    def get_amplitude(self, chip, channel) -> Optional[float]:
        if chip not in [1, 2] or channel not in ['X', 'Y']:
            self.error(f"chip = {chip}, channel = {channel} is invalid.")
            return
        
        return self.amplitudes[(chip, channel)]

    def master_reset(self):
        """Perform master reset of the system."""
        self._write_command(bytes([255, 254, 253, 12, 55, 1, 2, 3, 4, 5, 6]))
        self.info("Perfomed master reset of AC box.")

    def configure_control_register(self):
        """Configure device control registers."""
        self._write_command(bytes([255, 254, 253, 12, 7, 16, 68, 0, 32, 0, 0]))
        self.info("Configured AC box control register.")

    def _write_command(self, command: bytes):
        """Internal method to handle command writing with error recovery."""
        try:
            self.device.write_raw(command)
            self._flush_buffers()
        except Exception as e:
            self.error(f"Write error: {e}, attempting reconnect...")
            self.refresh()
            self.device.write_raw(command)
            self._flush_buffers()

    def _flush_buffers(self):
        """Clear communication buffers."""
        try:
            self.device.read_raw()
        except pyvisa.errors.VisaIOError:
            pass
        self.device.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
