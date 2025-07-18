"""
This class is an interface for communicating with the AD5984 AC box. The
instrument in question has an Arduino Uno connected to two Analog Devices 9854
evaluation boards that takes serial bus input to set AC outputs on 2 channels,
each of which consist of an X and Y output 90 degrees out of phase. The channel
phases can be controlled independently. The amplitude range is 0 to 120 mV with
12 bit resolution, the phase resolution is 14 bits, and the frequency range is
up to ~ 50 MHz with 48 bit resolution. 
"""

from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from math import floor

class ad9854(pyvisaDevice):
    """Class interface for communicating with the AD5984 AC box."""
    def __init__(self, logger=None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """

        self.pyvisa_config = {
            "resource_name": "ASRL3::INSTR",
            "baud_rate": 19200,
            "data_bits": 8,
            "parity": pyvisa.constants.Parity.none,
            "stop_bits": pyvisa.constants.StopBits.one,
            "flow_control": pyvisa.constants.VI_ASRL_FLOW_NONE,
            "write_buffer_size": 512
        }
            
        super().__init__(self.pyvisa_config, logger, instrument_id)
        
        self.master_reset()
        self.configure_control_register()

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

    def set_phase(self, chip: int, phase: float):
        """
        Set phase offset for specified chip.
        
        Parameters
        ----------
        chip : int
            chip number (1 or 2).
        phase : float
            phase offset in degrees.
        """

        if not chip in [1, 2]:
            self.error(f"Chip ({chip}) must be either 1 or 2.")
            return
        
        phase = phase % 360 # Wrap to [0, 360)
        ptw = int(floor(phase / 360 * 2**self.Nphase))
        pw1, pw2 = (ptw >> 8) & 0xFF, ptw & 0xFF
        command = bytes([255, 254, 253, chip, 0, pw1, pw2, 0, 0, 0, 0])
        self._write_command(command)
        self.phase = phase

        self.info(f"Set phase offset to {phase} degrees.")
        return phase

    def get_phase(self) -> float:
        """
        Returns
        -------
        phase : float
            phase offset in degrees.
        """
        return self.phase

    def set_amplitude(self, chip: int, channel: str, Vrms: float):
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
        self.amplitudes[(chip, channel)] = amplitude

        self.info(f"Set {channel}{chip} amplitude to {amplitude} V rms.")
        return amplitude

    def get_amplitude(self, chip, channel) -> float | None:
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
        
        return self.amplitudes[(chip, channel)]

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

    def configure_control_register(self):
        """Configure device control registers."""
        self._write_command(bytes([255, 254, 253, 12, 7, 16, 68, 0, 32, 0, 0]),
                            speedy = False)
        self.info("Configured AC box control register.")

    def _write_command(self, command: bytes, speedy: bool = True):
        """Internal method to handle command writing with error recovery."""
        try:
            self.device.write_raw(command)
            if not speedy: 
                self._flush_buffers()
        except Exception as e:
            self.error(f"Write error: {e}, attempting reconnect...")
            self.refresh()
            self.device.write_raw(command)
            if not speedy:
                self._flush_buffers()

    def _flush_buffers(self):
        """Clear communication buffers."""
        try:
            self.device.read_raw()
        except pyvisa.errors.VisaIOError:
            pass
        self.device.flush(pyvisa.constants.VI_WRITE_BUF_DISCARD)
