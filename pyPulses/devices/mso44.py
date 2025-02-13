
# This class is an interface for communicating with the MSO44 digital
# oscilloscope. It implements only a small fraction of the functionality offered
# by the instrument.

from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from typing import Optional
import numpy as np

class mso44(pyvisaDevice):
    def __init__(self, logger = None):
        self.config = {
            "resource_name" : "TCPIP0::169.254.9.11::4000::SOCKET",
        }

        super().__init__(self.config, logger)

        # Set the input buffer size to 2^14 bytes
        self.device.set_buffer(pyvisa.constants.VI_READ_BUF, 16384)

        # Right now, I've only implemented this for pulling data using ASCII
        self.device.write("DATA:WIDTh 4")
        self.device.write("DATA:ENCdg ASCII")

        self.get_status()

    def set_channel(self, ch: int):
        """Set the target channel to which other commands point."""
        self.device.write(f"DATa:SOUrce CH{ch}")
        self.get_status()
        self.info(f"MSO44: Set target channel to {ch}.")

    def get_channel(self) -> int:
        """Get the target channel to which other commands point."""
        return int(self.device.query("DATa:SOUrce?")[2:])
    
    def get_waveform(self):
        """Pull a time-domain waveform off the scope."""
        data = np.fromstring(self.device.query("CURVe?"), 
                             dtype = float, sep = ',')
        v = (data - self.YOF) * self.YMU + self.YZE
        t = np.arange(v.size) * self.XIN + self.XZE
        return t, v
    
    def get_waveform_parameters(self):
        """
        Update the parameters used to convert from raw waveform data to voltage.
        """
        self.XZE = float(self.device.query("WFMP:XZE?"))
        self.XIN = float(self.device.query("WFMP:XIN?"))
        self.YZE = float(self.device.query("WFMP:YZE?"))
        self.YMU = float(self.device.query("WFMP:YMU?"))
        self.YOF = float(self.device.query("WFMP:YOF?"))

    def fast_acquisition(self, on: Optional[bool] = None) -> Optional[bool]:
        """Query or set whether fast acquisition is enabled."""
        if on is None:
            return self.device.query("FASTAcq:STATE?") == 'ON\n'
        
        self.device.write(f"FASTAcq:STATE {'ON' if on else 'OFF'}")
        self.info(f"MSO44: Turned fast acquisition {'on' if on else 'off'}.")

    def set_acquisition_mode(self, mode):
        """Set the aquisition mode of the target channel."""
        if mode not in ['SAM', 'PEAK', 'HIR', 'AVE', 'ENV']:
            self.warn(f"MSO44: {mode} is not an acquisition mode.")
            self.warn("     Try 'SAM', 'PEAK', 'HIR', 'AVE', or 'ENV'")
            self.warn(
                "(sample, peak detection, high resolution, average, or envelope)"
            )
        
        self.device.write(f"ACQuire:MODE {mode}")
        self.info(f"MSO44: Set acquisition mode to {mode}.")

    def get_acquisition_mode(self):
        """Get the acquisition mode of the target channel."""
        return self.device.query("ACQuire:MODE?")[:-1]
    
    def set_num_averages(self, N: int):
        """Set the number of averages for averaging mode."""
        self.device.write(f"ACQuire:NUMAVg {N}")
        self.info(f"MSO44: Set number of averages to {N}.")

    def get_num_averages(self) -> int:
        """Query the number of averages for averaging mode."""
        return int(self.device.query("ACQuire:NUMAVg?"))

    def get_status(self):
        """
        Miscellaneous updates that should run when we change certain parameters.
        Note: right now this is bare-bones, but it's wrapped like this for
        future-proofing.
        """
        self.get_waveform_parameters()
