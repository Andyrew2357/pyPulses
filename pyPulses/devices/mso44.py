"""
This class is an interface for communicating with the MSO44 digital
oscilloscope. It implements only a small fraction of the functionality offered
by the instrument.
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
# import pyvisa.constants
from typing import Optional, Tuple
import numpy as np

class mso44(pyvisaDevice):
    def __init__(self, logger: Optional[str] = None, 
                 instrument_id: Optional[str] = None):
        self.config = {
            "resource_name" : "TCPIP0::169.254.9.11::inst0::INSTR",

            "timeout"           : 10000,
            "input_buffer_size"  : 16384,
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        # Right now, I've only implemented this for pulling data using ASCII
        self.device.write("DATA:WIDTh 4")
        self.device.write("DATA:ENCdg ASCII")

        self.set_channel(1)
        self.get_status()

    def run(self, on: bool):
        """Run/Stop acquisition"""
        self.device.write(f"ACQuire:STATE {1 if on else 0}")
        self.info(f"MSO44: {'Started' if on else 'Stopped'} acquisition.")

    def is_running(self) -> bool:
        """Query whether acquisitions are being taken."""
        return int(self.device.query("ACQuire:STATE?")) == 1

    def set_channel(self, ch: int):
        """Set the target channel to which other commands point."""
        self.device.write(f"DATa:SOUrce CH{ch}")
        self.get_status()
        self.info(f"MSO44: Set target channel to {ch}.")

    def get_channel(self) -> int:
        """Get the target channel to which other commands point."""
        return int(self.device.query("DATa:SOUrce?")[2:])
    
    def get_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
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

        self.info("MSO44: Updating waveform parameters...")
        self.info(f"    XZE = {self.XZE}")
        self.info(f"    XIN = {self.XIN}")
        self.info(f"    YZE = {self.YZE}")
        self.info(f"    YMU = {self.YMU}")
        self.info(f"    YOF = {self.YOF}")

    def clear_trace(self):
        """Clear the captured waveforms and all averages."""
        self.device.write("CLEAR")
        self.info("MSO44: Trace cleared.")

    def fast_acquisition(self, on: Optional[bool] = None) -> Optional[bool]:
        """Query or set whether fast acquisition is enabled."""
        if on is None:
            return int(self.device.query("FASTAcq:STATE?")) == 1
        
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
        running = self.is_running()
        if not running:
            self.run(True)

        try:
            self.get_waveform_parameters()
        except:
            self.warn("MSO44: Could not get waveform parameters.")

        if not running:
            self.run(False)
