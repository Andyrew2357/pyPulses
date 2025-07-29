"""
This class is an interface for communicating with the MSO44 digital
oscilloscope. It implements only a small fraction of the functionality offered
by the instrument.
"""

from .pyvisa_device import pyvisaDevice
from typing import Tuple
import numpy as np

class mso44(pyvisaDevice):
    def __init__(self, logger = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """
        self.pyvisa_config = {
            "resource_name" : "TCPIP0::169.254.9.11::inst0::INSTR",
            "timeout"           : 10000,
            "input_buffer_size" : 16384,
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        # Right now, I've only implemented this for pulling data using ASCII
        self.write("DATA:WIDTh 4")
        self.write("DATA:ENCdg ASCII")

        self.set_channel(1)
        self._get_status()

    def run(self, on: bool):
        """
        Run/Stop acquisition.
        
        Parameters
        ----------
        on : bool
        """
        self.write(f"ACQuire:STATE {1 if on else 0}")
        self.info(f"MSO44: {'Started' if on else 'Stopped'} acquisition.")

    def is_running(self) -> bool:
        """
        Query whether acquisitions are being taken.
        
        Returns
        -------
        bool
        """
        return int(self.query("ACQuire:STATE?")) == 1

    def set_channel(self, ch: int):
        """
        Set the target channel to which other commands point.
        
        Parameters
        ----------
        ch : int
        """
        self.write(f"DATa:SOUrce CH{ch}")
        self._get_status()
        self.info(f"MSO44: Set target channel to {ch}.")

    def get_channel(self) -> int:
        """
        Get the target channel to which other commands point.
        
        Returns
        -------
        int
        """
        return int(self.query("DATa:SOUrce?")[2:])
    
    def get_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pull a time-domain waveform off the scope.
        
        Returns
        -------
        t, v : np.ndarray
        """
        data = np.fromstring(self.query("CURVe?"), 
                             dtype = float, sep = ',')
        v = (data - self.YOF) * self.YMU + self.YZE
        t = np.arange(v.size) * self.XIN + self.XZE
        return t, v
    
    def get_waveform_parameters(self):
        """
        Update the parameters used to convert from raw waveform data to voltage.
        """
        self.XZE = float(self.query("WFMP:XZE?"))
        self.XIN = float(self.query("WFMP:XIN?"))
        self.YZE = float(self.query("WFMP:YZE?"))
        self.YMU = float(self.query("WFMP:YMU?"))
        self.YOF = float(self.query("WFMP:YOF?"))

        self.info("MSO44: Updating waveform parameters...")
        self.info(f"    XZE = {self.XZE}")
        self.info(f"    XIN = {self.XIN}")
        self.info(f"    YZE = {self.YZE}")
        self.info(f"    YMU = {self.YMU}")
        self.info(f"    YOF = {self.YOF}")

    def clear_trace(self):
        """Clear the captured waveforms and all averages."""
        self.write("CLEAR")
        self.info("MSO44: Trace cleared.")

    def fast_acquisition(self, on: bool = None) -> bool | None:
        """
        Query or set whether fast acquisition is enabled.
        
        Returns
        -------
        bool
        """
        if on is None:
            return int(self.query("FASTAcq:STATE?")) == 1
        
        self.write(f"FASTAcq:STATE {'ON' if on else 'OFF'}")
        self.info(f"MSO44: Turned fast acquisition {'on' if on else 'off'}.")

    def set_acquisition_mode(self, mode):
        """
        Set the aquisition mode of the target channel.
        
        Parameters
        ----------
        mode : str
            One of {'SAM', 'PEAK', 'HIR', 'AVE', 'ENV'}
        """
        if mode not in ['SAM', 'PEAK', 'HIR', 'AVE', 'ENV']:
            self.warn(f"MSO44: {mode} is not an acquisition mode.")
            self.warn("     Try 'SAM', 'PEAK', 'HIR', 'AVE', or 'ENV'")
            self.warn(
                "(sample, peak detection, high resolution, average, or envelope)"
            )
        
        self.write(f"ACQuire:MODE {mode}")
        self.info(f"MSO44: Set acquisition mode to {mode}.")

    def get_acquisition_mode(self) -> str:
        """
        Get the acquisition mode of the target channel.
        
        Returns
        -------
        mode : str
            {'SAM', 'PEAK', 'HIR', 'AVE', 'ENV'}
        """
        return self.query("ACQuire:MODE?")[:-1]
    
    def set_num_averages(self, N: int):
        """
        Set the number of averages for averaging mode.
        
        Parameters
        ----------
        N : int
        """
        self.write(f"ACQuire:NUMAVg {N}")
        self.info(f"MSO44: Set number of averages to {N}.")

    def get_num_averages(self) -> int:
        """
        Query the number of averages for averaging mode.
        
        Returns
        -------
        int
        """
        return int(self.query("ACQuire:NUMAVg?"))

    def _get_status(self):
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
