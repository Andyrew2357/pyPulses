# TODO This is severely lacking in functionality, in part because I don't expect
# to use this scope for long

from .pyvisa_device import pyvisaDevice
from .channel_adapter import ScopeChannelAdapter
from .registry import register_hardware_class

import time
import numpy as np
from logging import Logger
from typing import Tuple

@register_hardware_class("mso44")
class mso44(pyvisaDevice):
    """
    Class representation of the MSO44 digital
    oscilloscope. It implements only a small fraction of the functionality offered
    by the instrument.
    """

    DEFAULT_PYVISA_CONFIG = {
        'timeout': 10000,
        'input_buffer_size': 16384,
        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.05,
    }

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

        if not skip_connect:
            # Right now, I've only implemented this for pulling data using ASCII
            self.write("DATA:WIDTh 2")
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
    
    def get_waveform(self) -> np.ndarray:
        """
        Pull a time-domain waveform off the scope.
        
        Returns
        -------
        t, v : np.ndarray
        """
        try:
            data = np.fromstring(self.query("CURVe?"), dtype=float, sep=',')
        except Exception as e:
            self.warn(f"MSO44: CURVe? failed ({e}), flushing and retrying...")
            self.write("*CLS")
            self.query("*OPC?")
            data = np.fromstring(self.query("CURVe?"), dtype=float, sep=',')
        
        v = (data - self.YOF) * self.YMU + self.YZE
        return v
    
    def get_waveform_parameters(self):
        """
        Update the parameters used to convert from raw waveform data to voltage.
        """
        self.HPOS = 1e-2 *float(self.query("HORizontal:POSition?"))
        self.XZE = float(self.query("WFMP:XZE?"))
        self.XIN = float(self.query("WFMP:XIN?"))
        self.YZE = float(self.query("WFMP:YZE?"))
        self.YMU = float(self.query("WFMP:YMU?"))
        self.YOF = float(self.query("WFMP:YOF?"))

        self.info("MSO44: Updating waveform parameters...")
        self.info(f"    HPOS = {self.HPOS}")
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

    def resolve(self, accessor: str) -> mso44_trace_channel:
        if accessor == 'trace':
            return mso44_trace_channel(self)

class mso44_trace_channel(ScopeChannelAdapter):
    def __init__(self, parent: mso44):
        super().__init__(parent, 'trace')

    def __call__(self) -> Tuple[np.ndarray, float, float]:
        self._parent.run(False)
        self._parent.clear_trace()
        self._parent.run(True)
        time.sleep(0.6)
        v = self._parent.get_waveform()
        return v, self._parent.XIN, -self._parent.HPOS * self._parent.XIN * v.size