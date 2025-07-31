from .srs_lockin_base import SRSLockin
from .pyvisa_device import parse_IEEE_488_2


import numpy as np
import time
from math import log2
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

class sr830(SRSLockin):
    """Class interface for controlling the SR830 DSP Lock-in"""
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
            "resource_name"     : "",
            "output_buffer_size": 512,

            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
        }

        self.output_map = {
            'X': 1,
            'Y': 2,
            'R': 3,
            'T': 4
        }

        x = [5e-9, 2e-9, 1e-9]
        self.sens_vals = np.array(
            [1e0] + [val 
                for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
                for val in [factor*x[0], factor*x[1], factor*x[2]]]
        )[:-1][::-1]

        x = [1e-6, 3e-6]
        tau_vals = []
        for factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
            tau_vals.extend([factor*x[0], factor*x[1]])
        self.tau_vals = np.array(tau_vals)

        super().__init__(logger, instrument_id)

    def reference_input_impedance(self, imp: str = None) -> float | None:
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")

    def sine_output_offset(self, V: float = None) -> float | None:
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")
    
    def input_mode_current(self, I_mode: bool = None) -> bool | None:
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")
    
    def input_range(self, V: float = None) -> float | None:
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")
    
    def current_gain(self, gan: str = None) -> str | None:
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")
    
    def get_input_level_status(self) -> int:
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")
    
    def auto_range(self):
        """Not available for SR830."""
        raise AttributeError("SR830 does not offer this functionality.")

class sr844(SRSLockin):
    """Class interface for controlling the SR844 DSP Lock-in"""
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
            "resource_name"     : "",
            "output_buffer_size": 512,

            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
        }

        self.cmd_map = {
            'icpl': "INPZ",
            'oaux': "AUXI",
            'auxv': "AUXO"
        }

        self.output_map = {
            'X': 1,
            'Y': 2,
            'R': 3,
            'T': 5
        }

        x = [3e-9, 1e-9]
        self.sens_vals = np.array(
            [1e0] + [val 
                for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2]
                for val in [factor*x[0], factor*x[1]]]
        )[::-1]

        x = [1e-6, 3e-6]
        tau_vals = []
        for factor in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
            tau_vals.extend([factor*x[0], factor*x[1]])
        print(len(tau_vals))
        self.tau_vals = np.array(tau_vals)

        super().__init__(logger, instrument_id)

    def detection_harmonic(self, harm: int = None) -> int | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def reference_trigger(self, condition: str = None) -> str | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def sine_output_amplitude(self, V: float = None) -> float | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def sine_output_offset(self, V: float = None) -> float | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def input_configuration(self, src: str = None) -> str | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def input_shield_grounded(self, gnd: bool = None) -> bool | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def input_mode_current(self, I_mode: bool = None) -> bool | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")
    
    def input_range(self, V: float = None) -> float | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def current_gain(self, gan: str = None) -> str | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")
    
    def line_notch_filter(self, setting: str = None) -> str | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def get_input_level_status(self) -> int:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")
    
    def low_pass_filter_slope(self, slope: str = None) -> str | None:
        """
        Set or query the low-pass filter slope.

        Parameters
        ----------
        slope : str, optional
            One of {'0dB/oct', '6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct'}.

        Returns
        -------
        str or None
        """

        if slope is None:
            return ['0dB/oct', '6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct'][
                int(self.query(f"{self.cmd_map['ofsl']}?"))]
        opt = {'0dB/oct': 0, '6dB/oct': 1, '12dB/oct': 2, 
               '18dB/oct': 3, '24dB/oct': 4}
        self.write(f"{self.cmd_map['ofsl']} {opt[slope]}")
        self.info(f"Set low-pass filter slope to {slope}.")

    def sync_filter_state(self, on: bool = None) -> bool | None:
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

    def auto_range(self):
        """Not available for SR844."""
        raise AttributeError("SR844 does not offer this functionality.")

class sr850(SRSLockin):
    """Class interface for controlling the SR850 DSP Lock-in"""
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
            "resource_name"     : "",
            "output_buffer_size": 512,

            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
        }

        self.cmd_map = {
            'fmod': "RSRC"
        }

        self.output_map = {
            'X': 1,
            'Y': 2,
            'R': 3,
            'T': 4
        }

        x = [5e-9, 2e-9, 1e-9]
        self.sens_vals = np.array(
            [1e0] + [val 
                for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
                for val in [factor*x[0], factor*x[1], factor*x[2]]]
        )[:-1][::-1]

        x = [1e-6, 3e-6]
        tau_vals = []
        for factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
            tau_vals.extend([factor*x[0], factor*x[1]])
        self.tau_vals = np.array(tau_vals)

        super().__init__(logger, instrument_id)

    def reference_input_impedance(self, imp: str = None) -> float | None:
        """Not available for SR850."""
        raise AttributeError("SR850 does not offer this functionality.")

    def sine_output_offset(self, V: float = None) -> float | None:
        """Not available for SR850."""
        raise AttributeError("SR850 does not offer this functionality.")
    
    def input_mode_current(self, I_mode: bool = None) -> bool | None:
        """Not available for SR850."""
        raise AttributeError("SR850 does not offer this functionality.")

    def input_range(self, V: float = None) -> float | None:
        """Not available for SR850."""
        raise AttributeError("SR830 does not offer this functionality.")

    def get_input_level_status(self) -> int:
        """Not available for SR850."""
        raise AttributeError("SR850 does not offer this functionality.")
    
    def auto_range(self):
        """Not available for SR850."""
        raise AttributeError("SR850 does not offer this functionality.")

@dataclass
class srs_acquisition:
    ready       : bool  = False # Is the acquisition set up?
    running     : bool  = False # Is it currently running?
    buffer_size : int   = None  # buffer size in kB
    currsamp    : int   = None  # current samples taken
    sampint     : float = None  # sampling interval in seconds
    data_config : str   = None  # configuration of returned data
    timeout     : float = None  # fallback timeout

class sr860(SRSLockin):
    """Class interface for controlling the SR860 DSP Lock-in"""
    def __init__(self, logger = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """

        self._name = 'SR860'

        self.pyvisa_config = {
            "resource_name"     : "USB0::0xB506::0x2000::003931::INSTR",
            "output_buffer_size": 512,

            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
        }

        self.cmd_map = {
            'fmod': "RSRC",
            'rslp': "RTRG",
            'igan': "ICUR",
            'sens': "SCAL",
            'agan': "ASCL"
        }

        self.output_map = {
            'X': 0,
            'Y': 1,
            'R': 2,
            'T': 3
        }

        x = [5e-9, 2e-9, 1e-9]
        self.sens_vals = np.array(
            [1e0] + [val 
                for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
                for val in [factor*x[0], factor*x[1], factor*x[2]]]
        )
        
        x = [1e-6, 3e-6]
        tau_vals = []
        for factor in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
            tau_vals.extend([factor*x[0], factor*x[1]])
        self.tau_vals = np.array(tau_vals)

        self.irng_vals = np.array([1, 0.3, 0.1, 0.03, 0.01])

        super().__init__(logger, instrument_id)

        self._acquisition = srs_acquisition()

    def input_configuration(self, src: str = None) -> str | None:
        if src == 'I':
            raise ValueError(f"{self._name} does not support setting 'I'.")
        return super().input_configuration(src)

    def line_notch_filter(self, setting: str = None) -> str | None:
        """Not available for SR860."""
        raise AttributeError(f"{self._name} does not offer this functionality.")

    def is_input_overloaded(self) -> bool:
        """
        Check if the input signal is overloaded.
        
        Returns
        -------
        bool
        """
        status = int(self.query("CUROVLDSTAT?"))
        return bool((status >> 4) & 1) # extract bit 4
    
    def is_reference_unlocked(self) -> bool:
        """
        Check if the external reference is unlocked.
        
        Returns
        -------
        bool
        """
        status = int(self.query("CUROVLDSTAT?"))
        return bool((status >> 3) & 1) # extract bit 3

    def setup_data_acquisition(self, buffer_size: int, config: str = 'XY',
                               sample_rate: float = None, timeout = 10.0
                               ) -> float:
        """
        Configure the data acquisition system.

        Parameters
        ----------
        buffer_size : int
            size of the buffer in kilobytes.
        config : str
            One of {'X', 'XY', 'RT', 'XYRT'}.
        sample_rate : float
            desired sample rate in Hz.

        Returns
        -------
        sample_rate : float
            the actual sample rate used.
        """

        if self._acquisition.running:
            raise RuntimeError(
                "Cannot configure an acquisition while one is already running."
            )

        if not config in ['X', 'XY', 'RT', 'XYRT']:
            raise ValueError(
                "Unsupported config; must be 'X', 'XY', 'RT' or 'XYRT'."
            )
        
        if buffer_size < 1 or buffer_size > 4096:
            raise ValueError(
                "Buffer size must be between 1 and 4096 kB."
            )
        
        self.write(f"CAPTURECFG {config}")
        self.write(f"CAPTURELEN {buffer_size}")

        max_sample_rate = float(self.query("CAPTURERATEMAX?"))
        if sample_rate is not None:
            n = int(log2(max_sample_rate / sample_rate))
            if n < 0 or n > 20:
                raise ValueError(
                    f"Sample rate {sample_rate} is out of range."
                )
        else:
            n = 0

        sample_rate = max_sample_rate / (2**n)
        self.write(f"CAPTURERATE {n}")

        self._acquisition.buffer_size = buffer_size
        self._acquisition.currsamp = 0
        self._acquisition.sampint = 1/sample_rate
        self._acquisition.data_config = config
        self._acquisition.timeout = timeout
        self._acquisition.ready = True
        self._acquisition.running = False

        self.info(f"Set up data acquisition with parameters:")
        self.info(f"    config      = {config}")
        self.info(f"    buffer_size = {buffer_size} kB")
        self.info(f"    sample_rate = {sample_rate} Hz")
        return sample_rate
    
    def _start_acquisition(self):
        """Start data acquisition."""
        if not self._acquisition.ready:
            raise RuntimeError(
                "Cannot start acquisition which has not been configured."
            )
        if self._acquisition.running:
            raise RuntimeError(
                "Cannot start acquisition while one is already running."
            )
        
        self.write("CAPTURESTART ONE, IMM")
        self._acquisition.running = True
        self.info("Started data acquisition.")

    def _stop_acquisition(self):
        """Stop data acquisition."""
        if not self._acquisition.running:
            raise RuntimeError(
                "Cannot stop acquisition which is not currently running."
            )
        self.write("CAPTURESTOP")
        self._acquisition.running = False
        self.info("Stopped data acquisition.")

    def _get_buffered_data(self, max_tries: int = 5) -> np.ndarray:
        """
        Get buffered data.

        Returns
        -------
        np.ndarray
        """

        if not self._acquisition.running:
            raise RuntimeError(
                "Cannot get buffered from an acquisition not in progress."
            )

        # wait until enough points are available
        start_time = time.time()
        seen = defaultdict(int)
        while True:
            n = int(self.query("CAPTUREBYTES?"))
            if n >= 1024*self._acquisition.buffer_size:
                break

            # If no progress is being made, we should stop waiting
            seen[n] += 1
            if seen[n] > 3 or \
                time.time() - start_time > self._acquisition.timeout:
                
                self.error(f"Insufficient progess waiting for acquisition.")
                self._stop_acquisition()
                break
            
            # calculate time to wait
            delay = max((1024*self._acquisition.buffer_size - n) \
                        * self._acquisition.sampint / 16, 0.01)
            self.info(f"Waiting for {delay:.2f} s to acquire data.")
            time.sleep(delay)

        # end the acquisition and request data
        self._stop_acquisition()
        for i in range(max_tries):
            self.write(f"CAPTUREGET? 0,{self._acquisition.buffer_size}")
            time.sleep((self._acquisition.buffer_size + i - 1) * 10e-3)

            # read in and parse the binary data. Then reshape
            raw = self.read_raw()
            if len(raw) != 0:
                break
            
            # If we failed to retrieve the data, we request it again with a
            # longer delay between the request and raw read.
        else:
            self.error(
                "Failed to retrieve binary data when reading buffer."
            )
            return np.array([])

        data = parse_IEEE_488_2(raw)
        match self._acquisition.data_config:
            case 'X':
                data = data.reshape(-1, 1)
            case 'XY':
                data = data.reshape(-1, 2)
            case 'RT':
                data = data.reshape(-1, 2)
            case 'XYRT':
                data = data.reshape(-1, 4)

        self.info(f"Retrieved buffered data.")
        return data.T
    
    def get_average(self, auto_rescale: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample X,Y some desired number of times and return an average along with
        covariance in the measured values.

        (Assumes that you have already called setup_acquisition)

        Parameters
        ----------
        auto_rescale : bool, default=False
            whether to call `auto_rescale` prior to taking the average

        Returns
        -------
        mean, cov : np.ndarray
        """

        if not self._acquisition.ready:
            raise RuntimeError(
                "An acquisition must be configured before attempting to take "
                "an average."
            )

        if auto_rescale:
            self.auto_gain()

        # take the acquisition
        self._start_acquisition()
        samps = self._get_buffered_data()

        # TODO FIGURE OUT IF THIS DIVISION SHOULD ACTUALLY HAPPEN...
        return samps.mean(axis = 1), np.cov(samps) # / samps.shape[0]

class sr865a(sr860):
    """Class interface for controlling the SR865A DSP Lock-in"""
    def __init__(self, logger = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """

        super().__init__(logger, instrument_id)
        self._name = 'SR865A'
