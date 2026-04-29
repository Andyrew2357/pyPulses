from ..utils.stats import series_correlated_covariance

from .srs_lockin_base import SRSLockin
from .registry import register_hardware_class

import numpy as np
import time
from math import log2
from collections import defaultdict
from dataclasses import dataclass
from logging import Logger
from typing import Tuple

@register_hardware_class("sr830")
class sr830(SRSLockin):
    """Class representation of the SR830 DSP Lock-in"""

    DEFAULT_PYVISA_CONFIG = {
        "resource_name"     : "",
        "output_buffer_size": 512,

        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.05
    }

    out_aux_channels = [1, 2, 3, 4]

    output_map = {
        'X': 1,
        'Y': 2,
        'R': 3,
        'T': 4
    }

    sens_vals = np.array(
        [1e0] + [val 
            for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
            for val in [factor*5e-9, factor*2e-9, factor*1e-9]]
    )[:-1][::-1]

    tau_vals = []
    for factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        tau_vals.extend([factor*1e-6, factor*3e-6])
    tau_vals = np.array(tau_vals)

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

@register_hardware_class("sr844")
class sr844(SRSLockin):
    """Class representation of the SR844 DSP Lock-in"""

    DEFAULT_PYVISA_CONFIG = {
        "resource_name"     : "",
        "output_buffer_size": 512,

        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.1
    }

    out_aux_channels = [1, 2]

    cmd_map = {
        'icpl': "INPZ",
        'oaux': "AUXI",
        'auxv': "AUXO"
    }

    output_map = {
        'X': 1,
        'Y': 2,
        'R': 3,
        'T': 5
    }

    sens_vals = np.array(
        [1e0] + [val 
            for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2]
            for val in [factor*3e-9, factor*1e-9]]
    )[::-1]

    tau_vals = []
    for factor in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        tau_vals.extend([factor*1e-6, factor*3e-6])
    print(len(tau_vals))
    tau_vals = np.array(tau_vals)

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

    def detection_harmonic(self, harm: int = None) -> int | None:
        """
        Note that the syntax is different for the SR844.
        The options are 0 or 1, where 0 is the fundamental and 1 is the first
        harmonic.
        """
        return super().detection_harmonic(harm)

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

@register_hardware_class("sr850")
class sr850(SRSLockin):
    """Class representation of the SR850 DSP Lock-in"""

    DEFAULT_PYVISA_CONFIG = {
        'write_termination': '\n',
        'output_buffer_size': 512,
        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.1,
    }

    out_aux_channels = [1, 2, 3, 4]

    output_map = {
        'X': 1,
        'Y': 2,
        'R': 3,
        'T': 4
    }

    sens_vals = np.array(
        [1e0] + [val 
            for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
            for val in [factor*5e-9, factor*2e-9, factor*1e-9]]
    )[:-1][::-1]

    tau_vals = []
    for factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        tau_vals.extend([factor*1e-6, factor*3e-6])
    tau_vals = np.array(tau_vals)

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
        raise AttributeError("SR850 does not offer this functionality.")

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
    tau         : float = None

@register_hardware_class("sr860")
class sr860(SRSLockin):
    """Class representation of the SR860 DSP Lock-in"""

    _name = 'SR860'

    DEFAULT_PYVISA_CONFIG = {
        'output_buffer_size': 512,
        'input_buffer_size': 5 * 1024 * 1024,
        'timeout': 10_000,
        'read_termination': None,
        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.05,
    }

    out_aux_channels = [0, 1, 2, 3]

    cmd_map = {
        'fmod': "RSRC",
        'rslp': "RTRG",
        'igan': "ICUR",
        'sens': "SCAL",
        'agan': "ASCL"
    }

    output_map = {
        'X': 0,
        'Y': 1,
        'R': 2,
        'T': 3
    }

    sens_vals = np.array(
        [1e0] + [val 
            for factor in [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
            for val in [factor*5e-9, factor*2e-9, factor*1e-9]]
    )
    
    tau_vals = []
    for factor in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        tau_vals.extend([factor*1e-6, factor*3e-6])
    tau_vals = np.array(tau_vals)

    irng_vals = np.array([1, 0.3, 0.1, 0.03, 0.01])

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
        self._acquisition.tau = self.time_constant()
        self._acquisition.ready = True
        self._acquisition.running = False

        self.info(f"Set up data acquisition with parameters:")
        self.info(f"    config      = {config}")
        self.info(f"    buffer_size = {buffer_size} kB")
        self.info(f"    sample_rate = {sample_rate} Hz")
        return sample_rate
    
    def setup_data_acquisition_timed(self,
        duration:    float,
        config:      str   = 'XY',
        sample_rate: float = None,
        timeout:     float = None,
    ) -> float:
        """
        Configure the data acquisition system for a given measurement duration.
 
        Computes the required buffer size from the desired duration and sample
        rate, rounded up to the nearest kilobyte, and calls
        `setup_data_acquisition`.
 
        Parameters
        ----------
        duration : float
            Desired measurement duration in seconds.
        config : str
            One of {'X', 'XY', 'RT', 'XYRT'}.
        sample_rate : float, optional
            Desired sample rate in Hz. Defaults to maximum.
        timeout : float, optional
            Acquisition timeout in seconds. Defaults to 2 * duration + 5.
 
        Returns
        -------
        sample_rate : float
            The actual sample rate used.
        """
        import math
        from math import log2
 
        if timeout is None:
            timeout = 2 * duration + 5
 
        bytes_per_sample = {'X': 2, 'XY': 4, 'RT': 4, 'XYRT': 8}[config]
 
        max_sample_rate = float(self.query("CAPTURERATEMAX?"))
        if sample_rate is not None:
            n = max(0, min(20, int(log2(max_sample_rate / sample_rate))))
        else:
            n = 0
        actual_sample_rate = max_sample_rate / (2 ** n)
 
        n_samples = math.ceil(duration * actual_sample_rate)
        n_bytes   = n_samples * bytes_per_sample
        buffer_kb = max(1, min(4096, math.ceil(n_bytes / 1024)))
 
        return self.setup_data_acquisition(
            buffer_size = buffer_kb,
            config      = config,
            sample_rate = actual_sample_rate,
            timeout     = timeout,
        )

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
        
        self.write("CAPTURESTART 0,0")
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
        bytes_per_sample = {'X': 2, 'XY': 4, 'RT': 4, 'XYRT': 8}[self._acquisition.data_config]
        expected_time = (self._acquisition.buffer_size * 1024 \
                         * self._acquisition.sampint / bytes_per_sample)
        time.sleep(0.9 * expected_time)
        seen = defaultdict(int)
        while True:
            n = int(self.query("CAPTUREBYTES?"))
            if n >= 1024 * self._acquisition.buffer_size:
                break

            # If no progress is being made, we should stop waiting
            seen[n] += 1
            if seen[n] > 3 or \
                time.time() - start_time > self._acquisition.timeout:
                
                self.error(f"Insufficient progess waiting for acquisition.")
                self._stop_acquisition()
                break
            
            # calculate time to wait
            delay = max((1024 * self._acquisition.buffer_size - n) \
                        * self._acquisition.sampint / bytes_per_sample, 0.05)
            self.info(f"Waiting for {delay:.2f} s to acquire data.")
            time.sleep(delay)

        # end the acquisition and request data
        self._stop_acquisition()
        chunk_kb = 16 # read in 16 kB chunks
        total_kb = self._acquisition.buffer_size
        chunks = []
        for start_kb in range(0, total_kb, chunk_kb):
            size_kb = min(chunk_kb, total_kb - start_kb)
            self.write(f"CAPTUREGET? {start_kb},{size_kb}")
            chunk_data = self.device.read_binary_values(
                datatype = 'f', 
                is_big_endian = False,
                container = np.array,
            )
            chunks.append(chunk_data)
        data = np.concatenate(chunks)

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
        return data
    
    def get_average(self, auto_rescale: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample X,Y some desired number of times and return an average along with
        covariance in the measured values.

        (Assumes that you have already called setup_data_acquisition)

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
        cov = np.cov(samps.T)
        if self._acquisition.sampint < self._acquisition.tau:
            cov *= (self._acquisition.tau / self._acquisition.sampint)
        return samps.mean(axis = 0), cov
    
    def get_average_series_correlated(self, 
                                      auto_rescale: bool = False, 
                                      L: int = None
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample X,Y some desired number of times and return an average along with
        covariance in the measured values. This covariance is estimated 
        accounting for series correlations. See utils.stats for details.

        (Assumes that you have already called setup_data_acquisition)

        Parameters
        ----------
        auto_rescale : bool, default=False
            whether to call `auto_rescale` prior to taking the average
        L : int, optional
            maximum correlation length to consider for covariance estimation 
            (in units of sampling interval). By default, we try to use 20 time 
            constants.
            
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

        if L is None:
            L = int(20 * self._acquisition.tau / self._acquisition.sampint)
        return samps.mean(axis = 0), series_correlated_covariance(samps, L = L)

    def _serialize_state(self) -> dict:
        state = super()._serialize_state()
        if self._acquisition.ready:
            state['acquisition_args'] = {
                'buffer_size'   : self._acquisition.buffer_size,
                'sampint'       : self._acquisition.sampint,
                'data_config'   : self._acquisition.data_config,
                'timeout'       : self._acquisition.timeout,
                'tau'           : self._acquisition.tau
            }
        return state
    
    def _deserialize_state(self, state: dict):
        super()._deserialize_state(state)
        acq = state.get('acquisition_args')
        if acq:
            try:
                self.setup_data_acquisition(
                    buffer_size = acq['buffer_size'],
                    config      = acq['data_config'],
                    sample_rate = 1.0 / acq['sampint'],
                    timeout     = acq['timeout']
                )
            except:
                print("Failed to set up data acquisition while deserializing.")

    def resolve(self, accessor: str):
 
        _lockin_accessors = {
            'get_average':       (False, 1.0),
            'get_average_uV':    (False, 1e6),
            'get_average_sc':    (True,  1.0),
            'get_average_sc_uV': (True,  1e6),
        }
        if accessor in _lockin_accessors:
            series_corr, scale = _lockin_accessors[accessor]
            return sr860_lockin_channel(self, accessor, scale, series_corr)
 
        return super().resolve(accessor)
    
class sr860_lockin_channel():
    """
    LockInChannel for sr860.get_average / get_average_series_correlated.
 
    Parameters
    ----------
    parent : sr860
    accessor : str
    scale : float
        1.0 for raw volts, 1e6 for microvolts. Applied as mean*scale,
        cov*(scale**2) so that units are consistent throughout.
    series_corr : bool
        If True, use get_average_series_correlated; else get_average.
    """
    def __init__(self, parent, accessor: str, scale: float, series_corr: bool):
        self._parent      = parent
        self._accessor    = accessor
        self.scale        = scale
        self._series_corr = series_corr
 
    def format_ref(self):
        return self._parent, self._accessor
 
    def __call__(self):
        if self._series_corr:
            mean, cov = self._parent.get_average_series_correlated()
        else:
            mean, cov = self._parent.get_average()
        s = self.scale
        return mean * s, cov * (s * s)

@register_hardware_class("sr865")
class sr865a(sr860):
    """Class representation of the SR865A DSP Lock-in"""

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
        self._name = 'SR865A'
