"""
This class is an interface to the Stanford Research Systems model 865A lock-in 
amplifier.
"""

from .pyvisa_device import pyvisaDevice
from typing import Tuple
from collections import defaultdict
import numpy as np
from math import log2
import time

class sr865a(pyvisaDevice):
    """Class interface for communicating with the SR865A"""
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
            "resource_name"     : "USB0::0xB506::0x2000::003931::INSTR",
            "output_buffer_size": 512
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        # initialize data acquisition parameters
        self.currsamp    = 0
        self.sampint     = 0        # sampling interval in seconds
        self.buffer_size = 0        # size of the buffer in kB
        self.data_config = 'XY'     # data format (X, XY, RT, XYRT)

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

        # internal state to avoid wasteful querying while automatically setting 
        # range
        self.settings = {
            'sens_ind'  : None,
            'irng_ind'  : None,
            'time_const': None
        }

        self.upd_internal_state()

    # INPUT RELATED

    def get_x(self) -> float:
        """
        Get the X (in-phase) component of the measured signal.
        
        Returns
        -------
        X : float
        """
        return float(self.device.query("OUTP? 0"))
    
    def get_y(self) -> float:
        """
        Get the Y (quadrature) component of the measured signal.
        
        Returns
        -------
        Y : float
        """
        return float(self.device.query("OUTP? 1"))
    
    def get_r(self) -> float:
        """
        Get the R (magnitude) component of the measured signal.
        
        Returns
        -------
        R : float
        """
        return float(self.device.query("OUTP? 2"))
    
    def get_theta(self) -> float:
        """
        Get the θ (phase) component of the measured signal in degrees.
        
        Returns
        -------
        theta : float
        """
        return float(self.device.query("OUTP? 3"))
    
    def get_xy(self) -> Tuple[float, float]:
        """
        Get X and Y components simultaneously for better timing accuracy.
        
        Returns
        -------
        X, Y : float
        """
        result = self.device.query("SNAP? 0,1")
        return tuple(map(float, result.strip().split(',')))
    
    #REFERENCE RELATED

    def get_frequency(self) -> float:
        """
        Get the reference frequency in Hz.
        
        Returns
        -------
        f : float
        """
        return float(self.device.query("FREQ?"))
    
    def set_frequency(self, freq: float):
        """
        Set the reference frequency in Hz.
        
        Parameters
        ----------
        freq : float
        """
        self.device.write(f"FREQ {freq}")
        self.info(f"SR865A: Set reference frequency to {freq} Hz.")

    def get_phase(self) -> float:
        """
        Get the reference phase shift in degrees.
        
        Returns
        -------
        phase : float
        """
        return float(self.device.query("PHAS?"))
    
    def set_phase(self, phase: float):
        """
        Set the reference phase shift in degrees.
        
        Parameters
        ----------
        phase : float
        """
        self.device.write(f"PHAS {phase}")
        self.info(f"SR865A: Set reference phase shift to {phase} degrees.")

    def get_ref_amplitude(self) -> float:
        """
        Get the reference amplitude in volts.
        
        Returns
        -------
        V : float
        """
        return float(self.device.query("SLVL?"))
    
    def set_ref_amplitude(self, amplitude: float):
        """
        Set the reference amplitude in volts.
        
        Parameters
        ----------
        amplitude : float
        """
        self.device.write(f"SLVL {amplitude}")
        self.info(f"SR865A: Set reference amplitude to {amplitude} V.")

    # INPUT SETTINGS

    def get_input_range(self) -> float:
        """
        Get the input signal range in volts.
        
        Returns
        -------
        Vrange : float
        """
        idx = int(self.device.query("IRNG?"))
        self.settings['irng_ind'] = idx
        return self._input_range_value(idx)

    def set_input_range(self, range_v: float):
        """
        Set the input signal range in volts.

        Parameters
        ----------
        range_v : float
            rounded to 1.0, 0.3, 0.2, 0.03, or 0.01.
        """ 
        idx = self._input_range_index(range_v)
        self.device.write(f"IRNG {idx}")
        self.settings['irng_ind'] = idx
        self.info(
            f"SR865A: Set input range to {self._input_range_value(idx)} V."
        )

    def get_sensitivity(self) -> float:
        """
        Get the current sensitivity in volts.
        
        Returns
        -------
        sensitivity : float
        """
        idx = int(self.device.query("SCAL?"))
        self.settings['sens_ind'] = idx
        return self._sens_value(idx)
    
    def set_sensitivity(self, sensitivity: float):
        """
        Set the sensitivity in volts.
        
        Parameters
        ----------
        sensitivity : float
            rounded to the closest valid sensitivity value.
        """
        idx = self._sens_index(sensitivity)
        self.device.write(f"SCAL {idx}")
        self.settings['sens_ind'] = idx
        self.info(f"SR865A: Set sensitivity to {self._sens_value(idx)} V.")

    def get_time_const(self) -> float:
        """
        Get the time constant in seconds.
        
        Returns
        -------
        tau : float
        """
        idx = int(self.device.query("OFLT?"))
        self.settings['time_const'] = self._tau_value(idx)
        return self._tau_value(idx)
    
    def set_time_const(self, time_const: float):
        """
        Set the time constant in seconds.
        
        Parameters
        ----------
        time_const : float
            rounded to the closest valid time constant.
        """
        idx = self._tau_index(time_const)
        self.device.write(f"OFLT {idx}")
        self.settings['time_const'] = self._tau_value(idx)
        self.info(f"SR865A: Set time constant to {self._tau_value(idx)} s.")

    # SYNC FILTER CONTROL

    def get_sync_filter_status(self) -> bool:
        """
        Get the sync filter status.
        
        Returns
        -------
        bool
            true = enabled, false = disabled.
        """
        return bool(int(self.device.query("SYNC?")))
    
    def set_sync_filter(self, enabled: bool):
        """
        Enable or disable the sync filter.
        
        Parameters
        ----------
        enabled : bool
        """
        self.device.write(f"SYNC {1 if enabled else 0}")
        self.info(
            f"SR865A: {'Enabled' if enabled else 'Disabled'} sync filter."
        )

    # AUXILIARY I/O

    def get_aux_input(self, ch: int) -> float:
        """
        Get the voltage at an auxiliary input.
        
        Parameters
        ----------
        ch : int

        Returns
        -------
        float
        """
        if not ch in [1, 2, 3, 4]:
            raise ValueError("Aux input channel must be 1, 2, 3, or 4.")
        return float(self.device.query(f"AUXV? {ch}"))
    
    def get_aux_output(self, ch: int) -> float:
        """
        Get the voltage at an auxiliary output.
        
        Parameters
        ----------
        ch : int

        Returns
        -------
        float
        """
        if not ch in [1, 2, 3, 4]:
            raise ValueError("Aux output channel must be 1, 2, 3, or 4.")
        return float(self.device.query(f"OAUX? {ch}"))
    
    def set_aux_output(self, ch: int, V: float):
        """
        Set the voltage at an auxiliary output.
        
        Parameters
        ----------
        ch : int
        V : float
        """
        if not ch in [1, 2, 3, 4]:
            raise ValueError("Aux output channel must be 1, 2, 3, or 4.")
        if not -10.5 <= V <= 10.5:
            raise ValueError(
                "Aux output voltage must be between -10.5 and 10.5 V"
            )
        self.device.write(f"OAUX {ch} {V}")
        self.info(f"SR865A: Set channel {ch} auxiliary output to {V} V.")

    def get_dc_offset(self) -> float:
        """
        Get the DC offset in percent.
        
        Returns
        -------
        float
        """
        return float(self.device.query("SOFF?"))
    
    def set_dc_offset(self, off: float):
        """
        Set the DC offset in percent (-100 to 100%).
        
        Parameters
        ----------
        off : float
        """
        if not -100 <= off <= 100:
            raise ValueError("DC offset must be between -100 and 100%.")
        self.device.write(f"SOFF {off}")
        self.info(f"SR865A: Set DC offset to {off}%.")

    # STATUS / DIAGNOSTICS

    def is_input_overloaded(self) -> bool:
        """
        Check if the input signal is overloaded.
        
        Returns
        -------
        bool
        """
        status = int(self.device.query("CUROVLDSTAT?"))
        return bool((status >> 4) & 1) # extract bit 4
    
    def is_reference_unlocked(self) -> bool:
        """
        Check if the external reference is unlocked.
        
        Returns
        -------
        bool
        """
        status = int(self.device.query("CUROVLDSTAT?"))
        return bool((status >> 3) & 1) # extract bit 3
    
    def get_input_level(self) -> int:
        """
        Get the input signal level indicator.

        Returns
        -------
        int
            ranges from 0 to 4; 0 = weak signal, 4 = overload.
        """
        return int(self.device.query("ILVL?"))
    
    # DATA ACQUISITION
    def setup_data_acquisition(self, buffer_size: int, config: str = 'XY',
                               sample_rate: float = None) -> float:
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

        if not config in ['X', 'XY', 'RT', 'XYRT']:
            raise ValueError(
                "Unsupported config; must be 'X', 'XY', 'RT' or 'XYRT'."
            )
        
        if buffer_size < 1 or buffer_size > 4096:
            raise ValueError(
                "Buffer size must be between 1 and 4096 kB."
            )

        self.device.write(f"CAPTURECFG {config}")
        self.device.write(f"CAPTURELEN {buffer_size}")

        max_sample_rate = float(self.device.query("CAPTURERATEMAX?"))
        if sample_rate is not None:
            n = int(log2(max_sample_rate / sample_rate))
            if n < 0 or n > 20:
                raise ValueError(
                    f"Sample rate {sample_rate} is out of range."
                )
        else:
            n = 0

        sample_rate = max_sample_rate / (2**n)
        self.device.write(f"CAPTURERATE {n}")

        self.buffer_size = buffer_size
        self.currsamp = 0
        self.sampint = 1/sample_rate
        self.data_config = config

        self.info(f"SR865A: Setup data acquisition with parameters:")
        self.info(f"    config      = {config}")
        self.info(f"    buffer_size = {buffer_size} kB")
        self.info(f"    sample_rate = {sample_rate} Hz")
        return sample_rate
    
    def start_acquisition(self):
        """Start data acquisition."""
        self.device.write("CAPTURESTART ONE, IMM")
        self.info("SR865A: Started data acquisition.")

    def stop_acquisition(self):
        """Stop data acquisition."""
        self.device.write("CAPTURESTOP")
        self.info("SR865A: Stopped data acquisition.")

    def get_buffered_data(self, max_tries: int = 5) -> np.ndarray:
        """
        Get buffered data.

        Returns
        -------
        np.ndarray
        """
        # wait until enough points are available
        seen = defaultdict(int)
        while True:
            n = int(self.device.query("CAPTUREBYTES?"))
            if n >= 1024*self.buffer_size:
                break

            # If no progress is being made, we should stop waiting
            seen[n] += 1
            if seen[n] > 3:
                self.error(f"SR865A: No progess waiting for acquisition.")
                self.stop_acquisition()
                break
            
            # calculate time to wait
            delay = max((1024*self.buffer_size - n) * self.sampint / 16, 0.01)
            self.info(f"SR865A: Waiting for {delay:.2f} s to acquire data.")
            time.sleep(delay)

        # end the acquisition and request data
        self.stop_acquisition()
        for i in range(max_tries):
            self.device.write(f"CAPTUREGET? 0,{self.buffer_size}")
            time.sleep((self.buffer_size + i - 1)*10e-3)

            # read in and parse the binary data. Then reshape
            raw = self.device.read_raw()
            if len(raw) != 0:
                break
            
            # If we failed to retrieve the data, we request it again with a
            # longer delay between the request and raw read.
        else:
            self.error(
                "SR865A: Failed to retrieve binary data when reading buffer."
            )
            return np.array([])

        data = self._parse_binary_block(raw)
        match self.data_config:
            case 'X':
                data = data.reshape(-1, 1)
            case 'XY':
                data = data.reshape(-1, 2)
            case 'RT':
                data = data.reshape(-1, 2)
            case 'XYRT':
                data = data.reshape(-1, 4)

        self.info(f"SR865A: Retrieved buffered data.")
        return data.T

    # UTILITY

    def _parse_binary_block(self, data: bytes) -> np.ndarray:
        if not data.startswith(b'#'):
            raise ValueError("Invalid binary block format.")
        
        n = int(data[1:2]) # number of digits in length
        len_start = 2
        len_end = len_start + n
        length = int(data[len_start:len_end])
        binary_data = data[len_end:len_end + length]

        if len(binary_data) != length:
            raise ValueError("Invalid binary block length.")
        
        return np.frombuffer(binary_data, dtype = '<f4') # little-endian float32

    def _sens_value(self, sens_index: int) -> float:
        """Convert sensitivity index to actual value in volts."""
        return self.sens_vals[sens_index]
    
    def _sens_index(self, sens_val: float) -> int:
        """Convert sensitivity value to index (rounds to closest value)."""
        return np.argmin(np.abs(self.sens_vals - sens_val))
    
    def _tau_value(self, tau_index: int) -> float:
        """Convert time constant index to actual value in seconds."""
        return self.tau_vals[tau_index]
    
    def _tau_index(self, tau_val: float) -> int:
        """Convert time constant value to index (rounds to closest value)."""
        return np.argmin(np.abs(self.tau_vals - tau_val))
    
    def _input_range_value(self, irng_index: int) -> float:
        """Convert input range index to actual value in volts."""
        return self.irng_vals[irng_index]
    
    def _input_range_index(self, irng_val: float) -> int:
        """Convert input range value to index (rounds to closest value)."""
        return np.argmin(np.abs(self.irng_vals - irng_val))

    # HIGHER LEVEL ROUTINES, RELEVANT FOR pyPulses.routines.kap_bridge

    def upd_internal_state(self):
        self.get_sensitivity()
        self.get_input_range()
        self.get_time_const()

    def increment_sensitivity(self):
        """Up the sensitivity to the next available value (more sensitive)"""
        idx = self.settings['sens_ind']
        if idx is None:
            idx = self._sens_index(self.get_sensitivity())

        if idx < len(self.sens_vals) - 1:
            self.settings['sens_ind'] += 1
            self.device.write(f"SCAL {idx + 1}")
            self.info(
                f"SR865A: Set sensitivity to {self._sens_value(idx + 1)} V."
            )
            return True
        return False

    def decrement_sensitivity(self):
        """Reduce the sensitivity to the next available value (less sensitive)"""
        idx = self.settings['sens_ind']
        if idx is None:
            idx = self._sens_index(self.get_sensitivity())

        if idx > 0:
            self.settings['sens_ind'] -= 1
            self.device.write(f"SCAL {idx - 1}")
            self.info(
                f"SR865A: Set sensitivity to {self._sens_value(idx - 1)} V."
            )
            return True
        return False

    def increment_input_range(self):
        """
        Reduce the input range to the next available value (Note that we refer
        to this as incrementing so that it is similar to increment_sensitivity)
        """
        idx = self.settings['irng_ind']
        if idx is None:
            idx = self._input_range_index(self.get_input_range())

        if idx < len(self.irng_vals) - 1:
            self.settings['irng_ind'] += 1
            self.device.write(f"IRNG {idx + 1}")
            self.info(
                f"SR865A: Set input range to {self._input_range_value(idx + 1)} V."
            )
            return True
        return False

    def decrement_input_range(self):
        """
        Increase the input range to the next available value (Note that we refer
        to this as dencrementing so that it is similar to decrement_sensitivity)
        """
        idx = self.settings['irng_ind']
        if idx is None:
            idx = self._input_range_index(self.get_input_range())

        if idx > 0:
            self.settings['irng_ind'] -= 1
            self.device.write(f"IRNG {idx - 1}")
            self.info(
                f"SR865A: Set input range to {self._input_range_value(idx - 1)} V."
            )
            return True
        return False

    def auto_rescale(self, margin: float = 0.5, max_iters: int = 5
                     ) -> Tuple[float, float, float]:
        """
        Adjust sensitivity (and, if desired, input range) so that the current
        vector (X, Y) sits at about 'margin' of full scale.

        Parameters
        ----------
        margin : float
            fraction of full-scale you'd like to occupy
        max_iters : float
            how many times to try before giving up.
             
        Returns
        -------
        ratio_of_input_range_filled, sensitivity, input_range : float
        """

        tau = self.settings['time_const']
        if tau is None:
            tau = self.get_time_const()

        for _ in range(max_iters):
            r = self.get_r()
            #current sensitivity

            target_span = max(r / margin, self.sens_vals.min())
            self.set_sensitivity(target_span)

            time.sleep(tau * 5)
            if not self.is_input_overloaded():
                break

            # if overloaded, step input range
            if not self.decrement_input_range():
                self.warn(
                    'auto_rescale max input range reached; still overloading'
                )
                break

            time.sleep(tau * 2)
        
        else:
            self.warn('auto_rescale failed to find non-overloaded setting')

        r = self.get_r()
        irng = self._input_range_value(self.settings['irng_ind'])
        sens = self._sens_value(self.settings['sens_ind'])
        return r, sens, irng

    def get_average(self, auto_rescale: bool = False, rescale_args = None, 
                    rescale_kwargs = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample X,Y some desired number of times and return an average along with
        covariance in the measured values.

        (Assumes that you have already called setup_acquisition)

        Parameters
        ----------
        auto_rescale : bool, default=False
            whether to call `auto_rescale` prior to taking the average
        rescale_args : tuple, default=()
            args passed to `auto_rescale`
        rescale_kwargs : dict, default={}
            kwargs passed to `auto_rescale`

        Returns
        -------
        mean, cov : np.ndarray
        """

        rescale_args = rescale_args or ()
        rescale_kwargs = rescale_kwargs or {}
        # adjust the lock-in range so that the signal fills some prescribed
        # portion of the range
        if auto_rescale:
            self.auto_rescale(*rescale_args, **rescale_kwargs)

        # take the acquisition
        self.start_acquisition()
        samps = self.get_buffered_data()

        return samps.mean(axis = 1), np.cov(samps) # / samps.shape[0]
