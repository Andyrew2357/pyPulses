from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
from typing import Optional, Tuple
import numpy as np
from math import log2
import time

class sr865a(pyvisaDevice):
    def __init__(self, logger: Optional[str] = None, 
                 instrument_id: Optional[str] = None):
        self.config = {
            "resource_name"     : "USB0::0xB506::0x2000::003931::INSTR",
            "output_buffer_size": 512
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        # initialize data acquisition parameters
        self.currsamp   = 0
        self.sampint    = 0         # sampling interval in seconds
        self.buffer_sizes = [0, 0]  # buffer sizes for data channels 1/2

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

    # INPUT RELATED

    def get_x(self) -> float:
        """Get the X (in-phase) component of the measured signal."""
        return float(self.device.query("OUTP? 0"))
    
    def get_y(self) -> float:
        """Get the Y (quadrature) component of the measured signal."""
        return float(self.device.query("OUTP? 1"))
    
    def get_r(self) -> float:
        """Get the R (magnitude) component of the measured signal."""
        return float(self.device.query("OUTP? 2"))
    
    def get_theta(self) -> float:
        """Get the θ (phase) component of the measured signal in degrees."""
        return float(self.device.query("OUTP? 3"))
    
    def get_xy(self) -> Tuple[float, float]:
        """Get X and Y components simultaneously for better timing accuracy."""
        result = self.device.query("SNAP? 0,1")
        return tuple(map(float, result.strip().split(',')))
    
    #REFERENCE RELATED

    def get_frequency(self) -> float:
        """Get the reference frequency in Hz."""
        return float(self.device.query("FREQ?"))
    
    def set_frequency(self, freq: float):
        """Set the reference frequency in Hz."""
        self.device.write(f"FREQ {freq}")

    def get_phase(self) -> float:
        """Get the reference phase shift in degrees."""
        return float(self.device.query("PHAS?"))
    
    def set_phase(self, phase: float):
        """Set the reference phase shift in degrees."""
        self.device.write(f"PHAS {phase}")

    def get_ref_amplitude(self) -> float:
        """Get the reference amplitude in volts."""
        return float(self.device.query("SLVL?"))
    
    def set_ref_amplitude(self, amplitude: float):
        """Set the reference amplitude in volts."""
        self.device.write(f"SLVL {amplitude}")

    # INPUT SETTINGS

    def get_input_range(self) -> float:
        """Get the input signal range in volts."""
        idx = int(self.device.query("IRNG?"))

    def set_input_range(self, range_v: float):
        """
        Set the input signal range in volts.
        valid values are: 1.0, 0.3, 0.1, 0.03, 0.01 V
        The closest valid value will be selected
        """ 
        idx = self._input_range_index(range_v)
        self.device.write(f"IRNG {idx}")

    def get_sensitivity(self) -> float:
        """Get the current sensitivity in volts."""
        idx = int(self.device.query("SCAL?"))
        return self._sens_value(idx)
    
    def set_sensitivity(self, sensitivity: float):
        """
        Set the sensitivity in volts.
        The closest valid sensitivity value will be selected.
        """
        idx = self._sens_index(sensitivity)
        self.device.write(f"SCAL {idx}")

    def get_time_const(self) -> float:
        """Get the time constant in seconds."""
        idx = int(self.device.query("OFLT?"))
        return self._tau_value(idx)
    
    def set_time_const(self, time_const: float):
        """
        Set the time constant in seconds.
        The closest valid time constant will be selected.
        """
        idx = self._tau_index(time_const)
        self.device.write(f"OFLT {idx}")

    # SYNC FILTER CONTROL

    def get_sync_filter_status(self) -> bool:
        """Get the sync filter status (True = enabled, False = disabled)."""
        return bool(int(self.device.query("SYNC?")))
    
    def set_sync_filter(self, enabled: bool):
        """Enable or disable the sync filter."""
        self.device.write(f"SYNC {1 if enabled else 0}")

    # AUXILIARY I/O

    def get_aux_input(self, ch: int) -> float:
        """Get the voltage at an auxiliary input."""
        if not ch in [1, 2, 3, 4]:
            raise ValueError("Aux input channel must be 1, 2, 3, or 4.")
        return float(self.device.query(f"AUXV? {ch}"))
    
    def get_aux_output(self, ch: int) -> float:
        """Get the voltage at an auxiliary output."""
        if not ch in [1, 2, 3, 4]:
            raise ValueError("Aux output channel must be 1, 2, 3, or 4.")
        return float(self.device.query(f"OAUX? {ch}"))
    
    def set_aux_output(self, ch: int, V: float):
        """Set the voltage at an auxiliary output."""
        if not ch in [1, 2, 3, 4]:
            raise ValueError("Aux output channel must be 1, 2, 3, or 4.")
        if not -10.5 <= V <= 10.5:
            raise ValueError(
                "Aux output voltage must be between -10.5 and 10.5 V"
            )
        self.device.write(f"OAUX {ch} {V}")

    def get_dc_offset(self) -> float:
        """Get the DC offset in percent."""
        return float(self.device.query("SOFF?"))
    
    def set_dc_offset(self, off: float):
        """Set the DC offset in percent (-100 to 100%)."""
        if not -100 <= off <= 100:
            raise ValueError("DC offset must be between -100 and 100%.")
        self.device.write(f"SOFF {off}")

    # STATUS / DIAGNOSTICS

    def is_input_overloaded(self) -> bool:
        """Check if the input signal is overloaded."""
        status = int(self.device.query("CUROVLDSTAT?"))
        return bool((status >> 4) & 1) # extract bit 4
    
    def is_reference_unlocked(self) -> bool:
        """Check if the external reference is unlocked."""
        status = int(self.device.query("CUROVLDSTAT?"))
        return bool((status >> 3) & 1) # extract bit 3
    
    def get_input_level(self) -> int:
        """
        Get the input signal level indicator.
        This is an integer from 0 - 4 with 0 = weak signal, 4 = overload
        """
        return int(self.device.query("ILVL?"))
    
    # DATA ACQUISITION
    def setup_data_acquisition(self, buffer_size: int, sample_rate: float, 
                               sync_sampling: Optional[bool] = False) -> float:
        """
        Configure the data acquisition system.

        buffer_size = number of samples to store in each buffer
        sample_rate = desired sample rate in Hz
        sync_sampling = if true, use synchronized sampling

        Returns the actual sample rate used.
        """
        if sync_sampling:
            n = 14
        else:
            n = round(log2(sample_rate)) + 4
            sample_rate = 2**(n - 4)

            if n < 0 or n > 13:
                raise ValueError(
                    f"Sample rate {sample_rate} Hz is not supported by SR865A"
                )
            
        self.device.write(f"REST; SEND 1; TSTR 1; SRAT {n}")
        time.sleep(0.1)
        self.currsamp = 0
        self.sampint = 1/sample_rate
        self.buffer_sizes = [buffer_size, buffer_size]

        return sample_rate
    
    def start_acquisition(self):
        """Start data acquisition."""
        self.device.write("STRT")

    def reset_acquisition(self):
        """Reset data acquisition buffers."""
        self.device.write("REST")
        self.currsamp = 0
        time.sleep(0.1) # give instrument time before next trigger

    def get_buffered_data(self, ch: int) -> np.ndarray:
        """
        Get buffered data from the specified channel.
        ch must be 1 or 2
        Returns a numpy ndarray of acquired data points
        """
        if ch not in [1, 2]:
            raise ValueError("Channel must be 1 or 2.")
        
        npts = self.buffer_sizes[ch - 1]

        # wait until enough points are available
        while True:
            navail = int(self.device.query("SPTS?"))
            if navail >= npts + self.currsamp:
                break
            else:
                wait_time = 0.8 * (npts + self.currsamp - navail) * self.sampint
                time.sleep(wait_time)
        
        # fetch the data
        cmd = f"TRCB? {ch}, {self.currsamp}, {self.currsamp + npts}"
        self.device.write(cmd)

        # read binary data (implementation depends on the VISA library)
        # this uses PyVISA's read_binary_values method
        val = self.device.read_binary_values('f')[:npts]
        self.currsamp += npts

        return np.array(val)

    # UTILITY

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
