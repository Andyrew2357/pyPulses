"""
This class is an interface to the Zurich Instruments HF2LI lock-in amplifier.
"""

from .zhinst_device import zhinstDevice

import numpy as np
from typing import Tuple

class hf2li(zhinstDevice):
    def __init__(self, 
                 device_id: str, 
                 server_host: str = 'localhost',
                 server_port: int = 8004, 
                 API_level: int = 1,
                 logger = None):
        super().__init__(device_id, server_host, server_port, API_level, 
                         logger = logger)

        self.num_demods = 8
        self.num_outputs = 2

        self.settings = {
            'demods': [dict(time_const=None) for _ in range(self.num_demods)]
        }

        self.demod = [DemodChannel(self, i) for i in range(self.num_demods)]
        self.output = [OutputChannel(self, i) for i in range(self.num_outputs)]

    def get_average_all(self, duration=0.1) -> dict:
        """
        Acquire averaged X, Y + covariance from all demodulators.
        Returns a dict {index: (mean, cov)}
        """
        results = {}
        for i, d in enumerate(self.demod):
            try:
                results[i] = d.get_average(duration)
            except Exception as e:
                self.warn(f"HF2LI: Failed to read demod {i}: {e}")
        return results

    def get_r_all(self) -> dict:
        """Get the r signal from all demodulators"""
        return {i: d.get_r() for i, d in enumerate(self.demod)}

    def get_theta_all(self) -> dict:
        """Get the phase of the signal from all demodulators"""
        return {i: d.get_theta() for i, d in enumerate(self.demod)}

class DemodChannel:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"demods/{self.index}/{subpath}"

    def get_xy(self) -> Tuple[float, float]:
        """Get in and out of phase signals"""
        x = self.parent.get_param(self._p("sample/x"))
        y = self.parent.get_param(self._p("sample/y"))
        return x, y

    def get_r(self) -> float:
        """Get magnitude of signal"""
        x, y = self.get_xy()
        return np.sqrt(x**2 + y**2)

    def get_theta(self) -> float:
        """Get phase of signal"""
        x, y = self.get_xy()
        return np.degrees(np.arctan2(y, x))

    def get_frequency(self) -> float:
        """Get the frequency of the demodulator (internal/external reference)"""
        return self.parent.get_param(self._p("freq"))

    def set_frequency(self, freq: float):
        """Set the frequency of the demodulator (internal reference)"""
        self.parent.set_param(self._p("freq"), freq)

    def get_time_const(self) -> float:
        """Get the demodulator time constant"""
        tau = self.parent.get_param(self._p("timeconstant"))
        self.parent.settings['demods'][self.index]['time_const'] = tau
        return tau

    def set_time_const(self, tau: float):
        """Set the time constant of the demodulator"""
        self.parent.set_param(self._p("timeconstant"), tau)
        self.parent.settings['demods'][self.index]['time_const'] = tau

    def acquire_samples(self, duration=0.1, timeout=0.2) -> np.ndarray:
        """Poll samples from the demodulator for a given time duration"""
        path = self._p("sample")
        data = self.parent.subscribe_and_poll(path, duration=duration, 
                                              poll_timeout=timeout)
        full_path = self.parent._get_full_path(path)
        samples = data[full_path]
        return np.array(samples["x"]) + 1j * np.array(samples["y"])

    def get_average(self, duration=0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Acquire an average and covariance from polling the demodulator"""
        samples = self.acquire_samples(duration)
        mean = np.mean(samples)
        cov = np.cov([samples.real, samples.imag])
        return np.array([mean.real, mean.imag]), cov


class OutputChannel:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"sigouts/{self.index}/{subpath}"

    def enable(self, on: bool = True):
        self.parent.set_param(self._p("on"), int(on))

    def is_enabled(self) -> bool:
        return bool(self.parent.get_param(self._p("on")))

    def set_amplitude(self, value: float):
        self.parent.set_param(self._p("amplitude"), value)

    def get_amplitude(self) -> float:
        return self.parent.get_param(self._p("amplitude"))

    def set_offset(self, value: float):
        self.parent.set_param(self._p("offset"), value)

    def get_offset(self) -> float:
        return self.parent.get_param(self._p("offset"))

    def set_range(self, value: float):
        self.parent.set_param(self._p("range"), value)

    def get_range(self) -> float:
        return self.parent.get_param(self._p("range"))
    
    def set_phase(self, degrees: float):
        self.parent.set_param(self._p("phase"), degrees)

    def get_phase(self) -> float:
        return self.parent.get_param(self._p("phase"))

    def connect_to_demod(self, demod_index: int):
        self.parent.set_param(self._p(f"enables/{demod_index}"), 1)

    def disconnect_demod(self, demod_index: int):
        self.parent.set_param(self._p(f"enables/{demod_index}"), 0)
