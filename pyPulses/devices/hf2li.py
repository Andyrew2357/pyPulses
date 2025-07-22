"""
This class is an interface to the Zurich Instruments HF2LI lock-in amplifier.
"""

from .zhinst_device import zhinstDevice

import numpy as np
from typing import Tuple

class hf2li(zhinstDevice):
    """Class interface for communicating with the HF2LI lock-in amplifier."""
    def __init__(self, 
                 device_id: str = 'dev1616', 
                 server_host: str = 'localhost',
                 server_port: int = 8005, 
                 API_level: int = 1,
                 logger = None):
        """
        Parameters
        ----------
        device_id : str, default='dev1616'
        server_host : str, default='localhost'
        server_port : int, default=8005
        API_level : int, default=1
        logger : Logger, optional
        """
        super().__init__(device_id, server_host, server_port, API_level, 
                         logger = logger)

        self.num_demods = 8
        self.num_outputs = 2
        self.num_oscillators = 6

        self.settings = {
            'demods': [dict(time_const=None) for _ in range(self.num_demods)]
        }

        self.demod = [hf2liDemodChannel(self, i) for i in range(self.num_demods)]
        self.output = [hf2liOutputChannel(self, i) for i in range(self.num_outputs)]
        self.oscillator = [hf2liOscillator(self, i) for i in range(self.num_oscillators)]

    def get_average_all(self, duration=0.1) -> dict:
        """
        Return an average and covariance for signals taken from all demodulators.

        Parameters
        ----------
        duration : float, default=0.1
            duration of the average in s.

        Returns
        -------
        Dict[int, Tuple[ndarray, ndarray]]
            maps demodulator index to average and covariance matrix for (X,Y).
        """
        results = {}
        for i, d in enumerate(self.demod):
            try:
                results[i] = d.get_average(duration)
            except Exception as e:
                self.warn(f"HF2LI: Failed to read demod {i}: {e}")
        return results

    def get_r_all(self) -> dict:
        """
        Return the magnitude of the signal on all demodulators.
        
        Returns
        -------
        Dict[int, float]
        """
        return {i: d.get_r() for i, d in enumerate(self.demod)}

    def get_theta_all(self) -> dict:
        """
        Return the phase of the signal on all demodulators.
        
        Returns
        -------
        Dict[int, float]
        """
        return {i: d.get_theta() for i, d in enumerate(self.demod)}

    def get_signal_routing(self) -> dict:
        """
        Get a dict summary of the signal routing for outputs and oscillators.

        Returns
        -------
        dict
        """
        return {
            'outputs': {
                i: [j for j in range(self.num_demods)
                    if self.get_int(f"sigouts/{i}/enables/{j}")]
                for i in range(self.num_outputs)
            },
            'demod_oscillators': {
                i: (self.get_int(f"demods/{i}/oscselect") if i < 6 else i - 6)
                for i in range(self.num_demods)
            },
            'ref_modes': {
                i: self.demod[i].get_reference_mode()
                for i in range(6, 8)
            }
        }

    def summary(self) -> str:
        """
        Summarize the routing information from get_signal_routing in a string.

        Returns
        -------
        str
        """
        routing = self.get_signal_routing()
        lines = ["\nHF2LI Routing Summary:"]
        lines.append("Outputs:")
        for out, demods in routing['outputs'].items():
            lines.append(f"  Output {out}: demods {demods}")

        lines.append("Demodulators:")
        for demod, osc in routing['demod_oscillators'].items():
            line = f"  Demod {demod}: uses Oscillator {osc}"
            if demod >= 6:
                line += f" ({routing['ref_modes'][demod]})"
            lines.append(line)
        return "\n".join(lines)

class hf2liDemodChannel:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"demods/{self.index}/{subpath}"

    def get_xy(self) -> Tuple[float | None, float | None]:
        """
        Sample the in and out of phase input signal.
        
        Returns
        -------
        X, Y : float or None
        """
        data = self.parent.get_sample(self._p("sample"))
        return data.get('x'), data.get('y')

    def get_r(self) -> float | None:
        """
        Sample the magnitude of the input signal.
        
        Returns
        -------
        float or None
        """
        x, y = self.get_xy()
        return np.sqrt(x**2 + y**2)

    def get_theta(self) -> float | None:
        """
        Sample the phase of the input signal.
        
        Returns
        -------
        float
        """
        x, y = self.get_xy()
        return np.degrees(np.arctan2(y, x))

    def get_oscillator(self) -> int:
        """
        Query which is the associated oscillator.
        
        Returns
        -------
        int
        """
        return self.parent.get_int(self._p("oscselect")) if self.index < 6 \
                                                            else self.index - 6

    def set_oscillator(self, osc_index: int):
        """
        Set the associated oscillator (if it can be set).
        
        Parameters
        ----------
        osc_index : int
        """
        if self.index < 6:
            self.parent.set_int(self._p("oscselect"), osc_index)
            self.parent.info(
                f"Set demod {self.index} to oscillator {osc_index}."
            )
        else:
            raise RuntimeError(
                f"Demod {self.index} is locked to osc {self.index - 6}"
        )

    def get_reference_mode(self) -> str:
        """
        Get the reference mode.
        
        Returns
        -------
        str
            One of {'internal', 'external'}.
        """
        if self.index >= 6:
            mode = self.parent.get_int(self._p("rfm"))
            return "external" if mode == 1 else "internal"
        raise RuntimeError(
            "Reference mode control only valid for demods 6 and 7"
        )

    def set_reference_mode(self, mode: str):
        """
        Set the reference mode (if it can be set). Only 6 and 7 can be set 
        directly.

        Parameters
        ----------
        mode : str
            One of {'internal', 'external'}.
        """
        if self.index >= 6:
            if mode not in ["internal", "external"]:
                raise ValueError("Mode must be 'internal' or 'external'")
            self.parent.set_int(self._p("rfm"), 1 if mode == "external" else 0)
            self.parent.info(
                f"Set demod {self.index} reference mode to {mode}."
            )
        else:
            raise RuntimeError(
                "Reference mode control only valid for demods 6 and 7"
            )

    def get_time_const(self) -> float:
        """
        Returns
        -------
        float
        """
        tau = self.parent.get_double(self._p("timeconstant"))
        self.parent.settings['demods'][self.index]['time_const'] = tau
        return tau

    def set_time_const(self, tau: float):
        """
        Parameters
        ----------
        tau : float
        """
        self.parent.set_double(self._p("timeconstant"), tau)
        self.parent.settings['demods'][self.index]['time_const'] = tau
        self.parent.info(f"Set demod {self.index} time constant to {tau} s.")

    def acquire_samples(self, duration=0.1, timeout=0.2) -> np.ndarray:
        """
        Poll samples of x and y over a specified duration.
        
        Parameters
        ----------
        duration : float, default=0.1
        timeout : float, default=0.2

        Returns
        -------
        ndarray
            raw samples returned as an array of complex numbers.
        """
        path = self._p("sample")
        data = self.parent.subscribe_and_poll(path, duration=duration, 
                                              poll_timeout=timeout)
        full_path = self.parent._get_full_path(path)
        samples = data[full_path]
        return np.array(samples["x"]) + 1j * np.array(samples["y"])

    def get_average(self, duration=0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtain an average and covariance of samples over a specified duration.

        Parameters
        ----------
        duration : float, default=0.1

        Returns
        -------
        mean, cov : ndarray
        """
        samples = self.acquire_samples(duration)
        mean = np.mean(samples)
        cov = np.cov([samples.real, samples.imag])
        return np.array([mean.real, mean.imag]), cov

    def get_frequency(self) -> float:
        """
        Get the frequency of the associated oscillator.
        
        Returns
        -------
        float
        """
        return self.parent.oscillator[self.get_oscillator()].get_frequency()

    def set_frequency(self, freq: float):
        """
        Set the frequency of the associated oscillator (if it can be set).
        
        Parameters
        ----------
        freq : float
        """
        self.parent.oscillator[self.get_oscillator()].set_frequency(freq)
        self.parent.info(
            f"Configuring oscillators to set demod {self.index}"
            f" frequency to {freq} Hz."
        )

class hf2liOutputChannel:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"sigouts/{self.index}/{subpath}"

    def enable(self, on: bool = True):
        """
        Enable or disable external output.

        Parameters
        ----------
        on : bool
        """
        self.parent.set_int(self._p("on"), int(on))
        self.parent.info(
            f"{'En' if on else 'Dis'}abled external output {self.index}."
        )

    def is_enabled(self) -> bool:
        """
        Returns
        -------
        bool
        """
        return bool(self.parent.get_int(self._p("on")))

    def set_amplitude(self, value: float):
        """
        Set the peak to peak amplitude.

        Parameters
        ----------
        value : float
        """
        self.parent.set_double(self._p("amplitude"), value)
        self.parent.info(f"Set output {self.index} amplitude to {value} Vpp.")

    def get_amplitude(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("amplitude"))

    def set_offset(self, value: float):
        """
        Set the DC offset.
        
        Parameters
        ----------
        value : float
        """
        self.parent.set_double(self._p("offset"), value)
        self.parent.info(f"Set output {self.index} DC offset to {value} V.")

    def get_offset(self) -> float:
        """
        Get the DC offset.
        
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("offset"))

    def set_range(self, value: float):
        """
        Set the output range (usually done automatically)
        
        Parameters
        ----------
        value : float
        """
        self.parent.set_double(self._p("range"), value)
        self.parent.info(
            f"Manually set range of output {self.index} to {value} Vpp."
        )

    def get_range(self) -> float:
        """
        Get the output range
        
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("range"))

    def set_phase(self, degrees: float):
        """
        Set the output phase relative to the oscillator.

        Parameters
        ----------
        degrees : float
        """
        self.parent.set_double(self._p("phase"), degrees)
        self.parent.info(f"Set output {self.index} phase to {degrees} degrees.")

    def get_phase(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("phase"))

    def connect_to_demod(self, demod_index: int):
        """
        Tie the output to a given demodulator.
        
        Parameters
        ----------
        demod_index : int
        """
        self.parent.set_int(self._p(f"enables/{demod_index}"), 1)
        self.parent.info(
            f"Assigned output {self.index} to demod {demod_index}."
        )

    def disconnect_demod(self, demod_index: int):
        """
        Disassociate from a given demodulator.
        
        Parameters
        ----------
        demod_index : int
        """
        self.parent.set_int(self._p(f"enables/{demod_index}"), 0)

class hf2liOscillator:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"oscs/{self.index}/{subpath}"

    def set_frequency(self, freq: float):
        """
        Set the oscillator frequency in Hz.

        Parameters
        ----------
        freq : float
        """
        self.parent.set_double(self._p("freq"), freq)
        self.parent.info(f"Set oscillator {self.index} to {freq} Hz.")

    def get_frequency(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("freq"))

"""This class is an abstraction to make handling the output signals easier"""

class hf2liACout:
    def __init__(self, parent: hf2li, output_index: int, oscillator_index: int):
        self.parent = parent
        self.output = parent.output[output_index]
        self.oscillator = parent.oscillator[oscillator_index]
        self.output_index = output_index
        self.oscillator_index = oscillator_index

        # Assume demod is routed to this output and uses this oscillator
        self.associated_demods = [
            i for i in range(parent.num_demods)
            if parent.get_int(f"sigouts/{output_index}/enables/{i}")
            and parent.demod[i].get_oscillator() == oscillator_index
        ]

    def enable(self, state: bool = True):
        """
        Enable or disable external output.

        Parameters
        ----------
        state : bool, default=True
        """
        self.output.enable(state)

    def set_frequency(self, freq: float):
        """
        Set the oscillator frequency in Hz.

        Parameters
        ----------
        freq : float
        """
        self.oscillator.set_frequency(freq)

    def get_frequency(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.oscillator.get_frequency()

    def set_amplitude(self, value: float):
        """
        Set the peak to peak amplitude.

        Parameters
        ----------
        value : float
        """
        self.output.set_amplitude(value)

    def get_amplitude(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.output.get_amplitude()

    def set_phase(self, value: float):
        """
        Set the phase relative to the oscillator (in degrees).

        Parameters
        ----------
        value : float
        """
        self.output.set_phase(value)

    def get_phase(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.output.get_phase()

    def summary(self) -> str:
        """
        Get a string summarizing the AC output settings.

        Returns
        -------
        str
        """
        return (
            f"AC Output {self.output_index} using Oscillator"
            f" {self.oscillator_index}:\n"
            f"  Enabled: {self.output.is_enabled()}\n"
            f"  Frequency: {self.get_frequency()} Hz\n"
            f"  Amplitude: {self.get_amplitude()} V\n"
            f"  Phase: {self.get_phase()} deg\n"
            f"  Associated Demods: {self.associated_demods}"
        )
