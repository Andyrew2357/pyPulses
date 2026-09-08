"""
Control class for the Keysight N9010A (X-Series) spectrum analyzer.

This is a partial driver covering common, simple use cases only (narrow-band
noise/PSD acquisition via swept traces, integrated-power/shot-noise readings,
and noise-floor benchmarking via markers) rather than the full instrument
command set. See ``N9010A_PLAN.md`` at the repo root for the design/planning
document this driver is built from, including the condensed SCPI command
reference it was written against.

This module implements Phases 1-4 of that plan: connection handling (inherited
from ``pyvisaDevice``), identification/reset, the error queue, mode selection,
swept-trace acquisition (frequency/span, RBW/VBW, sweep control,
detector/trace-type/averaging, front-end amplitude settings, and
``get_trace()``), the Channel Power measurement (integrated/total power in a
band, for shot-noise readings), marker functions (peak search, plus the
noise marker / band density functions used for noise-floor benchmarking), and
JSON state serialization (see `_STATE_SETTINGS`/`_serialize_state` for what is
and isn't captured).
"""

from .pyvisa_device import pyvisaDevice, parse_IEEE_488_2
from .registry import register_hardware_class

from logging import Logger
from typing import Any, Dict, List, Tuple

import numpy as np


@register_hardware_class("n9010a")
class n9010a(pyvisaDevice):
    """
    Class representation of the Keysight N9010A spectrum analyzer.

    Talks plain SCPI over VISA (LAN/GPIB), so this class follows the same
    conventions as the other SCPI instrument drivers in this package: a
    get-or-set method per setting (``None`` queries, a value writes), and
    numpy-style docstrings.
    """

    DEFAULT_PYVISA_CONFIG = {
        'timeout': 10000,
        'write_termination': '\n',
        'read_termination': '\n',
        'min_interval': 0.0,
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
            VISA resource name (e.g. ``"TCPIP0::192.168.1.10::inst0::INSTR"``,
            ``"GPIB0::18::INSTR"``), or ``"DEBUG"`` for a hardware-free
            self-test instance.
        registry_id : str, optional
            Name to register this instance under in the HardwareRegistry.
        logger : Logger, optional
            Logger used by abstractDevice.
        skip_connect : bool, default False
        **kwargs
        """
        super().__init__(
            resource_name=resource_name,
            registry_id=registry_id,
            logger=logger,
            skip_connect=skip_connect,
            **kwargs,
        )

    """
    -------------------------------------------------------------------------
    General / common commands
    -------------------------------------------------------------------------
    """

    def idn(self) -> str:
        """
        Query the instrument identification string (``*IDN?``).

        Returns
        -------
        str
        """
        return self.query("*IDN?").strip()

    def reset(self):
        """Reset the instrument to its default state (``*RST``)."""
        self.write("*RST")
        self.info("N9010A: Reset to default state.")

    def clear_status(self):
        """Clear the status/event registers and error queue (``*CLS``)."""
        self.write("*CLS")

    def get_error(self) -> Tuple[int, str]:
        """
        Pop and return the next entry from the error queue (``SYST:ERR?``).

        Returns
        -------
        code, message : int, str
            ``(0, "No error")`` when the queue is empty.
        """
        raw = self.query("SYST:ERR?").strip()
        code_str, _, message = raw.partition(',')
        return int(code_str), message.strip().strip('"')

    def get_all_errors(self) -> List[Tuple[int, str]]:
        """
        Drain the error queue, returning every entry until it is empty.

        Returns
        -------
        list of (code, message)
        """
        errors = []
        while True:
            code, message = self.get_error()
            if code == 0:
                break
            errors.append((code, message))
        return errors

    def select_sa_mode(self):
        """Select base Spectrum Analyzer mode (``INSTrument:SELect SA``)."""
        self.write("INST:SEL SA")
        self.info("N9010A: Selected Spectrum Analyzer mode.")

    """
    -------------------------------------------------------------------------
    Sweep configuration (frequency, bandwidth, sweep control)
    -------------------------------------------------------------------------
    """

    def center_frequency(self, f: float = None) -> float:
        """
        Set or query the center frequency in Hz (``FREQuency:CENTer``).

        Parameters
        ----------
        f : float, optional

        Returns
        -------
        float
        """
        if f is None:
            return float(self.query("FREQ:CENT?"))
        self.write(f"FREQ:CENT {f:g}")
        self.info(f"N9010A: Set center frequency to {f:g} Hz.")
        return f

    def span(self, span: float = None) -> float:
        """
        Set or query the frequency span in Hz (``FREQuency:SPAN``).

        A span of 0 Hz selects zero span (time-domain sweep at a fixed
        frequency).

        Parameters
        ----------
        span : float, optional

        Returns
        -------
        float
        """
        if span is None:
            return float(self.query("FREQ:SPAN?"))
        self.write(f"FREQ:SPAN {span:g}")
        self.info(f"N9010A: Set span to {span:g} Hz.")
        return span

    def start_frequency(self, f: float = None) -> float:
        """
        Set or query the sweep start frequency in Hz (``FREQuency:STARt``).

        Parameters
        ----------
        f : float, optional

        Returns
        -------
        float
        """
        if f is None:
            return float(self.query("FREQ:STAR?"))
        self.write(f"FREQ:STAR {f:g}")
        self.info(f"N9010A: Set start frequency to {f:g} Hz.")
        return f

    def stop_frequency(self, f: float = None) -> float:
        """
        Set or query the sweep stop frequency in Hz (``FREQuency:STOP``).

        Parameters
        ----------
        f : float, optional

        Returns
        -------
        float
        """
        if f is None:
            return float(self.query("FREQ:STOP?"))
        self.write(f"FREQ:STOP {f:g}")
        self.info(f"N9010A: Set stop frequency to {f:g} Hz.")
        return f

    def resolution_bandwidth(self, rbw: float = None, auto: bool = None) -> float:
        """
        Set or query the resolution bandwidth in Hz (``BANDwidth[:RESolution]``).

        Parameters
        ----------
        rbw : float, optional
            If given, sets the RBW (which also disables RBW:AUTO on the
            instrument, per its normal SCPI behavior).
        auto : bool, optional
            If given (and `rbw` is None), sets RBW:AUTO on/off without
            changing the current value.

        Returns
        -------
        float
            The resulting/queried RBW in Hz.
        """
        if auto is not None and rbw is None:
            self.write(f"BAND:AUTO {'ON' if auto else 'OFF'}")
            self.info(f"N9010A: Set RBW auto to {auto}.")
        if rbw is not None:
            self.write(f"BAND {rbw:g}")
            self.info(f"N9010A: Set resolution bandwidth to {rbw:g} Hz.")
            return rbw
        return float(self.query("BAND?"))

    def video_bandwidth(self, vbw: float = None, auto: bool = None) -> float:
        """
        Set or query the video bandwidth in Hz (``BANDwidth:VIDeo``).

        Parameters
        ----------
        vbw : float, optional
            If given, sets the VBW (which also disables VBW:AUTO).
        auto : bool, optional
            If given (and `vbw` is None), sets VBW:AUTO on/off.

        Returns
        -------
        float
        """
        if auto is not None and vbw is None:
            self.write(f"BAND:VID:AUTO {'ON' if auto else 'OFF'}")
            self.info(f"N9010A: Set VBW auto to {auto}.")
        if vbw is not None:
            self.write(f"BAND:VID {vbw:g}")
            self.info(f"N9010A: Set video bandwidth to {vbw:g} Hz.")
            return vbw
        return float(self.query("BAND:VID?"))

    def sweep_time(self, t: float = None, auto: bool = None) -> float:
        """
        Set or query the sweep time in seconds (``SWEep:TIME``).

        Parameters
        ----------
        t : float, optional
            If given, sets the sweep time (which also disables
            SWEep:TIME:AUTO).
        auto : bool, optional
            If given (and `t` is None), sets SWEep:TIME:AUTO on/off.

        Returns
        -------
        float
        """
        if auto is not None and t is None:
            self.write(f"SWE:TIME:AUTO {'ON' if auto else 'OFF'}")
            self.info(f"N9010A: Set sweep time auto to {auto}.")
        if t is not None:
            self.write(f"SWE:TIME {t:g}")
            self.info(f"N9010A: Set sweep time to {t:g} s.")
            return t
        return float(self.query("SWE:TIME?"))

    def sweep_points(self, n: int = None) -> int:
        """
        Set or query the number of sweep points (``SWEep:POINts``).

        Parameters
        ----------
        n : int, optional

        Returns
        -------
        int
        """
        if n is None:
            return int(float(self.query("SWE:POIN?")))
        self.write(f"SWE:POIN {int(n)}")
        self.info(f"N9010A: Set sweep points to {int(n)}.")
        return int(n)

    def continuous_sweep(self, state: bool = None) -> bool:
        """
        Set or query continuous vs single sweep (``INITiate:CONTinuous``).

        Set False before using `single_sweep()` for triggered acquisitions.

        Parameters
        ----------
        state : bool, optional

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query("INIT:CONT?")))
        self.write(f"INIT:CONT {'ON' if state else 'OFF'}")
        self.info(f"N9010A: Set continuous sweep to {state}.")
        return state

    def single_sweep(self):
        """
        Trigger one sweep and block until it completes.

        Sends ``INITiate:IMMediate;*OPC?`` as a single query, which the
        instrument only answers once the sweep finishes. Requires
        ``continuous_sweep(False)`` to have been set first, otherwise the
        instrument is already sweeping continuously and this has no useful
        effect.
        """
        self.query("INIT:IMM;*OPC?")
        self.info("N9010A: Triggered single sweep.")

    """
    -------------------------------------------------------------------------
    Trace configuration and acquisition
    -------------------------------------------------------------------------
    """

    _DETECTOR_MODES = {
        "NORM", "AVER", "POS", "SAMP", "NEG", "QPE", "EAV", "RAV",
    }

    def detector(self, mode: str = None, trace: int = 1) -> str:
        """
        Set or query the detector for a trace (``DETector:TRACe<n>``).

        Parameters
        ----------
        mode : str, optional
            One of {'NORM', 'AVER', 'POS', 'SAMP', 'NEG', 'QPE', 'EAV', 'RAV'}
            (normal, average, positive peak, sample, negative peak, quasi
            peak, EMI average, EMI RMS average).
        trace : int, default 1
            Trace number (1-6).

        Returns
        -------
        str
        """
        if mode is None:
            return self.query(f"DET:TRAC{trace}?").strip()
        mode = mode.upper()
        if mode not in self._DETECTOR_MODES:
            raise ValueError(f"mode must be one of {sorted(self._DETECTOR_MODES)}.")
        self.write(f"DET:TRAC{trace} {mode}")
        self.info(f"N9010A: Set trace {trace} detector to {mode}.")
        return mode

    _TRACE_TYPES = {"WRIT", "AVER", "MAXH", "MINH"}

    def trace_type(self, mode: str = None, trace: int = 1) -> str:
        """
        Set or query the trace mode (``TRACe<n>:TYPE``).

        Parameters
        ----------
        mode : str, optional
            One of {'WRIT', 'AVER', 'MAXH', 'MINH'} (clear-write, average,
            max hold, min hold).
        trace : int, default 1
            Trace number (1-6).

        Returns
        -------
        str
        """
        if mode is None:
            return self.query(f"TRAC{trace}:TYPE?").strip()
        mode = mode.upper()
        if mode not in self._TRACE_TYPES:
            raise ValueError(f"mode must be one of {sorted(self._TRACE_TYPES)}.")
        self.write(f"TRAC{trace}:TYPE {mode}")
        self.info(f"N9010A: Set trace {trace} type to {mode}.")
        return mode

    def average_count(self, n: int = None) -> int:
        """
        Set or query the trace averaging count (``AVERage:COUNt``).

        Parameters
        ----------
        n : int, optional

        Returns
        -------
        int
        """
        if n is None:
            return int(float(self.query("AVER:COUN?")))
        self.write(f"AVER:COUN {int(n)}")
        self.info(f"N9010A: Set average count to {int(n)}.")
        return int(n)

    _AVERAGE_TYPES = {"RMS", "LOG", "SCAL"}

    def average_type(self, type_: str = None) -> str:
        """
        Set or query the averaging type (``AVERage:TYPE``).

        Parameters
        ----------
        type_ : str, optional
            One of {'RMS', 'LOG', 'SCAL'} (power averaging, log-power
            averaging, voltage/scalar averaging).

        Returns
        -------
        str
        """
        if type_ is None:
            return self.query("AVER:TYPE?").strip()
        type_ = type_.upper()
        if type_ not in self._AVERAGE_TYPES:
            raise ValueError(f"type_ must be one of {sorted(self._AVERAGE_TYPES)}.")
        self.write(f"AVER:TYPE {type_}")
        self.info(f"N9010A: Set average type to {type_}.")
        return type_

    def average_clear(self):
        """Restart trace averaging from zero (``AVERage:CLEar``)."""
        self.write("AVER:CLE")
        self.info("N9010A: Cleared trace average.")

    def get_trace(self, trace: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch the current trace as (frequency, amplitude) arrays.

        Requests a 32-bit real binary block (``FORMat:DATA REAL,32`` with
        little-endian byte order) for ``TRACe:DATA?`` and reconstructs the
        (linear) frequency axis from the current start/stop frequency, since
        the trace query itself returns amplitude values only. This does not
        trigger a new sweep — call `single_sweep()` first for a fresh,
        settled acquisition (or leave the instrument in continuous sweep).

        Parameters
        ----------
        trace : int, default 1
            Trace number (1-6).

        Returns
        -------
        freq, amplitude : np.ndarray
            `freq` in Hz; `amplitude` in the current Y-axis unit (see
            `amplitude_unit()`).
        """
        self.write("FORM:DATA REAL,32")
        self.write("FORM:BORD SWAP")
        self.write(f"TRAC:DATA? TRACE{trace}")
        raw = self.read_raw()
        amplitude = parse_IEEE_488_2(raw)

        f_start = self.start_frequency()
        f_stop = self.stop_frequency()
        freq = np.linspace(f_start, f_stop, amplitude.size)
        return freq, amplitude

    """
    -------------------------------------------------------------------------
    Amplitude / front-end settings
    -------------------------------------------------------------------------
    """

    def reference_level(self, level: float = None) -> float:
        """
        Set or query the reference level, in the current Y-axis unit
        (``DISPlay:WINDow:TRACe:Y:SCALe:RLEVel``).

        Parameters
        ----------
        level : float, optional

        Returns
        -------
        float
        """
        if level is None:
            return float(self.query("DISP:WIND:TRAC:Y:SCAL:RLEV?"))
        self.write(f"DISP:WIND:TRAC:Y:SCAL:RLEV {level:g}")
        self.info(f"N9010A: Set reference level to {level:g}.")
        return level

    def attenuation(self, atten: float = None, auto: bool = None) -> float:
        """
        Set or query the input (mechanical) attenuation in dB
        (``POWer[:RF]:ATTenuation``).

        Parameters
        ----------
        atten : float, optional
            If given, sets the attenuation (which also disables
            attenuation:AUTO).
        auto : bool, optional
            If given (and `atten` is None), sets attenuation:AUTO on/off.

        Returns
        -------
        float
        """
        if auto is not None and atten is None:
            self.write(f"POW:ATT:AUTO {'ON' if auto else 'OFF'}")
            self.info(f"N9010A: Set attenuation auto to {auto}.")
        if atten is not None:
            self.write(f"POW:ATT {atten:g}")
            self.info(f"N9010A: Set attenuation to {atten:g} dB.")
            return atten
        return float(self.query("POW:ATT?"))

    def input_coupling(self, coupling: str = None) -> str:
        """
        Set or query the RF input coupling (``INPut:COUPling``).

        Parameters
        ----------
        coupling : str, optional
            One of {'AC', 'DC'}.

        Returns
        -------
        str
        """
        if coupling is None:
            return self.query("INP:COUP?").strip()
        coupling = coupling.upper()
        if coupling not in ("AC", "DC"):
            raise ValueError("coupling must be 'AC' or 'DC'.")
        self.write(f"INP:COUP {coupling}")
        self.info(f"N9010A: Set input coupling to {coupling}.")
        return coupling

    _AMPLITUDE_UNITS = {
        "DBM", "DBMV", "DBMA", "V", "W", "A",
        "DBUV", "DBUA", "DBPW", "DBUVM", "DBUAM", "DBPT", "DBG",
    }

    def amplitude_unit(self, unit: str = None) -> str:
        """
        Set or query the Y-axis amplitude unit (``UNIT:POWer``).

        Parameters
        ----------
        unit : str, optional
            One of {'DBM', 'DBMV', 'DBMA', 'V', 'W', 'A', 'DBUV', 'DBUA',
            'DBPW', 'DBUVM', 'DBUAM', 'DBPT', 'DBG'}.

        Returns
        -------
        str
        """
        if unit is None:
            return self.query("UNIT:POW?").strip()
        unit = unit.upper()
        if unit not in self._AMPLITUDE_UNITS:
            raise ValueError(f"unit must be one of {sorted(self._AMPLITUDE_UNITS)}.")
        self.write(f"UNIT:POW {unit}")
        self.info(f"N9010A: Set amplitude unit to {unit}.")
        return unit

    def display_enable(self, state: bool = None) -> bool:
        """
        Set or query whether the display updates during remote operation
        (``DISPlay:ENABle``).

        Turning this off speeds up remote acquisitions.

        Parameters
        ----------
        state : bool, optional

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query("DISP:ENAB?")))
        self.write(f"DISP:ENAB {'ON' if state else 'OFF'}")
        self.info(f"N9010A: Set display enable to {state}.")
        return state

    """
    -------------------------------------------------------------------------
    Channel Power measurement (integrated power in a band, for shot noise)
    -------------------------------------------------------------------------
    """

    def configure_channel_power(self, span: float = None, bandwidth: float = None):
        """
        Select the Channel Power measurement (``CONFigure:CHPower``) and,
        optionally, set its integration span/bandwidth.

        Switches the instrument's active measurement away from the swept
        trace (Spectrum Analyzer) measurement; call `configure_swept_sa()` to
        switch back.

        Parameters
        ----------
        span : float, optional
            Measurement span in Hz (see `channel_power_span`).
        bandwidth : float, optional
            Integration bandwidth in Hz (see `channel_power_bandwidth`).
        """
        self.write("CONF:CHP")
        self.info("N9010A: Configured Channel Power measurement.")
        if span is not None:
            self.channel_power_span(span)
        if bandwidth is not None:
            self.channel_power_bandwidth(bandwidth)

    def configure_swept_sa(self):
        """
        Return to the default Swept SA (trace) measurement
        (``CONFigure:SANalyzer``), undoing `configure_channel_power()`.
        """
        self.write("CONF:SAN")
        self.info("N9010A: Configured Swept SA measurement.")

    def channel_power_span(self, span: float = None, auto: bool = None) -> float:
        """
        Set or query the Channel Power measurement span in Hz
        (``CHPower:FREQuency:SPAN``) — the frequency range the measurement
        looks at, as distinct from the integration bandwidth actually summed
        over (see `channel_power_bandwidth`).

        Parameters
        ----------
        span : float, optional
            If given, sets the span (which also disables span:AUTO).
        auto : bool, optional
            If given (and `span` is None), sets span:AUTO on/off.

        Returns
        -------
        float
        """
        if auto is not None and span is None:
            self.write(f"CHP:FREQ:SPAN:AUTO {'ON' if auto else 'OFF'}")
            self.info(f"N9010A: Set channel power span auto to {auto}.")
        if span is not None:
            self.write(f"CHP:FREQ:SPAN {span:g}")
            self.info(f"N9010A: Set channel power span to {span:g} Hz.")
            return span
        return float(self.query("CHP:FREQ:SPAN?"))

    def channel_power_bandwidth(self, bandwidth: float = None, auto: bool = None) -> float:
        """
        Set or query the Channel Power integration bandwidth in Hz
        (``CHPower:BANDwidth[:RESolution]``) — this is the bandwidth the
        instrument actually integrates power over.

        Parameters
        ----------
        bandwidth : float, optional
            If given, sets the integration bandwidth (which also disables
            bandwidth:AUTO).
        auto : bool, optional
            If given (and `bandwidth` is None), sets bandwidth:AUTO on/off.

        Returns
        -------
        float
        """
        if auto is not None and bandwidth is None:
            self.write(f"CHP:BAND:AUTO {'ON' if auto else 'OFF'}")
            self.info(f"N9010A: Set channel power bandwidth auto to {auto}.")
        if bandwidth is not None:
            self.write(f"CHP:BAND {bandwidth:g}")
            self.info(f"N9010A: Set channel power integration bandwidth to "
                      f"{bandwidth:g} Hz.")
            return bandwidth
        return float(self.query("CHP:BAND?"))

    def channel_power_average_count(self, n: int = None) -> int:
        """
        Set or query the Channel Power averaging count
        (``CHPower:AVERage:COUNt``).

        Parameters
        ----------
        n : int, optional

        Returns
        -------
        int
        """
        if n is None:
            return int(float(self.query("CHP:AVER:COUN?")))
        self.write(f"CHP:AVER:COUN {int(n)}")
        self.info(f"N9010A: Set channel power average count to {int(n)}.")
        return int(n)

    def channel_power_init(self):
        """
        Trigger a Channel Power acquisition and block until it completes.

        Sends ``INITiate:CHPower;*OPC?`` as a single query. Requires
        `configure_channel_power()` to have been called first.
        """
        self.query("INIT:CHP;*OPC?")
        self.info("N9010A: Triggered channel power acquisition.")

    def get_channel_power(self) -> float:
        """
        Fetch the integrated channel power from the last acquisition, in the
        current Y-axis unit (``FETCh:CHPower:CHPower?``).

        Does not trigger a new acquisition — call `channel_power_init()`
        first for a fresh reading.

        Returns
        -------
        float
        """
        return float(self.query("FETC:CHP:CHP?"))

    def get_channel_power_density(self) -> float:
        """
        Fetch the channel power spectral density from the last acquisition
        (power normalized per Hz of the integration bandwidth), in the
        current Y-axis unit per Hz (``FETCh:CHPower:DENSity?``).

        Does not trigger a new acquisition — call `channel_power_init()`
        first for a fresh reading.

        Returns
        -------
        float
        """
        return float(self.query("FETC:CHP:DENS?"))

    def measure_channel_power(self) -> Tuple[float, float]:
        """
        Trigger a Channel Power acquisition and fetch both results.

        Convenience wrapper equivalent to `channel_power_init()` followed by
        `get_channel_power()` and `get_channel_power_density()` — both
        results come from the same acquisition, so only one trigger is
        needed. Requires `configure_channel_power()` to have been called
        first.

        Returns
        -------
        power, density : float
            Integrated channel power and its spectral density, both in the
            current Y-axis unit (density per Hz).
        """
        self.channel_power_init()
        return self.get_channel_power(), self.get_channel_power_density()

    """
    -------------------------------------------------------------------------
    Markers (incl. noise marker / band density, for noise-floor benchmarking)
    -------------------------------------------------------------------------
    """

    def marker_state(self, n: int = 1, state: bool = None) -> bool:
        """
        Set or query whether marker `n` is on (``CALCulate:MARKer<n>:STATe``).

        Parameters
        ----------
        n : int, default 1
            Marker number (1-12).
        state : bool, optional

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query(f"CALC:MARK{n}:STAT?")))
        self.write(f"CALC:MARK{n}:STAT {'ON' if state else 'OFF'}")
        self.info(f"N9010A: Set marker {n} state to {state}.")
        return state

    def marker_x(self, n: int = 1, x: float = None) -> float:
        """
        Set or query marker `n`'s X (frequency) position in Hz
        (``CALCulate:MARKer<n>:X``).

        Parameters
        ----------
        n : int, default 1
        x : float, optional

        Returns
        -------
        float
        """
        if x is None:
            return float(self.query(f"CALC:MARK{n}:X?"))
        self.write(f"CALC:MARK{n}:X {x:g}")
        self.info(f"N9010A: Set marker {n} X to {x:g} Hz.")
        return x

    def marker_y(self, n: int = 1) -> float:
        """
        Read marker `n`'s Y (result) value (``CALCulate:MARKer<n>:Y?``).

        This is the marker's ordinary amplitude readout, and is also how the
        result of `marker_function` (e.g. the noise marker) is read back.

        Parameters
        ----------
        n : int, default 1

        Returns
        -------
        float
        """
        return float(self.query(f"CALC:MARK{n}:Y?"))

    _PEAK_SEARCH_MODES = {
        "max"  : "MAX",
        "next" : "MAX:NEXT",
        "left" : "MAX:LEFT",
        "right": "MAX:RIGH",
    }

    def marker_peak_search(self, n: int = 1, mode: str = "max"):
        """
        Move marker `n` to a peak (``CALCulate:MARKer<n>:MAXimum...``).

        Parameters
        ----------
        n : int, default 1
        mode : str, default 'max'
            One of {'max', 'next', 'left', 'right'} — the global peak, the
            next-highest peak, or the next peak to the left/right of the
            marker's current position.
        """
        mode = mode.lower()
        if mode not in self._PEAK_SEARCH_MODES:
            raise ValueError(f"mode must be one of {sorted(self._PEAK_SEARCH_MODES)}.")
        self.write(f"CALC:MARK{n}:{self._PEAK_SEARCH_MODES[mode]}")
        self.info(f"N9010A: Marker {n} peak search ({mode}).")

    _MARKER_FUNCTIONS = {"NOIS", "BPOW", "BDEN", "OFF"}

    def marker_function(self, n: int = 1, func: str = None) -> str:
        """
        Set or query marker `n`'s function (``CALCulate:MARKer<n>:FUNCtion``).

        Parameters
        ----------
        n : int, default 1
        func : str, optional
            One of {'NOIS', 'BPOW', 'BDEN', 'OFF'}: noise marker (result
            normalized to a 1 Hz bandwidth), band power (total power in a
            marker-defined band, see `marker_band_span`), band density (band
            power normalized by the band's noise bandwidth), or off. Read the
            result with `marker_y`.

        Returns
        -------
        str
        """
        if func is None:
            return self.query(f"CALC:MARK{n}:FUNC?").strip()
        func = func.upper()
        if func not in self._MARKER_FUNCTIONS:
            raise ValueError(f"func must be one of {sorted(self._MARKER_FUNCTIONS)}.")
        self.write(f"CALC:MARK{n}:FUNC {func}")
        self.info(f"N9010A: Set marker {n} function to {func}.")
        return func

    def marker_band_span(self, n: int = 1, span: float = None) -> float:
        """
        Set or query the band width used by marker `n`'s BPOWer/BDENsity
        function, in Hz (``CALCulate:MARKer<n>:FUNCtion:BAND:SPAN``).

        Parameters
        ----------
        n : int, default 1
        span : float, optional

        Returns
        -------
        float
        """
        if span is None:
            return float(self.query(f"CALC:MARK{n}:FUNC:BAND:SPAN?"))
        self.write(f"CALC:MARK{n}:FUNC:BAND:SPAN {span:g}")
        self.info(f"N9010A: Set marker {n} band span to {span:g} Hz.")
        return span

    def get_noise_marker(self, freq: float, n: int = 1) -> float:
        """
        Convenience: place marker `n` at `freq`, set it to the noise-marker
        function, and read back the result.

        Parameters
        ----------
        freq : float
            Frequency in Hz to place the marker at.
        n : int, default 1

        Returns
        -------
        float
            Noise density at `freq`, in the current Y-axis unit per Hz
            (e.g. dBm/Hz).
        """
        self.marker_state(n, True)
        self.marker_x(n, freq)
        self.marker_function(n, "NOIS")
        return self.marker_y(n)

    def get_band_density(self, center: float, span: float, n: int = 1) -> float:
        """
        Convenience: place marker `n` at `center`, set it to the band-density
        function with the given band `span`, and read back the result.

        Parameters
        ----------
        center : float
            Band center frequency in Hz.
        span : float
            Band width in Hz.
        n : int, default 1

        Returns
        -------
        float
            Band power density, in the current Y-axis unit per Hz.
        """
        self.marker_state(n, True)
        self.marker_x(n, center)
        self.marker_function(n, "BDEN")
        self.marker_band_span(n, span)
        return self.marker_y(n)

    """
    -------------------------------------------------------------------------
    Serialization
    -------------------------------------------------------------------------
    """

    def save_state_json(self, path: str):
        """Save the spectrum analyzer state to JSON locally."""
        super().save_state_json(path)

    def load_state_json(self, path: str):
        """Load the spectrum analyzer state from JSON locally."""
        super().load_state_json(path)

    # Settings serialized as get/set accessors (trace/channel index 1 only).
    _STATE_SETTINGS = [
        "start_frequency", "stop_frequency",
        "resolution_bandwidth", "video_bandwidth",
        "sweep_time", "sweep_points", "continuous_sweep",
        "detector", "trace_type", "average_count", "average_type",
        "reference_level", "attenuation", "input_coupling",
        "amplitude_unit", "display_enable",
    ]

    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize connection config plus the swept-SA-mode settings listed
        in `_STATE_SETTINGS` (trace 1 only).

        Channel Power configuration and marker state are deliberately *not*
        captured: they describe an alternate/orthogonal measurement setup
        this driver has no way of knowing was actually the "active" one
        when the snapshot was taken, so blindly restoring them alongside
        the swept-trace settings below could silently leave the instrument
        in a measurement mode the caller didn't ask for. Reconfigure those
        explicitly via `configure_channel_power()`/the marker methods after
        restoring, if needed. Also note `resolution_bandwidth`/
        `video_bandwidth`/`sweep_time` only round-trip their numeric value —
        whether AUTO mode was on for any of them is not itself queryable
        through this driver, so is not captured either.
        """
        state = super()._serialize_state()
        for setting in self._STATE_SETTINGS:
            try:
                state[setting] = getattr(self, setting)()
            except Exception:
                continue
        return state

    def _deserialize_state(self, state: Dict[str, Any]):
        """
        Restore settings from serialized state (see `_serialize_state` for
        what is and isn't captured).
        """
        super()._deserialize_state(state)
        self.select_sa_mode()
        for setting in self._STATE_SETTINGS:
            if setting in state:
                try:
                    getattr(self, setting)(state[setting])
                except Exception:
                    continue

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "n9010a":
        """
        Construct from serialized config.

        Parameters
        ----------
        config : dict
            Output from `_serialize_state()`, plus 'registry_id'.
        """
        registry_id = config.pop('registry_id')
        resource_name = config.pop('resource_name')
        instance = cls(
            resource_name=resource_name,
            registry_id=registry_id,
            skip_connect=False,
            **config,
        )
        instance._deserialize_state(config)
        return instance


if __name__ == '__main__':
    """Example self-test of the N9010A driver using dummyResource."""

    import logging
    import os
    import struct
    import sys
    import tempfile

    logger = logging.getLogger('n9010a_test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sa = n9010a(resource_name='DEBUG', logger=logger)

    def get_idn(obj):
        return "Keysight Technologies,N9010A,MY00000000,A.30.00\n"
    sa.device.add_command(r'\*IDN\?', get_idn)

    sa.device.attr['errors'] = ['+0,"No error"\n']
    def get_error(obj):
        if len(obj.attr['errors']) > 1:
            return obj.attr['errors'].pop(0)
        return obj.attr['errors'][0]
    sa.device.add_command(r'SYST:ERR\?', get_error)

    def reset(obj):
        obj.attr['errors'] = ['+0,"No error"\n']
    sa.device.add_command(r'\*RST', reset)

    def select_mode(obj, mode):
        obj.attr['mode'] = mode
    sa.device.add_command(r'INST:SEL (\S+)', select_mode)

    print(f"IDN: {sa.idn()}")

    sa.reset()
    print(f"Error queue after reset: {sa.get_all_errors()}")

    sa.device.attr['errors'] = ['+113,"Undefined header"\n', '+0,"No error"\n']
    print(f"Injected errors: {sa.get_all_errors()}")

    sa.select_sa_mode()
    print(f"Mode selected: {sa.device.attr['mode']}")

    # ── Phase 2: sweep/trace/front-end settings + trace fetch ──────────────

    def add_scalar(query_re, set_re, key, default, caster=float):
        """Wire up a simple SCPI get/set pair backed by sa.device.attr[key]."""
        sa.device.attr[key] = default
        def get(obj):
            return f"{obj.attr[key]}\n"
        sa.device.add_command(query_re, get)
        def set_(obj, val):
            obj.attr[key] = caster(val)
        sa.device.add_command(set_re, set_)

    def add_bool(query_re, set_re, key, default=1):
        sa.device.attr[key] = default
        def get(obj):
            return f"{obj.attr[key]}\n"
        sa.device.add_command(query_re, get)
        def set_(obj, val):
            obj.attr[key] = 1 if val in ('ON', '1') else 0
        sa.device.add_command(set_re, set_)

    add_scalar(r'FREQ:CENT\?', r'FREQ:CENT (\S+)', 'center', 1.0e9)
    add_scalar(r'FREQ:SPAN\?', r'FREQ:SPAN (\S+)', 'span', 1.0e8)
    add_scalar(r'FREQ:STAR\?', r'FREQ:STAR (\S+)', 'start', 0.0)
    add_scalar(r'FREQ:STOP\?', r'FREQ:STOP (\S+)', 'stop', 2.0e9)
    add_scalar(r'BAND\?', r'BAND (\S+)', 'rbw', 1.0e6)
    add_scalar(r'BAND:VID\?', r'BAND:VID (\S+)', 'vbw', 3.0e6)
    add_scalar(r'SWE:TIME\?', r'SWE:TIME (\S+)', 'sweep_time', 1.0e-3)
    add_scalar(r'SWE:POIN\?', r'SWE:POIN (\S+)', 'points', 1001, caster=int)
    add_bool(r'INIT:CONT\?', r'INIT:CONT (\S+)', 'cont', 1)
    add_scalar(r'AVER:COUN\?', r'AVER:COUN (\S+)', 'avg_count', 10, caster=int)
    add_scalar(r'POW:ATT\?', r'POW:ATT (\S+)', 'atten', 10.0)
    add_scalar(r'DISP:WIND:TRAC:Y:SCAL:RLEV\?', r'DISP:WIND:TRAC:Y:SCAL:RLEV (\S+)',
               'rlev', 0.0)
    add_scalar(r'DET:TRAC1\?', r'DET:TRAC1 (\S+)', 'detector', 'NORM', caster=str)
    add_scalar(r'TRAC1:TYPE\?', r'TRAC1:TYPE (\S+)', 'trace_type', 'WRIT', caster=str)
    add_scalar(r'AVER:TYPE\?', r'AVER:TYPE (\S+)', 'avg_type', 'LOG', caster=str)
    add_scalar(r'INP:COUP\?', r'INP:COUP (\S+)', 'coupling', 'DC', caster=str)
    add_scalar(r'UNIT:POW\?', r'UNIT:POW (\S+)', 'unit', 'DBM', caster=str)
    add_bool(r'DISP:ENAB\?', r'DISP:ENAB (\S+)', 'disp_enable', 1)

    def opc(obj):
        return "1\n"
    sa.device.add_command(r'INIT:IMM;\*OPC\?', opc)

    def average_clear(obj):
        obj.attr['avg_count'] = 0
    sa.device.add_command(r'AVER:CLE', average_clear)

    def build_trace_response(obj, trace):
        # 11 fake amplitude points, packed as a #<n><len> IEEE 488.2 block.
        values = [-90.0 + i for i in range(11)]
        payload = struct.pack(f'<{len(values)}f', *values)
        header = f"#{len(str(len(payload)))}{len(payload)}"
        return header.encode() + payload
    sa.device.add_command(r'TRAC:DATA\? TRACE(\d+)', build_trace_response)

    print(f"Center frequency: {sa.center_frequency(1.5e9)} Hz "
          f"(query: {sa.center_frequency()} Hz)")
    print(f"Span: {sa.span(5e8)} Hz")
    sa.start_frequency(1e6)
    sa.stop_frequency(11e6)
    print(f"Start/stop: {sa.start_frequency()} Hz / {sa.stop_frequency()} Hz")
    print(f"RBW: {sa.resolution_bandwidth(1e3)} Hz (auto off)")
    sa.resolution_bandwidth(auto=True)
    print(f"VBW: {sa.video_bandwidth(3e3)} Hz")
    print(f"Sweep time: {sa.sweep_time(0.05)} s")
    print(f"Sweep points: {sa.sweep_points(1001)}")

    sa.continuous_sweep(False)
    sa.single_sweep()
    print(f"Continuous sweep: {sa.continuous_sweep()}")

    print(f"Detector: {sa.detector('AVER')}")
    print(f"Trace type: {sa.trace_type('AVER')}")
    print(f"Average count: {sa.average_count(50)}")
    print(f"Average type: {sa.average_type('RMS')}")
    sa.average_clear()

    print(f"Reference level: {sa.reference_level(-10.0)} dBm")
    print(f"Attenuation: {sa.attenuation(20.0)} dB")
    print(f"Input coupling: {sa.input_coupling('DC')}")
    print(f"Amplitude unit: {sa.amplitude_unit('DBM')}")
    print(f"Display enable: {sa.display_enable(False)}")

    freq, amplitude = sa.get_trace()
    print(f"Trace: {amplitude.size} points, "
          f"{freq[0]/1e6:.2f}-{freq[-1]/1e6:.2f} MHz, "
          f"amplitude[0:3]={amplitude[:3]}")

    # ── Phase 3: Channel Power measurement ──────────────────────────────────

    def select_measurement(obj, meas):
        obj.attr['measurement'] = meas
    sa.device.add_command(r'CONF:(CHP|SAN)', select_measurement)

    add_scalar(r'CHP:FREQ:SPAN\?', r'CHP:FREQ:SPAN (\S+)', 'chp_span', 1e6)
    add_scalar(r'CHP:BAND\?', r'CHP:BAND (\S+)', 'chp_bw', 3e5)
    add_scalar(r'CHP:AVER:COUN\?', r'CHP:AVER:COUN (\S+)', 'chp_avg_count', 10,
               caster=int)

    def chp_opc(obj):
        return "1\n"
    sa.device.add_command(r'INIT:CHP;\*OPC\?', chp_opc)

    # Read-only fetch results (no setter, unlike the settings above).
    sa.device.attr['chp_power'] = -42.5
    sa.device.add_command(r'FETC:CHP:CHP\?', lambda obj: f"{obj.attr['chp_power']}\n")
    sa.device.attr['chp_density'] = -75.3
    sa.device.add_command(r'FETC:CHP:DENS\?', lambda obj: f"{obj.attr['chp_density']}\n")

    sa.configure_channel_power(span=2e6, bandwidth=5e5)
    print(f"Channel power span: {sa.channel_power_span()} Hz")
    print(f"Channel power bandwidth: {sa.channel_power_bandwidth()} Hz")
    print(f"Channel power average count: {sa.channel_power_average_count(20)}")

    power, density = sa.measure_channel_power()
    print(f"Channel power: {power} dBm, density: {density} dBm/Hz")

    sa.configure_swept_sa()
    print(f"Measurement after configure_swept_sa: {sa.device.attr['measurement']}")

    # ── Phase 4: markers ─────────────────────────────────────────────────────

    add_bool(r'CALC:MARK1:STAT\?', r'CALC:MARK1:STAT (\S+)', 'mkr_state', 0)
    add_scalar(r'CALC:MARK1:X\?', r'CALC:MARK1:X (\S+)', 'mkr_x', 0.0)
    add_scalar(r'CALC:MARK1:FUNC\?', r'CALC:MARK1:FUNC (\S+)', 'mkr_func', 'OFF',
               caster=str)
    add_scalar(r'CALC:MARK1:FUNC:BAND:SPAN\?', r'CALC:MARK1:FUNC:BAND:SPAN (\S+)',
               'mkr_band_span', 0.0)

    # Marker Y result depends on which function is active (mocks a Y readout
    # that differs for NOISe vs BDENsity vs a plain marker, as the real
    # instrument would).
    def get_marker_y(obj):
        return {"NOIS": -160.0, "BDEN": -150.0}.get(obj.attr['mkr_func'], -40.0)
    sa.device.add_command(r'CALC:MARK1:Y\?', lambda obj: f"{get_marker_y(obj)}\n")

    def peak_search(obj, kind):
        obj.attr['mkr_x'] = 5.0e6  # pretend the peak is at 5 MHz
    sa.device.add_command(r'CALC:MARK1:(MAX(?::NEXT|:LEFT|:RIGH)?)', peak_search)

    print(f"Marker state: {sa.marker_state(state=True)}")
    sa.marker_x(x=3e6)
    print(f"Marker X: {sa.marker_x()} Hz")
    sa.marker_peak_search(mode="max")
    print(f"Marker X after peak search: {sa.marker_x()} Hz")
    print(f"Marker Y: {sa.marker_y()}")

    print(f"Marker function: {sa.marker_function(func='NOIS')}")
    print(f"Noise marker at 5 MHz: {sa.get_noise_marker(5e6)} dBm/Hz")

    print(f"Marker function: {sa.marker_function(func='BDEN')}")
    print(f"Marker band span: {sa.marker_band_span(span=2e5)} Hz")
    print(f"Band density at 5 MHz over 200 kHz: "
          f"{sa.get_band_density(5e6, 2e5)} dBm/Hz")

    # ── Serialization round trip ─────────────────────────────────────────────

    state = sa._serialize_state()
    captured = {k: state[k] for k in sa._STATE_SETTINGS if k in state}
    print(f"Serialized settings: {captured}")

    sa.resolution_bandwidth(999.0)  # perturb before restoring
    sa._deserialize_state(state)
    print(f"RBW after _deserialize_state round trip: "
          f"{sa.resolution_bandwidth()} Hz (should match the captured value)")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "n9010a_state.json")
        sa.save_state_json(json_path)
        print(f"Saved state JSON to {json_path} "
              f"({os.path.getsize(json_path)} bytes).")

        sa.resolution_bandwidth(999.0)  # perturb again before reloading
        sa.load_state_json(json_path)
        print(f"RBW after save_state_json/load_state_json round trip: "
              f"{sa.resolution_bandwidth()} Hz")

    config = {'registry_id': 'n9010a_test_2', 'resource_name': 'DEBUG'}
    sa2 = n9010a.from_config(config)
    print(f"from_config constructed a fresh instance: {type(sa2).__name__}, "
          f"resource_name={sa2.resource_name!r}")
