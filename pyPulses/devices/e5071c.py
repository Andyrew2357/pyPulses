"""
Control class for the Agilent/Keysight E5071C ENA vector network analyzer.

This is a partial driver covering common, simple use cases only: S-parameter
sweeps, the built-in (Option 010) time-domain/TDR transform, and basic
calibration control, rather than the full instrument command set. See
``E5071C_PLAN.md`` at the repo root for the design/planning document this
driver is built from, including the SCPI command reference it was written
against (sourced from the ENA Help System's Programming chapter, with enum
values spot-checked against the alphabetical Command Reference).

This module implements Phases 1-4 of that plan: connection handling
(inherited from ``pyvisaDevice``), identification/reset/preset, the error
queue, the channel/trace/measurement setup primitives every other command
depends on (this instrument, unlike the N9010A, requires an explicit
Channel -> Trace -> Measurement setup — assigning an S-parameter to a
numbered trace on a numbered channel, then selecting it — before any sweep
or data command works; there is no PNA-style
``DISPlay:WINDow:TRACe:FEED``), sweep configuration, data acquisition
(``get_trace()``/``get_stimulus()`` as complex S-parameter/frequency arrays,
plus a handful of light, generic, lossless conversions of that data — log
magnitude, phase, VSWR, group delay), the built-in (Option 010)
time-domain/TDR transform, basic calibration control, and JSON state
serialization (see `_STATE_SETTINGS`/`_serialize_state` for what is and
isn't captured).

Note on calibration: this instrument has no PNA-style command to activate a
saved calibration set by name (confirmed absent from the manual). Recalling
a calibration therefore means recalling a full instrument-state file that
already has one baked in (`load_state()`) — see that method's docstring.
This driver deliberately does not implement a guided SOLT/ECal calibration
sequence (out of scope; see ``E5071C_PLAN.md``).

Note on TDR: this instrument's "TDR" name covers two distinct, separately
licensed things. This driver targets **Option 010** ("Time Domain Analysis"),
a normal SCPI-controllable transform built into the instrument's own
firmware. There is also a separate "Enhanced Time Domain Analysis" (Option
TDR) PC application, controlled over its own independent VISA/SICL-LAN
session rather than the main instrument session — that one is out of scope
for this driver entirely (see ``E5071C_PLAN.md``).
"""

from .pyvisa_device import pyvisaDevice, parse_IEEE_488_2
from .registry import register_hardware_class

from logging import Logger
from typing import Any, Dict, List, Tuple

import numpy as np


@register_hardware_class("e5071c")
class e5071c(pyvisaDevice):
    """
    Class representation of the Agilent/Keysight E5071C ENA vector network
    analyzer.

    Talks plain SCPI over VISA (LAN/GPIB), so this class follows the same
    conventions as the other SCPI instrument drivers in this package: a
    get-or-set method per setting (``None`` queries, a value writes), and
    numpy-style docstrings.

    Unlike a simple spectrum analyzer, this instrument requires an explicit
    Channel -> Trace -> Measurement setup before any sweep/data command
    works: pick a channel number (``ch``, default 1 throughout this driver),
    decide how many traces that channel has, and assign an S-parameter to a
    specific trace number (``tr``, default 1) before it can be swept or
    read. Every method that touches a channel or trace takes ``ch``/``tr``
    as leading parameters for this reason.
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
            VISA resource name (e.g. ``"TCPIP0::192.168.1.20::inst0::INSTR"``,
            ``"GPIB0::16::INSTR"``), or ``"DEBUG"`` for a hardware-free
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
        """Reset via the standard IEEE-488.2 command (``*RST``)."""
        self.write("*RST")
        self.info("E5071C: Reset to default state (*RST).")

    def preset(self):
        """
        Preset the instrument to its default state (``SYSTem:PRESet``).

        This is the reset form used throughout the manual's own sample
        programs, and is preferred over `reset()` for this instrument. Note
        that neither this nor `reset()` resets the `data_format` setting
        (added in Phase 2) — that persists across presets/resets.
        """
        self.write("SYST:PRES")
        self.info("E5071C: Preset to default state (SYST:PRES).")

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

    """
    -------------------------------------------------------------------------
    Channel / trace / measurement setup
    -------------------------------------------------------------------------
    """

    def trace_count(self, n: int = None, ch: int = 1) -> int:
        """
        Set or query the number of traces on channel `ch`
        (``CALCulate<ch>:PARameter:COUNt``).

        Parameters
        ----------
        n : int, optional
        ch : int, default 1

        Returns
        -------
        int
        """
        if n is None:
            return int(float(self.query(f"CALC{ch}:PAR:COUN?")))
        self.write(f"CALC{ch}:PAR:COUN {int(n)}")
        self.info(f"E5071C: Set channel {ch} trace count to {int(n)}.")
        return int(n)

    def define_parameter(self, sparam: str = None, ch: int = 1, tr: int = 1) -> str:
        """
        Set or query the S-parameter assigned to trace `tr` on channel `ch`
        (``CALCulate<ch>:PARameter<tr>:DEFine``).

        Parameters
        ----------
        sparam : str, optional
            e.g. ``'S11'``, ``'S21'``, ``'S12'``, ``'S22'`` (or ``'Smn'`` for
            m,n up to the instrument's port count on 3/4-port units).
        ch : int, default 1
        tr : int, default 1

        Returns
        -------
        str
        """
        if sparam is None:
            return self.query(f"CALC{ch}:PAR{tr}:DEF?").strip()
        sparam = sparam.upper()
        self.write(f"CALC{ch}:PAR{tr}:DEF {sparam}")
        self.info(f"E5071C: Set channel {ch} trace {tr} parameter to {sparam}.")
        return sparam

    def select_trace(self, ch: int = 1, tr: int = 1):
        """
        Select/activate trace `tr` on channel `ch`
        (``CALCulate<ch>:PARameter<tr>:SELect``).

        Many channel-scoped commands (e.g. trace format, data readout, added
        in later phases) implicitly act on whichever trace was selected
        last, so this must be called before them whenever more than one
        trace exists on the channel.

        Parameters
        ----------
        ch : int, default 1
        tr : int, default 1
        """
        self.write(f"CALC{ch}:PAR{tr}:SEL")
        self.info(f"E5071C: Selected channel {ch} trace {tr}.")

    def configure_measurement(self, sparam: str, ch: int = 1, tr: int = 1):
        """
        Convenience: ensure channel `ch` has at least `tr` traces, assign
        `sparam` to trace `tr`, and select it.

        Only grows the trace count (never shrinks it), so it won't disturb
        other already-configured traces on the same channel.

        Parameters
        ----------
        sparam : str
            e.g. ``'S11'``, ``'S21'``, ``'S12'``, ``'S22'``.
        ch : int, default 1
        tr : int, default 1
        """
        if self.trace_count(ch=ch) < tr:
            self.trace_count(tr, ch=ch)
        self.define_parameter(sparam, ch=ch, tr=tr)
        self.select_trace(ch=ch, tr=tr)

    """
    -------------------------------------------------------------------------
    Sweep configuration
    -------------------------------------------------------------------------
    """

    _SWEEP_TYPES = {"LIN", "LOG", "SEGM", "POW"}

    def sweep_type(self, mode: str = None, ch: int = 1) -> str:
        """
        Set or query the sweep type on channel `ch`
        (``SENSe<ch>:SWEep:TYPE``).

        Parameters
        ----------
        mode : str, optional
            One of {'LIN', 'LOG', 'SEGM', 'POW'} (linear frequency, log
            frequency, segment sweep, power sweep). Power-sweep-specific
            setup (``SOURce:POWer:STARt``/etc.) and the segment sweep table
            (``SENSe:SEGMent:DATA``) are not implemented by this driver.
        ch : int, default 1

        Returns
        -------
        str
        """
        if mode is None:
            return self.query(f"SENS{ch}:SWE:TYPE?").strip()
        mode = mode.upper()
        if mode not in self._SWEEP_TYPES:
            raise ValueError(f"mode must be one of {sorted(self._SWEEP_TYPES)}.")
        self.write(f"SENS{ch}:SWE:TYPE {mode}")
        self.info(f"E5071C: Set channel {ch} sweep type to {mode}.")
        return mode

    def center_frequency(self, f: float = None, ch: int = 1) -> float:
        """
        Set or query the center frequency in Hz on channel `ch`
        (``SENSe<ch>:FREQuency:CENTer``).

        Parameters
        ----------
        f : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if f is None:
            return float(self.query(f"SENS{ch}:FREQ:CENT?"))
        self.write(f"SENS{ch}:FREQ:CENT {f:g}")
        self.info(f"E5071C: Set channel {ch} center frequency to {f:g} Hz.")
        return f

    def span(self, span: float = None, ch: int = 1) -> float:
        """
        Set or query the frequency span in Hz on channel `ch`
        (``SENSe<ch>:FREQuency:SPAN``).

        Parameters
        ----------
        span : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if span is None:
            return float(self.query(f"SENS{ch}:FREQ:SPAN?"))
        self.write(f"SENS{ch}:FREQ:SPAN {span:g}")
        self.info(f"E5071C: Set channel {ch} span to {span:g} Hz.")
        return span

    def start_frequency(self, f: float = None, ch: int = 1) -> float:
        """
        Set or query the sweep start frequency in Hz on channel `ch`
        (``SENSe<ch>:FREQuency:STARt``).

        Parameters
        ----------
        f : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if f is None:
            return float(self.query(f"SENS{ch}:FREQ:STAR?"))
        self.write(f"SENS{ch}:FREQ:STAR {f:g}")
        self.info(f"E5071C: Set channel {ch} start frequency to {f:g} Hz.")
        return f

    def stop_frequency(self, f: float = None, ch: int = 1) -> float:
        """
        Set or query the sweep stop frequency in Hz on channel `ch`
        (``SENSe<ch>:FREQuency:STOP``).

        Parameters
        ----------
        f : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if f is None:
            return float(self.query(f"SENS{ch}:FREQ:STOP?"))
        self.write(f"SENS{ch}:FREQ:STOP {f:g}")
        self.info(f"E5071C: Set channel {ch} stop frequency to {f:g} Hz.")
        return f

    def sweep_points(self, n: int = None, ch: int = 1) -> int:
        """
        Set or query the number of sweep points on channel `ch`
        (``SENSe<ch>:SWEep:POINts``).

        Parameters
        ----------
        n : int, optional
        ch : int, default 1

        Returns
        -------
        int
        """
        if n is None:
            return int(float(self.query(f"SENS{ch}:SWE:POIN?")))
        self.write(f"SENS{ch}:SWE:POIN {int(n)}")
        self.info(f"E5071C: Set channel {ch} sweep points to {int(n)}.")
        return int(n)

    def if_bandwidth(self, bw: float = None, ch: int = 1) -> float:
        """
        Set or query the IF bandwidth in Hz on channel `ch`
        (``SENSe<ch>:BANDwidth``).

        Parameters
        ----------
        bw : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if bw is None:
            return float(self.query(f"SENS{ch}:BAND?"))
        self.write(f"SENS{ch}:BAND {bw:g}")
        self.info(f"E5071C: Set channel {ch} IF bandwidth to {bw:g} Hz.")
        return bw

    def power(self, level: float = None, ch: int = 1) -> float:
        """
        Set or query the source power in dBm on channel `ch`
        (``SOURce<ch>:POWer``).

        Parameters
        ----------
        level : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if level is None:
            return float(self.query(f"SOUR{ch}:POW?"))
        self.write(f"SOUR{ch}:POW {level:g}")
        self.info(f"E5071C: Set channel {ch} source power to {level:g} dBm.")
        return level

    def average_state(self, state: bool = None, ch: int = 1) -> bool:
        """
        Set or query whether sweep averaging is enabled on channel `ch`
        (``SENSe<ch>:AVERage``).

        Parameters
        ----------
        state : bool, optional
        ch : int, default 1

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query(f"SENS{ch}:AVER?")))
        self.write(f"SENS{ch}:AVER {'ON' if state else 'OFF'}")
        self.info(f"E5071C: Set channel {ch} averaging to {state}.")
        return state

    def average_count(self, n: int = None, ch: int = 1) -> int:
        """
        Set or query the averaging count on channel `ch`
        (``SENSe<ch>:AVERage:COUNt``).

        Parameters
        ----------
        n : int, optional
        ch : int, default 1

        Returns
        -------
        int
        """
        if n is None:
            return int(float(self.query(f"SENS{ch}:AVER:COUN?")))
        self.write(f"SENS{ch}:AVER:COUN {int(n)}")
        self.info(f"E5071C: Set channel {ch} average count to {int(n)}.")
        return int(n)

    def average_clear(self, ch: int = 1):
        """
        Restart sweep averaging from zero on channel `ch`
        (``SENSe<ch>:AVERage:CLEar``).

        Parameters
        ----------
        ch : int, default 1
        """
        self.write(f"SENS{ch}:AVER:CLE")
        self.info(f"E5071C: Cleared channel {ch} averaging.")

    def continuous_sweep(self, state: bool = None, ch: int = 1) -> bool:
        """
        Set or query continuous vs. hold sweep mode on channel `ch`
        (``INITiate<ch>:CONTinuous``).

        Set False before using `single_sweep()` for triggered acquisitions.

        Parameters
        ----------
        state : bool, optional
        ch : int, default 1

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query(f"INIT{ch}:CONT?")))
        self.write(f"INIT{ch}:CONT {'ON' if state else 'OFF'}")
        self.info(f"E5071C: Set channel {ch} continuous sweep to {state}.")
        return state

    _TRIGGER_SOURCES = {"INT", "EXT", "MAN", "BUS"}

    def trigger_source(self, source: str = None) -> str:
        """
        Set or query the trigger source (``TRIGger[:SEQuence]:SOURce``).

        This setting is global (not per-channel), unlike most other sweep
        settings in this driver.

        Parameters
        ----------
        source : str, optional
            One of {'INT', 'EXT', 'MAN', 'BUS'}. Use 'BUS' before
            `single_sweep()`.

        Returns
        -------
        str
        """
        if source is None:
            return self.query("TRIG:SOUR?").strip()
        source = source.upper()
        if source not in self._TRIGGER_SOURCES:
            raise ValueError(f"source must be one of {sorted(self._TRIGGER_SOURCES)}.")
        self.write(f"TRIG:SOUR {source}")
        self.info(f"E5071C: Set trigger source to {source}.")
        return source

    def single_sweep(self, ch: int = 1):
        """
        Trigger one sweep on channel `ch` and block until it completes.

        Arms the channel (``INITiate<ch>``), then sends a bus trigger
        combined with ``*OPC?`` as a single query, which the instrument only
        answers once the sweep finishes. Requires `trigger_source('BUS')`
        and `continuous_sweep(False, ch)` to have been set first.

        Parameters
        ----------
        ch : int, default 1
        """
        self.write(f"INIT{ch}")
        self.query("TRIG:SING;*OPC?")
        self.info(f"E5071C: Triggered single sweep on channel {ch}.")

    def abort(self):
        """
        Abort any sweep in progress, on all channels (``ABORt``).
        """
        self.write("ABOR")
        self.info("E5071C: Aborted sweep.")

    """
    -------------------------------------------------------------------------
    Data acquisition
    -------------------------------------------------------------------------
    """

    def data_format(self, fmt: str = None) -> str:
        """
        Set or query the numeric transfer format for binary trace/stimulus
        reads (``FORMat:DATA``).

        `get_trace`/`get_stimulus` set ``REAL,32`` internally regardless of
        this setting, so it rarely needs to be called directly. Note that
        neither `reset()`/`*RST` nor `preset()`/``SYST:PRES`` resets this
        setting.

        Parameters
        ----------
        fmt : str, optional
            One of {'ASC', 'REAL32', 'REAL64'} — ASCII, 32-bit real, or
            64-bit real.

        Returns
        -------
        str
        """
        _to = {"ASC": "ASC", "REAL32": "REAL,32", "REAL64": "REAL,64"}
        if fmt is None:
            return self.query("FORM:DATA?").strip()
        fmt = fmt.upper()
        if fmt not in _to:
            raise ValueError(f"fmt must be one of {sorted(_to)}.")
        self.write(f"FORM:DATA {_to[fmt]}")
        self.info(f"E5071C: Set data format to {fmt}.")
        return fmt

    def data_byte_order(self, order: str = None) -> str:
        """
        Set or query the byte order for binary data transfers
        (``FORMat:BORDer``).

        Parameters
        ----------
        order : str, optional
            One of {'NORM', 'SWAP'} (big-endian, little-endian).

        Returns
        -------
        str
        """
        if order is None:
            return self.query("FORM:BORD?").strip()
        order = order.upper()
        if order not in ("NORM", "SWAP"):
            raise ValueError("order must be 'NORM' or 'SWAP'.")
        self.write(f"FORM:BORD {order}")
        self.info(f"E5071C: Set data byte order to {order}.")
        return order

    _TRACE_FORMATS = {
        "MLOG", "PHAS", "GDEL", "SLIN", "SLOG", "SCOM", "SMIT", "SADM",
        "PLIN", "PLOG", "POL", "MLIN", "SWR", "REAL", "IMAG", "UPH", "PPH",
    }

    def trace_format(self, fmt: str = None, ch: int = 1) -> str:
        """
        Set or query the display/data format of the selected trace on
        channel `ch` (``CALCulate<ch>[:SELected]:FORMat``).

        Acts on whichever trace was last selected via `select_trace`. Not
        required for `get_trace()`, which always reads corrected complex
        data (``:DATA:SDATa?``) regardless of this setting — but does
        matter for the on-instrument display, and is required before
        reading real-part time-domain data in TDR mode (added in a later
        phase).

        Parameters
        ----------
        fmt : str, optional
            One of {'MLOG', 'PHAS', 'GDEL', 'SLIN', 'SLOG', 'SCOM', 'SMIT',
            'SADM', 'PLIN', 'PLOG', 'POL', 'MLIN', 'SWR', 'REAL', 'IMAG',
            'UPH', 'PPH'} (log magnitude, phase, group delay, Smith
            linear/log/complex, Smith admittance, polar linear/log, polar,
            linear magnitude, SWR, real part, imaginary part, unwrapped/
            positive phase).
        ch : int, default 1

        Returns
        -------
        str
        """
        if fmt is None:
            return self.query(f"CALC{ch}:FORM?").strip()
        fmt = fmt.upper()
        if fmt not in self._TRACE_FORMATS:
            raise ValueError(f"fmt must be one of {sorted(self._TRACE_FORMATS)}.")
        self.write(f"CALC{ch}:FORM {fmt}")
        self.info(f"E5071C: Set channel {ch} trace format to {fmt}.")
        return fmt

    def get_stimulus(self, ch: int = 1) -> np.ndarray:
        """
        Fetch the stimulus (frequency) axis for channel `ch`, in Hz
        (``SENSe<ch>:FREQuency:DATA?``).

        Requests a 32-bit real binary block (little-endian byte order).

        Parameters
        ----------
        ch : int, default 1

        Returns
        -------
        np.ndarray
        """
        self.write("FORM:DATA REAL,32")
        self.write("FORM:BORD SWAP")
        self.write(f"SENS{ch}:FREQ:DATA?")
        raw = self.read_raw()
        return parse_IEEE_488_2(raw)

    def get_trace(self, ch: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch the corrected S-parameter data for the selected trace on
        channel `ch`, as (frequency, complex S-parameter) arrays.

        Acts on whichever trace was last selected via `select_trace` — call
        that first if the channel has more than one trace. Requests a
        32-bit real binary block (``CALCulate<ch>:DATA:SDATa?`` returns
        interleaved real/imaginary pairs) and reassembles it into a complex
        array. Does not trigger a new sweep — call `single_sweep()` first
        for a fresh, settled acquisition (or leave the instrument in
        continuous sweep).

        Parameters
        ----------
        ch : int, default 1

        Returns
        -------
        freq, s : np.ndarray
            `freq` in Hz; `s` is complex (error-corrected if correction is
            on — `correction_state` is added in a later phase).
        """
        self.write("FORM:DATA REAL,32")
        self.write("FORM:BORD SWAP")
        self.write(f"CALC{ch}:DATA:SDAT?")
        raw = self.read_raw()
        data = parse_IEEE_488_2(raw)
        s = data[0::2] + 1j * data[1::2]

        freq = self.get_stimulus(ch)
        return freq, s

    """
    -------------------------------------------------------------------------
    Conversions (static, no I/O) — light, generic, lossless conversions of a
    complex S-parameter array; not tied to any particular experimental setup
    -------------------------------------------------------------------------
    """

    @staticmethod
    def log_mag(s: np.ndarray) -> np.ndarray:
        """
        Convert complex S-parameter data to log magnitude in dB
        (``20*log10(|s|)``).

        Parameters
        ----------
        s : np.ndarray
            Complex S-parameter array (e.g. from `get_trace`).

        Returns
        -------
        np.ndarray
        """
        return 20.0 * np.log10(np.abs(s))

    @staticmethod
    def phase_rad(s: np.ndarray, unwrap: bool = False) -> np.ndarray:
        """
        Convert complex S-parameter data to phase in radians.

        Parameters
        ----------
        s : np.ndarray
        unwrap : bool, default False
            If True, unwrap the phase (remove 2*pi discontinuities) before
            returning.

        Returns
        -------
        np.ndarray
        """
        phase = np.angle(s)
        return np.unwrap(phase) if unwrap else phase

    @staticmethod
    def phase_deg(s: np.ndarray, unwrap: bool = False) -> np.ndarray:
        """
        Convert complex S-parameter data to phase in degrees.

        Parameters
        ----------
        s : np.ndarray
        unwrap : bool, default False
            If True, unwrap the phase (in radians) before converting to
            degrees.

        Returns
        -------
        np.ndarray
        """
        return np.degrees(e5071c.phase_rad(s, unwrap=unwrap))

    @staticmethod
    def vswr(s: np.ndarray) -> np.ndarray:
        """
        Convert complex (reflection) S-parameter data to VSWR.

        Only meaningful for reflection parameters (S11, S22, ...); ``|s|``
        should be < 1 for a passive DUT.

        Parameters
        ----------
        s : np.ndarray

        Returns
        -------
        np.ndarray
        """
        mag = np.abs(s)
        return (1.0 + mag) / (1.0 - mag)

    @staticmethod
    def group_delay(freq: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Compute group delay in seconds from complex S-parameter data, via a
        finite-difference derivative of unwrapped phase with respect to
        angular frequency: ``-d(phase)/d(omega)``.

        Only meaningful for transmission parameters (S21, S12, ...).

        Parameters
        ----------
        freq : np.ndarray
            Frequency axis in Hz (e.g. from `get_stimulus`), same length as
            `s`.
        s : np.ndarray
            Complex S-parameter array.

        Returns
        -------
        np.ndarray
            Same length as `freq`/`s`; endpoints use a one-sided difference
            (see `np.gradient`).
        """
        phase = np.unwrap(np.angle(s))
        omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
        return -np.gradient(phase, omega)

    """
    -------------------------------------------------------------------------
    Time-domain / TDR (built-in Option 010 transform only — see the module
    docstring for the separate, out-of-scope "Enhanced Time Domain Analysis"
    PC application)
    -------------------------------------------------------------------------
    """

    def transform_state(self, state: bool = None, ch: int = 1) -> bool:
        """
        Set or query whether the time-domain transform is enabled on
        channel `ch` (``CALCulate<ch>:TRANsform:TIME:STATe``).

        Requires Option 010 (Time Domain Analysis). Once enabled,
        `get_trace()` on the selected trace returns the transformed
        time-domain data instead of frequency-domain S-parameters.

        Parameters
        ----------
        state : bool, optional
        ch : int, default 1

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query(f"CALC{ch}:TRAN:TIME:STAT?")))
        self.write(f"CALC{ch}:TRAN:TIME:STAT {'ON' if state else 'OFF'}")
        self.info(f"E5071C: Set channel {ch} time-domain transform to {state}.")
        return state

    _TRANSFORM_TYPES = {"BPAS", "LPAS"}

    def transform_type(self, mode: str = None, ch: int = 1) -> str:
        """
        Set or query the time-domain transform type on channel `ch`
        (``CALCulate<ch>:TRANsform:TIME``).

        Parameters
        ----------
        mode : str, optional
            One of {'BPAS', 'LPAS'} (bandpass, lowpass). Lowpass mode
            requires `align_to_lowpass_frequencies()` to have been called
            (and the channel re-swept) so the frequency points land on
            harmonics of the start frequency, and additionally needs
            `transform_stimulus()` set.
        ch : int, default 1

        Returns
        -------
        str
        """
        if mode is None:
            return self.query(f"CALC{ch}:TRAN:TIME?").strip()
        mode = mode.upper()
        if mode not in self._TRANSFORM_TYPES:
            raise ValueError(f"mode must be one of {sorted(self._TRANSFORM_TYPES)}.")
        self.write(f"CALC{ch}:TRAN:TIME {mode}")
        self.info(f"E5071C: Set channel {ch} transform type to {mode}.")
        return mode

    _TRANSFORM_STIMULI = {"IMP", "STEP"}

    def transform_stimulus(self, mode: str = None, ch: int = 1) -> str:
        """
        Set or query the lowpass-mode stimulus type on channel `ch`
        (``CALCulate<ch>:TRANsform:TIME:STIMulus``).

        Only meaningful when `transform_type` is 'LPAS' (lowpass).

        Parameters
        ----------
        mode : str, optional
            One of {'IMP', 'STEP'} (impulse response, step response).
        ch : int, default 1

        Returns
        -------
        str
        """
        if mode is None:
            return self.query(f"CALC{ch}:TRAN:TIME:STIM?").strip()
        mode = mode.upper()
        if mode not in self._TRANSFORM_STIMULI:
            raise ValueError(f"mode must be one of {sorted(self._TRANSFORM_STIMULI)}.")
        self.write(f"CALC{ch}:TRAN:TIME:STIM {mode}")
        self.info(f"E5071C: Set channel {ch} transform stimulus to {mode}.")
        return mode

    def align_to_lowpass_frequencies(self, ch: int = 1):
        """
        Force the swept frequency points on channel `ch` to harmonics of
        the start frequency (``CALCulate<ch>:TRANsform:TIME:LPFRequency``),
        required before using lowpass-mode transforms (`transform_type`
        'LPAS'). Re-sweep (e.g. `single_sweep`) after calling this.

        Parameters
        ----------
        ch : int, default 1
        """
        self.write(f"CALC{ch}:TRAN:TIME:LPFR")
        self.info(f"E5071C: Aligned channel {ch} frequency points to "
                  f"lowpass harmonics.")

    def transform_window(self, beta: float = None, ch: int = 1) -> float:
        """
        Set or query the time-domain transform's Kaiser-Bessel window
        parameter on channel `ch`
        (``CALCulate<ch>:TRANsform:TIME:KBESsel``).

        A continuous parameter from 0 (minimum sidelobe suppression, best
        time resolution) to 13 (maximum sidelobe suppression), replacing
        the front-panel's discrete minimum/normal/maximum window choices.

        Parameters
        ----------
        beta : float, optional
            0-13.
        ch : int, default 1

        Returns
        -------
        float
        """
        if beta is None:
            return float(self.query(f"CALC{ch}:TRAN:TIME:KBES?"))
        self.write(f"CALC{ch}:TRAN:TIME:KBES {beta:g}")
        self.info(f"E5071C: Set channel {ch} transform window (KBessel) to "
                  f"{beta:g}.")
        return beta

    def transform_time_start(self, t: float = None, ch: int = 1) -> float:
        """
        Set or query the time-domain display start time in seconds on
        channel `ch` (``CALCulate<ch>:TRANsform:TIME:STARt``).

        Parameters
        ----------
        t : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if t is None:
            return float(self.query(f"CALC{ch}:TRAN:TIME:STAR?"))
        self.write(f"CALC{ch}:TRAN:TIME:STAR {t:g}")
        self.info(f"E5071C: Set channel {ch} transform start time to {t:g} s.")
        return t

    def transform_time_stop(self, t: float = None, ch: int = 1) -> float:
        """
        Set or query the time-domain display stop time in seconds on
        channel `ch` (``CALCulate<ch>:TRANsform:TIME:STOP``).

        Parameters
        ----------
        t : float, optional
        ch : int, default 1

        Returns
        -------
        float
        """
        if t is None:
            return float(self.query(f"CALC{ch}:TRAN:TIME:STOP?"))
        self.write(f"CALC{ch}:TRAN:TIME:STOP {t:g}")
        self.info(f"E5071C: Set channel {ch} transform stop time to {t:g} s.")
        return t

    def configure_tdr(self, mode: str = "LPAS", stimulus: str = "IMP",
        window: float = 13.0, t_start: float = None, t_stop: float = None,
        ch: int = 1,
    ):
        """
        Convenience: configure and enable the time-domain transform on
        channel `ch` in one call, following the sequence used in the
        manual's own TDR example.

        For lowpass mode ('LPAS', the default, appropriate for TDR/
        impedance-discontinuity work), call
        `align_to_lowpass_frequencies()` and re-sweep *before* calling this
        — it is not done automatically here, since it would otherwise
        silently trigger an extra sweep as a side effect. Sets the trace
        format to 'REAL' for lowpass mode (time-axis data has no phase
        information there); leaves the trace format alone for bandpass
        mode.

        Parameters
        ----------
        mode : str, default 'LPAS'
            One of {'BPAS', 'LPAS'}.
        stimulus : str, default 'IMP'
            One of {'IMP', 'STEP'}; only used (and only sent) for lowpass
            mode.
        window : float, default 13.0
            Kaiser-Bessel window parameter, 0-13.
        t_start : float, optional
            Time-domain display start time in seconds.
        t_stop : float, optional
            Time-domain display stop time in seconds.
        ch : int, default 1
        """
        mode = mode.upper()
        self.transform_type(mode, ch=ch)
        if mode == "LPAS":
            self.transform_stimulus(stimulus, ch=ch)
        self.transform_window(window, ch=ch)
        if t_start is not None:
            self.transform_time_start(t_start, ch=ch)
        if t_stop is not None:
            self.transform_time_stop(t_stop, ch=ch)
        self.transform_state(True, ch=ch)
        if mode == "LPAS":
            self.trace_format("REAL", ch=ch)
        self.info(f"E5071C: Configured {mode} time-domain transform on "
                  f"channel {ch}.")

    """
    -------------------------------------------------------------------------
    Basic calibration control (select/recall + state query only — not a
    guided SOLT/ECal calibration sequence; see the module docstring)
    -------------------------------------------------------------------------
    """

    def correction_state(self, state: bool = None, ch: int = 1) -> bool:
        """
        Set or query whether error correction is applied on channel `ch`
        (``SENSe<ch>:CORRection:STATe``).

        Parameters
        ----------
        state : bool, optional
        ch : int, default 1

        Returns
        -------
        bool
        """
        if state is None:
            return bool(int(self.query(f"SENS{ch}:CORR:STAT?")))
        self.write(f"SENS{ch}:CORR:STAT {'ON' if state else 'OFF'}")
        self.info(f"E5071C: Set channel {ch} correction state to {state}.")
        return state

    def correction_type(self, ch: int = 1) -> str:
        """
        Query the calibration method currently active on channel `ch`
        (``SENSe<ch>:CORRection:COLLect:METHod:TYPE?``), read-only.

        Parameters
        ----------
        ch : int, default 1

        Returns
        -------
        str
            One of {'AREM', 'ERES', 'NONE', 'RESPO', 'RESPS', 'RESPT',
            'SOLT1', 'SOLT2', 'SOLT3', 'SOLT4', 'TRL2', 'TRL3', 'TRL4'}
            (adapter removal, enhanced response, none, response cal
            open/short/thru, full 1/2/3/4-port SOLT, or 2/3/4-port TRL).
        """
        return self.query(f"SENS{ch}:CORR:COLL:METH:TYPE?").strip()

    def store_type(self, type_: str = None) -> str:
        """
        Set or query what a subsequent `save_state()` includes
        (``MMEMory:STORe:STYPe``), e.g. whether calibration coefficients
        are saved along with the rest of the instrument state.

        The exact legal tokens for this command were not confirmed against
        the manual during development — only that it exists and gates
        what `save_state()` includes. Query it with `type_=None` first (or
        check the instrument's own `MMEMory` documentation) before relying
        on a specific value.

        Parameters
        ----------
        type_ : str, optional

        Returns
        -------
        str
        """
        if type_ is None:
            return self.query("MMEM:STOR:STYP?").strip()
        self.write(f"MMEM:STOR:STYP {type_}")
        self.info(f"E5071C: Set MMEMory store type to {type_}.")
        return type_

    def load_state(self, path: str):
        """
        Recall a full instrument state file (``MMEMory:LOAD``).

        This is this driver's route to "recalling a saved calibration": the
        E5071C has no PNA-style command to activate a saved calibration set
        by name — calibration is either part of a full instrument-state
        file (this method), or built live via a guided SOLT/ECal sequence
        (``SENSe:CORRection:COLLect:METHod:*`` + measuring standards +
        ``:SAVE``, not implemented by this driver). Use
        `correction_state()`/`correction_type()` after loading to confirm a
        working calibration actually came along with the state.

        Parameters
        ----------
        path : str
            Instrument-side file path, e.g. ``"D:\\State01.sta"``.
        """
        self.write(f'MMEM:LOAD "{path}"')
        self.info(f"E5071C: Loaded instrument state from {path!r}.")

    def save_state(self, path: str):
        """
        Save the current instrument state to a file (``MMEMory:STORe``).

        Whether the saved state includes calibration coefficients depends
        on `store_type()` (see its docstring) — check it before relying on
        a later `load_state()` to recall a working calibration.

        Parameters
        ----------
        path : str
            Instrument-side file path, e.g. ``"D:\\State01.sta"``.
        """
        self.write(f'MMEM:STOR "{path}"')
        self.info(f"E5071C: Saved instrument state to {path!r}.")

    """
    -------------------------------------------------------------------------
    JSON state serialization

    Distinct from `save_state()`/`load_state()` above, which drive the
    instrument's own on-board `MMEMory:STORe`/`:LOAD` file mechanism. These
    methods instead serialize to/from a local JSON file via `abstractDevice`,
    matching the convention used throughout the rest of this package (see
    e.g. `dtg5274`, the SRS lock-in classes).
    -------------------------------------------------------------------------
    """

    def save_state_json(self, path: str):
        """Save the network analyzer state to JSON locally."""
        super().save_state_json(path)

    def load_state_json(self, path: str):
        """Load the network analyzer state from JSON locally."""
        super().load_state_json(path)

    # Settings serialized as get/set accessors (channel 1 / trace 1 only).
    _STATE_SETTINGS = [
        "sweep_type",
        "start_frequency", "stop_frequency",
        "sweep_points", "if_bandwidth", "power",
        "average_state", "average_count",
        "continuous_sweep",
        "trace_format",
        "correction_state",
    ]

    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize connection config plus channel-1/trace-1 settings: which
        S-parameter is assigned to that trace, and the settings listed in
        `_STATE_SETTINGS`.

        Only channel 1 / trace 1 is captured (this driver's default scope
        throughout). Time-domain/TDR transform state and calibration
        coefficients are deliberately *not* captured: TDR is an alternate
        configuration layered on top of the base S-parameter sweep that
        this driver has no safe way to blindly reapply (lowpass mode in
        particular needs a physical re-sweep after
        `align_to_lowpass_frequencies()`, which isn't safe to trigger
        automatically from deserialization), and calibration data lives in
        the instrument's own state-file mechanism (`load_state()`/
        `save_state()`) rather than in this JSON snapshot — only whether
        correction is currently *on* is captured here, not the calibration
        itself.
        """
        state = super()._serialize_state()
        try:
            state['sparam'] = self.define_parameter(ch=1, tr=1)
        except Exception:
            pass
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
        if 'sparam' in state:
            try:
                self.define_parameter(state['sparam'], ch=1, tr=1)
                self.select_trace(ch=1, tr=1)
            except Exception:
                pass
        for setting in self._STATE_SETTINGS:
            if setting in state:
                try:
                    getattr(self, setting)(state[setting])
                except Exception:
                    continue

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "e5071c":
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
    """Example self-test of the E5071C driver using dummyResource."""

    import logging
    import os
    import struct
    import sys
    import tempfile

    logger = logging.getLogger('e5071c_test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    vna = e5071c(resource_name='DEBUG', logger=logger)

    def get_idn(obj):
        return "Agilent Technologies,E5071C,MY00000000,A.09.60\n"
    vna.device.add_command(r'\*IDN\?', get_idn)

    vna.device.attr['errors'] = ['+0,"No error"\n']
    def get_error(obj):
        if len(obj.attr['errors']) > 1:
            return obj.attr['errors'].pop(0)
        return obj.attr['errors'][0]
    vna.device.add_command(r'SYST:ERR\?', get_error)

    def reset(obj):
        obj.attr['errors'] = ['+0,"No error"\n']
    vna.device.add_command(r'\*RST', reset)
    vna.device.add_command(r'SYST:PRES', reset)

    print(f"IDN: {vna.idn()}")

    vna.reset()
    print(f"Error queue after reset: {vna.get_all_errors()}")
    vna.preset()
    print(f"Error queue after preset: {vna.get_all_errors()}")

    vna.device.attr['errors'] = ['+221,"Settings conflict"\n', '+0,"No error"\n']
    print(f"Injected errors: {vna.get_all_errors()}")

    # ── Channel / trace / measurement setup ─────────────────────────────────

    vna.device.attr['par_count'] = {1: 1}
    def get_par_count(obj, ch):
        return f"{obj.attr['par_count'].get(int(ch), 1)}\n"
    vna.device.add_command(r'CALC(\d+):PAR:COUN\?', get_par_count)
    def set_par_count(obj, ch, n):
        obj.attr['par_count'][int(ch)] = int(n)
    vna.device.add_command(r'CALC(\d+):PAR:COUN (\d+)', set_par_count)

    vna.device.attr['par_def'] = {}
    def get_par_def(obj, ch, tr):
        return f"{obj.attr['par_def'].get((int(ch), int(tr)), 'S11')}\n"
    vna.device.add_command(r'CALC(\d+):PAR(\d+):DEF\?', get_par_def)
    def set_par_def(obj, ch, tr, sparam):
        obj.attr['par_def'][(int(ch), int(tr))] = sparam
    vna.device.add_command(r'CALC(\d+):PAR(\d+):DEF (\S+)', set_par_def)

    vna.device.attr['selected'] = None
    def select_trace(obj, ch, tr):
        obj.attr['selected'] = (int(ch), int(tr))
    vna.device.add_command(r'CALC(\d+):PAR(\d+):SEL', select_trace)

    print(f"Trace count: {vna.trace_count(ch=1)}")
    print(f"Define S21 on trace 1: {vna.define_parameter('S21', ch=1, tr=1)}")
    print(f"Query parameter: {vna.define_parameter(ch=1, tr=1)}")
    vna.select_trace(ch=1, tr=1)
    print(f"Selected trace: {vna.device.attr['selected']}")

    vna.configure_measurement('S11', ch=1, tr=2)
    print(f"Trace count after configure_measurement(tr=2): {vna.trace_count(ch=1)}")
    print(f"Parameter on trace 2: {vna.define_parameter(ch=1, tr=2)}")
    print(f"Selected trace: {vna.device.attr['selected']}")

    # ── Phase 2: sweep configuration + data acquisition ─────────────────────

    def add_ch_scalar(query_re, set_re, key, default, caster=float):
        """Per-channel SCPI get/set pair backed by vna.device.attr[key][ch]."""
        vna.device.attr[key] = {}
        def get(obj, ch):
            return f"{obj.attr[key].get(int(ch), default)}\n"
        vna.device.add_command(query_re, get)
        def set_(obj, ch, val):
            obj.attr[key][int(ch)] = caster(val)
        vna.device.add_command(set_re, set_)

    def add_ch_bool(query_re, set_re, key, default=1):
        vna.device.attr[key] = {}
        def get(obj, ch):
            return f"{obj.attr[key].get(int(ch), default)}\n"
        vna.device.add_command(query_re, get)
        def set_(obj, ch, val):
            obj.attr[key][int(ch)] = 1 if val in ('ON', '1') else 0
        vna.device.add_command(set_re, set_)

    def add_scalar(query_re, set_re, key, default, caster=str):
        """Global (non-channel) SCPI get/set pair backed by vna.device.attr[key]."""
        vna.device.attr[key] = default
        def get(obj):
            return f"{obj.attr[key]}\n"
        vna.device.add_command(query_re, get)
        def set_(obj, val):
            obj.attr[key] = caster(val)
        vna.device.add_command(set_re, set_)

    add_ch_scalar(r'SENS(\d+):SWE:TYPE\?', r'SENS(\d+):SWE:TYPE (\S+)', 'sweep_type',
                  'LIN', caster=str)
    add_ch_scalar(r'SENS(\d+):FREQ:CENT\?', r'SENS(\d+):FREQ:CENT (\S+)', 'center', 1.5e9)
    add_ch_scalar(r'SENS(\d+):FREQ:SPAN\?', r'SENS(\d+):FREQ:SPAN (\S+)', 'span', 1e9)
    add_ch_scalar(r'SENS(\d+):FREQ:STAR\?', r'SENS(\d+):FREQ:STAR (\S+)', 'start', 1e9)
    add_ch_scalar(r'SENS(\d+):FREQ:STOP\?', r'SENS(\d+):FREQ:STOP (\S+)', 'stop', 2e9)
    add_ch_scalar(r'SENS(\d+):SWE:POIN\?', r'SENS(\d+):SWE:POIN (\d+)', 'points', 201,
                  caster=int)
    add_ch_scalar(r'SENS(\d+):BAND\?', r'SENS(\d+):BAND (\S+)', 'bw', 1e3)
    add_ch_scalar(r'SOUR(\d+):POW\?', r'SOUR(\d+):POW (\S+)', 'power', -10.0)
    add_ch_bool(r'SENS(\d+):AVER\?', r'SENS(\d+):AVER (\S+)', 'avg_state', 0)
    add_ch_scalar(r'SENS(\d+):AVER:COUN\?', r'SENS(\d+):AVER:COUN (\d+)', 'avg_count', 1,
                  caster=int)
    add_ch_bool(r'INIT(\d+):CONT\?', r'INIT(\d+):CONT (\S+)', 'cont', 1)
    add_ch_scalar(r'CALC(\d+):FORM\?', r'CALC(\d+):FORM (\S+)', 'trace_fmt', 'MLOG',
                  caster=str)

    add_scalar(r'TRIG:SOUR\?', r'TRIG:SOUR (\S+)', 'trig_source', 'INT')
    add_scalar(r'FORM:DATA\?', r'FORM:DATA (.+)', 'data_format', 'ASC')
    add_scalar(r'FORM:BORD\?', r'FORM:BORD (\S+)', 'byte_order', 'NORM')

    def average_clear(obj, ch):
        obj.attr.setdefault('avg_count', {})[int(ch)] = 0
    vna.device.add_command(r'SENS(\d+):AVER:CLE', average_clear)

    def arm(obj, ch):
        pass  # INIT<ch> just arms a sweep; no response needed.
    vna.device.add_command(r'INIT(\d+)$', arm)

    def opc(obj):
        return "1\n"
    vna.device.add_command(r'TRIG:SING;\*OPC\?', opc)

    def abort(obj):
        pass
    vna.device.add_command(r'ABOR', abort)

    def build_freq_response(obj, ch):
        # 6 fake stimulus points, 1.0-1.5 GHz.
        freqs = [1.0e9 + i * 1.0e8 for i in range(6)]
        payload = struct.pack(f'<{len(freqs)}f', *freqs)
        header = f"#{len(str(len(payload)))}{len(payload)}"
        return header.encode() + payload
    vna.device.add_command(r'SENS(\d+):FREQ:DATA\?', build_freq_response)

    def build_sdata_response(obj, ch):
        # 6 fake complex S-parameter points, packed as interleaved Re/Im.
        values = []
        for i in range(6):
            values += [0.5 - 0.05 * i, 0.1 * i]
        payload = struct.pack(f'<{len(values)}f', *values)
        header = f"#{len(str(len(payload)))}{len(payload)}"
        return header.encode() + payload
    vna.device.add_command(r'CALC(\d+):DATA:SDAT\?', build_sdata_response)

    print(f"Sweep type: {vna.sweep_type('LIN', ch=1)}")
    print(f"Center frequency: {vna.center_frequency(1.5e9, ch=1)} Hz")
    print(f"Span: {vna.span(1e9, ch=1)} Hz")
    vna.start_frequency(1e9, ch=1)
    vna.stop_frequency(2e9, ch=1)
    print(f"Start/stop: {vna.start_frequency(ch=1)} Hz / {vna.stop_frequency(ch=1)} Hz")
    print(f"Sweep points: {vna.sweep_points(201, ch=1)}")
    print(f"IF bandwidth: {vna.if_bandwidth(1e3, ch=1)} Hz")
    print(f"Power: {vna.power(-10.0, ch=1)} dBm")
    print(f"Average state: {vna.average_state(True, ch=1)}")
    print(f"Average count: {vna.average_count(16, ch=1)}")
    vna.average_clear(ch=1)

    vna.continuous_sweep(False, ch=1)
    print(f"Continuous sweep: {vna.continuous_sweep(ch=1)}")
    print(f"Trigger source: {vna.trigger_source('BUS')}")
    vna.single_sweep(ch=1)
    vna.abort()

    print(f"Data format: {vna.data_format('REAL32')}")
    print(f"Data byte order: {vna.data_byte_order('SWAP')}")
    print(f"Trace format: {vna.trace_format('MLOG', ch=1)}")

    freq = vna.get_stimulus(ch=1)
    print(f"Stimulus: {freq.size} points, "
          f"{freq[0] / 1e9:.2f}-{freq[-1] / 1e9:.2f} GHz")

    freq2, s = vna.get_trace(ch=1)
    print(f"Trace: {s.size} points, s[0:2]={s[:2]}")
    print(f"log_mag[0:2] = {vna.log_mag(s[:2])}")
    print(f"phase_deg[0:2] = {vna.phase_deg(s[:2])}")
    print(f"vswr[0:2] = {vna.vswr(s[:2])}")
    print(f"group_delay[0:2] = {vna.group_delay(freq2[:2], s[:2])}")

    # ── Phase 3: time-domain / TDR (Option 010) ──────────────────────────────

    add_ch_bool(r'CALC(\d+):TRAN:TIME:STAT\?', r'CALC(\d+):TRAN:TIME:STAT (\S+)',
                'tdr_state', 0)
    add_ch_scalar(r'CALC(\d+):TRAN:TIME\?', r'CALC(\d+):TRAN:TIME (\S+)', 'tdr_type',
                  'BPAS', caster=str)
    add_ch_scalar(r'CALC(\d+):TRAN:TIME:STIM\?', r'CALC(\d+):TRAN:TIME:STIM (\S+)',
                  'tdr_stim', 'IMP', caster=str)
    add_ch_scalar(r'CALC(\d+):TRAN:TIME:KBES\?', r'CALC(\d+):TRAN:TIME:KBES (\S+)',
                  'tdr_kbes', 2.0)
    add_ch_scalar(r'CALC(\d+):TRAN:TIME:STAR\?', r'CALC(\d+):TRAN:TIME:STAR (\S+)',
                  'tdr_tstar', 0.0)
    add_ch_scalar(r'CALC(\d+):TRAN:TIME:STOP\?', r'CALC(\d+):TRAN:TIME:STOP (\S+)',
                  'tdr_tstop', 1e-8)

    def lpfr(obj, ch):
        pass
    vna.device.add_command(r'CALC(\d+):TRAN:TIME:LPFR', lpfr)

    print(f"Transform state (before configure_tdr): {vna.transform_state(ch=1)}")

    vna.align_to_lowpass_frequencies(ch=1)
    vna.single_sweep(ch=1)
    vna.configure_tdr(mode="LPAS", stimulus="IMP", window=13.0,
                       t_start=0.0, t_stop=1e-8, ch=1)

    print(f"Transform state: {vna.transform_state(ch=1)}")
    print(f"Transform type: {vna.transform_type(ch=1)}")
    print(f"Transform stimulus: {vna.transform_stimulus(ch=1)}")
    print(f"Transform window: {vna.transform_window(ch=1)}")
    print(f"Transform time range: {vna.transform_time_start(ch=1)} - "
          f"{vna.transform_time_stop(ch=1)} s")
    print(f"Trace format after configure_tdr: {vna.trace_format(ch=1)}")

    # ── Phase 4: basic calibration control ───────────────────────────────────

    add_ch_bool(r'SENS(\d+):CORR:STAT\?', r'SENS(\d+):CORR:STAT (\S+)', 'corr_state', 0)

    def get_corr_type(obj, ch):
        return "SOLT2\n"
    vna.device.add_command(r'SENS(\d+):CORR:COLL:METH:TYPE\?', get_corr_type)

    add_scalar(r'MMEM:STOR:STYP\?', r'MMEM:STOR:STYP (\S+)', 'store_type', 'DEF')

    def mmem_load(obj, path):
        obj.attr['loaded_path'] = path
    vna.device.add_command(r'MMEM:LOAD "([^"]*)"', mmem_load)

    def mmem_store(obj, path):
        obj.attr['stored_path'] = path
    vna.device.add_command(r'MMEM:STOR "([^"]*)"', mmem_store)

    print(f"Correction state: {vna.correction_state(True, ch=1)}")
    print(f"Correction state (query): {vna.correction_state(ch=1)}")
    print(f"Correction type: {vna.correction_type(ch=1)}")

    print(f"Store type: {vna.store_type('DEF')}")
    vna.save_state("D:\\State01.sta")
    print(f"Stored path: {vna.device.attr['stored_path']}")
    vna.load_state("D:\\State01.sta")
    print(f"Loaded path: {vna.device.attr['loaded_path']}")

    # ── Serialization round trip ─────────────────────────────────────────────

    state = vna._serialize_state()
    captured = {k: state[k] for k in vna._STATE_SETTINGS if k in state}
    captured['sparam'] = state.get('sparam')
    print(f"Serialized settings: {captured}")

    vna.if_bandwidth(999.0, ch=1)  # perturb before restoring
    vna._deserialize_state(state)
    print(f"IF bandwidth after _deserialize_state round trip: "
          f"{vna.if_bandwidth(ch=1)} Hz (should match the captured value)")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "e5071c_state.json")
        vna.save_state_json(json_path)
        print(f"Saved state JSON to {json_path} "
              f"({os.path.getsize(json_path)} bytes).")

        vna.if_bandwidth(999.0, ch=1)  # perturb again before reloading
        vna.load_state_json(json_path)
        print(f"IF bandwidth after save_state_json/load_state_json round "
              f"trip: {vna.if_bandwidth(ch=1)} Hz")

    config = {'registry_id': 'e5071c_test_2', 'resource_name': 'DEBUG'}
    vna2 = e5071c.from_config(config)
    print(f"from_config constructed a fresh instance: {type(vna2).__name__}, "
          f"resource_name={vna2.resource_name!r}")
