"""
Control class for the AMETEK / Signal Recovery Model 7280 DSP lock-in amplifier.

The 7280 does not speak the SCPI-like protocol of the Stanford Research Systems
instruments, so this class is *not* an ``SRSLockin`` subclass. It derives from
``pyvisaDevice`` directly (the same base ``SRSLockin`` uses) and re-implements
the same public surface — identical accessor names, serialization, registry,
``resolve`` channels, and web-GUI hooks — on top of the 7280 command set. That
makes it a drop-in for the existing GUI and orchestration layer without
inheriting SCPI assumptions that do not hold for this instrument.

Protocol notes specific to the 7280
------------------------------------
* There is no ``?`` query syntax. A command sent bare (e.g. ``SEN``) returns the
  current value; the same command with an argument sets it.
* "Dotted" commands (``X.``, ``FRQ.``, ``OA.``, ``MP.`` ...) return calibrated
  floating-point values; the un-dotted forms return raw integers/indices.
* ``SEN`` (sensitivity) and ``TC`` (time constant) are *index* commands whose
  index-to-value mapping depends on the input mode (voltage vs current,
  fast vs slow, noise vs normal). Value<->index conversion therefore queries the
  relevant mode first rather than relying on a single static table.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .srs_lockin.gui import SRSLockinGUI

from .pyvisa_device import pyvisaDevice
from .channel_adapter import ScalarChannelAdapter
from .registry import register_hardware_class

# Re-use the SRS channel adapters so `resolve` returns the same objects the rest
# of the ecosystem already understands.
from .srs_lockin.base import SRSLockin_sensitivity_channel
from .srs_lockin.lockin import series_correlated_covariance, sr860_lockin_channel

import time
from math import log10
from logging import Logger
from typing import Any, Dict, List, Tuple

import numpy as np


@register_hardware_class("model7280")
class model7280(pyvisaDevice):
    """
    Class representation of the AMETEK / Signal Recovery Model 7280 lock-in.

    Presents the same interface as the SRS lock-in classes (so the shared GUI and
    experiment-orchestration code treat it identically) plus the 7280-specific
    controls: reference source selection, AC gain / dynamic reserve, fast and
    noise modes, current-input bandwidth, auto-measure/offset, effective noise
    bandwidth, and analog-output routing.
    """

    _name = "7280"

    # peak-to-RMS conversion used by the 7280 gain/reserve relationships.
    _PEAK_TO_RMS = 0.707106781

    # ── VISA defaults ──────────────────────────────────────────────────────
    #
    # NOTE: the read/write terminators must match the instrument's GPIB `GP`
    # terminator setting. CR is a common choice; change here (or reconfigure the
    # instrument) if reads hang or return malformed values.
    DEFAULT_PYVISA_CONFIG = {
        "read_termination" : "\r",
        "write_termination": "\r",
        "timeout"          : 10_000,
        "max_retries"      : 3,
        "retry_delay"      : 0.1,
        "min_interval"     : 0.05,
    }

    out_aux_channels: List[int] = [1, 2, 3, 4]   # DAC outputs
    in_aux_channels : List[int] = [1, 2, 3, 4]   # ADC inputs

    # Present only for parity with the SRS classes; the 7280 uses dedicated
    # dotted read commands rather than an OUTP/SNAP map.
    output_map = {"X": 0, "Y": 1, "R": 2, "T": 3}

    # The 7280 has no SR860-style front-end input range; leave empty so the GUI
    # skips that dropdown.
    irng_vals = np.array([])

    # ── 7280 index tables (from the instrument manual / legacy driver) ───────
    #
    # Each entry is (instrument index, full-scale value). Voltage tables are in
    # volts, current tables in amps.

    _SENS_VOLTAGE: List[Tuple[int, float]] = [
        (3, 10e-9), (4, 20e-9), (5, 50e-9), (6, 100e-9), (7, 200e-9), (8, 500e-9),
        (9, 1e-6), (10, 2e-6), (11, 5e-6), (12, 10e-6), (13, 20e-6), (14, 50e-6),
        (15, 100e-6), (16, 200e-6), (17, 500e-6), (18, 1e-3), (19, 2e-3),
        (20, 5e-3), (21, 10e-3), (22, 20e-3), (23, 50e-3), (24, 100e-3),
        (25, 200e-3), (26, 500e-3), (27, 1.0),
    ]

    # High-bandwidth current mode (IMODE 1). Ported from the legacy driver —
    # verify against your 7280 manual before trusting current-mode readings.
    _SENS_CURRENT_HIGH: List[Tuple[int, float]] = [
        (3, 1e-12), (4, 2e-12), (5, 5e-12), (6, 10e-12), (7, 20e-12), (8, 50e-12),
        (9, 100e-12), (10, 200e-12), (11, 500e-12), (12, 1e-9), (13, 2e-9),
        (14, 5e-9), (15, 10e-9), (16, 20e-9), (17, 50e-9), (18, 100e-9),
        (19, 200e-9), (20, 500e-9), (21, 1e-6), (22, 2e-6), (23, 5e-6),
        (24, 10e-6), (25, 20e-6), (26, 50e-6), (27, 100e-6),
    ]

    # Low-bandwidth current mode (IMODE 2). Same caveat as above.
    _SENS_CURRENT_LOW: List[Tuple[int, float]] = [
        (9, 10e-15), (10, 20e-15), (11, 50e-15), (12, 100e-15), (13, 200e-15),
        (14, 500e-15), (15, 1e-12), (16, 2e-12), (17, 5e-12), (18, 10e-12),
        (19, 20e-12), (20, 50e-12), (21, 100e-12), (22, 200e-12), (23, 500e-12),
        (24, 1e-9), (25, 2e-9), (26, 5e-9), (27, 10e-9),
    ]

    # Time-constant tables. The active table depends on FASTMODE and NOISEMODE.
    _TC_FAST: List[Tuple[int, float]] = [
        (0, 1e-6), (1, 2e-6), (2, 5e-6), (3, 10e-6), (4, 20e-6), (5, 50e-6),
        (6, 100e-6), (7, 200e-6), (8, 500e-6), (9, 1e-3), (10, 2e-3), (11, 4e-3),
        (12, 10e-3), (13, 20e-3), (14, 50e-3), (15, 100e-3), (16, 200e-3),
        (17, 500e-3), (18, 1.0), (19, 2.0), (20, 5.0), (21, 10.0), (22, 20.0),
        (23, 50.0), (24, 100.0), (25, 200.0), (26, 500.0), (27, 1e3), (28, 2e3),
        (29, 5e3), (30, 10e3), (31, 20e3), (32, 50e3), (33, 100e3),
    ]
    _TC_NORMAL: List[Tuple[int, float]] = [
        (8, 500e-6), (9, 1e-3), (10, 2e-3), (11, 5e-3), (12, 10e-3), (13, 20e-3),
        (14, 50e-3), (15, 100e-3), (16, 200e-3), (17, 500e-3), (18, 1.0),
        (19, 2.0), (20, 5.0), (21, 10.0), (22, 20.0), (23, 50.0), (24, 100.0),
        (25, 200.0), (26, 500.0), (27, 1e3), (28, 2e3), (29, 5e3), (30, 10e3),
        (31, 20e3), (32, 50e3), (33, 100e3),
    ]
    _TC_NOISE: List[Tuple[int, float]] = [
        (8, 500e-6), (9, 1e-3), (10, 2e-3), (11, 5e-3), (12, 10e-3),
    ]
    _TC_FASTNOISE: List[Tuple[int, float]] = [
        (8, 500e-6), (9, 1e-3), (10, 2e-3), (11, 4e-3), (12, 10e-3),
    ]

    # AC gain: (ACGAIN index, input full-scale limit [V], gain [dB]).
    _AC_GAIN: List[Tuple[int, float, float]] = [
        (0, 1.6, 0), (1, 0.800, 6), (2, 0.320, 14), (3, 0.160, 20),
        (4, 0.080, 26), (5, 0.032, 34), (6, 0.016, 40), (7, 0.008, 46),
        (8, 0.0032, 54), (9, 0.0016, 60), (10, 0.0008, 66),
    ]

    _SLOPE_OPTIONS = ["6dB/oct", "12dB/oct", "18dB/oct", "24dB/oct"]  # SLOPE 0..3

    # ── Curve-buffer bit map (CBD / DC / DCT / DCB) ─────────────────────────
    #
    # The CBD command parameter is the decimal sum of ``1 << bit`` over the
    # curves selected for storage; the DC/DCB argument is the *bit number* of the
    # single curve to dump. (Manual section 6.4.09, pp. 6-23..6-27.)
    _CURVE_BIT: Dict[str, int] = {
        "X": 0, "Y": 1, "MAG": 2, "PHA": 3, "SEN": 4, "NOISE": 5,
        "RATIO": 6, "LOGRATIO": 7,
        "ADC1": 8, "ADC2": 9, "ADC3": 10, "ADC4": 11, "DAC1": 12, "DAC2": 13,
    }
    # User-facing tokens -> canonical names (R/T/N mirror get_r/get_t/get_noise).
    _CURVE_ALIAS: Dict[str, str] = {
        "X": "X", "Y": "Y",
        "R": "MAG", "MAG": "MAG",
        "T": "PHA", "P": "PHA", "PHA": "PHA", "PHASE": "PHA",
        "N": "NOISE", "NOISE": "NOISE",
        "RATIO": "RATIO", "LOGRATIO": "LOGRATIO",
        "ADC1": "ADC1", "ADC2": "ADC2", "ADC3": "ADC3", "ADC4": "ADC4",
        "DAC1": "DAC1", "DAC2": "DAC2",
    }
    # Floating-point (``DC.``) dumps of these curves require the SEN curve to be
    # co-stored so the instrument can convert %FS -> volts/amps. Without it the
    # dotted dump has no calibration reference. (Manual, DC[.] description.)
    _CURVE_NEEDS_SEN = frozenset({"X", "Y", "MAG", "NOISE"})
    _CURVE_BUFFER_TOTAL = 32768   # storage points shared across stored curves

    # ── Construction ─────────────────────────────────────────────────────────

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
            VISA resource name (e.g. ``"GPIB0::12::INSTR"``).
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
        # Line-notch center-frequency region: 0 -> 60/120 Hz, 1 -> 50/100 Hz.
        self._line_freq_code = 0
        # Averaging configuration (set via setup_data_acquisition).
        self._acq: Dict[str, Any] | None = None

    # ── Low-level query helpers ──────────────────────────────────────────────

    def _q(self, cmd: str) -> str:
        """Query and clean the 7280's response (strips trailing null/prompt)."""
        return self.query(cmd).replace("\x00", "").strip()

    def _qf(self, cmd: str) -> float:
        return float(self._q(cmd))

    def _qi(self, cmd: str) -> int:
        # Values are ASCII integers, but coerce through float to be safe.
        return int(float(self._q(cmd)))

    def _qpair(self, cmd: str) -> Tuple[float, float]:
        a, b = self._q(cmd).replace(",", " ").split()[:2]
        return float(a), float(b)

    @staticmethod
    def _table_value(table: List[Tuple[int, float]], idx: int) -> float:
        """Return the value in `table` at instrument index `idx` (nearest)."""
        for i, v in table:
            if i == idx:
                return v
        indices = np.array([i for i, _ in table])
        return table[int(np.argmin(np.abs(indices - idx)))][1]

    @staticmethod
    def _nearest_index(table: List[Tuple[int, float]], target: float) -> int:
        """Return the instrument index whose value is closest to `target`."""
        vals = np.array([v for _, v in table])
        return table[int(np.argmin(np.abs(vals - target)))][0]

    # ── Data readout (calibrated dotted commands) ────────────────────────────

    def get_x(self) -> float:
        """X output in volts (or amps in current mode)."""
        return self._qf("X.")

    def get_y(self) -> float:
        """Y output in volts (or amps in current mode)."""
        return self._qf("Y.")

    def get_r(self) -> float:
        """Magnitude R in volts (or amps in current mode)."""
        return self._qf("MAG.")

    def get_t(self) -> float:
        """Phase theta in degrees."""
        return self._qf("PHA.")

    def get_xy(self) -> Tuple[float, float]:
        """(X, Y) in a single acquisition."""
        return self._qpair("XY.")

    def get_rt(self) -> Tuple[float, float]:
        """(R, theta) in a single acquisition."""
        return self._qpair("MP.")

    def get_noise(self) -> float:
        """Noise spectral density at the reference in V/sqrt(Hz) (``NHZ.``)."""
        return self._qf("NHZ.")

    def get_ratio(self) -> float:
        """Ratio X / ADC1 (``RT.``)."""
        return self._qf("RT.")

    def get_log_ratio(self) -> float:
        """Log ratio log10(X / ADC1) (``LR.``)."""
        return self._qf("LR.")

    # ── Reference / oscillator ───────────────────────────────────────────────

    def reference_phase(self, phase: float = None) -> float | None:
        """
        Set or query the reference phase shift in degrees (``REFP.``).

        Parameters
        ----------
        phase : float, optional

        Returns
        -------
        float or None
        """
        if phase is None:
            return self._qf("REFP.")
        self.write(f"REFP. {phase:.2f}")
        self.info(f"Set reference phase shift to {phase} degrees.")

    def reference_source(self, mode: str = None) -> str | None:
        """
        Set or query the reference source (``IE``).

        Parameters
        ----------
        mode : str, optional
            One of {'internal', 'external_ttl', 'external_sine'}.

        Returns
        -------
        str or None
        """
        _to = {"internal": 0, "external_ttl": 1, "external_sine": 2}
        _from = {0: "internal", 1: "external_ttl", 2: "external_sine"}
        if mode is None:
            return _from.get(self._qi("IE"), "internal")
        if mode not in _to:
            raise ValueError(
                "mode must be 'internal', 'external_ttl' or 'external_sine'."
            )
        self.write(f"IE {_to[mode]}")
        self.info(f"Set reference source to {mode}.")

    def internal_reference(self, on: bool = None) -> bool | None:
        """
        Set or query whether the reference is internal (SRS-compatible).

        When switching to external, the previously selected external sub-mode is
        kept; if the instrument was internal, external-sine is used as a default.
        Use ``reference_source`` for explicit TTL/sine selection.

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return self.reference_source() == "internal"
        if on:
            self.reference_source("internal")
        elif self.reference_source() == "internal":
            self.reference_source("external_sine")
        self.info(f"{'En' if on else 'Dis'}abled internal reference.")

    def reference_trigger(self, condition: str = None) -> str | None:
        """
        Set or query the external reference trigger condition (SRS-compatible).

        Parameters
        ----------
        condition : str, optional
            One of {'sine', 'pos', 'neg'}. On the 7280 both 'pos' and 'neg' map
            to TTL/logic triggering (the instrument does not distinguish edge
            polarity through this control), and 'sine' maps to analog
            zero-crossing triggering.

        Returns
        -------
        str or None
        """
        if condition is None:
            return "pos" if self.reference_source() == "external_ttl" else "sine"
        if condition == "sine":
            self.reference_source("external_sine")
        elif condition in ("pos", "neg"):
            self.reference_source("external_ttl")
        else:
            raise ValueError("condition must be 'sine', 'pos' or 'neg'.")
        self.info(f"Set external reference trigger condition to {condition}.")

    def reference_frequency(self, freq: float = None) -> float | None:
        """
        Set or query the reference frequency in Hz.

        Query reads the measured frequency (``FRQ.``). Set programs the internal
        oscillator (``OF``, argument in mHz) and is only applied when using the
        internal reference; a ``LOCK`` is issued afterwards to refresh the
        frequency-dependent gain calibration.

        Parameters
        ----------
        freq : float, optional

        Returns
        -------
        float or None
        """
        if freq is None:
            return self._qf("FRQ.")
        if self.internal_reference():
            self.write(f"OF {round(freq * 1000)}")   # OF takes mHz
            self.info(f"Set oscillator frequency to {freq} Hz.")
            self.lock()

    def detection_harmonic(self, harm: int = None) -> int | None:
        """
        Set or query the detection harmonic (``REFN``).

        Parameters
        ----------
        harm : int, optional

        Returns
        -------
        int or None
        """
        if harm is None:
            return self._qi("REFN")
        self.write(f"REFN {int(harm)}")
        self.info(f"Set detection harmonic to {harm}.")

    def sine_output_amplitude(self, V: float = None) -> float | None:
        """
        Set or query the oscillator (sine) output amplitude in volts rms (``OA.``).

        Parameters
        ----------
        V : float, optional

        Returns
        -------
        float or None
        """
        if V is None:
            return self._qf("OA.")
        self.write(f"OA. {V:g}")
        self.info(f"Set sine output amplitude to {V} V.")

    def lock(self):
        """Recompute frequency-dependent gain corrections (``LOCK``)."""
        self.write("LOCK")

    # ── Input configuration ──────────────────────────────────────────────────

    def input_configuration(self, src: str = None) -> str | None:
        """
        Set or query the input configuration (SRS-compatible subset).

        Parameters
        ----------
        src : str, optional
            One of {'A', 'A-B', 'I'}. 'I' selects high-bandwidth current mode;
            use ``current_input`` to choose the bandwidth explicitly, and
            ``voltage_input`` for the '-B' and 'ground' voltage settings.

        Returns
        -------
        str or None
        """
        if src is None:
            if self._qi("IMODE") != 0:
                return "I"
            return "A-B" if self._qi("VMODE") == 3 else "A"
        if src == "A":
            self.write("IMODE 0;VMODE 1")
        elif src == "A-B":
            self.write("IMODE 0;VMODE 3")
        elif src == "I":
            self.write("IMODE 1")
        else:
            raise ValueError("src must be 'A', 'A-B' or 'I'.")
        self.info(f"Set input configuration to {src}.")

    def voltage_input(self, mode: str = None) -> str | None:
        """
        Set or query the voltage input mode (``IMODE 0`` + ``VMODE``).

        Parameters
        ----------
        mode : str, optional
            One of {'ground', 'A', '-B', 'A-B'}.

        Returns
        -------
        str or None
            The current voltage mode, or None if the instrument is in a current
            input mode.
        """
        _to = {"ground": 0, "A": 1, "-B": 2, "A-B": 3}
        _from = {0: "ground", 1: "A", 2: "-B", 3: "A-B"}
        if mode is None:
            if self._qi("IMODE") != 0:
                return None
            return _from.get(self._qi("VMODE"), "A")
        if mode not in _to:
            raise ValueError("mode must be 'ground', 'A', '-B' or 'A-B'.")
        self.write(f"IMODE 0;VMODE {_to[mode]}")
        self.info(f"Set voltage input to {mode}.")

    def current_input(self, bandwidth: str = None) -> str | None:
        """
        Set or query the current input bandwidth (``IMODE``).

        Parameters
        ----------
        bandwidth : str, optional
            One of {'high', 'low', 'off'}. 'off' returns to voltage mode.

        Returns
        -------
        str or None
            'high', 'low', or None if the instrument is in voltage mode.
        """
        _to = {"high": 1, "low": 2}
        _from = {1: "high", 2: "low"}
        if bandwidth is None:
            return _from.get(self._qi("IMODE"))
        if bandwidth == "off":
            self.write("IMODE 0")
        elif bandwidth in _to:
            self.write(f"IMODE {_to[bandwidth]}")
        else:
            raise ValueError("bandwidth must be 'high', 'low' or 'off'.")
        self.info(f"Set current input bandwidth to {bandwidth}.")

    def input_mode_current(self, I_mode: bool = None) -> bool | None:
        """
        Set or query whether the input is current mode (SRS-compatible).

        Enabling defaults to high-bandwidth current mode; use ``current_input``
        for explicit bandwidth control.

        Parameters
        ----------
        I_mode : bool, optional

        Returns
        -------
        bool or None
        """
        if I_mode is None:
            return self._qi("IMODE") != 0
        self.write("IMODE 1" if I_mode else "IMODE 0")
        self.info(f"Set input to {'current' if I_mode else 'voltage'} mode.")

    def input_shield_grounded(self, gnd: bool = None) -> bool | None:
        """
        Set or query whether the input connector shield is grounded (``FLOAT``).

        Parameters
        ----------
        gnd : bool, optional

        Returns
        -------
        bool or None
        """
        if gnd is None:
            return self._qi("FLOAT") == 0
        self.write(f"FLOAT {0 if gnd else 1}")
        self.info(f"Set input shield {'to ground' if gnd else 'floating'}.")

    def input_coupling_DC(self, dc_couple: bool = None) -> bool | None:
        """
        Set or query DC vs AC input coupling (``CP``: 0 = AC, 1 = DC).

        Parameters
        ----------
        dc_couple : bool, optional

        Returns
        -------
        bool or None
        """
        if dc_couple is None:
            return self._qi("CP") == 1
        self.write(f"CP {1 if dc_couple else 0}")
        self.info(f"Set input coupling to {'DC' if dc_couple else 'AC'}.")

    def line_notch_filter(self, setting: str = None) -> str | None:
        """
        Set or query the line-frequency notch filter (``LF``).

        Parameters
        ----------
        setting : str, optional
            One of {'Out', 'In', '2xIn', 'Both'} — off, line, 2x line, or both.
            The notch center region (50/60 Hz) is set with
            ``line_notch_frequency``.

        Returns
        -------
        str or None
        """
        _to = {"Out": 0, "In": 1, "2xIn": 2, "Both": 3}
        _from = {0: "Out", 1: "In", 2: "2xIn", 3: "Both"}
        if setting is None:
            n1 = int(self._q("LF").replace(",", " ").split()[0])
            return _from.get(n1, "Out")
        if setting not in _to:
            raise ValueError("setting must be 'Out', 'In', '2xIn' or 'Both'.")
        self.write(f"LF {_to[setting]} {self._line_freq_code}")
        self.info(f"Set line notch filter status to {setting}.")

    def line_notch_frequency(self, hz: int = None) -> int | None:
        """
        Set or query the line-notch center region (``LF`` second argument).

        Parameters
        ----------
        hz : int, optional
            50 or 60.

        Returns
        -------
        int or None
        """
        parts = self._q("LF").replace(",", " ").split()
        n1 = int(parts[0])
        n2 = int(parts[1]) if len(parts) > 1 else self._line_freq_code
        if hz is None:
            return 50 if n2 == 1 else 60
        if hz not in (50, 60):
            raise ValueError("hz must be 50 or 60.")
        self._line_freq_code = 1 if hz == 50 else 0
        self.write(f"LF {n1} {self._line_freq_code}")
        self.info(f"Set line notch center frequency to {hz} Hz.")

    # ── Sensitivity, time constant, filter, gain/reserve ─────────────────────

    def _sens_table(self) -> List[Tuple[int, float]]:
        """Active sensitivity table for the current input mode."""
        try:
            imode = self._qi("IMODE")
        except Exception:
            imode = 0
        if imode == 1:
            return self._SENS_CURRENT_HIGH
        if imode == 2:
            return self._SENS_CURRENT_LOW
        return self._SENS_VOLTAGE

    def _tau_table(self) -> List[Tuple[int, float]]:
        """Active time-constant table for the current fast/noise mode."""
        try:
            fast = self._qi("FASTMODE") != 0
            noise = self._qi("NOISEMODE") != 0
        except Exception:
            fast, noise = False, False
        if fast:
            return self._TC_FASTNOISE if noise else self._TC_FAST
        return self._TC_NOISE if noise else self._TC_NORMAL

    def input_sensitivity(self, val: float = None, units: str = "V") -> float | None:
        """
        Set or query the full-scale input sensitivity (``SEN``).

        The value<->index mapping depends on the input mode; this method queries
        the mode as needed.

        Parameters
        ----------
        val : float, optional
        units : str, default 'V'
            One of {'pV','nV','uV','mV','V','pA','nA','uA','mA','A'}.

        Returns
        -------
        float or None
        """
        coeff = 1 if units[-1] == "V" else 1e-6
        if len(units) == 2:
            coeff *= {"p": 1e12, "n": 1e9, "u": 1e6, "m": 1e3}[units[0]]

        table = self._sens_table()
        if val is None:
            return coeff * self._table_value(table, self._qi("SEN"))
        idx = self._nearest_index(table, val / coeff)
        self.write(f"SEN {idx}")
        self.info(f"Set sensitivity to {self._table_value(table, idx)} "
                  f"{'A' if units[-1] == 'A' else 'V'} full scale.")

    def time_constant(self, tau: float = None) -> float | None:
        """
        Set or query the output time constant in seconds (``TC``).

        The available values depend on fast/noise mode; this method queries those
        modes as needed.

        Parameters
        ----------
        tau : float, optional

        Returns
        -------
        float or None
        """
        table = self._tau_table()
        if tau is None:
            return self._table_value(table, self._qi("TC"))
        idx = self._nearest_index(table, tau)
        self.write(f"TC {idx}")
        self.info(f"Set time constant to {self._table_value(table, idx)} s.")

    def low_pass_filter_slope(self, slope: str = None) -> str | None:
        """
        Set or query the output low-pass filter slope (``SLOPE``).

        Parameters
        ----------
        slope : str, optional
            One of {'6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct'}.

        Returns
        -------
        str or None
        """
        if slope is None:
            return self._SLOPE_OPTIONS[self._qi("SLOPE")]
        if slope not in self._SLOPE_OPTIONS:
            raise ValueError(f"slope must be one of {self._SLOPE_OPTIONS}.")
        self.write(f"SLOPE {self._SLOPE_OPTIONS.index(slope)}")
        self.info(f"Set low-pass filter slope to {slope}.")

    def sync_filter_state(self, on: bool = None) -> bool | None:
        """
        Set or query the synchronous filter (``SYNC``), effective below ~200 Hz.

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return self._qi("SYNC") == 1
        self.write(f"SYNC {1 if on else 0}")
        self.info(f"{'En' if on else 'Dis'}abled sync filter.")

    def _acgain_by_index(self, idx: int) -> Tuple[int, float, float]:
        for g in self._AC_GAIN:
            if g[0] == idx:
                return g
        return self._AC_GAIN[-1]

    def ac_gain(self, dB: float = None) -> float | None:
        """
        Set or query the AC signal-channel gain in dB (``ACGAIN``).

        On set, the nearest available gain is chosen from those whose input
        full-scale limit is still compatible with the current sensitivity.

        Parameters
        ----------
        dB : float, optional

        Returns
        -------
        float or None
        """
        if dB is None:
            return self._acgain_by_index(self._qi("ACGAIN"))[2]
        sens = self.input_sensitivity()
        candidates = [g for g in self._AC_GAIN if g[1] >= sens] or list(self._AC_GAIN)
        best = min(candidates, key=lambda g: abs(g[2] - dB))
        self.write(f"ACGAIN {best[0]}")
        self.info(f"Set AC gain to {best[2]} dB.")

    def dynamic_reserve(self, dB: float = None) -> float | None:
        """
        Set or query the dynamic reserve in dB.

        On the 7280 reserve is not a direct setting: it is the difference between
        the full-scale gain implied by the current sensitivity and the AC gain,
        ``DR = 20*log10(peakToRMS * 1.6 / sensitivity) - ACGain``. Setting a
        reserve therefore programs the AC gain accordingly.

        Parameters
        ----------
        dB : float, optional

        Returns
        -------
        float or None
        """
        sens = self.input_sensitivity()
        full_scale_gain = 20.0 * log10(self._PEAK_TO_RMS * 1.6 / sens)
        if dB is None:
            return full_scale_gain - self.ac_gain()
        self.ac_gain(full_scale_gain - dB)
        self.info(f"Set dynamic reserve to {dB} dB.")

    def automatic_gain(self, on: bool = None) -> bool | None:
        """
        Set or query automatic AC-gain control (``AUTOMATIC``).

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return self._qi("AUTOMATIC") != 0
        self.write(f"AUTOMATIC {1 if on else 0}")
        self.info(f"{'En' if on else 'Dis'}abled automatic gain.")

    def fast_mode(self, on: bool = None) -> bool | None:
        """
        Set or query fast output mode (``FASTMODE``).

        Note: fast mode changes which time-constant table is active.

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return self._qi("FASTMODE") != 0
        self.write(f"FASTMODE {1 if on else 0}")
        self.info(f"{'En' if on else 'Dis'}abled fast mode.")

    def noise_mode(self, on: bool = None) -> bool | None:
        """
        Set or query noise-measurement mode (``NOISEMODE``).

        Note: noise mode changes which time-constant table is active.

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return self._qi("NOISEMODE") != 0
        self.write(f"NOISEMODE {1 if on else 0}")
        self.info(f"{'En' if on else 'Dis'}abled noise mode.")

    # ── Auxiliary I/O ────────────────────────────────────────────────────────

    def get_aux_input(self, idx: int) -> float:
        """
        Query auxiliary ADC input `idx` in volts (``ADC.``).

        Parameters
        ----------
        idx : int

        Returns
        -------
        float
        """
        if idx not in self.in_aux_channels:
            raise IndexError(f"ADC channel {idx} out of range.")
        return self._qf(f"ADC. {idx}")

    def aux_output(self, idx: int, V: float = None) -> float | None:
        """
        Set or query auxiliary DAC output `idx` in volts (``DAC.``).

        Parameters
        ----------
        idx : int
        V : float, optional

        Returns
        -------
        float or None
        """
        if idx not in self.out_aux_channels:
            raise IndexError(f"DAC channel {idx} out of range.")
        if V is None:
            return self._qf(f"DAC. {idx}")
        self.write(f"DAC. {idx} {V:.3f}")
        self.info(f"Set auxiliary output {idx} to {V} V.")

    # ── Auto functions ───────────────────────────────────────────────────────

    def auto_phase(self):
        """Auto-phase: null Y into X (``AQN``)."""
        self.write("AQN")

    def auto_offset(self):
        """Auto-offset the X/Y outputs (``AXO``)."""
        self.write("AXO")

    def auto_measure(self):
        """
        Auto-measure (``ASM``): set sensitivity so the magnitude is 30-90% of
        full scale, then auto-phase. Note this also changes the phase.
        """
        self.write("ASM")

    # ── Status ───────────────────────────────────────────────────────────────

    def status_byte(self) -> int:
        """
        Read the 7280 status byte via serial poll (falls back to ``ST``).

        Bit 4 (0x10): input/output overload; bit 3 (0x08): reference unlock.
        """
        try:
            return int(self.device.read_stb())
        except Exception:
            return self._qi("ST")

    def is_overloaded(self) -> bool:
        """True if the input or an output is overloaded."""
        return bool(self.status_byte() & 0x10)

    def is_reference_unlocked(self) -> bool:
        """True if the reference PLL is unlocked."""
        return bool(self.status_byte() & 0x08)

    def is_locked(self) -> bool:
        """True if the reference PLL is locked."""
        return not self.is_reference_unlocked()

    # ── Misc device control ──────────────────────────────────────────────────

    def sample_rate(self, hz: float = None) -> float | None:
        """
        Set or query the curve-storage sample rate in Hz (``STR``, interval in ms).

        Parameters
        ----------
        hz : float, optional

        Returns
        -------
        float or None
        """
        if hz is None:
            ms = self._qi("STR")
            return 1000.0 / ms if ms > 0 else 0.0
        ms = int(1000.0 / hz)
        ms -= ms % 5                       # instrument rounds to multiples of 5 ms
        ms = max(ms, 0)
        self.write(f"STR {ms}")
        actual = 1000.0 / ms if ms > 0 else 0.0
        self.info(f"Set sample rate to {actual} Hz.")

    def effective_noise_bandwidth(self) -> float:
        """Effective noise bandwidth of the current output filter in Hz (``ENBW.``)."""
        return self._qf("ENBW.")

    def route_analog_output(self, channel: int, quantity: str):
        """
        Route a measured quantity to a rear-panel analog output (``CH``).

        Parameters
        ----------
        channel : int
        quantity : str
            One of {'x', 'y', 'r', 'p', 'n'} (X, Y, magnitude, phase, noise).
        """
        codes = {"x": 0, "y": 1, "r": 2, "p": 3, "n": 5}
        if quantity not in codes:
            raise ValueError("quantity must be one of 'x','y','r','p','n'.")
        self.write(f"CH {channel} {codes[quantity]}")
        self.info(f"Routed {quantity} to analog output {channel}.")

    def remote(self):
        """Engage GPIB remote lockout (``REMOTE 1``)."""
        self.write("REMOTE 1")

    def local(self):
        """Release GPIB remote lockout (``REMOTE 0``)."""
        self.write("REMOTE 0")

    def restore_defaults(self):
        """Restore the instrument's factory default settings (``ADF 0``)."""
        self.write("ADF 0")
        self.info("Restored factory default settings.")

    def front_panel_display(self, on: bool):
        """Enable/disable the front-panel display (``LTS``)."""
        self.write(f"LTS {1 if on else 0}")

    # ── Averaging (software polling; matches the sr860 return contract) ──────

    def setup_data_acquisition(self, n_samples: int,
                               sample_interval: float = None) -> float:
        """
        Configure a software-polled X/Y average.

        Parameters
        ----------
        n_samples : int
            Number of (X, Y) samples to average (>= 2).
        sample_interval : float, optional
            Seconds between samples. Defaults to the current time constant.

        Returns
        -------
        sample_interval : float
            The interval that will be used.
        """
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2.")
        tau = self.time_constant()
        if sample_interval is None:
            sample_interval = max(tau, 1e-3)
        self._acq = {
            "n"    : int(n_samples),
            "dt"   : float(sample_interval),
            "tau"  : float(tau),
            "ready": True,
        }
        self.info(f"Configured averaging: n={n_samples}, dt={sample_interval:g}s.")
        return sample_interval

    def _poll_xy(self) -> np.ndarray:
        """Poll (X, Y) `n` times at the configured interval. Returns (n, 2)."""
        if not (self._acq and self._acq.get("ready")):
            raise RuntimeError("Call setup_data_acquisition before averaging.")
        n, dt = self._acq["n"], self._acq["dt"]
        samples = np.empty((n, 2))
        for i in range(n):
            samples[i] = self.get_xy()
            if dt > 0 and i < n - 1:
                time.sleep(dt)
        return samples

    def get_average(self, auto_rescale: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Poll X/Y and return the mean and covariance.

        Parameters
        ----------
        auto_rescale : bool, default False
            If True, run an auto-measure (``ASM``) first. Note this also
            re-phases the instrument.

        Returns
        -------
        mean, cov : np.ndarray
            mean has shape (2,) = (X, Y); cov has shape (2, 2).
        """
        if auto_rescale:
            self.auto_measure()
        samples = self._poll_xy()
        cov = np.cov(samples.T)
        if self._acq["dt"] < self._acq["tau"]:
            cov = cov * (self._acq["tau"] / self._acq["dt"])
        return samples.mean(axis=0), cov

    def get_average_series_correlated(self, auto_rescale: bool = False,
                                      L: int = None
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Poll X/Y and return the mean and a serial-correlation-aware covariance.

        Parameters
        ----------
        auto_rescale : bool, default False
            If True, run an auto-measure (``ASM``) first.
        L : int, optional
            Maximum autocorrelation length (in samples). Defaults to ~20 tau.

        Returns
        -------
        mean, cov : np.ndarray
        """
        if auto_rescale:
            self.auto_measure()
        samples = self._poll_xy()
        if L is None:
            L = int(20 * self._acq["tau"] / self._acq["dt"])
        return samples.mean(axis=0), series_correlated_covariance(samples, L=L)

    def _curve_status(self) -> Tuple[int, int, int, int]:
        """
        Parse the ``M`` curve-acquisition monitor into
        ``(status, sweeps, status_byte, points_acquired)``.

        status: 0 idle, 1 TD running, 2 TDC running, 5 TD halted, 6 TDC halted.
        points_acquired is zeroed by ``NC`` and increments once per stored
        sample, so it is an unambiguous completion counter after an ``NC``.
        (Manual, ``M`` command, p. 6-25.)
        """
        parts = self._q("M").replace(",", " ").split()
        vals = [int(float(v)) for v in parts[:4]]
        vals += [0] * (4 - len(vals))
        return vals[0], vals[1], vals[2], vals[3]

    def _read_curve_values(self, n_points: int, timeout: float = 120.0) -> np.ndarray:
        """
        Read ``n_points`` values of a single curve dumped by ``DC[.]``.

        Follows the manual's GPIB read protocol: serial-poll for bit 7 (data
        available) before each read, until ``n_points`` values have arrived.
        The tight loop talks to the raw VISA resource so it bypasses the
        wrapper's ``min_interval`` throttle — otherwise every point would be
        paced by that interval, making a 1024-point dump take tens of seconds.
        (Manual, ``DC[.]`` description, p. 6-26.)
        """
        dev = self.device
        vals: List[float] = []
        deadline = time.time() + timeout
        while len(vals) < n_points:
            if time.time() > deadline:
                raise TimeoutError(
                    f"curve dump timed out at {len(vals)}/{n_points} points."
                )
            try:
                data_ready = bool(int(dev.read_stb()) & 0x80)   # bit 7
            except Exception:
                data_ready = True         # no serial poll -> fall back to a read
            if not data_ready:
                time.sleep(0.001)
                continue
            chunk = dev.read().replace("\x00", "").replace(",", " ").split()
            vals.extend(float(v) for v in chunk)
        return np.asarray(vals[:n_points])

    def acquire_curve(self, curves="XY", length: int = 1024,
                      sample_rate: float = None,
                      timeout: float = 120.0) -> Dict[str, np.ndarray]:
        """
        Hardware curve-buffer capture via the on-instrument buffer
        (``NC`` / ``CBD`` / ``LEN`` / ``STR`` / ``TD`` + ``DC.`` readout).

        Stores the requested curves — plus the SEN curve when any of X/Y/R/noise
        is requested, since the floating-point ``DC.`` conversion to volts/amps
        needs it — runs a single ``TD`` acquisition at the configured sample
        rate, waits for completion via the ``M`` monitor, then dumps each
        requested curve in calibrated floating point.

        Still worth a one-time bench check: the exact GPIB read framing on your
        VISA stack (whether ``DC.`` returns one value per read or a block), and
        the current-mode SEN co-store. ``get_average`` remains the
        software-polled, tested-by-construction default.

        Parameters
        ----------
        curves : str or sequence of str
            Curves to capture. Tokens: 'X', 'Y', 'R' (= MAG), 'T' (= phase),
            'N' (= noise), 'RATIO', 'LOGRATIO', 'ADC1'..'ADC4', 'DAC1', 'DAC2'.
            A bare string such as ``"XY"`` is read as individual single-letter
            tokens; pass a list for the multi-letter names.
        length : int
            Points per curve.
        sample_rate : float, optional
            Samples per second (sets ``STR``); defaults to the current setting.
        timeout : float
            Per-phase timeout in seconds, applied to the acquisition wait and to
            each curve dump.

        Returns
        -------
        dict[str, np.ndarray]
            Canonical curve name -> calibrated values, each of shape (length,).
            Note 'R' maps to key 'MAG', 'T' to 'PHA', 'N' to 'NOISE'.
        """
        # Normalize the requested curves to canonical names (order preserved).
        tokens = list(curves)
        names: List[str] = []
        for tok in tokens:
            key = str(tok).upper()
            if key not in self._CURVE_ALIAS:
                raise ValueError(f"unknown curve token {tok!r}.")
            name = self._CURVE_ALIAS[key]
            if name not in names:
                names.append(name)
        if not names:
            raise ValueError("request at least one curve.")

        # Build the CBD storage set: requested curves + SEN if any need it for
        # the %FS -> volts/amps conversion on the dotted dump.
        store = list(names)
        if any(n in self._CURVE_NEEDS_SEN for n in names) and "SEN" not in store:
            store.append("SEN")

        n_curves = len(store)
        if length * n_curves > self._CURVE_BUFFER_TOTAL:
            max_len = self._CURVE_BUFFER_TOTAL // n_curves
            raise ValueError(
                f"length {length} x {n_curves} stored curves exceeds the "
                f"{self._CURVE_BUFFER_TOTAL}-point buffer (max {max_len})."
            )
        mask = sum(1 << self._CURVE_BIT[n] for n in store)

        if sample_rate is not None:
            self.sample_rate(sample_rate)

        # Order matters (manual p. 6-24): NC resets the buffer and counters,
        # CBD defines the curves, LEN sets the length against that curve count.
        self.write("NC")
        self.write(f"CBD {mask}")
        self.write(f"LEN {int(length)}")
        self.write("TD")

        # Wait for completion. points_acquired reaching `length` is the
        # unambiguous signal (counters were just zeroed by NC); a return to idle
        # with points already logged also counts. Bail on the halted states.
        deadline = time.time() + timeout
        while True:
            status, _sweeps, _sb, points = self._curve_status()
            if points >= length:
                break
            if status == 0 and points > 0:
                break
            if status in (5, 6):
                raise RuntimeError(
                    f"curve acquisition halted at {points}/{length} points."
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"curve acquisition timed out at {points}/{length} points."
                )
            time.sleep(0.05)

        # Dump each requested data curve (never the SEN curve) in floating point.
        out: Dict[str, np.ndarray] = {}
        for name in names:
            self.write(f"DC. {self._CURVE_BIT[name]}")
            out[name] = self._read_curve_values(length, timeout=timeout)
        return out

    # ── Serialization ────────────────────────────────────────────────────────

    def save_state_json(self, path: str):
        """Save the lock-in state to JSON locally."""
        super().save_state_json(path)

    def load_state_json(self, path: str):
        """Load the lock-in state from JSON locally."""
        super().load_state_json(path)

    # Settings serialized as get/set accessors, in a safe application order.
    _STATE_SETTINGS = [
        "reference_source", "internal_reference", "reference_trigger",
        "reference_phase", "detection_harmonic", "sine_output_amplitude",
        "input_mode_current", "input_configuration", "input_shield_grounded",
        "input_coupling_DC", "line_notch_frequency", "line_notch_filter",
        "fast_mode", "noise_mode", "ac_gain", "automatic_gain",
        "input_sensitivity", "time_constant", "low_pass_filter_slope",
        "sync_filter_state", "sample_rate", "reference_frequency",
    ]

    def _serialize_state(self) -> dict:
        state = super()._serialize_state()
        for setting in self._STATE_SETTINGS:
            if hasattr(self, setting):
                try:
                    state[setting] = getattr(self, setting)()
                except Exception:
                    continue
        state["aux_output"] = []
        for ch in self.out_aux_channels:
            try:
                state["aux_output"].append((ch, self.aux_output(ch)))
            except Exception:
                continue
        return state

    def _deserialize_state(self, state: dict):
        super()._deserialize_state(state)
        # Apply in _STATE_SETTINGS order: input mode before sensitivity, fast/
        # noise before time constant, reference source before frequency — so the
        # index tables and gating resolve against the correct mode.
        for setting in self._STATE_SETTINGS:
            if setting in state and hasattr(self, setting):
                try:
                    getattr(self, setting)(state[setting])
                except Exception:
                    continue
        for ch, val in state.get("aux_output", []):
            try:
                self.aux_output(ch, val)
            except Exception:
                continue

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "model7280":
        """
        Construct from serialized config.

        Parameters
        ----------
        config : dict
            Output from _serialize_state(), plus 'registry_id'.
        """
        registry_id = config.pop("registry_id")
        resource_name = config.pop("resource_name")
        instance = cls(
            resource_name=resource_name,
            registry_id=registry_id,
            skip_connect=False,
            **config,
        )
        instance._deserialize_state(config)
        return instance

    # ── resolve() channels (shared with the SRS ecosystem) ──────────────────

    def resolve(self, accessor: str) -> ScalarChannelAdapter:
        _lockin_accessors = {
            "get_average":       (False, 1.0),
            "get_average_uV":    (False, 1e6),
            "get_average_sc":    (True,  1.0),
            "get_average_sc_uV": (True,  1e6),
        }
        if accessor in _lockin_accessors:
            series_corr, scale = _lockin_accessors[accessor]
            return sr860_lockin_channel(self, accessor, scale, series_corr)
        if accessor == "input_sensitivity":
            return SRSLockin_sensitivity_channel(self, lockin_scale=1.0)
        if accessor == "input_sensitivity_uV":
            return SRSLockin_sensitivity_channel(self, lockin_scale=1e6)
        raise ValueError(
            f"{self.__class__.__name__} cannot resolve accessor: {accessor!r}"
        )

    # ── GUI-facing discrete tables (mode-aware) ─────────────────────────────

    @property
    def sens_vals(self) -> np.ndarray:
        """Full-scale sensitivity values for the active input mode (for the GUI)."""
        try:
            table = self._sens_table()
        except Exception:
            table = self._SENS_VOLTAGE
        return np.array([v for _, v in table])

    @property
    def tau_vals(self) -> np.ndarray:
        """Time-constant values for the active fast/noise mode (for the GUI)."""
        try:
            table = self._tau_table()
        except Exception:
            table = self._TC_NORMAL
        return np.array([v for _, v in table])

    # ── GUI lifecycle ────────────────────────────────────────────────────────

    def launch_gui(self, port: int = 8760, poll_interval: float = 0.5) -> "SRSLockinGUI":
        """
        Launch a localhost web GUI for this instrument.
        Sets self.gui and returns the SRSLockinGUI instance.
        """
        from .srs_lockin.gui import SRSLockinGUI
        gui = getattr(self, "gui", None)
        if gui is not None:
            print(f"GUI already running → http://localhost:{gui._port}")
            return gui
        return SRSLockinGUI(self, port=port, poll_interval=poll_interval).start()

    def kill_gui(self):
        """Stop the running GUI, if any."""
        gui = getattr(self, "gui", None)
        if gui is not None:
            gui.stop()