"""
Control classes for Stanford Research Systems DSP Lock-in Amplifiers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gui import SRSLockinGUI

from ..pyvisa_device import pyvisaDevice
from ..channel_adapter import ScalarChannelAdapter

import numpy as np
from logging import Logger
from typing import Any, Dict, List, Tuple


class SRSLockin(pyvisaDevice):
    """
    Base control class for 800 series Stanford Research Systems lock-in
    amplifiers. Contains only functionality common to all models.

    Model-specific capabilities are provided through mixins defined in this
    module:
        ReferenceTriggerMixin         - reference_trigger (SR830, SR850, SR860/865A)
        ReferenceInputImpedanceMixin  - reference_input_impedance (SR844, SR860/865A)
        SineOutputAmplitudeMixin      - sine_output_amplitude (SR830, SR850, SR860/865A)
        VoltageInputMixin             - input_configuration, input_shield_grounded
                                        (SR830, SR850, SR860/865A)
        CurrentGainMixin              - current_gain (SR850, SR860/865A)
        LineNotchFilterMixin          - line_notch_filter (SR830, SR850)
        InputLevelStatusMixin         - get_input_level_status (SR860/865A)
        SyncFilterMixin               - sync_filter_state (SR830, SR850, SR860/865A)
    """

    out_aux_channels: List[int]
    output_map: Dict[str, int]
    sens_vals: np.ndarray
    tau_vals: np.ndarray
    irng_vals: np.ndarray = np.array([])
    cmd_map: Dict[str, str] = {}

    DEFAULT_CMDS = {
        'phas': "PHAS",
        'fmod': "FMOD",
        'freq': "FREQ",
        'harm': "HARM",
        'rslp': "RSLP",
        'refz': "REFZ",
        'slvl': "SLVL",
        'soff': "SOFF",
        'isrc': "ISRC",
        'ignd': "IGND",
        'icpl': "ICPL",
        'ivmd': "IVMD",
        'irng': "IRNG",
        'igan': "IGAN",
        'ilin': "ILIN",
        'ilvl': "ILVL",
        'sens': "SENS",
        'oflt': "OFLT",
        'ofsl': "OFSL",
        'sync': "SYNC",
        'oaux': "OAUX",
        'auxv': "AUXV",
        'aphs': "APHS",
        'arng': "ARNG",
        'agan': "AGAN",
    }

    def __init__(self,
        resource_name: str,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_connect: bool = False,
        **kwargs,
    ):
        super().__init__(
            resource_name=resource_name,
            registry_id=registry_id,
            logger=logger,
            skip_connect=skip_connect,
            **kwargs
        )

        for k, v in self.DEFAULT_CMDS.items():
            self.cmd_map.setdefault(k, v)

    # QUERYING DATA

    def _get_output(self, parm: str) -> float:
        return float(self.query(f"OUTP? {self.output_map[parm]}"))

    def _get_snap(self, parms: str) -> Tuple[float, ...]:
        args = str([self.output_map[ch] for ch in parms])[1:-1]
        return tuple(map(float, self.query(f"SNAP? {args}").strip().split(',')))

    def get_x(self) -> float:
        return self._get_output('X')

    def get_y(self) -> float:
        return self._get_output('Y')

    def get_r(self) -> float:
        return self._get_output('R')

    def get_t(self) -> float:
        return self._get_output('T')

    def get_xy(self) -> Tuple[float, float]:
        return self._get_snap('XY')

    def get_rt(self) -> Tuple[float, float]:
        return self._get_snap('RT')

    # TIMEBASE SETTINGS

    def reference_phase(self, phase: float = None) -> float | None:
        """
        Set or query the reference phase shift in degrees.

        Parameters
        ----------
        phase : float, optional

        Returns
        -------
        float or None
        """
        if phase is None:
            return float(self.query(f"{self.cmd_map['phas']}?"))
        self.write(f"{self.cmd_map['phas']} {phase}")
        self.info(f"Set reference phase shift to {phase} degrees.")

    def internal_reference(self, on: bool = None) -> bool | None:
        """
        Set or query whether the reference is internal or external.

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return int(self.query(f"{self.cmd_map['fmod']}?")) == 1
        self.write(f"{self.cmd_map['fmod']} {int(on)}")
        self.info(f"{'En' if on else 'Dis'}abled internal reference.")

    def reference_frequency(self, freq: float = None) -> float | None:
        """
        Set or query the reference frequency in Hz (set only works if using an
        internal reference).

        Parameters
        ----------
        freq : float, optional

        Returns
        -------
        float or None
        """
        if freq is None:
            return float(self.query(f"{self.cmd_map['freq']}?"))
        if self.internal_reference():
            self.write(f"{self.cmd_map['freq']} {freq}")
            self.info(f"Set reference frequency to {freq} Hz.")

    def detection_harmonic(self, harm: int = None) -> int | None:
        """
        Set or query the detection harmonic.

        Parameters
        ----------
        harm : int, optional

        Returns
        -------
        int or None
        """
        if harm is None:
            return int(self.query(f"{self.cmd_map['harm']}?"))
        self.write(f"{self.cmd_map['harm']} {harm}")
        self.info(f"Set detection harmonic to {harm}")

    # INPUT / SIGNAL SETTINGS

    def input_coupling_DC(self, dc_couple: bool = None) -> bool | None:
        """
        Set or query whether the input is DC or AC coupled.

        Parameters
        ----------
        dc_couple : bool, optional

        Returns
        -------
        bool or None
        """
        if dc_couple is None:
            return int(self.query(f"{self.cmd_map['icpl']}?")) == 1
        self.write(f"{self.cmd_map['icpl']} {int(dc_couple)}")
        self.info(f"Set input coupling to {'DC' if dc_couple else 'AC'}.")

    # GAIN AND TIME CONSTANT SETTINGS

    def _input_range_value(self, irng_index: int) -> float:
        """Convert input range index to actual value in volts."""
        return self.irng_vals[irng_index]

    def _input_range_index(self, irng_val: float) -> int:
        """Convert input range value to index (rounds to closest value)."""
        return np.argmin(np.abs(self.irng_vals - irng_val))

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

    def input_sensitivity(self, val: float = None, units: str = 'V') -> float | None:
        """
        Set or query the input sensitivity.

        Parameters
        ----------
        val : float, optional
        units : str, default='V'
            One of {'pV', 'nV', 'uV', 'mV', 'V', 'pA', 'nA', 'uA', 'mA', 'A'}

        Returns
        -------
        float or None
        """
        coeff = 1 if units[-1] == 'V' else 1e-6
        if len(units) == 2:
            coeff *= {'p': 1e12, 'n': 1e9, 'u': 1e6, 'm': 1e3}[units[0]]

        if val is None:
            return coeff * self._sens_value(
                int(self.query(f"{self.cmd_map['sens']}?"))
            )
        self.write(f"{self.cmd_map['sens']} {self._sens_index(val / coeff)}")
        self.info(
            f"Set sensitivity to {self._sens_value(self._sens_index(val))} {units}."
        )

    def time_constant(self, tau: float = None) -> float | None:
        """
        Set or query the time constant in seconds.

        Parameters
        ----------
        tau : float, optional

        Returns
        -------
        float or None
        """
        if tau is None:
            return self._tau_value(int(self.query(f"{self.cmd_map['oflt']}?")))
        self.write(f"{self.cmd_map['oflt']} {self._tau_index(tau)}")
        self.info(
            f"Set time constant to {self._tau_value(self._tau_index(tau))} s."
        )

    def low_pass_filter_slope(self, slope: str = None) -> str | None:
        """
        Set or query the low-pass filter slope.

        Parameters
        ----------
        slope : str, optional
            One of {'6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct'}.

        Returns
        -------
        str or None
        """
        if slope is None:
            return ['6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct'][
                int(self.query(f"{self.cmd_map['ofsl']}?"))]
        opt = {'6dB/oct': 0, '12dB/oct': 1, '18dB/oct': 2, '24dB/oct': 3}
        self.write(f"{self.cmd_map['ofsl']} {opt[slope]}")
        self.info(f"Set low-pass filter slope to {slope}.")

    # AUXILIARY I/O

    def get_aux_input(self, idx: int) -> float:
        """
        Query an auxiliary input in Volts.

        Parameters
        ----------
        idx : int

        Returns
        -------
        float
        """
        return float(self.query(f"{self.cmd_map['oaux']}? {idx}"))

    def aux_output(self, idx: int, V: float = None) -> float | None:
        """
        Set or query an auxiliary output in Volts.

        Parameters
        ----------
        idx : int
        V : float, optional

        Returns
        -------
        float or None
        """
        if not idx in self.out_aux_channels:
            raise IndexError
        if V is None:
            return float(self.query(f"{self.cmd_map['auxv']}? {idx}"))
        self.write(f"{self.cmd_map['auxv']} {idx}, {V}")
        self.info(f"Set auxiliary output {idx} to {V} V")

    # AUTO FUNCTIONS

    def auto_phase(self):
        """Automatically set the phase offset."""
        self.write(self.cmd_map['aphs'])

    def auto_gain(self):
        """Automatically set the input gain."""
        self.write(self.cmd_map['agan'])

    # SERIALIZATION AND DESERIALIZATION

    def save_state_json(self, path: str):
        """
        Save the Lock-in state to JSON locally.

        Parameters
        ----------
        path : str
            directory at which to save.
        """
        super().save_state_json(path)

    def load_state_json(self, path: str):
        """
        Load the Lock-in state from JSON locally.

        Parameters
        ----------
        path : str
            directory from which to load.
        """
        super().load_state_json(path)

    def _serialize_state(self) -> dict:
        _all_settings = [
            # timebase settings
            'reference_phase',
            'internal_reference',
            'reference_frequency',
            'detection_harmonic',
            'reference_trigger',
            'reference_input_impedance',
            'sine_output_amplitude',
            'sine_output_offset',
            # input settings
            'input_configuration',
            'input_shield_grounded',
            'input_coupling_DC',
            'input_mode_current',
            'input_range',
            'current_gain',
            'line_notch_filter',
            # gain and time constant settings
            'input_sensitivity',
            'time_constant',
            'low_pass_filter_slope',
            'sync_filter_state',
        ]

        state = super()._serialize_state()
        for setting in _all_settings:
            if hasattr(self, setting):
                try:
                    state[setting] = getattr(self, setting)()
                except Exception:
                    continue

        state['aux_output'] = []
        for ch in self.out_aux_channels:
            try:
                state['aux_output'].append((ch, self.aux_output(ch)))
            except Exception:
                continue

        return state

    def _deserialize_state(self, state: dict):
        super()._deserialize_state(state)

        _all_settings = [
            # timebase settings
            'reference_phase',
            'internal_reference',
            'reference_frequency',
            'detection_harmonic',
            'reference_trigger',
            'reference_input_impedance',
            'sine_output_amplitude',
            'sine_output_offset',
            # input settings
            'input_configuration',
            'input_shield_grounded',
            'input_coupling_DC',
            'input_mode_current',
            'input_range',
            'current_gain',
            'line_notch_filter',
            # gain and time constant settings
            'input_sensitivity',
            'time_constant',
            'low_pass_filter_slope',
            'sync_filter_state',
        ]

        for setting in _all_settings:
            if setting in state and hasattr(self, setting):
                try:
                    getattr(self, setting)(state[setting])
                except Exception:
                    continue

        for ch, val in state.get('aux_output', []):
            try:
                self.aux_output(ch, val)
            except Exception:
                continue

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SRSLockin':
        """
        Construct from serialized config.

        Parameters
        ----------
        config : dict
            Output from _serialize_state(), plus 'registry_id'.
        """
        registry_id = config.pop('registry_id')
        resource_name = config.pop('resource_name')

        instance = cls(
            resource_name=resource_name,
            registry_id=registry_id,
            skip_connect=False,
            **config
        )
        instance._deserialize_state(config)

        return instance

    def resolve(self, accessor: str) -> ScalarChannelAdapter:
        if accessor == 'input_sensitivity':
            return SRSLockin_sensitivity_channel(self, lockin_scale=1.0)
        if accessor == 'input_sensitivity_uV':
            return SRSLockin_sensitivity_channel(self, lockin_scale=1e6)
        raise ValueError(f"{self.__class__.__name__} cannot resolve accessor: {accessor!r}")

    # GUI

    def launch_gui(self, port: int = 8760, poll_interval: float = 0.5) -> 'SRSLockinGUI':
        """
        Launch a localhost web GUI for this instrument.
        Sets self.gui and returns the SRSLockinGUI instance.
        """
        from .gui import SRSLockinGUI
        gui = getattr(self, 'gui', None)
        if gui is not None:
            print(f"GUI already running → http://localhost:{gui._port}")
            return gui
        return SRSLockinGUI(self, port=port, poll_interval=poll_interval).start()

    def kill_gui(self):
        """Stop the running GUI, if any."""
        gui = getattr(self, 'gui', None)
        if gui is not None:
            gui.stop()


"""
─────────────────────────── CAPABILITY MIXINS ────────────────────────────

Each mixin provides one logical capability group. A concrete instrument
class inherits only the mixins for features its hardware actually supports.

Mixins are listed before SRSLockin in each class's bases so that super()
chains resolve mixin methods correctly under Python's MRO.
"""

class ReferenceTriggerMixin:
    """reference_trigger — SR830, SR850, SR860/865A"""

    def reference_trigger(self, condition: str = None) -> str | None:
        """
        Set or query the reference trigger condition when using an external
        reference.

        Parameters
        ----------
        condition : str, optional
            One of {'sine', 'pos', 'neg'}.

        Returns
        -------
        str or None
        """
        if condition is None:
            return ['sine', 'pos', 'neg'][
                int(self.query(f"{self.cmd_map['rslp']}?"))]
        opt = {'sine': 0, 'pos': 1, 'neg': 2}
        self.write(f"{self.cmd_map['rslp']} {opt[condition]}")
        self.info(f"Set external reference trigger condition to {condition}.")


class ReferenceInputImpedanceMixin:
    """reference_input_impedance — SR844, SR860/865A"""

    def reference_input_impedance(self, imp: str = None) -> str | None:
        """
        Set or query the reference signal input impedance.

        Parameters
        ----------
        imp : str, optional
            One of {'low', 'high'}.

        Returns
        -------
        str or None
        """
        if imp is None:
            return ['low', 'high'][int(self.query(f"{self.cmd_map['refz']}?"))]
        opt = {'low': 0, 'high': 1}
        self.write(f"{self.cmd_map['refz']} {opt[imp]}")
        self.info(f"Set external reference input impedance {imp}.")


class SineOutputAmplitudeMixin:
    """sine_output_amplitude — SR830, SR850, SR860/865A"""

    def sine_output_amplitude(self, V: float = None) -> float | None:
        """
        Set or query the sine output amplitude in Volts.

        Parameters
        ----------
        V : float, optional

        Returns
        -------
        float or None
        """
        if V is None:
            return float(self.query(f"{self.cmd_map['slvl']}?"))
        self.write(f"{self.cmd_map['slvl']} {V}")
        self.info(f"Set sine output amplitude to {V} V.")


class VoltageInputMixin:
    """input_configuration, input_shield_grounded — SR830, SR850, SR860/865A"""

    def input_configuration(self, src: str = None) -> str | None:
        """
        Set or query the input configuration.

        Parameters
        ----------
        src : str, optional
            One of {'A', 'A-B', 'I'}.

        Returns
        -------
        str or None
        """
        if src is None:
            return ['A', 'A-B', 'I'][
                int(self.query(f"{self.cmd_map['isrc']}?"))]
        opt = {'A': 0, 'A-B': 1, 'I': 2}
        self.write(f"{self.cmd_map['isrc']} {opt[src]}")
        self.info(f"Set input configuration to {src}.")

    def input_shield_grounded(self, gnd: bool = None) -> bool | None:
        """
        Set or query whether the shield of the input is grounded or floating.

        Parameters
        ----------
        gnd : bool, optional

        Returns
        -------
        bool or None
        """
        if gnd is None:
            return int(self.query(f"{self.cmd_map['ignd']}?")) == 1
        self.write(f"{self.cmd_map['ignd']} {int(gnd)}")
        self.info(f"Set input shield {'to ground' if gnd else 'floating'}.")


class CurrentGainMixin:
    """current_gain — SR850, SR860/865A"""

    def current_gain(self, gan: str = None) -> str | None:
        """
        Set or query the current input gain.

        Parameters
        ----------
        gan : str, optional
            One of {'1MEG', '100MEG'}.

        Returns
        -------
        str or None
        """
        if gan is None:
            return ['1MEG', '100MEG'][
                int(self.query(f"{self.cmd_map['igan']}?"))]
        opt = {'1MEG': 0, '100MEG': 1}
        self.write(f"{self.cmd_map['igan']} {opt[gan]}")
        self.info(f"Set input current gain to {opt}.")


class LineNotchFilterMixin:
    """line_notch_filter — SR830, SR850"""

    def line_notch_filter(self, setting: str = None) -> str | None:
        """
        Set or query the input line notch filter status.

        Parameters
        ----------
        setting : str, optional
            One of {'Out', 'In', '2xIn', 'Both'}.

        Returns
        -------
        str or None
        """
        if setting is None:
            return ['Out', 'In', '2xIn', 'Both'][
                int(self.query(f"{self.cmd_map['ilin']}?"))]
        opt = {'Out': 0, 'In': 1, '2xIn': 2, 'Both': 3}
        self.write(f"{self.cmd_map['ilin']} {opt[setting]}")
        self.info(f"Set line notch filter status to {setting}.")


class InputLevelStatusMixin:
    """get_input_level_status — SR860/865A only"""

    def get_input_level_status(self) -> int:
        """
        Query the input level indicator from lowest (0) to overload (4).

        Returns
        -------
        int
        """
        return int(self.query(f"{self.cmd_map['ilvl']}?"))


class SyncFilterMixin:
    """sync_filter_state — SR830, SR850, SR860/865A"""

    def sync_filter_state(self, on: bool = None) -> bool | None:
        """
        Set or query the sync filter status.

        Parameters
        ----------
        on : bool, optional

        Returns
        -------
        bool or None
        """
        if on is None:
            return int(self.query(f"{self.cmd_map['sync']}?")) == 1
        self.write(f"{self.cmd_map['sync']} {int(on)}")
        self.info(f"{'En' if on else 'Dis'}abled sync filter.")


class SRSLockin_sensitivity_channel(ScalarChannelAdapter):
    """
    ScalarChannel for lock-in input sensitivity in volts.

    Parameters
    ----------
    parent : SRSLockin
    lockin_scale : float
        The scale factor used by the accompanying lockin_call channel
        (e.g. 1e6 if readings are in µV). Sensitivity targets derived
        from lock-in readings must be divided by this before being passed
        to input_sensitivity().
    """
    def __init__(self, parent: SRSLockin, lockin_scale: float = 1.0):
        super().__init__(parent, 'input_sensitivity')
        self.lockin_scale = lockin_scale

    def get_output(self) -> float:
        return self._parent.input_sensitivity() * self.lockin_scale

    def set_output(self, value: float):
        self._parent.input_sensitivity(value / self.lockin_scale)