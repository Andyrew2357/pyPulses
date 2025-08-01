"""
Control classes for Stanford Research Systems DSP Lock-in Amplifiers.
"""

from .pyvisa_device import pyvisaDevice

import numpy as np
from typing import Tuple

class SRSLockin(pyvisaDevice):
    """
    Base control class for 800 series Stanford Research Systems lock-in 
    amplifiers.
    """
    def __init__(self, logger = None, instrument_id: str = None):
        
        assert hasattr(self, 'pyvisa_config')
        assert hasattr(self, 'out_aux_channels')
        assert hasattr(self, 'output_map')
        assert hasattr(self, 'sens_vals')
        assert hasattr(self, 'tau_vals')
        if not hasattr(self, 'irng_vals'):
            self.irng_vals = []

        default_cmds = {
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

        if not hasattr(self, 'cmd_map'):
            self.cmd_map =  {}
        for k, v in default_cmds.items():
            self.cmd_map.setdefault(k, v)

        super().__init__(self.pyvisa_config, logger, instrument_id)

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
        Set or query whether the reference is internal or external

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
        Set or query the detection harmonic

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

    def reference_input_impedance(self, imp: str = None) -> str | None:
        """
        Set or query the reference signal input impedance.

        Parameters
        ----------
        imp : str, optional
            One of {'low', 'high'}

        Returns
        -------
        float or None
        """

        if imp is None:
            return ['low', 'high'][int(self.query(f"{self.cmd_map['refz']}?"))]
        opt = {'low': 0, 'high': 1}
        self.write(f"{self.cmd_map['refz']} {opt[imp]}")
        self.info(f"Set external reference input impedance {imp}.")

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

    def sine_output_offset(self, V: float = None) -> float | None:
        """
        Set or query the sine output DC offset in Volts.

        Parameters
        ----------
        V : float, optional

        Returns
        -------
        float or None
        """

        if V is None:
            return float(self.query(f"{self.cmd_map['soff']}?"))
        self.write(f"{self.cmd_map['soff']} {V}")
        self.info(f"Set sine output DC offset to {V} V.")
    

    # INPUT / SIGNAL SETTINGS
    
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

    def input_mode_current(self, I_mode: bool = None) -> bool | None:
        """
        Set or query whether the input mode is current or voltage.

        Parameters
        ----------
        I_mode : bool, optional
        
        Returns
        -------
        bool or None
        """

        if I_mode is None:
            return int(self.query(f"{self.cmd_map['ivmd']}?")) == 1
        self.write(f"{self.cmd_map['ivmd']} {int(I_mode)}")
        self.info(f"Set input to {'current' if I_mode else 'voltage'} mode.")

    def _input_range_value(self, irng_index: int) -> float:
        """Convert input range index to actual value in volts."""
        return self.irng_vals[irng_index]
    
    def _input_range_index(self, irng_val: float) -> int:
        """Convert input range value to index (rounds to closest value)."""
        return np.argmin(np.abs(self.irng_vals - irng_val))

    def input_range(self, V: float = None) -> float | None:
        """
        Set or query the input range.

        Parameters
        ----------
        V : float, optional

        Returns
        -------
        float or None
        """

        if V is None:
            return self._input_range_value(
                int(self.query(f"{self.cmd_map['irng']}?")))
        self.write(f"{self.cmd_map['irng']} {self._input_range_index(V)}")
        self.info(
            f"Set input range to {
            self._input_range_value(self._input_range_index(V))} V."
        )

    def current_gain(self, gan: str = None) -> str | None:
        """
        Set or query the current input gain

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

    def line_notch_filter(self, setting: str = None) -> str | None:
        """
        Sets or queries the input line notch filter status.

        Parameters
        ----------
        setting : str, optional
            One of {'Out', 'In', '2xIn', 'Both'}

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

    def get_input_level_status(self) -> int:
        """
        Query the input level indicator from lowest (0) to overload (4).

        Returns
        -------
        int
        """

        return int(self.query(f"{self.cmd_map['ilvl']}?"))

    # GAIN AND TIME CONSTANT SETTINGS

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

    def input_sensitivity(self, V: float = None) -> float | None:
        """
        Set or query the input sensitivity in Volts.

        Parameters
        ----------
        V : float, optional

        Returns
        -------
        float or None
        """

        if V is None:
            return self._sens_value(int(self.query(f"{self.cmd_map['sens']}?")))
        self.write(f"{self.cmd_map['sens']} {self._sens_index(V)}")
        self.info(
            f"Set sensitivity to {self._sens_value(self._sens_index(V))} V."
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

    # AUXILIARY I/O

    def get_aux_input(self, idx: int) -> float:
        """
        Set or query the auxiliary input in Volts.

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
        Set or query the auxiliary output in Volts.

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

    def auto_range(self):
        """Automatically set the input range."""
        self.write(self.cmd_map['arng'])

    # SERIALIZATION AND DESERIALIZATION

    # Plan to add this here (generally good for instruments with a ton of settings)
    # may avoid applying it to auxiliary outputs since indices differ across models 
    # (maybe add an attribute that reflects this, wouldn't be too hard)
    # Will also need to override it to add acquisition configurations for sr860/sr865a

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

        simple_settings = [
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

        state = {}
        for setting in simple_settings:
            try:
                state[setting] = getattr(self, setting, lambda: None)()
            except:
                continue

        # Handle auxiliary outputs
        state['aux_output'] = []
        for ch in self.out_aux_channels:
            try:
                state['aux_output'].append((ch, self.aux_output(ch)))
            except:
                continue

        return state

    def _deserialize_state(self, state: dict):
        
        simple_settings = [
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

        for setting in simple_settings:
            try:
                getattr(self, setting)(state[setting])
            except:
                continue

        for ch, val in state['aux_output']:
            try:
                self.aux_output(ch, val)
            except:
                continue
