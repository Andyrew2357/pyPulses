"""
Zurich Instruments HF2LI lock-in amplifier.

Layout (verified on dev1616 with the MF option, planning/HF2LI_CHECK.ipynb):

    6 demodulators     demods/0-5, each on any of the 6 oscillators
    6 oscillators      oscs/0-5
    2 signal outputs   sigouts/0-1, each summing 8 mixer channels:
                         channels 0-5 belong to demods 0-5 (and use their
                           oscillator and phase shift),
                         channels 6-7 belong to PLL 0-1, which run on
                           oscillators 0-1 and have no phase control.
                       By default output 0 plays channel 6, output 1 channel 7.
    2 signal inputs    sigins/0-1

Units, as measured:
    sigouts/n/amplitudes/m  peak volts as a fraction of sigouts/n/range
    sigouts/n/offset        volts as a fraction of sigouts/n/range
    demod X, Y              volts RMS
    demods/m/phaseshift     degrees; rotates mixer channel m's output AND
                            demod m's reference together

The driver works in volts (peak) throughout and converts with the range.

Bridge use
----------
configure_bridge() does the routing and returns the channels:

    hf2 = hf2li('dev1616', registry_id='hf2')
    br = hf2.configure_bridge(
        std = (0, 1),            # reference capacitor: output 0, mixer ch 1
        ex  = (1, 2),            # sample:              output 1, mixer ch 2
        measure_demod = 0, input = 0, oscillator = 0,
        frequency = 1e6, output_range = 1.0, duration = 0.1,
    )
    br['Vex'](0.01)              # V peak
    ctx = CapContext(
        Vstd        = br['Vstd'],          # V peak  ('out0_ch1_amplitude')
        Theta       = br['Theta'],         # deg     ('out0_ch1_phase')
        lockin_call = br['stream_average'],
        Vex         = 0.01,                # same units as Vstd (V peak)
        ...
    )
    with hf2.streaming():          # optional: otherwise starts on first read
        runner.run()

The reference drive's phase is its demodulator's phase shift, which also
rotates that demodulator's reference; so Vstd and Vex sit on channels whose
demodulators are not the measuring one, all on one oscillator. Channels 6-7
(the HF2's default output channels) have no phase control and are refused.
The output range must cover CapContext.Vstd_max: the amplitude accessors
refuse values beyond it.

Reading
-------
On the HF2 every node read costs ~12 ms and sync() ~70 ms (measured), so a
'get_average' read (sync + fresh subscription) carries ~120 ms of overhead.
'stream_average' keeps the demodulator subscribed and costs ~20-30 ms per
read; each read takes the next round(rate * duration) samples with
timestamps after the latest sample at call time, so it only sees data
recorded after the call starts (minus the ~10-30 ms transfer latency). Wait
for settling after moving anything, as with any lock-in.
"""

from .zhinst_device import zhinstDevice
from .registry import register_hardware_class
from .channel_adapter import ScalarChannelAdapter
from .lockin.channels import series_correlated_covariance

import re
import numpy as np
from contextlib import contextmanager
from logging import Logger
from typing import Any, Dict, List, Tuple

@register_hardware_class("hf2li")
class hf2li(zhinstDevice):
    """Class representation of the HF2LI lock-in amplifier."""

    num_demods = 6
    num_oscillators = 6
    num_outputs = 2
    num_inputs = 2
    num_mixer_channels = 8

    # Mixer channels without a demodulator: channel -> PLL index
    PLL_CHANNELS = {6: 0, 7: 1}

    OUTPUT_RANGES = (0.01, 0.1, 1.0, 10.0)

    def __init__(self,
        device_id: str = 'dev1616',
        server_host: str = 'localhost',
        server_port: int = 8005,
        api_level: int = 1,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_connect: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        device_id : str, default='dev1616'
        server_host : str, default='localhost'
        server_port : int, default=8005
            HF2 data server port.
        api_level : int, default=1
            The HF2 only supports API level 1.
        registry_id : str, optional
        logger : Logger, optional
        skip_connect : bool, default=False
        """
        super().__init__(device_id, server_host, server_port, api_level,
                         registry_id, logger, skip_connect)

        self.demod = [hf2liDemodChannel(self, i) for i in range(self.num_demods)]
        self.output = [hf2liOutputChannel(self, i) for i in range(self.num_outputs)]
        self.input = [hf2liInputChannel(self, i) for i in range(self.num_inputs)]
        self.oscillator = [hf2liOscillator(self, i) for i in range(self.num_oscillators)]

        # Used by get_average() / the 'get_average*' accessors
        self.measure_demod = 0
        self.average_duration = 0.1

    """Averaging (LockinLike interface used by lockin_channel)"""

    def setup_average(self, demod: int = 0, duration: float = 0.1):
        """
        Choose the demodulator and averaging time used by get_average().

        Parameters
        ----------
        demod : int, default=0
        duration : float, default=0.1
            Averaging time in s.
        """
        self._check_demod(demod)
        if duration <= 0:
            raise ValueError("duration must be positive.")
        if demod != self.measure_demod and self.demod[self.measure_demod].streaming:
            self.demod[self.measure_demod].stop_stream()
        self.measure_demod = demod
        self.average_duration = float(duration)
        self.info(f"HF2LI: averaging demod {demod} for {duration} s.")

    def get_average(self, auto_rescale: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Average X, Y on the measurement demodulator (see setup_average).

        Parameters
        ----------
        auto_rescale : bool, default=False
            Not supported on the HF2LI.

        Returns
        -------
        mean, cov : ndarray
        """
        if auto_rescale:
            raise NotImplementedError("HF2LI: auto_rescale is not supported.")
        return self.demod[self.measure_demod].get_average(self.average_duration)

    def get_average_series_correlated(self, auto_rescale: bool = False,
                                      L: int | None = None
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like get_average, with a covariance that accounts for serial
        correlations (see lockin.channels.series_correlated_covariance).

        Parameters
        ----------
        auto_rescale : bool, default=False
            Not supported on the HF2LI.
        L : int, optional
            Maximum correlation length in samples. Defaults to 20 time
            constants.

        Returns
        -------
        mean, cov : ndarray
        """
        if auto_rescale:
            raise NotImplementedError("HF2LI: auto_rescale is not supported.")
        return self.demod[self.measure_demod].get_average_series_correlated(
            self.average_duration, L)

    def stream_average(self, auto_rescale: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like get_average, but from a persistent subscription on the
        measurement demodulator (started on first use): ~20-30 ms overhead
        per call instead of ~120 ms. Each call uses only samples recorded after
        it starts (minus at most the ~10-30 ms transfer latency).

        Parameters
        ----------
        auto_rescale : bool, default=False
            Not supported on the HF2LI.

        Returns
        -------
        mean, cov : ndarray
        """
        if auto_rescale:
            raise NotImplementedError("HF2LI: auto_rescale is not supported.")
        return self.demod[self.measure_demod].stream_average(self.average_duration)

    def stream_average_series_correlated(self, auto_rescale: bool = False,
                                         L: int | None = None
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like get_average_series_correlated, from the stream (see
        stream_average).

        Parameters
        ----------
        auto_rescale : bool, default=False
            Not supported on the HF2LI.
        L : int, optional

        Returns
        -------
        mean, cov : ndarray
        """
        if auto_rescale:
            raise NotImplementedError("HF2LI: auto_rescale is not supported.")
        return self.demod[self.measure_demod].stream_average(
            self.average_duration, series_corr=True, L=L)

    @contextmanager
    def streaming(self):
        """
        Stream the measurement demodulator for the duration of a block:

            with hf2.streaming():
                runner.run()
        """
        d = self.demod[self.measure_demod]
        d.start_stream()
        try:
            yield d
        finally:
            d.stop_stream()

    def get_average_all(self, duration=0.1) -> dict:
        """
        Return an average and covariance from every enabled demodulator.

        Parameters
        ----------
        duration : float, default=0.1
            duration of the average in s.

        Returns
        -------
        Dict[int, Tuple[ndarray, ndarray]]
            maps demodulator index to average and covariance matrix for (X,Y).
        """
        return {d.index: d.get_average(duration)
                for d in self.demod if d.is_enabled()}

    """Accessors"""

    _ACCESSOR_RE = re.compile(r'out(\d)_ch(\d)_(amplitude|phase)$')

    def resolve(self, accessor: str):
        """
        Resolve an accessor to a channel object.

        'get_average', 'get_average_uV', 'get_average_sc', 'get_average_sc_uV'
            LockInChannel on the measurement demodulator (setup_average),
            fresh subscription per call.
        'stream_average', 'stream_average_uV', 'stream_average_sc',
        'stream_average_sc_uV'
            The same from a persistent subscription (for scans and balances
            that read repeatedly). Starts on first call; stop with
            stop_all_streams() or use `with hf2.streaming():`.
        'out{o}_ch{m}_amplitude'
            ScalarChannel: peak amplitude (V) of mixer channel m on output o.
        'out{o}_ch{m}_phase'
            ScalarChannel: phase (deg) of mixer channel m (0-5) on output o.
        'osc{n}_freq'
            ScalarChannel: frequency (Hz) of oscillator n.
        """
        m = re.match(r'(get|stream)_average(_sc)?(_uV)?$', accessor)
        if m:
            return hf2li_lockin_channel(
                self, accessor,
                scale = 1e6 if m[3] else 1.0,
                series_corr = bool(m[2]),
                stream = m[1] == 'stream',
            )

        m = self._ACCESSOR_RE.match(accessor)
        if m:
            out, ch, kind = int(m[1]), int(m[2]), m[3]
            ac = hf2liACout(self, out, ch)
            if kind == 'phase' and ch in self.PLL_CHANNELS:
                raise ValueError(
                    f"{accessor}: mixer channel {ch} belongs to a PLL and has "
                    f"no phase control; use a channel 0-5."
                )
            cls = hf2li_amplitude_channel if kind == 'amplitude' \
                else hf2li_phase_channel
            return cls(self, accessor, ac)

        m = re.match(r'osc(\d)_freq$', accessor)
        if m and int(m[1]) < self.num_oscillators:
            return hf2li_frequency_channel(self, accessor, int(m[1]))

        raise ValueError(f"hf2li cannot resolve accessor: {accessor}")

    """Bridge setup"""

    def configure_bridge(self,
        std: Tuple[int, int] = (0, 1),
        ex: Tuple[int, int] = (1, 2),
        measure_demod: int = 0,
        input: int = 0,
        oscillator: int = 0,
        frequency: float | None = None,
        output_range: float = 1.0,
        duration: float = 0.1,
        rate: float | None = None,
    ) -> Dict[str, Any]:
        """
        Route a capacitance bridge and return its channels.

        The reference capacitor is driven from mixer channel std[1] on output
        std[0], the sample from channel ex[1] on output ex[0], and the bridge
        signal is measured on `measure_demod` from signal input `input`. All
        three share `oscillator`. Afterwards:

          - every other mixer channel on the two outputs is disabled and
            zeroed (the HF2's defaults, channels 6 and 7, would otherwise add
            to them);
          - both drive amplitudes are 0, the outputs are at `output_range`
            and switched on, so nothing is driven until you set Vex/Vstd;
          - get_average / stream_average read `measure_demod` for `duration`.

        Parameters
        ----------
        std, ex : (output, mixer channel)
            Channels 0-5 only (6-7 have no phase), different from each other
            and from `measure_demod` (a channel's phase also rotates its
            demodulator's reference).
        measure_demod : int, default=0
        input : int, default=0
            Signal input feeding the measurement demodulator.
        oscillator : int, default=0
        frequency : float, optional
            Set the oscillator frequency (Hz); unchanged if None.
        output_range : float, default=1.0
            Range of both outputs (V); must cover CapContext.Vstd_max and the
            largest Vex.
        duration : float, default=0.1
            Averaging time per reading (s).
        rate : float, optional
            Measurement demod sample rate (Sa/s); unchanged if None.

        Returns
        -------
        dict
            'Vstd', 'Theta', 'Vex', 'Vex_phase', 'freq' (ScalarChannels),
            'stream_average', 'get_average' (LockInChannels), and 'accessors'
            (the same as resolve() strings).
        """
        (so, sc), (eo, ec) = std, ex
        for o in (so, eo):
            if not 0 <= o < self.num_outputs:
                raise ValueError(f"HF2LI has no output {o}.")
        for c in (sc, ec):
            if c in self.PLL_CHANNELS or not 0 <= c < self.num_demods:
                raise ValueError(f"Bridge channels must be 0-5 (got {c}).")
        if len({sc, ec, measure_demod}) < 3:
            raise ValueError(
                "std channel, ex channel and measure_demod must all differ."
            )
        if not 0 <= input < self.num_inputs:
            raise ValueError(f"HF2LI has no input {input}.")
        if not 0 <= oscillator < self.num_oscillators:
            raise ValueError(f"No oscillator {oscillator}.")

        # Nothing driven while rerouting
        for o, c in ((so, sc), (eo, ec)):
            self.output[o].set_amplitude(0.0, c)
        for o in {so, eo}:
            keep = {c for oo, c in ((so, sc), (eo, ec)) if oo == o}
            out = self.output[o]
            for m in range(self.num_mixer_channels):
                if m not in keep:
                    out.enable_channel(m, False)
                    out.set_amplitude(0.0, m)
            out.set_offset(0.0)
            out.set_range(output_range)

        for d in (sc, ec, measure_demod):
            self.demod[d].set_oscillator(oscillator)
        if frequency is not None:
            self.oscillator[oscillator].set_frequency(frequency)

        dm = self.demod[measure_demod]
        dm.set_input(input)
        if rate is not None:
            dm.set_rate(rate)
        dm.enable(True)
        self.setup_average(measure_demod, duration)

        for o, c in ((so, sc), (eo, ec)):
            self.output[o].enable_channel(c, True)
        for o in {so, eo}:
            self.output[o].enable(True)
        self.sync()

        accessors = {
            'Vstd':           f'out{so}_ch{sc}_amplitude',
            'Theta':          f'out{so}_ch{sc}_phase',
            'Vex':            f'out{eo}_ch{ec}_amplitude',
            'Vex_phase':      f'out{eo}_ch{ec}_phase',
            'freq':           f'osc{oscillator}_freq',
            'stream_average': 'stream_average',
            'get_average':    'get_average',
        }
        self.info(f"HF2LI bridge: {accessors}")
        return {**{k: self.resolve(a) for k, a in accessors.items()},
                'accessors': accessors}

    """Routing"""

    def _check_demod(self, i: int):
        if not 0 <= i < self.num_demods:
            raise ValueError(f"HF2LI has demods 0-{self.num_demods - 1}, not {i}.")

    def _check_channel(self, m: int):
        if not 0 <= m < self.num_mixer_channels:
            raise ValueError(
                f"HF2LI has mixer channels 0-{self.num_mixer_channels - 1}, not {m}."
            )

    def channel_oscillator(self, channel: int) -> int:
        """
        Oscillator that drives a mixer channel.

        Parameters
        ----------
        channel : int
            0-5 follow their demodulator, 6-7 their PLL.

        Returns
        -------
        int
        """
        self._check_channel(channel)
        if channel in self.PLL_CHANNELS:
            return self.get_int(f"plls/{self.PLL_CHANNELS[channel]}/oscselect")
        return self.demod[channel].get_oscillator()

    def get_signal_routing(self) -> dict:
        """
        Get a dict summary of the output mixers and demodulator oscillators.

        Returns
        -------
        dict
        """
        return {
            'outputs': {
                o.index: {
                    'on': o.is_enabled(),
                    'range': o.get_range(),
                    'channels': {m: o.get_amplitude(m)
                                 for m in o.get_enabled_channels()},
                }
                for o in self.output
            },
            'demod_oscillators': {d.index: d.get_oscillator() for d in self.demod},
            'pll_oscillators': {
                m: self.channel_oscillator(m) for m in self.PLL_CHANNELS
            },
        }

    def summary(self) -> str:
        """
        Summarize the routing information from get_signal_routing in a string.

        Returns
        -------
        str
        """
        routing = self.get_signal_routing()
        lines = ["\nHF2LI Routing Summary:", "Outputs:"]
        for out, r in routing['outputs'].items():
            chans = ", ".join(
                f"ch{m} {v:.4g} Vpk (osc {self.channel_oscillator(m)})"
                for m, v in r['channels'].items()
            ) or "none"
            lines.append(f"  Output {out} ({'on' if r['on'] else 'off'}, "
                         f"range {r['range']} V): {chans}")
        lines.append("Demodulators:")
        for demod, osc in routing['demod_oscillators'].items():
            mark = "  <- measured" if demod == self.measure_demod else ""
            lines.append(f"  Demod {demod}: oscillator {osc}{mark}")
        return "\n".join(lines)

    """Serialization"""

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

    def _serialize_state(self) -> Dict[str, Any]:
        state = super()._serialize_state()
        state['measure_demod'] = self.measure_demod
        state['average_duration'] = self.average_duration
        state['demod'] = [
            {
                'oscillator'    : d.get_oscillator(),
                'time_constant' : d.get_time_const(),
                'order'         : d.get_order(),
                'phase'         : d.get_phase(),
                'input'         : d.get_input(),
                'rate'          : d.get_rate(),
                'enabled'       : d.is_enabled(),
            }
            for d in self.demod
        ]
        state['oscillator'] = [
            {'frequency': osc.get_frequency()}
            for osc in self.oscillator
        ]
        state['input'] = [
            {
                'range': i.get_range(),
                'ac'   : i.get_ac(),
                'imp50': i.get_imp50(),
                'diff' : i.get_diff(),
            }
            for i in self.input
        ]
        # Raw fractions of range, so the restore is exact
        state['output'] = [
            {
                'range'     : out.get_range(),
                'offset'    : out._get_frac('offset'),
                'amplitudes': [out._get_frac(f'amplitudes/{m}')
                               for m in range(self.num_mixer_channels)],
                'enables'   : [out.is_channel_enabled(m)
                               for m in range(self.num_mixer_channels)],
                'enabled'   : out.is_enabled(),
            }
            for out in self.output
        ]
        return state

    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)

        self.measure_demod = state.get('measure_demod', self.measure_demod)
        self.average_duration = state.get('average_duration', self.average_duration)

        for s, osc in zip(state.get('oscillator', []), self.oscillator):
            osc.set_frequency(s['frequency'])

        for s, d in zip(state.get('demod', []), self.demod):
            d.set_oscillator(s['oscillator'])
            d.set_time_const(s['time_constant'])
            d.set_order(s['order'])
            d.set_phase(s['phase'])
            d.set_input(s['input'])
            d.set_rate(s['rate'])
            d.enable(s['enabled'])

        for s, i in zip(state.get('input', []), self.input):
            i.set_range(s['range'])
            i.set_ac(s['ac'])
            i.set_imp50(s['imp50'])
            i.set_diff(s['diff'])

        # Output last, and switched on last, so it never plays a partial state
        for s, out in zip(state.get('output', []), self.output):
            out.enable(False)
            self.set_double(out._p('range'), s['range'])
            out._set_frac('offset', s['offset'])
            for m, (a, on) in enumerate(zip(s['amplitudes'], s['enables'])):
                out._set_frac(f'amplitudes/{m}', a)
                out.enable_channel(m, on)
            out.enable(s['enabled'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'hf2li':
        """
        Construct from serialized config.

        Parameters
        ----------
        config : dict
            Output from _serialize_state(), plus 'registry_id'.
        """

        # Extract required fields
        device_id = config.pop('device_id')
        server_host = config.pop('server_host')
        server_port = config.pop('server_port')
        api_level = config.pop('api_level')
        registry_id = config.pop('registry_id')

        # Construct instance
        instance = cls(
            device_id=device_id,
            server_host=server_host,
            server_port=server_port,
            api_level=api_level,
            registry_id=registry_id,
            skip_connect=False,
            **config,
        )
        instance._deserialize_state(config)

        return instance

class hf2liDemodChannel:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"demods/{self.index}/{subpath}"

    """Readings"""

    def get_xy(self) -> Tuple[float, float]:
        """
        Latest in- and out-of-phase reading (V RMS).

        Returns
        -------
        X, Y : float
        """
        data = self.parent.get_sample(self._p("sample"))
        return float(np.ravel(data['x'])[0]), float(np.ravel(data['y'])[0])

    def get_r(self) -> float:
        """
        Latest magnitude (V RMS).

        Returns
        -------
        float
        """
        x, y = self.get_xy()
        return float(np.hypot(x, y))

    def get_theta(self) -> float:
        """
        Latest phase (degrees).

        Returns
        -------
        float
        """
        x, y = self.get_xy()
        return float(np.degrees(np.arctan2(y, x)))

    def acquire_samples(self, duration=0.1) -> np.ndarray:
        """
        Record X + iY for `duration` seconds.

        Parameters
        ----------
        duration : float, default=0.1

        Returns
        -------
        ndarray
            complex samples (V RMS), about rate * duration of them.
        """
        if self.streaming:
            return self.stream_samples(duration)
        data = self.parent.poll_node(self._p("sample"), duration=duration)
        if data is None or len(data['x']) == 0:
            if not self.is_enabled():
                raise RuntimeError(
                    f"HF2LI demod {self.index} is not enabled; call "
                    f"demod[{self.index}].enable() first."
                )
            raise RuntimeError(
                f"HF2LI demod {self.index} returned no samples in {duration} s."
            )
        return np.asarray(data['x']) + 1j * np.asarray(data['y'])

    @staticmethod
    def _stats(samples: np.ndarray, tau: float, rate: float,
               series_corr: bool, L: int | None
               ) -> Tuple[np.ndarray, np.ndarray]:
        # Per-sample covariance, scaled up by tau/dt when samples are closer
        # than a time constant (the SRS driver's convention), or the
        # series-correlated estimate.
        xy = np.column_stack([samples.real, samples.imag])
        if series_corr:
            if L is None:
                L = int(20 * tau * rate)
            return xy.mean(axis=0), series_correlated_covariance(xy, L=L)
        cov = np.cov(xy.T)
        if 1.0 / rate < tau:
            cov *= tau * rate
        return xy.mean(axis=0), cov

    def get_average(self, duration=0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean of X, Y over `duration` (fresh subscription; ~120 ms overhead
        per call on the HF2, see stream_average), with the per-sample
        covariance scaled up by tau/dt when samples are closer than a time
        constant (the same convention as the SRS driver).

        Parameters
        ----------
        duration : float, default=0.1

        Returns
        -------
        mean, cov : ndarray
        """
        samples = self.acquire_samples(duration)
        return self._stats(samples, self.get_time_const(), self.get_rate(),
                           False, None)

    def get_average_series_correlated(self, duration=0.1, L: int | None = None
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean of X, Y over `duration` (fresh subscription), with a covariance
        that accounts for serial correlations.

        Parameters
        ----------
        duration : float, default=0.1
        L : int, optional
            Maximum correlation length in samples. Defaults to 20 time
            constants.

        Returns
        -------
        mean, cov : ndarray
        """
        samples = self.acquire_samples(duration)
        return self._stats(samples, self.get_time_const(), self.get_rate(),
                           True, L)

    """Streaming (persistent subscription)"""

    @property
    def streaming(self) -> bool:
        return self.parent.is_streaming(self._p("sample"))

    def start_stream(self):
        """
        Subscribe persistently, so repeated reads skip the per-call sync and
        subscription (~20-30 ms overhead per read instead of ~120 ms). The time
        constant and rate are cached here; the driver's own setters keep the
        cache current, but call refresh_stream() after changing them from
        elsewhere (e.g. the LabOne UI).
        """
        if not self.is_enabled():
            raise RuntimeError(
                f"HF2LI demod {self.index} is not enabled; call "
                f"demod[{self.index}].enable() first."
            )
        self.refresh_stream()
        self.parent.start_stream(self._p("sample"))

    def refresh_stream(self):
        """Re-read the time constant and rate used by the stream reads."""
        self._stream_tau = self.get_time_const()
        self._stream_rate = self.get_rate()

    def stop_stream(self):
        self.parent.stop_stream(self._p("sample"))

    def stream_samples(self, duration=0.1) -> np.ndarray:
        """
        X + iY for `duration` seconds, recorded after the call starts
        (starts the stream if needed).

        Parameters
        ----------
        duration : float, default=0.1

        Returns
        -------
        ndarray
            complex samples (V RMS), round(rate * duration) of them.
        """
        if not self.streaming:
            self.start_stream()
        n = max(2, int(round(duration * self._stream_rate)))
        data = self.parent.read_stream(self._p("sample"), n, self._stream_rate)
        return np.asarray(data['x']) + 1j * np.asarray(data['y'])

    def stream_average(self, duration=0.1, series_corr: bool = False,
                       L: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like get_average / get_average_series_correlated, from the stream.

        Parameters
        ----------
        duration : float, default=0.1
        series_corr : bool, default=False
        L : int, optional

        Returns
        -------
        mean, cov : ndarray
        """
        samples = self.stream_samples(duration)
        return self._stats(samples, self._stream_tau, self._stream_rate,
                           series_corr, L)

    """Settings"""

    def enable(self, on: bool = True):
        """
        Turn the sample stream on or off (needed for any reading).

        Parameters
        ----------
        on : bool, default=True
        """
        if not on and self.streaming:
            self.stop_stream()
        self.parent.set_int(self._p("enable"), int(on))
        self.parent.info(f"{'En' if on else 'Dis'}abled demod {self.index}.")

    def is_enabled(self) -> bool:
        """
        Returns
        -------
        bool
        """
        return bool(self.parent.get_int(self._p("enable")))

    def set_rate(self, rate: float):
        """
        Set the sample rate sent to the computer (rounded by the device).

        Parameters
        ----------
        rate : float
            Samples per second.
        """
        self.parent.set_double(self._p("rate"), rate)
        self.parent.info(f"Set demod {self.index} rate to {rate} Sa/s.")
        if self.streaming:
            self.refresh_stream()

    def get_rate(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("rate"))

    def get_oscillator(self) -> int:
        """
        Query which is the associated oscillator.

        Returns
        -------
        int
        """
        return self.parent.get_int(self._p("oscselect"))

    def set_oscillator(self, osc_index: int):
        """
        Set the associated oscillator.

        Parameters
        ----------
        osc_index : int
        """
        if not 0 <= osc_index < self.parent.num_oscillators:
            raise ValueError(f"No oscillator {osc_index}.")
        self.parent.set_int(self._p("oscselect"), osc_index)
        self.parent.info(f"Set demod {self.index} to oscillator {osc_index}.")

    def get_phase(self) -> float:
        """
        Phase shift in degrees. This rotates both demod `index`'s reference
        and mixer channel `index`'s output.

        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("phaseshift"))

    def set_phase(self, degrees: float):
        """
        Set the phase shift. This rotates both demod `index`'s reference and
        mixer channel `index`'s output.

        Parameters
        ----------
        degrees : float
        """
        self.parent.set_double(self._p("phaseshift"), degrees)
        self.parent.info(f"Set demod {self.index} phase shift to {degrees} deg.")

    def get_input(self) -> int:
        """
        Input source (adcselect): 0/1 Signal In 1/2, 2/3 Aux In 1/2,
        4/5 DIO 0/1.

        Returns
        -------
        int
        """
        return self.parent.get_int(self._p("adcselect"))

    def set_input(self, source: int):
        """
        Parameters
        ----------
        source : int
            See get_input.
        """
        self.parent.set_int(self._p("adcselect"), source)
        self.parent.info(f"Set demod {self.index} input to {source}.")

    def get_order(self) -> int:
        """
        Low-pass filter order (1-8).

        Returns
        -------
        int
        """
        return self.parent.get_int(self._p("order"))

    def set_order(self, order: int):
        """
        Parameters
        ----------
        order : int
            1-8 (6 to 48 dB/oct).
        """
        self.parent.set_int(self._p("order"), order)
        self.parent.info(f"Set demod {self.index} filter order to {order}.")

    def get_time_const(self) -> float:
        """
        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("timeconstant"))

    def set_time_const(self, tau: float):
        """
        Set the time constant (rounded by the device to an allowed value).

        Parameters
        ----------
        tau : float
        """
        self.parent.set_double(self._p("timeconstant"), tau)
        self.parent.info(f"Set demod {self.index} time constant to {tau} s.")
        if self.streaming:
            self.refresh_stream()

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
        Set the frequency of the associated oscillator (this changes every
        demod and mixer channel on that oscillator).

        Parameters
        ----------
        freq : float
        """
        self.parent.oscillator[self.get_oscillator()].set_frequency(freq)

class hf2liOutputChannel:
    """
    One signal output. Its signal is the sum of 8 mixer channels plus an
    offset; amplitudes and offset are in volts here and stored on the device
    as fractions of the range.
    """

    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"sigouts/{self.index}/{subpath}"

    def _get_frac(self, subpath: str) -> float:
        return self.parent.get_double(self._p(subpath))

    def _set_frac(self, subpath: str, frac: float):
        self.parent.set_double(self._p(subpath), frac)

    def _to_frac(self, volts: float, what: str) -> float:
        rng = self.get_range()
        if abs(volts) > rng * (1 + 1e-9):
            raise ValueError(
                f"Output {self.index} {what} {volts} V exceeds the {rng} V range."
            )
        return volts / rng

    def enable(self, on: bool = True):
        """
        Enable or disable the output.

        Parameters
        ----------
        on : bool
        """
        self.parent.set_int(self._p("on"), int(on))
        self.parent.info(
            f"{'En' if on else 'Dis'}abled signal output {self.index}."
        )

    def is_enabled(self) -> bool:
        """
        Returns
        -------
        bool
        """
        return bool(self.parent.get_int(self._p("on")))

    def get_range(self) -> float:
        """
        Get the output range (V).

        Returns
        -------
        float
        """
        return self.parent.get_double(self._p("range"))

    def set_range(self, value: float):
        """
        Set the output range, keeping every channel amplitude and the offset
        at the same voltage. The order of writes makes sure the output never
        exceeds its old or new value while switching.

        Parameters
        ----------
        value : float
            One of 0.01, 0.1, 1, 10 V.
        """
        if not any(np.isclose(value, r) for r in self.parent.OUTPUT_RANGES):
            raise ValueError(
                f"Output range must be one of {self.parent.OUTPUT_RANGES}."
            )
        old = self.get_range()
        if np.isclose(old, value):
            return

        n = self.parent.num_mixer_channels
        fracs = {f'amplitudes/{m}': self._get_frac(f'amplitudes/{m}') for m in range(n)}
        fracs['offset'] = self._get_frac('offset')
        too_big = [k for k, f in fracs.items() if abs(f * old) > value]
        if too_big:
            raise ValueError(
                f"Output {self.index}: {too_big} would exceed the {value} V range."
            )

        def rescale():
            for k, f in fracs.items():
                if f != 0.0:
                    self._set_frac(k, f * old / value)

        if value > old:      # shrink fractions first, then widen the range
            rescale()
            self.parent.set_double(self._p("range"), value)
        else:                # narrow the range first, then grow fractions
            self.parent.set_double(self._p("range"), value)
            rescale()
        self.parent.info(f"Set output {self.index} range to {value} V.")

    def set_amplitude(self, volts: float, channel: int):
        """
        Set a mixer channel's peak amplitude.

        Parameters
        ----------
        volts : float
            Peak amplitude in V, within the output range.
        channel : int
            Mixer channel 0-7.
        """
        self.parent._check_channel(channel)
        self._set_frac(f"amplitudes/{channel}", self._to_frac(volts, 'amplitude'))
        self.parent.info(
            f"Set output {self.index} channel {channel} amplitude to {volts} Vpk."
        )

    def get_amplitude(self, channel: int) -> float:
        """
        Parameters
        ----------
        channel : int

        Returns
        -------
        float
            Peak amplitude in V.
        """
        self.parent._check_channel(channel)
        return self._get_frac(f"amplitudes/{channel}") * self.get_range()

    def set_offset(self, volts: float):
        """
        Set the DC offset.

        Parameters
        ----------
        volts : float
        """
        self._set_frac("offset", self._to_frac(volts, 'offset'))
        self.parent.info(f"Set output {self.index} DC offset to {volts} V.")

    def get_offset(self) -> float:
        """
        Get the DC offset (V).

        Returns
        -------
        float
        """
        return self._get_frac("offset") * self.get_range()

    def enable_channel(self, channel: int, on: bool = True):
        """
        Add (or remove) a mixer channel to this output.

        Parameters
        ----------
        channel : int
        on : bool, default=True
        """
        self.parent._check_channel(channel)
        self.parent.set_int(self._p(f"enables/{channel}"), int(on))
        self.parent.info(
            f"{'En' if on else 'Dis'}abled output {self.index} channel {channel}."
        )

    def is_channel_enabled(self, channel: int) -> bool:
        """
        Parameters
        ----------
        channel : int

        Returns
        -------
        bool
        """
        self.parent._check_channel(channel)
        return bool(self.parent.get_int(self._p(f"enables/{channel}")))

    def get_enabled_channels(self) -> List[int]:
        """
        Returns
        -------
        list of int
        """
        return [m for m in range(self.parent.num_mixer_channels)
                if self.is_channel_enabled(m)]

class hf2liInputChannel:
    def __init__(self, parent: hf2li, index: int):
        self.parent = parent
        self.index = index

    def _p(self, subpath: str) -> str:
        return f"sigins/{self.index}/{subpath}"

    def get_range(self) -> float:
        """
        Returns
        -------
        float
            Input range in V.
        """
        return self.parent.get_double(self._p("range"))

    def set_range(self, value: float):
        """
        Parameters
        ----------
        value : float
            Input range in V (rounded by the device).
        """
        self.parent.set_double(self._p("range"), value)
        self.parent.info(f"Set input {self.index} range to {value} V.")

    def get_ac(self) -> bool:
        """
        Returns
        -------
        bool
            True if AC coupled.
        """
        return bool(self.parent.get_int(self._p("ac")))

    def set_ac(self, on: bool):
        """
        Parameters
        ----------
        on : bool
            AC (True) or DC (False) coupling.
        """
        self.parent.set_int(self._p("ac"), int(on))

    def get_imp50(self) -> bool:
        """
        Returns
        -------
        bool
            True for 50 Ohm, False for 1 MOhm.
        """
        return bool(self.parent.get_int(self._p("imp50")))

    def set_imp50(self, on: bool):
        """
        Parameters
        ----------
        on : bool
            50 Ohm (True) or 1 MOhm (False) input impedance.
        """
        self.parent.set_int(self._p("imp50"), int(on))

    def get_diff(self) -> bool:
        """
        Returns
        -------
        bool
            True if differential.
        """
        return bool(self.parent.get_int(self._p("diff")))

    def set_diff(self, on: bool):
        """
        Parameters
        ----------
        on : bool
            Differential (True) or single-ended (False).
        """
        self.parent.set_int(self._p("diff"), int(on))

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
    """
    One AC source: mixer channel `channel` on signal output `output_index`.

    Amplitude is peak volts. Phase is only available for channels 0-5 and is
    demod `channel`'s phase shift, so it also rotates that demod's reference;
    measure on a different demod.
    """

    def __init__(self, parent: hf2li, output_index: int, channel: int):
        if not 0 <= output_index < parent.num_outputs:
            raise ValueError(f"HF2LI has no output {output_index}.")
        parent._check_channel(channel)
        self.parent = parent
        self.output = parent.output[output_index]
        self.output_index = output_index
        self.channel = channel

    @property
    def oscillator(self) -> hf2liOscillator:
        return self.parent.oscillator[self.parent.channel_oscillator(self.channel)]

    def enable(self, state: bool = True):
        """
        Add this channel to the output (or remove it). Does not switch the
        output itself on or off.

        Parameters
        ----------
        state : bool, default=True
        """
        self.output.enable_channel(self.channel, state)

    def is_enabled(self) -> bool:
        """
        Returns
        -------
        bool
            True if the channel is enabled and the output is on.
        """
        return self.output.is_channel_enabled(self.channel) \
            and self.output.is_enabled()

    def set_frequency(self, freq: float):
        """
        Set the oscillator frequency in Hz (shared with everything on that
        oscillator).

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
        Set the peak amplitude in V.

        Parameters
        ----------
        value : float
        """
        self.output.set_amplitude(value, self.channel)

    def get_amplitude(self) -> float:
        """
        Returns
        -------
        float
            Peak amplitude in V.
        """
        return self.output.get_amplitude(self.channel)

    def _phase_demod(self) -> hf2liDemodChannel:
        if self.channel in self.parent.PLL_CHANNELS:
            raise RuntimeError(
                f"Mixer channel {self.channel} belongs to a PLL and has no "
                f"phase control; use a channel 0-5."
            )
        return self.parent.demod[self.channel]

    def set_phase(self, value: float):
        """
        Set the phase in degrees (demod `channel`'s phase shift).

        Parameters
        ----------
        value : float
        """
        if self.channel == self.parent.measure_demod:
            raise RuntimeError(
                f"Demod {self.channel} is the measurement demod; its phase "
                f"shift would also rotate the measurement. Drive on another "
                f"channel or measure on another demod (setup_average)."
            )
        self._phase_demod().set_phase(value)

    def get_phase(self) -> float:
        """
        Returns
        -------
        float
        """
        return self._phase_demod().get_phase()

    def summary(self) -> str:
        """
        Get a string summarizing the AC output settings.

        Returns
        -------
        str
        """
        phase = "n/a (PLL channel)" if self.channel in self.parent.PLL_CHANNELS \
            else f"{self.get_phase()} deg"
        return (
            f"AC Output {self.output_index}, mixer channel {self.channel} "
            f"(oscillator {self.parent.channel_oscillator(self.channel)}):\n"
            f"  Output on: {self.output.is_enabled()}  "
            f"Channel enabled: {self.output.is_channel_enabled(self.channel)}\n"
            f"  Frequency: {self.get_frequency()} Hz\n"
            f"  Amplitude: {self.get_amplitude()} Vpk "
            f"(range {self.output.get_range()} V)\n"
            f"  Phase: {phase}"
        )

"""Channel adapters (for resolve)"""

class hf2li_lockin_channel:
    """
    LockInChannel on the measurement demodulator: () -> (mean, cov), with
    mean scaled by `scale` and cov by scale**2.
    """
    def __init__(self, parent: hf2li, accessor: str, scale: float,
                 series_corr: bool, stream: bool):
        self._parent = parent
        self._accessor = accessor
        self.scale = scale
        self._series_corr = series_corr
        self._stream = stream

    def format_ref(self):
        return self._parent, self._accessor

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self._parent
        if self._stream:
            fn = p.stream_average_series_correlated if self._series_corr \
                else p.stream_average
        else:
            fn = p.get_average_series_correlated if self._series_corr \
                else p.get_average
        mean, cov = fn()
        s = self.scale
        return mean * s, cov * (s * s)

class hf2li_amplitude_channel(ScalarChannelAdapter):
    def __init__(self, parent: hf2li, accessor: str, ac: hf2liACout):
        super().__init__(parent, accessor)
        self.ac = ac

    def get_output(self) -> float:
        return self.ac.get_amplitude()

    def set_output(self, value: float):
        self.ac.set_amplitude(value)

class hf2li_phase_channel(ScalarChannelAdapter):
    def __init__(self, parent: hf2li, accessor: str, ac: hf2liACout):
        super().__init__(parent, accessor)
        self.ac = ac

    def get_output(self) -> float:
        return self.ac.get_phase()

    def set_output(self, value: float):
        self.ac.set_phase(value)

class hf2li_frequency_channel(ScalarChannelAdapter):
    def __init__(self, parent: hf2li, accessor: str, osc: int):
        super().__init__(parent, accessor)
        self.osc = osc

    def get_output(self) -> float:
        return self._parent.oscillator[self.osc].get_frequency()

    def set_output(self, value: float):
        self._parent.oscillator[self.osc].set_frequency(value)
