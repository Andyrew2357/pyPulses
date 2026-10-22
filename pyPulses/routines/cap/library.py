"""
pyPulses.routines.cap.library
=============================
Precomputed per-excitation calibrations for off-balance measurements.

CapCrosstalkTransition + CapRebalanceTransition recalibrate live whenever the
excitation (frequency, amplitude, ...) changes. That forces a trip to the
reference point (and an amp off/on cycle) at every excitation change, so the
excitation has to be the outer scan coordinate.

Here every excitation state is calibrated once, up front, at a single parked
point, and stored in a CapCalibrationLibrary. CapCalibrationTransition then
only has to change the excitation and load the stored state, which is cheap
enough to put the excitation on an inner loop:

    location (metal ref, gap ref, cut) x excitation x points along the cut

A stored calibration is everything a live rebalance leaves behind: the ctx
attrs set from `ctx_attrs` (Vex, Vstd_range, ...), the crosstalk model
(ctx.crosstalk) and the sample-path three-point balance (seeded into
ctx.cap_filter, which cap_measure reads K_matrix from), with Vstd parked at
its V0. Loading an entry leaves ctx in the same state a live rebalance at the
build point would have.

Usage
-----
    # Build, parked at the reference point (gates are never moved)
    lib = CapCalibrationLibrary(trigger=['freq', 'Vex_amp'])
    cap_build_library(
        ctx, lib, states=[{'freq': 1e6, 'Vex_amp': 0.01}, ...],
        excitation_channels = [freq_ch, vex_ch],
        amp_channels = [amp_gate_ch],
        amp_on = {'amp_VG': -0.2}, amp_off = {'amp_VG': -5.0},
        ctx_attrs = {'Vex': 'Vex_amp',
                     'Vstd_range': ('Vex_amp', lambda v: 2.0 * v)},
        ref = {'Vtg': 0.0, 'Vbg': 2.0},
        path = 'cap_library.json',          # saved after every state
    )

    # Later (any session)
    lib = CapCalibrationLibrary.load_json('cap_library.json')
    load = CapCalibrationTransition(ctx, lib, settle=0.2)
    runner = Runner(location_scan * exc_scan * cut_scan, meas,
                    transitions=[load], ...)
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Sequence

import numpy as np

from .context import _NumpyEncoder
from .crosstalk import CrosstalkCompensatedLockIn, _fit_crosstalk
from .initialize import (
    ThreePointBalanceResult,
    cap_initialize_filter,
    three_point_to_dict,
    three_point_from_dict,
)
from .transition import (
    Coords, AttrSpec, _compile_attr_spec, _apply_ctx_attrs, _same_state,
    _balance_three_point,
)
from ...core.job import checkpoint
from ...core.tandem_sweep import tandemSweep, SweepResult
from ...devices.sweepable_channel import SweepableChannel

if TYPE_CHECKING:
    from .context import CapContext
    from ...core.scan import ScanBase


def _now() -> str:
    return datetime.datetime.now().isoformat()


@dataclass
class CapCalibration:
    """
    Stored calibration for one excitation state.

    Attributes
    ----------
    state : dict
        Trigger coordinate -> value.
    ctx : dict
        CapContext attribute -> value, set before the balance was taken
        (Vex, Vstd_range, ...).
    crosstalk : ThreePointBalanceResult or None
        Crosstalk model fitted with the amp off. None if crosstalk
        compensation is not used.
    balance : ThreePointBalanceResult or None
        Sample-path three-point balance. None while the build is in progress.
    meta : dict
        Timestamps, build point and anything else worth keeping.
    """
    state: Coords
    ctx: Dict[str, float] = field(default_factory=dict)
    crosstalk: ThreePointBalanceResult | None = None
    balance: ThreePointBalanceResult | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def check(self, ctx: 'CapContext') -> None:
        """Raise if this entry can't be applied to `ctx`. Touches nothing."""
        if self.balance is None:
            raise ValueError(f"Calibration for {self.state} has no balance yet.")
        if self.crosstalk is not None and \
                not isinstance(ctx.lockin_call, CrosstalkCompensatedLockIn):
            raise TypeError(
                f"Calibration for {self.state} includes a crosstalk model, but "
                "ctx.lockin_call is not a CrosstalkCompensatedLockIn."
            )
        bad = [a for a in self.ctx if not hasattr(ctx, a)]
        if bad:
            raise ValueError(f"CapContext has no attribute(s) {bad}.")

    def apply(self, ctx: 'CapContext', park: bool = True) -> None:
        """
        Load this calibration into `ctx`: attrs, crosstalk, filter (K_matrix,
        A, P) and extrapolator. If `park`, set Vstd/Theta to the balance V0.
        Does not touch the excitation channels.
        """
        self.check(ctx)
        for attr, val in self.ctx.items():
            setattr(ctx, attr, val)
        ctx.crosstalk = self.crosstalk
        cap_initialize_filter(ctx, self.balance)
        if park:
            ctx.set_Vstd_complex(self.balance.V0)

    def to_dict(self) -> dict:
        return dict(
            state = dict(self.state),
            ctx = dict(self.ctx),
            crosstalk = None if self.crosstalk is None
                else three_point_to_dict(self.crosstalk),
            balance = None if self.balance is None
                else three_point_to_dict(self.balance),
            meta = dict(self.meta),
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'CapCalibration':
        return cls(
            state = {k: float(v) for k, v in d['state'].items()},
            ctx = {k: float(v) for k, v in d.get('ctx', {}).items()},
            crosstalk = None if d.get('crosstalk') is None
                else three_point_from_dict(d['crosstalk']),
            balance = None if d.get('balance') is None
                else three_point_from_dict(d['balance']),
            meta = dict(d.get('meta', {})),
        )


class CapCalibrationLibrary:
    """
    A set of CapCalibrations keyed by the values of the `trigger` coordinates.

    Lookups accept any mapping containing the trigger coordinates (extra keys,
    e.g. gate coordinates of a scan target, are ignored) and match values to
    within `rtol`.

    Parameters
    ----------
    trigger : sequence of str
        Coordinates that define an excitation state (scan coordinate names).
    rtol : float
        Relative tolerance for matching a state.
    """

    FORMAT = 'pyPulses.cap.CapCalibrationLibrary/1'

    def __init__(self, trigger: Sequence[str], rtol: float = 1e-9):
        self.trigger = list(trigger)
        if not self.trigger:
            raise ValueError("trigger must name at least one coordinate.")
        self.rtol = rtol
        self.entries: List[CapCalibration] = []

    """Lookup"""

    def key(self, state: Dict[str, float]) -> Coords:
        """Restrict `state` to the trigger coordinates, as floats."""
        missing = [n for n in self.trigger if n not in state]
        if missing:
            raise KeyError(f"State {dict(state)} is missing trigger coords {missing}.")
        return {n: float(state[n]) for n in self.trigger}

    def get(self, state: Dict[str, float]) -> CapCalibration | None:
        k = self.key(state)
        for e in self.entries:
            if _same_state(k, e.state, self.rtol):
                return e
        return None

    def __getitem__(self, state: Dict[str, float]) -> CapCalibration:
        e = self.get(state)
        if e is None:
            raise KeyError(
                f"No calibration for {self.key(state)}. "
                f"Library has: {self.states()}"
            )
        return e

    def __contains__(self, state: Dict[str, float]) -> bool:
        return self.get(state) is not None

    def setdefault(self, state: Dict[str, float]) -> CapCalibration:
        """Return the entry for `state`, creating an empty one if needed."""
        e = self.get(state)
        if e is None:
            e = CapCalibration(state=self.key(state))
            self.entries.append(e)
        return e

    def states(self) -> List[Coords]:
        return [dict(e.state) for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CapCalibration]:
        return iter(self.entries)

    """Serialization"""

    def to_dict(self) -> dict:
        return dict(
            format = self.FORMAT,
            trigger = list(self.trigger),
            rtol = float(self.rtol),
            entries = [e.to_dict() for e in self.entries],
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'CapCalibrationLibrary':
        if d.get('format') != cls.FORMAT:
            raise ValueError(
                f"Unrecognized library format {d.get('format')!r} "
                f"(expected {cls.FORMAT!r})."
            )
        lib = cls(d['trigger'], rtol=d.get('rtol', 1e-9))
        lib.entries = [CapCalibration.from_dict(e) for e in d['entries']]
        return lib

    def save_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=_NumpyEncoder)

    @classmethod
    def load_json(cls, path: str) -> 'CapCalibrationLibrary':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


"""Builders"""

def _sweep(channels: Sequence[SweepableChannel], target: Coords,
           min_wait: float | None) -> None:
    res = tandemSweep(list(channels), target, min_wait=min_wait)
    if res is not SweepResult.SUCCEEDED:
        raise RuntimeError(f"Sweep to {target} did not succeed ({res.name}).")


def _check_excitation_channels(
    library: CapCalibrationLibrary,
    channels: Sequence[SweepableChannel],
) -> None:
    names = [ch.name for ch in channels]
    if sorted(names) != sorted(library.trigger):
        raise ValueError(
            f"excitation_channels {names} must be exactly the library "
            f"trigger coordinates {library.trigger}."
        )


def _compile_ctx_attrs(ctx, trigger, ctx_attrs):
    fns = {}
    for attr, spec in (ctx_attrs or {}).items():
        if not hasattr(ctx, attr):
            raise ValueError(f"CapContext has no attribute {attr!r}.")
        fns[attr] = _compile_attr_spec(trigger, attr, spec)
    return fns


def cap_calibrate_crosstalk(
    ctx: 'CapContext',
    library: CapCalibrationLibrary,
    states: Sequence[Dict[str, float]],
    excitation_channels: Sequence[SweepableChannel],
    amp_channels: Sequence[SweepableChannel],
    amp_on: Coords,
    amp_off: Coords,
    ctx_attrs: Dict[str, AttrSpec] | None = None,
    settle: float = 0.0,
    settle_off: float = 0.0,
    settle_on: float = 0.0,
    three_point_kwargs: Dict[str, Any] | None = None,
    min_wait: float | None = 0.0,
    path: str | None = None,
    overwrite: bool = False,
) -> CapCalibrationLibrary:
    """
    Fit the crosstalk model for every state with a single amp off/on cycle.

    For each state not already holding crosstalk (all states if `overwrite`):
    set the excitation, apply `ctx_attrs`, wait `settle`, fit as
    CapCrosstalkTransition does. The amp is turned back on even if this
    raises. Re-measuring crosstalk on an entry clears its balance (it was
    taken against the old model). Gates are never moved.

    Parameters
    ----------
    ctx : CapContext
        ctx.lockin_call must be a CrosstalkCompensatedLockIn.
    library : CapCalibrationLibrary
        Updated in place (and returned).
    states : sequence of dict
        Excitation states, each containing the library trigger coordinates.
    excitation_channels : sequence of SweepableChannel
        One channel per trigger coordinate, named like the coordinate.
    amp_channels, amp_on, amp_off
        As in CapCrosstalkTransition.
    ctx_attrs : dict, optional
        Same spec mini-language as CapRebalanceTransition.
    settle, settle_off, settle_on : float
        Waits after each excitation change, after turning the amp off, and
        after turning it back on.
    three_point_kwargs : dict, optional
        Forwarded to cap_balance_three_point. Defaults as in
        CapCrosstalkTransition (wait=ctx.settle_time, samples=5, plot=False,
        move_to_balance=False).
    min_wait : float or None, default 0.0
        Floor on the per-step wait passed to tandemSweep. 0.0 lets channels
        without SweepConfig timing (e.g. a frequency) be set directly.
    path : str, optional
        If given, the library is saved here after every state.
    overwrite : bool, default False
        Re-measure states that already have crosstalk.
    """
    if not isinstance(ctx.lockin_call, CrosstalkCompensatedLockIn):
        raise TypeError(
            "cap_calibrate_crosstalk requires ctx.lockin_call to be a "
            "CrosstalkCompensatedLockIn."
        )
    _check_excitation_channels(library, excitation_channels)
    attr_fns = _compile_ctx_attrs(ctx, library.trigger, ctx_attrs)
    kw = {'samples': 5, 'plot': False, 'move_to_balance': False,
          **(three_point_kwargs or {})}

    keys = [library.key(s) for s in states]
    todo = [k for k in keys if overwrite or
            (e := library.get(k)) is None or e.crosstalk is None]
    if not todo:
        return library

    raw = ctx.lockin_call.raw
    _sweep(amp_channels, amp_off, min_wait)
    try:
        if settle_off > 0:
            time.sleep(settle_off)
        for state in todo:
            checkpoint()
            _sweep(excitation_channels, state, min_wait)
            ctx_set = _apply_ctx_attrs(ctx, attr_fns, state)
            if settle > 0:
                time.sleep(settle)

            result = _fit_crosstalk(ctx, raw, kw)
            ctx.crosstalk = result

            entry = library.setdefault(state)
            if entry.balance is not None:
                ctx.log(f"Crosstalk remeasured for {state}; clearing its balance.")
            entry.crosstalk = result
            entry.balance = None
            entry.meta['crosstalk_time'] = _now()
            entry.meta['crosstalk_ctx'] = dict(ctx_set)
            ctx.log(f"Crosstalk calibrated for {state}: V0={result.V0:.5e}")
            if path is not None:
                library.save_json(path)
    finally:
        _sweep(amp_channels, amp_on, min_wait)
        if settle_on > 0:
            time.sleep(settle_on)

    return library


def cap_calibrate_balance(
    ctx: 'CapContext',
    library: CapCalibrationLibrary,
    states: Sequence[Dict[str, float]],
    excitation_channels: Sequence[SweepableChannel],
    ctx_attrs: Dict[str, AttrSpec] | None = None,
    settle: float = 0.0,
    ref: Coords | None = None,
    three_point_kwargs: Dict[str, Any] | None = None,
    min_wait: float | None = 0.0,
    path: str | None = None,
    overwrite: bool = False,
) -> CapCalibrationLibrary:
    """
    Take the sample-path three-point balance for every state, at the current
    (parked) gate point.

    For each state not already holding a balance (all states if
    `overwrite`): set the excitation, apply `ctx_attrs`, load the entry's
    crosstalk model, wait `settle`, balance as CapRebalanceTransition does and
    seed the filter with it. Afterwards ctx is left loaded with the last
    state processed. Gates are never moved; `ref` is only recorded.

    If ctx.lockin_call is a CrosstalkCompensatedLockIn, every state must
    already have crosstalk (run cap_calibrate_crosstalk first); if it isn't,
    no state may have crosstalk. Both are checked before anything moves.

    Parameters are as in cap_calibrate_crosstalk, plus:

    ref : dict, optional
        Gate coordinates of the parked point, stored in each entry's meta.
    three_point_kwargs : dict, optional
        Defaults as in CapRebalanceTransition (wait=ctx.settle_time,
        samples=5, plot=False).
    """
    _check_excitation_channels(library, excitation_channels)
    attr_fns = _compile_ctx_attrs(ctx, library.trigger, ctx_attrs)
    kw = {'samples': 5, 'plot': False, **(three_point_kwargs or {})}

    keys = [library.key(s) for s in states]
    todo = [k for k in keys if overwrite or
            (e := library.get(k)) is None or e.balance is None]
    if not todo:
        return library

    compensated = isinstance(ctx.lockin_call, CrosstalkCompensatedLockIn)
    if compensated:
        missing = [k for k in todo
                   if (e := library.get(k)) is None or e.crosstalk is None]
        if missing:
            raise ValueError(
                f"No crosstalk model for {missing}; run "
                "cap_calibrate_crosstalk first (ctx.lockin_call is a "
                "CrosstalkCompensatedLockIn)."
            )
    else:
        have = [k for k in todo
                if (e := library.get(k)) is not None and e.crosstalk is not None]
        if have:
            raise TypeError(
                f"States {have} have crosstalk models, but ctx.lockin_call "
                "is not a CrosstalkCompensatedLockIn."
            )

    for state in todo:
        checkpoint()
        entry = library.setdefault(state)
        _sweep(excitation_channels, state, min_wait)
        ctx_set = _apply_ctx_attrs(ctx, attr_fns, state)
        ctx.crosstalk = entry.crosstalk
        if settle > 0:
            time.sleep(settle)

        bal = _balance_three_point(ctx, kw, state)
        cap_initialize_filter(ctx, bal)

        entry.ctx = dict(ctx_set)
        entry.balance = bal
        entry.meta['balance_time'] = _now()
        entry.meta['ref'] = None if ref is None else \
            {k: float(v) for k, v in ref.items()}
        ctx.log(f"Balance calibrated for {state}: V0={bal.V0:.5e}, "
                f"residual={bal.error}")
        if path is not None:
            library.save_json(path)

    return library


def cap_build_library(
    ctx: 'CapContext',
    library: CapCalibrationLibrary,
    states: Sequence[Dict[str, float]],
    excitation_channels: Sequence[SweepableChannel],
    amp_channels: Sequence[SweepableChannel] | None = None,
    amp_on: Coords | None = None,
    amp_off: Coords | None = None,
    ctx_attrs: Dict[str, AttrSpec] | None = None,
    settle: float = 0.0,
    settle_off: float = 0.0,
    settle_on: float = 0.0,
    ref: Coords | None = None,
    crosstalk_kwargs: Dict[str, Any] | None = None,
    balance_kwargs: Dict[str, Any] | None = None,
    min_wait: float | None = 0.0,
    path: str | None = None,
    overwrite: bool = False,
) -> CapCalibrationLibrary:
    """
    cap_calibrate_crosstalk for all states (skipped if `amp_channels` is
    None), then cap_calibrate_balance for all states. Resumable: re-running
    with the same arguments only does what is missing. `crosstalk_kwargs` /
    `balance_kwargs` are the respective `three_point_kwargs`.
    """
    if amp_channels is not None:
        if amp_on is None or amp_off is None:
            raise ValueError("amp_on and amp_off are required with amp_channels.")
        cap_calibrate_crosstalk(
            ctx, library, states, excitation_channels,
            amp_channels, amp_on, amp_off,
            ctx_attrs = ctx_attrs,
            settle = settle,
            settle_off = settle_off,
            settle_on = settle_on,
            three_point_kwargs = crosstalk_kwargs,
            min_wait = min_wait,
            path = path,
            overwrite = overwrite,
        )
    return cap_calibrate_balance(
        ctx, library, states, excitation_channels,
        ctx_attrs = ctx_attrs,
        settle = settle,
        ref = ref,
        three_point_kwargs = balance_kwargs,
        min_wait = min_wait,
        path = path,
        overwrite = overwrite,
    )


"""Transition"""

class CapCalibrationTransition:
    """
    A Runner Transition that loads a precomputed calibration whenever the
    excitation state (the library's trigger coordinates) changes.

    On a trigger-state change:
      1. Look up the entry; a missing or incomplete entry raises before
         anything moves.
      2. Change the excitation in place (scan.move_to with the gates where
         they are; the runner's own move_to(target) finishes the trip).
      3. Apply the entry to ctx (attrs, crosstalk, filter/extrapolator, Vstd
         parked at V0 if `park`), then wait `settle`.

    Parameters
    ----------
    ctx : CapContext
    library : CapCalibrationLibrary
    settle : float, default 0.0
        Wait after changing the excitation and loading (lock-in re-lock,
        filter settling, ...).
    park : bool, default True
        Set Vstd/Theta to the stored balance V0.
    db : DatabaseLogger, optional
        If given, the library is written to its global attrs
        ('cap_calibration_library') once, and the load history
        ('cap_calibration_load') after every load.

    Attributes
    ----------
    history : list of dict
        One entry per load: time, scan idx, state, and the entry's build
        timestamps.
    loaded : dict or None
        Trigger state currently loaded.
    """

    def __init__(self,
        ctx: 'CapContext',
        library: CapCalibrationLibrary,
        settle: float = 0.0,
        park: bool = True,
        db=None,
    ):
        self.ctx = ctx
        self.library = library
        self.settle = settle
        self.park = park
        self.db = db

        self.loaded: Coords | None = None
        self.history: List[dict] = []
        self._db_with_library = None

    @property
    def trigger(self) -> List[str]:
        return self.library.trigger

    """Transition interface"""

    def __call__(self,
        scan: 'ScanBase',
        idx: np.ndarray,
        target: Coords,
        min_wait: float | None,
    ) -> None:
        state = self.library.key(target)
        if self.loaded is not None and \
                _same_state(state, self.loaded, self.library.rtol):
            return

        # Invalid until the load succeeds; a raise stops the scan and a
        # resume re-triggers here.
        self.loaded = None

        missing = set(self.trigger) - set(scan.coord_names)
        if missing:
            raise ValueError(f"Trigger coords not in the scan: {sorted(missing)}")
        entry = self.library[state]
        entry.check(self.ctx)

        here = dict(scan._get_current_coords())
        scan.move_to({**here, **state}, min_wait=min_wait)

        entry.apply(self.ctx, park=self.park)
        if self.settle > 0:
            time.sleep(self.settle)

        self.loaded = state
        self._record(idx, state, entry)

    """Internals"""

    def _record(self, idx, state, entry: CapCalibration):
        self.history.append(dict(
            time = _now(),
            idx = [int(i) for i in idx],
            state = dict(state),
            crosstalk_time = entry.meta.get('crosstalk_time'),
            balance_time = entry.meta.get('balance_time'),
        ))
        self.ctx.log(f"Loaded calibration at idx {self.history[-1]['idx']} "
                     f"for {state}.")
        if self.db is not None:
            attrs = {'cap_calibration_load': self.history}
            if self._db_with_library is not self.db:
                attrs['cap_calibration_library'] = self.library.to_dict()
                self._db_with_library = self.db
            self.db.set_global_attrs(attrs)


if __name__ == '__main__':
    """
    Self-test against a synthetic bridge (no real hardware), following the
    FakeChannel convention in crosstalk.py's self-test.

    Model: raw_lockin() = Kx(f) @ Vstd + (Ks(f) @ (Vstd - V0s(Vg)) if amp on)
    with V0s(Vg) = -Vex * (c(Vg) + i*loss) / Cstd, so cap_measure should
    return Cex = c(Vg), Closs = loss. Both the crosstalk and the sample gain
    depend on the frequency, so a wrong or stale calibration shows up.
    """

    import os
    import tempfile

    from .context import CapContext
    from .crosstalk import CapCrosstalkTransition
    from .measure import cap_measure
    from .transition import CapRebalanceTransition

    def check(label: str, cond: bool):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        assert cond, label

    class FakeChannel:
        def __init__(self, value: float = 0.0):
            self._v = value
        def __call__(self, v: float | None = None) -> float | None:
            if v is None:
                return self._v
            self._v = v

    class FakeSweepable:
        """Minimal SweepableChannel-like stand-in for tandemSweep."""
        def __init__(self, name, value=0.0, on_set=None):
            self.name = name
            self._v = value
            self._on_set = on_set

            class _Cfg:
                max_step = None
                min_step = None
                tolerance = 0.0
                max_rate = None
                settle_time = 0.0
                async_set = False
            self.config = _Cfg()

        def get_output(self) -> float:
            return self._v

        def set_output(self, value: float) -> None:
            self._v = float(value)
            if self._on_set is not None:
                self._on_set(self.name, self._v)

    class FakeScan:
        def __init__(self, chans):
            self.chans = {c.name: c for c in chans}
            self.coord_names = list(self.chans)
            self.moves = []
        def _get_current_coords(self):
            return {n: c.get_output() for n, c in self.chans.items()}
        def move_to(self, coords, min_wait=None):
            self.moves.append(dict(coords))
            for n, v in coords.items():
                self.chans[n].set_output(v)

    def rot(deg, g):
        t = np.deg2rad(deg)
        return g * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def Kx(f):       # deliberately not complex-linear
        return np.array([[0.3 * f, 0.1], [0.05, 0.02 * f]])
    def Ks(f):
        return rot(20.0 * f, 2.0 + 0.5 * f)
    def c_of(Vg):
        return 0.3 + 0.4 * Vg
    LOSS = 0.05

    amp_toggles = []
    def on_set(name, v):
        if name == 'amp_VG':
            amp_toggles.append(v > -1.0)

    freq_ch = FakeSweepable('freq', 1.0)
    vex_ch = FakeSweepable('Vex_amp', 0.1)
    vg_ch = FakeSweepable('Vg', 0.0)
    amp_ch = FakeSweepable('amp_VG', -0.2, on_set=on_set)
    Vstd_raw, Theta_raw = FakeChannel(0.0), FakeChannel(0.0)
    n_reads = {'n': 0}

    def raw_lockin_call():
        n_reads['n'] += 1
        f, vex, Vg = freq_ch.get_output(), vex_ch.get_output(), vg_ch.get_output()
        r, th = Vstd_raw(), np.deg2rad(Theta_raw())
        V = np.array([r * np.cos(th), r * np.sin(th)])
        L = Kx(f) @ V
        if amp_ch.get_output() > -1.0:
            V0s = -vex * np.array([c_of(Vg), LOSS])
            L = L + Ks(f) @ (V - V0s)
        return L, np.eye(2) * 1e-12

    def make_ctx():
        ctx = CapContext(
            Vstd=Vstd_raw, Theta=Theta_raw, lockin_call=raw_lockin_call,
            Vstd_range=0.2, Vex=0.1, Cstd=1.0, settle_time=0.0,
        )
        ctx.lockin_call = CrosstalkCompensatedLockIn(raw_lockin_call, ctx)
        return ctx

    trigger = ['freq', 'Vex_amp']
    ctx_attrs = {'Vex': 'Vex_amp',
                 'Vstd_range': ('Vex_amp', lambda v: 2.0 * v)}
    states = [{'freq': 1.0, 'Vex_amp': 0.1},
              {'freq': 2.0, 'Vex_amp': 0.1},
              {'freq': 3.0, 'Vex_amp': 0.2}]
    tp = {'wait': 0.0, 'samples': 1}
    REF, MEAS = 0.5, -0.25

    # --- Build, parked at the reference point ----------------------------
    vg_ch.set_output(REF)
    ctx_lib = make_ctx()
    lib = CapCalibrationLibrary(trigger)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'lib.json')
    cap_build_library(
        ctx_lib, lib, states, [freq_ch, vex_ch],
        amp_channels=[amp_ch], amp_on={'amp_VG': -0.2},
        amp_off={'amp_VG': -5.0}, ctx_attrs=ctx_attrs, ref={'Vg': REF},
        crosstalk_kwargs=tp, balance_kwargs=tp, path=path,
    )
    check('library has every state, complete',
          len(lib) == 3 and all(e.crosstalk and e.balance for e in lib))
    check('amp turned off and on exactly once', amp_toggles == [False, True])
    check('gate never moved during build', vg_ch.get_output() == REF)

    # --- Resume: nothing left to do --------------------------------------
    reads_before, toggles_before = n_reads['n'], len(amp_toggles)
    cap_build_library(
        ctx_lib, lib, states, [freq_ch, vex_ch],
        amp_channels=[amp_ch], amp_on={'amp_VG': -0.2},
        amp_off={'amp_VG': -5.0}, ctx_attrs=ctx_attrs,
        crosstalk_kwargs=tp, balance_kwargs=tp,
    )
    check('re-running the build does no hardware work',
          n_reads['n'] == reads_before and len(amp_toggles) == toggles_before)

    # --- JSON round trip --------------------------------------------------
    lib2 = CapCalibrationLibrary.load_json(path)
    check('JSON round trip preserves every entry',
          lib2.to_dict() == lib.to_dict())

    # --- Load transition vs. live transitions -----------------------------
    scan = FakeScan([freq_ch, vex_ch, vg_ch])
    ctx_live = make_ctx()
    xtalk = CapCrosstalkTransition(
        ctx_live, trigger, amp_channels=[amp_ch],
        amp_on={'amp_VG': -0.2}, amp_off={'amp_VG': -5.0},
        ctx_attrs=ctx_attrs, three_point_kwargs=tp, min_wait=0.0,
    )
    rebal = CapRebalanceTransition(
        ctx_live, trigger, ref_coords={'Vg': REF},
        ctx_attrs=ctx_attrs, three_point_kwargs=tp,
    )
    ctx_load = make_ctx()
    load = CapCalibrationTransition(ctx_load, lib2)

    for i, st in enumerate(states):
        target = {**st, 'Vg': MEAS}

        xtalk(scan, np.array([i, 0]), target, 0.0)
        rebal(scan, np.array([i, 0]), target, 0.0)
        scan.move_to(target)
        live = cap_measure(ctx_live)
        V_live = ctx_live.get_Vstd_complex()

        scan.move_to({**target, 'freq': 9.0, 'Vex_amp': 0.05})  # elsewhere
        load(scan, np.array([i, 0]), target, 0.0)
        check(f'{st}: excitation set by the load transition',
              freq_ch.get_output() == st['freq']
              and vex_ch.get_output() == st['Vex_amp'])
        scan.move_to(target)
        loaded = cap_measure(ctx_load)

        same = (
            np.isclose(ctx_load.Vex, ctx_live.Vex)
            and np.isclose(ctx_load.Vstd_range, ctx_live.Vstd_range)
            and np.allclose(ctx_load.crosstalk.A_matrix, ctx_live.crosstalk.A_matrix)
            and np.allclose(ctx_load.cap_filter.K_matrix, ctx_live.cap_filter.K_matrix)
            and np.allclose(ctx_load.cap_filter.get_A(), ctx_live.cap_filter.get_A())
            and np.isclose(ctx_load.get_Vstd_complex(), V_live)
        )
        check(f'{st}: loaded ctx matches live rebalance', same)
        check(f'{st}: loaded cap_measure matches live and truth',
              np.isclose(loaded.Cex, live.Cex, atol=1e-9)
              and np.isclose(loaded.Closs, live.Closs, atol=1e-9)
              and np.isclose(loaded.Cex, c_of(MEAS), atol=1e-6)
              and np.isclose(loaded.Closs, LOSS, atol=1e-6))

    # Same state again: no-op
    n_moves = len(scan.moves)
    load(scan, np.array([9, 0]), {**states[-1], 'Vg': 0.0}, 0.0)
    check('unchanged state does not reload', len(scan.moves) == n_moves)

    # Missing state: raises before moving
    try:
        load(scan, np.array([10, 0]), {'freq': 7.0, 'Vex_amp': 0.1, 'Vg': 0.0}, 0.0)
        raised = False
    except KeyError:
        raised = True
    check('missing state raises before any move',
          raised and len(scan.moves) == n_moves)

    # --- Balance refuses to run without crosstalk when compensated --------
    try:
        cap_calibrate_balance(make_ctx(), CapCalibrationLibrary(trigger),
                              states, [freq_ch, vex_ch], three_point_kwargs=tp)
        raised = False
    except ValueError:
        raised = True
    check('balance without crosstalk refused when compensated', raised)

    # --- Old Crosstalk_kwargs format still loads --------------------------
    old = {'Crosstalk_kwargs': dict(
        status=True, V0=[0.01, -0.02], A_matrix=[[1.0, 0.1], [0.2, 0.9]],
        P=[[1e-6, 0.0], [0.0, 1e-6]], Vex=0.1, Cstd=1.0,
    )}
    ctx_old = make_ctx()
    ctx_old._deserialize_state(old)
    check('old Crosstalk_kwargs block loads',
          ctx_old.crosstalk is not None
          and ctx_old.crosstalk.V0 == complex(0.01, -0.02))
    # (Full _serialize_state needs registered devices; round-trip the block.)
    ctx_rt = make_ctx()
    ctx_rt._deserialize_state(json.loads(json.dumps(
        {'Crosstalk_kwargs': three_point_to_dict(ctx_old.crosstalk)})))
    check('Crosstalk_kwargs survives a save/load round trip',
          np.allclose(ctx_rt.crosstalk.A_matrix, ctx_old.crosstalk.A_matrix))

    print("\nAll checks passed.")
