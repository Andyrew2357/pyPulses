"""
pyPulses.routines.cap.crosstalk
================================
Compensation for excitation-line crosstalk that bypasses the amplifier.

When the excitation lines carry attenuators, the pre-attenuation drive
amplitude on Vstd (and/or Vex) is larger than what actually reaches the
sample/amp -- and some of that larger upstream signal couples directly to
the output line, never passing through the sample or the amplifier at all.

This crosstalk is linear in Vstd and adds *before* any balance math runs, so
the "gain matrix" a three-point balance recovers against uncompensated
readings is actually (K_sample + K_crosstalk), not K_sample alone. Unless
K_crosstalk happens to be a pure complex-linear (rotation+scale) map -- no
physical reason it should be, since it's a different coupling path than the
amplifier -- this is a genuine distortion of the recovered gain that a
post-hoc phase/rotation correction cannot undo. It has to be measured and
subtracted *before* the sample-path calibration runs.

Key reuse insight
-----------------
With the amplifier disabled, the lock-in only sees the direct leakage path:

    L(Vstd) = K_crosstalk @ Vstd + offset

which is algebraically identical to K_crosstalk @ (Vstd - V0_crosstalk) for
V0_crosstalk = -K_crosstalk^{-1} @ offset -- exactly the model
cap_balance_three_point already fits. So no new linear algebra is needed:
running cap_balance_three_point(..., move_to_balance=False) with the amp off
*is* the crosstalk calibration. CrosstalkCompensatedLockIn just evaluates
that fitted model at the current Vstd and subtracts it from every raw
reading, so every existing consumer of ctx.lockin_call (cap_balance_three_point
for the sample path, cap_measure, cap_balance) transparently sees
crosstalk-free data.

Usage
-----
    raw_lockin = lockin.resolve('get_average')
    ctx = CapContext(Vstd=..., Theta=..., lockin_call=raw_lockin, ...)

    # One-off, at the current excitation (wraps ctx.lockin_call if needed),
    # then balance as usual against the compensated readings
    cap_initialize_crosstalk(
        ctx,
        amp_channels = [amp_gate_ch],
        amp_on       = {'amp_gate': -0.2},
        amp_off      = {'amp_gate': -5.0},
    )
    cap_initialize(ctx)

    # In a scan that changes the excitation
    xtalk = CapCrosstalkTransition(
        ctx,
        trigger      = [freq_ch.name],
        amp_channels = [amp_gate_ch],
        amp_on       = {'amp_gate': -0.2},
        amp_off      = {'amp_gate': -5.0},
    )
    rebal = CapRebalanceTransition(ctx, trigger=[freq_ch.name], ref_coords=...)

    # xtalk MUST come before rebal: crosstalk must be refit before the
    # sample-path three-point balance runs on the same trigger change.
    runner = Runner(scan, meas, transitions=[xtalk, rebal], ...)
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np

from .initialize import cap_balance_three_point, ThreePointBalanceResult
from .transition import (
    Coords, AttrSpec, _compile_attr_spec, _apply_ctx_attrs, _same_state,
)
from ...core.tandem_sweep import tandemSweep, SweepResult
from ...devices.channel_adapter import LockInChannel
from ...devices.sweepable_channel import SweepableChannel

if TYPE_CHECKING:
    from .context import CapContext
    from ...core.scan import ScanBase


def _fit_crosstalk(
    ctx: 'CapContext',
    raw_lockin_call,
    three_point_kwargs: Dict[str, Any],
) -> ThreePointBalanceResult:
    """
    Fit the crosstalk model with the amp already off. Shared by
    CapCrosstalkTransition and the calibration-library builders (library.py).
    """
    kw = {'wait': ctx.settle_time, **three_point_kwargs}
    return cap_balance_three_point(
        ctx.Vstd, ctx.Theta, raw_lockin_call,
        Vex = ctx.Vex,
        Cstd = 1.0,  # unused for crosstalk purposes -- only A_matrix/V0 matter
        Vstd_range = ctx.Vstd_range,
        Vstd_max = ctx.Vstd_max,
        logger = ctx.logger,
        **kw,
    )


class CrosstalkCompensatedLockIn:
    """
    Wraps a raw LockInChannel, subtracting a linear crosstalk model from
    every reading.

    The crosstalk model is a ThreePointBalanceResult fitted with the
    amplifier disabled (see CapCrosstalkTransition) and stored on
    `ctx.crosstalk`. It predicts the direct-leakage contribution at the
    current Vstd via the same L = A_matrix @ (Vstd - V0) form
    cap_balance_three_point already uses, and subtracts it before returning.

    If `ctx.crosstalk` is None (not yet calibrated), this is a transparent
    passthrough to the raw channel.

    Parameters
    ----------
    raw : LockInChannel
        The underlying, uncompensated lock-in call.
    ctx : CapContext
        Supplies the current Vstd (via get_Vstd_complex()) and the active
        crosstalk model (ctx.crosstalk).
    """

    def __init__(self, raw: LockInChannel, ctx: 'CapContext'):
        self.raw = raw
        self.ctx = ctx

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        mean, cov = self.raw()
        model = self.ctx.crosstalk
        if model is None:
            return mean, cov

        V_now = self.ctx.get_Vstd_complex()
        dV = np.array([
            V_now.real - model.V0.real,
            V_now.imag - model.V0.imag,
        ])
        correction = np.asarray(model.A_matrix, dtype=float) @ dV
        mean = np.asarray(mean, dtype=float).reshape(2) - correction
        return mean, cov

    def format_ref(self):
        """
        Not independently serializable -- CapContext special-cases this
        wrapper in its own (de)serialization, referencing `.raw` directly.
        See CapContext._serialize_state.
        """
        raise NotImplementedError(
            "CrosstalkCompensatedLockIn is not registry-serializable on its "
            "own; CapContext._serialize_state formats its .raw channel "
            "instead."
        )


def _ensure_compensated(ctx: 'CapContext') -> CrosstalkCompensatedLockIn:
    """Wrap ctx.lockin_call in a CrosstalkCompensatedLockIn if it isn't one."""
    if not isinstance(ctx.lockin_call, CrosstalkCompensatedLockIn):
        ctx.lockin_call = CrosstalkCompensatedLockIn(ctx.lockin_call, ctx)
    return ctx.lockin_call


def _sweep(channels: Sequence[SweepableChannel], target: Coords,
           min_wait: float | None) -> None:
    res = tandemSweep(list(channels), target, min_wait=min_wait)
    if res is not SweepResult.SUCCEEDED:
        raise RuntimeError(f"Sweep to {target} did not succeed ({res.name}).")


def cap_initialize_crosstalk(
    ctx: 'CapContext',
    amp_channels: Sequence[SweepableChannel],
    amp_on: Coords,
    amp_off: Coords,
    settle_off: float = 0.0,
    settle_on: float = 0.0,
    three_point_kwargs: Dict[str, Any] | None = None,
    min_wait: float | None = None,
) -> ThreePointBalanceResult:
    """
    Measure the crosstalk model once, at the current excitation and gate
    point, and store it on `ctx.crosstalk`.

    The one-off counterpart of CapCrosstalkTransition, meant to sit in the
    initialization steps right before cap_initialize (whose balance then
    sees compensated readings). Wraps `ctx.lockin_call` in a
    CrosstalkCompensatedLockIn if it isn't one already.

    Steps: ramp `amp_channels` to `amp_off`, wait `settle_off`, fit against
    the raw (uncompensated) channel, ramp back to `amp_on`, wait
    `settle_on`. The amp is turned back on even if the fit raises.

    Parameters
    ----------
    ctx : CapContext
        Vex/Vstd_range must already reflect the current excitation.
    amp_channels : sequence of SweepableChannel
        Channels that control the amplifier's on/off state.
    amp_on, amp_off : dict
        Target values for `amp_channels`, keyed by channel name.
    settle_off, settle_on : float
        Waits after turning the amp off and after turning it back on.
    three_point_kwargs : dict, optional
        Forwarded to cap_balance_three_point. `wait` defaults to
        ctx.settle_time, `samples` to 5, `plot` to False, `move_to_balance`
        to False.
    min_wait : float, optional
        Floor on the derived wait time passed to tandemSweep.

    Returns
    -------
    ThreePointBalanceResult
        The crosstalk model (also stored on ctx.crosstalk).
    """
    wrapper = _ensure_compensated(ctx)
    kw = {'samples': 5, 'plot': False, 'move_to_balance': False,
          **(three_point_kwargs or {})}

    _sweep(amp_channels, amp_off, min_wait)
    try:
        if settle_off > 0:
            time.sleep(settle_off)
        result = _fit_crosstalk(ctx, wrapper.raw, kw)
        ctx.crosstalk = result
    finally:
        _sweep(amp_channels, amp_on, min_wait)
        if settle_on > 0:
            time.sleep(settle_on)

    ctx.log(f"Crosstalk measured: V0={result.V0:.5e}")
    return result


class CapCrosstalkTransition:
    """
    A Runner Transition that re-measures excitation-line crosstalk whenever
    a set of "excitation" coordinates (amplitude, frequency, ...) changes.

    Assumes crosstalk bypasses the sample/amplifier entirely and so is
    independent of gate voltage / DUT state -- only of the excitation
    configuration (the `trigger` coordinates). Unlike CapRebalanceTransition,
    this does not need a `ref_coords` gate move; it recalibrates wherever
    the gates currently are. If crosstalk turns out to have gate-state
    dependence after all, a ref_coords step can be added following the same
    pattern as CapRebalanceTransition.

    On a trigger-state change:
      0. Change the excitation to the new state in place (scan.move_to with
         the gates where they are), so the fit sees the new excitation.
      1. Apply `ctx_attrs` (so e.g. Vex/Vstd_range reflect the new state
         before sizing the calibration sweep).
      2. cap_initialize_crosstalk: amp off, fit K_crosstalk/V0_crosstalk
         against the wrapper's raw (uncompensated) channel -- with the amp
         off, this *is* the crosstalk calibration (see module docstring) --
         store it on `ctx.crosstalk`, amp back on.

    Wraps `ctx.lockin_call` in a CrosstalkCompensatedLockIn on construction
    if it isn't one already.

    **Usage ordering**: list this transition *before* CapRebalanceTransition
    in `Runner(transitions=[...])` -- crosstalk must be refit before the
    sample-path three-point balance runs on the same trigger change.

    Parameters
    ----------
    ctx : CapContext
    trigger : sequence of str
        Scan coordinates whose change requires a crosstalk remeasurement.
    amp_channels : sequence of SweepableChannel
        Channels that control the amplifier's on/off state (bias, enable
        relay, ...).
    amp_on, amp_off : dict
        Target values for `amp_channels`, keyed by channel name, passed to
        `tandemSweep`.
    ctx_attrs : dict, optional
        Same spec mini-language as CapRebalanceTransition's ctx_attrs.
    settle_off : float, default 0.0
        Wait after ramping to amp_off, before the calibration sweep.
    settle_on : float, default 0.0
        Wait after ramping back to amp_on.
    three_point_kwargs : dict, optional
        Forwarded to cap_balance_three_point. `wait` defaults to
        ctx.settle_time, `samples` to 5, `plot` to False, `move_to_balance`
        to False (there is no physical point to park at -- V0_crosstalk is
        just the reference point of the fitted model).
    min_wait : float, optional
        Floor on the derived wait time passed to tandemSweep.
    db : DatabaseLogger, optional
        If given, the history is written to its global attrs
        ('cap_crosstalk') after every remeasurement.
    rtol : float
        Relative tolerance for deciding a trigger value is unchanged.

    Attributes
    ----------
    history : list of dict
        One entry per remeasurement: time, scan idx, trigger state, context
        values set, and the fitted crosstalk model.
    measured : dict or None
        Trigger state the crosstalk model is currently valid for.
    """

    def __init__(self,
        ctx: 'CapContext',
        trigger: Sequence[str],
        amp_channels: Sequence[SweepableChannel],
        amp_on: Coords,
        amp_off: Coords,
        ctx_attrs: Dict[str, AttrSpec] | None = None,
        settle_off: float = 0.0,
        settle_on: float = 0.0,
        three_point_kwargs: Dict[str, Any] | None = None,
        min_wait: float | None = None,
        db=None,
        rtol: float = 1e-9,
    ):
        _ensure_compensated(ctx)

        self.ctx = ctx
        self.trigger = list(trigger)
        if not self.trigger:
            raise ValueError("trigger must name at least one coordinate.")

        self.amp_channels = list(amp_channels)
        if not self.amp_channels:
            raise ValueError("amp_channels must name at least one channel.")
        self.amp_on = dict(amp_on)
        self.amp_off = dict(amp_off)

        self.ctx_attrs: Dict[str, Any] = {}
        for attr, spec in (ctx_attrs or {}).items():
            if not hasattr(ctx, attr):
                raise ValueError(f"CapContext has no attribute {attr!r}.")
            self.ctx_attrs[attr] = _compile_attr_spec(self.trigger, attr, spec)

        self.settle_off = settle_off
        self.settle_on = settle_on
        self.three_point_kwargs = {
            'samples': 5, 'plot': False, 'move_to_balance': False,
            **(three_point_kwargs or {}),
        }
        self.min_wait = min_wait
        self.db = db
        self.rtol = rtol

        self.measured: Coords | None = None
        self.history: List[dict] = []

    """Transition interface"""

    def __call__(self,
        scan: 'ScanBase',
        idx: np.ndarray,
        target: Coords,
        min_wait: float | None,
    ) -> None:
        state = {n: float(target[n]) for n in self.trigger}
        if self.measured is not None and _same_state(state, self.measured, self.rtol):
            return

        # Invalid until the remeasurement succeeds; a raise stops the scan
        # and a resume re-triggers here.
        self.measured = None

        # Crosstalk depends on the excitation, and the runner only applies
        # the target after all transitions have run -- so change it here,
        # in place (gates untouched).
        here = dict(scan._get_current_coords())
        scan.move_to({**here, **state}, min_wait=min_wait)

        ctx_set = _apply_ctx_attrs(self.ctx, self.ctx_attrs, state)

        result = cap_initialize_crosstalk(
            self.ctx, self.amp_channels, self.amp_on, self.amp_off,
            settle_off = self.settle_off,
            settle_on = self.settle_on,
            three_point_kwargs = self.three_point_kwargs,
            min_wait = min_wait if min_wait is not None else self.min_wait,
        )

        self.measured = state
        self._record(idx, state, ctx_set, result)

    """Internals"""

    def _record(self, idx, state, ctx_set, result: ThreePointBalanceResult):
        entry = dict(
            time = datetime.datetime.now().isoformat(),
            idx = [int(i) for i in idx],
            state = dict(state),
            ctx = dict(ctx_set),
            V0 = [float(result.V0.real), float(result.V0.imag)],
            K = np.asarray(result.A_matrix, dtype=float).tolist(),
            P = np.asarray(result.P, dtype=float).tolist(),
        )
        self.history.append(entry)
        self.ctx.log(
            f"Crosstalk remeasured at idx {entry['idx']} for {state}: "
            f"V0={result.V0:.5e}"
        )
        if self.db is not None:
            self.db.set_global_attrs({'cap_crosstalk': self.history})
