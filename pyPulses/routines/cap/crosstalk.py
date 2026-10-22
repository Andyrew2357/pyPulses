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
    ctx.lockin_call = CrosstalkCompensatedLockIn(raw_lockin, ctx)

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
from ...core.tandem_sweep import tandemSweep
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
      2. Ramp `amp_channels` to `amp_off` (disabling the amplifier) via the
         standalone `tandemSweep`.
      3. Fit K_crosstalk/V0_crosstalk with cap_balance_three_point against
         the wrapper's raw (uncompensated) channel -- with the amp off,
         this *is* the crosstalk calibration (see module docstring).
      4. Store the result on `ctx.crosstalk`.
      5. Ramp `amp_channels` back to `amp_on`.

    Requires `ctx.lockin_call` to already be a CrosstalkCompensatedLockIn,
    so this can reach its `.raw` channel.

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
        if not isinstance(ctx.lockin_call, CrosstalkCompensatedLockIn):
            raise TypeError(
                "CapCrosstalkTransition requires ctx.lockin_call to already "
                "be a CrosstalkCompensatedLockIn (so it can fit against the "
                "raw, uncompensated channel). Wrap it first: "
                "ctx.lockin_call = CrosstalkCompensatedLockIn(ctx.lockin_call, ctx)"
            )

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

        wait = min_wait if min_wait is not None else self.min_wait
        wrapper: CrosstalkCompensatedLockIn = self.ctx.lockin_call

        tandemSweep(self.amp_channels, self.amp_off, min_wait=wait)
        if self.settle_off > 0:
            time.sleep(self.settle_off)

        result = self._fit(wrapper.raw, state)
        self.ctx.crosstalk = result

        tandemSweep(self.amp_channels, self.amp_on, min_wait=wait)
        if self.settle_on > 0:
            time.sleep(self.settle_on)

        self.measured = state
        self._record(idx, state, ctx_set, result)

    """Internals"""

    def _fit(self, raw_lockin_call, state: Coords) -> ThreePointBalanceResult:
        return _fit_crosstalk(self.ctx, raw_lockin_call, self.three_point_kwargs)

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


if __name__ == '__main__':
    """
    Self-test against a synthetic linear bridge + crosstalk model (no real
    hardware), following the FakeChannel convention used in
    noise_probe.py/prep_amp.py's self-tests.

    Model: raw_lockin() = K_crosstalk @ Vstd + offset_vex
                           + (K_sample @ (Vstd - V0_sample) if amp enabled)

    K_crosstalk is deliberately NOT complex-linear (asymmetric: a pure real
    leakage into LX only), so a naive uncompensated calibration should pick
    up a biased, non-conformal gain -- and should show spurious loss at a
    synthetic "hard gap" (V0_sample purely real) that grows with the
    injected sample signal. After crosstalk compensation, the recovered
    gain should match K_sample and the hard-gap loss should vanish.
    """

    from .context import CapContext

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
        def __init__(self, name, channel: FakeChannel):
            self.name = name
            self._channel = channel

            class _Cfg:
                max_step = None
                min_step = None
                tolerance = 0.0
                max_rate = None
                settle_time = 0.0
                async_set = False
            self.config = _Cfg()

        def get_output(self) -> float:
            return self._channel()

        def set_output(self, value: float) -> None:
            self._channel(value)

    K_sample = np.array([[2.0, -0.5], [0.5, 2.0]])     # ideal (complex-linear)
    V0_sample = complex(0.3, 0.0)                        # "hard gap": zero loss
    K_crosstalk = np.array([[0.8, 0.3], [0.1, 0.1]])    # deliberately asymmetric, invertible
    # Kept near zero so the "scales with Cex" check below isolates the
    # non-conformal-K_crosstalk mechanism (a term proportional to the true
    # signal) from a separate, already-well-understood constant floor (a
    # fixed Vex-proportional leakage would show up as a size-independent
    # offset, which any correct gap/band-style calibration already handles).
    offset_vex = np.array([1e-6, -1e-6])
    Vex, Vstd_range = 1.0, 1.0

    Vstd_raw, Theta_raw = FakeChannel(0.0), FakeChannel(0.0)
    amp_bias = FakeChannel(-0.2)   # "on" value; set very negative to disable
    amp_enabled = {'v': True}

    def raw_lockin_call():
        r, th = Vstd_raw(), np.deg2rad(Theta_raw())
        Vstd_vec = np.array([r * np.cos(th), r * np.sin(th)])
        L = K_crosstalk @ Vstd_vec + offset_vex
        if amp_enabled['v']:
            dV = np.array([Vstd_vec[0] - V0_sample.real, Vstd_vec[1] - V0_sample.imag])
            L = L + K_sample @ dV
        return L, np.eye(2) * 1e-12

    ctx = CapContext(
        Vstd=Vstd_raw, Theta=Theta_raw,
        lockin_call=raw_lockin_call,
        Vstd_range=Vstd_range, Vex=Vex, Cstd=1.0,
        settle_time=0.0,
    )

    # --- Baseline: uncompensated three-point balance is biased ----------
    bal_uncompensated = cap_balance_three_point(
        ctx.Vstd, ctx.Theta, ctx.lockin_call,
        Vex=ctx.Vex, Cstd=ctx.Cstd, Vstd_range=ctx.Vstd_range,
        wait=0.0, move_to_balance=True,
    )
    check('uncompensated calibration status OK', bal_uncompensated.status)
    check('uncompensated gain differs from true K_sample (crosstalk leaks in)',
          not np.allclose(bal_uncompensated.A_matrix, K_sample, atol=1e-6))

    # Read loss at the uncompensated "balance" point for a couple of
    # different injected sample magnitudes -- should NOT be ~0, and should
    # grow with the injected Cex-like signal (scales with V0_sample.real).
    def uncompensated_loss_at(v0_real):
        nonlocal_V0 = complex(v0_real, 0.0)

        def lockin_call():
            r, th = Vstd_raw(), np.deg2rad(Theta_raw())
            Vstd_vec = np.array([r * np.cos(th), r * np.sin(th)])
            L = K_crosstalk @ Vstd_vec + offset_vex
            dV = np.array([Vstd_vec[0] - nonlocal_V0.real, Vstd_vec[1] - nonlocal_V0.imag])
            L = L + K_sample @ dV
            return L, np.eye(2) * 1e-12

        bal = cap_balance_three_point(
            ctx.Vstd, ctx.Theta, lockin_call,
            Vex=ctx.Vex, Cstd=ctx.Cstd, Vstd_range=ctx.Vstd_range,
            wait=0.0, move_to_balance=True,
        )
        return bal.Closs

    loss_small = abs(uncompensated_loss_at(0.1))
    loss_large = abs(uncompensated_loss_at(0.3))
    check('uncompensated hard-gap loss is spuriously nonzero', loss_small > 1e-4)
    check('uncompensated spurious loss grows with the injected signal (Cex)',
          loss_large > 2 * loss_small)

    # --- Crosstalk compensation fixes it ---------------------------------
    ctx.lockin_call = CrosstalkCompensatedLockIn(raw_lockin_call, ctx)

    amp_ch = FakeSweepable('amp_bias', amp_bias)

    def _toggle_amp(targets):
        amp_enabled['v'] = targets.get('amp_bias', amp_bias()) > -1.0

    orig_set_output = amp_ch.set_output
    def _set_output_and_toggle(value):
        orig_set_output(value)
        amp_enabled['v'] = value > -1.0
    amp_ch.set_output = _set_output_and_toggle

    xtalk = CapCrosstalkTransition(
        ctx,
        trigger=['freq'],
        amp_channels=[amp_ch],
        amp_on={'amp_bias': -0.2},
        amp_off={'amp_bias': -5.0},
        three_point_kwargs={'wait': 0.0, 'samples': 1},
    )

    class _FakeScan:
        coord_names = ['freq']
        def _get_current_coords(self):
            return {'freq': 0.0}
        def move_to(self, coords, min_wait=None):
            pass

    xtalk(_FakeScan(), np.array([0]), {'freq': 1.0e6}, 0.0)

    check('crosstalk model stored on ctx', ctx.crosstalk is not None)
    check('amp restored to "on" after crosstalk measurement', amp_enabled['v'])

    bal_compensated = cap_balance_three_point(
        ctx.Vstd, ctx.Theta, ctx.lockin_call,
        Vex=ctx.Vex, Cstd=ctx.Cstd, Vstd_range=ctx.Vstd_range,
        wait=0.0, move_to_balance=True,
    )
    check('compensated calibration status OK', bal_compensated.status)
    check('compensated gain recovers true K_sample',
          np.allclose(bal_compensated.A_matrix, K_sample, atol=1e-6))
    check('compensated hard-gap loss is ~0',
          abs(bal_compensated.Closs) < 1e-6)

    print("\nAll checks passed.")
