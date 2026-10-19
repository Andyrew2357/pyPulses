"""
pyPulses.routines.cap.transition
================================
A Runner Transition that rebalances the capacitance bridge at a reference
point whenever a set of "excitation" coordinates (amplitude, frequency, ...)
changes.

Path taken when triggered (all through scan.move_to):

    previous point --> reference point --> [excitation changed here]
                   --> three-point balance --> (runner) --> target point

Two gate ramps, the minimum given that the reference must be visited.
Nothing moves back; the runner's own move_to(target) finishes the trip, and
Measurement's time_per_point settle applies as usual.

Every rebalance is a full three-point balance: V0, the full K matrix, A and P
are all re-measured and pushed into the context via cap_initialize_filter.

Usage
-----
    exc_scan = ScanPoints([vex_ch, freq_ch], points=[...])   # outer, rare
    scan     = exc_scan * gate_scan

    rebal = CapRebalanceTransition(
        ctx,
        trigger    = [vex_ch.name, freq_ch.name],
        ref_coords = {'Vtg': 0.0, 'Vbg': 2.0},
        ctx_attrs  = {
            'Vex':        vex_ch.name,
            'Vstd_range': (vex_ch.name, lambda v: 2.0 * v),
        },
    )
    runner = Runner(scan, meas, transitions=[rebal], ...)
    db = DatabaseLogger.from_runner(runner, path)
    rebal.db = db                   # history also goes to the DB global attrs
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .initialize import cap_balance_three_point, cap_initialize_filter

if TYPE_CHECKING:
    from .context import CapContext
    from ...core.scan import ScanBase

Coords = Dict[str, float]
AttrSpec = (
    str                                          # copy a trigger coordinate
    | Tuple[str, Callable[[float], float]]       # transform one coordinate
    | Callable[[Coords], float]                  # any function of the state
)


class CapRebalanceTransition:
    """
    Parameters
    ----------
    ctx : CapContext
    trigger : sequence of str
        Scan coordinates whose change requires a rebalance (amplitude,
        frequency, ...). Any number, no built-in meaning.
    ref_coords : dict or callable(target) -> dict
        Coordinates of the reference point. Channels not named here stay where
        they are while balancing. Use a callable to make the reference depend
        on the target, e.g. lambda t: {'Vtg': 0, 'Vbg': 2, 'B': t['B']}.
        Must not contain trigger coordinates.
    ctx_attrs : dict, optional
        CapContext attribute -> how to compute it from the new trigger state.
        Set before balancing, so the balance uses the new values. A spec is:
          'coord'             copy the value of a trigger coordinate
          ('coord', f)        f(value of that coordinate)
          f                   f(state), state = {trigger coord: value}
        e.g.
          {'Vex':        ('Vex_amp', lambda v: v * divider),
           'Vstd_range': ('Vex_amp', lambda v: 2.0 * v * divider)}
        Any number of attributes may depend on the same coordinate.
        cap_initialize_filter copies ctx.Vstd_range into the extrapolator, so
        a changed Vstd_range propagates there as well.
    change_at_ref : bool, default True
        If True, arrive at the reference with the old excitation and change it
        there. If False, the excitation ramps in tandem with the gates.
    settle : float, default 0.0
        Wait after changing the excitation, before balancing (lock-in
        re-lock after a frequency change, filter settling, ...).
    three_point_kwargs : dict, optional
        Forwarded to cap_balance_three_point. `wait` defaults to
        ctx.settle_time, `samples` to 5, `plot` to False (plot=True would
        clear the runner's sidecar frame).
    db : DatabaseLogger, optional
        If given, the history is written to its global attrs ('cap_rebalance')
        after every rebalance.
    rtol : float
        Relative tolerance for deciding a trigger value is unchanged.

    Attributes
    ----------
    history : list of dict
        One entry per rebalance: time, scan idx, trigger state, reference,
        context values set, and the full balance result.
    balanced : dict or None
        Trigger state the bridge is currently balanced for.
    """

    def __init__(self,
        ctx: 'CapContext',
        trigger: Sequence[str],
        ref_coords: Coords | Callable[[Coords], Coords],
        ctx_attrs: Dict[str, AttrSpec] | None = None,
        change_at_ref: bool = True,
        settle: float = 0.0,
        three_point_kwargs: Dict[str, Any] | None = None,
        db=None,
        rtol: float = 1e-9,
    ):
        self.ctx = ctx
        self.trigger = list(trigger)
        if not self.trigger:
            raise ValueError("trigger must name at least one coordinate.")

        self.ref_coords = ref_coords

        # Normalized to attr -> f(state)
        self.ctx_attrs: Dict[str, Callable[[Coords], float]] = {}
        for attr, spec in (ctx_attrs or {}).items():
            if not hasattr(ctx, attr):
                raise ValueError(f"CapContext has no attribute {attr!r}.")
            self.ctx_attrs[attr] = self._compile_spec(attr, spec)

        self.change_at_ref = change_at_ref
        self.settle = settle
        self.three_point_kwargs = {'samples': 5, 'plot': False,
                                   **(three_point_kwargs or {})}
        self.db = db
        self.rtol = rtol

        self.balanced: Coords | None = None
        self.history: List[dict] = []

    """Transition interface"""

    def __call__(self,
        scan: 'ScanBase',
        idx: np.ndarray,
        target: Coords,
        min_wait: float | None,
    ) -> None:
        state = {n: float(target[n]) for n in self.trigger}
        if self.balanced is not None and self._same(state, self.balanced):
            return

        # Invalid until the rebalance succeeds; a raise stops the scan and a
        # resume re-triggers here.
        self.balanced = None

        ref = self._resolve_ref(target, scan)
        here = dict(scan._get_current_coords())

        # Go to the reference, then change the excitation there.
        exc_changes = not self._same(state, {n: here[n] for n in self.trigger})
        if self.change_at_ref and exc_changes:
            scan.move_to({**here, **ref}, min_wait=min_wait)
        scan.move_to({**here, **ref, **state}, min_wait=min_wait)
        if self.settle > 0:
            time.sleep(self.settle)

        ctx_set = self._update_ctx(state)
        bal = self._balance(state)
        cap_initialize_filter(self.ctx, bal)   # A, P, K_matrix, extrapolator,
                                               # clears ctx._Vstd_now
        self.balanced = state
        self._record(idx, state, ref, ctx_set, bal)

    """Internals"""

    def _compile_spec(self, attr: str, spec: AttrSpec) -> Callable[[Coords], float]:
        def need(coord):
            if coord not in self.trigger:
                raise ValueError(
                    f"ctx_attrs[{attr!r}] refers to {coord!r}, "
                    f"which is not in trigger {self.trigger}."
                )
        if isinstance(spec, str):
            need(spec)
            return lambda st, c=spec: st[c]
        if isinstance(spec, tuple):
            coord, fn = spec
            need(coord)
            return lambda st, c=coord, f=fn: f(st[c])
        if callable(spec):
            return spec
        raise TypeError(f"ctx_attrs[{attr!r}]: unsupported spec {spec!r}.")

    def _update_ctx(self, state: Coords) -> Dict[str, float]:
        # Compute everything first so a failing spec leaves ctx untouched.
        values = {attr: float(fn(state)) for attr, fn in self.ctx_attrs.items()}
        if 'Vstd_range' in values and not values['Vstd_range'] > 0:
            raise ValueError(
                f"Vstd_range must be positive, got {values['Vstd_range']} "
                f"for {state}."
            )
        for attr, val in values.items():
            setattr(self.ctx, attr, val)
        return values

    def _balance(self, state: Coords):
        ctx = self.ctx
        kw = {'wait': ctx.settle_time, **self.three_point_kwargs}
        bal = cap_balance_three_point(
            ctx.Vstd, ctx.Theta, ctx.lockin_call,
            Vex = ctx.Vex,
            Cstd = ctx.Cstd,
            Vstd_range = ctx.Vstd_range,
            logger = ctx.logger,
            **kw,
        )
        if not bal.status:
            raise RuntimeError(
                f"Three-point balance out of range at {state}: "
                f"|V0|={abs(bal.V0):.3e} > Vstd_range={ctx.Vstd_range:.3e}."
            )
        return bal

    def _resolve_ref(self, target: Coords, scan: 'ScanBase') -> Coords:
        ref = self.ref_coords(target) if callable(self.ref_coords) \
            else self.ref_coords
        ref = {k: float(v) for k, v in ref.items()}
        bad = set(ref) - set(scan.coord_names)
        if bad:
            raise ValueError(f"ref_coords has keys not in the scan: {sorted(bad)}")
        clash = set(ref) & set(self.trigger)
        if clash:
            raise ValueError(f"ref_coords must not set trigger coords: {sorted(clash)}")
        return ref

    def _same(self, a: Coords, b: Coords) -> bool:
        return a.keys() == b.keys() and all(
            np.isclose(a[k], b[k], rtol=self.rtol, atol=0.0) for k in a
        )

    def _record(self, idx, state, ref, ctx_set, bal):
        entry = dict(
            time = datetime.datetime.now().isoformat(),
            idx = [int(i) for i in idx],
            state = dict(state),
            ref = dict(ref),
            ctx = dict(ctx_set),
            Vex = float(bal.Vex),
            Cstd = float(bal.Cstd),
            V0 = [float(bal.V0.real), float(bal.V0.imag)],
            A = [float(bal.A_complex.real), float(bal.A_complex.imag)],
            K = np.asarray(bal.A_matrix, dtype=float).tolist(),
            P = np.asarray(bal.P, dtype=float).tolist(),
            Cex_ref = None if bal.Cex is None else float(bal.Cex),
            Closs_ref = None if bal.Closs is None else float(bal.Closs),
            residual = None if bal.error is None else [float(e) for e in bal.error],
        )
        self.history.append(entry)
        self.ctx.log(f"Rebalanced at idx {entry['idx']} for {state}: "
                     f"V0={bal.V0:.5e}, residual={entry['residual']}")
        if self.db is not None:
            self.db.set_global_attrs({'cap_rebalance': self.history})