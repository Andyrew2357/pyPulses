"""
pyPulses.routines.amp_tune.optimize
=====================================
amp_optimize_snr: constrained Bayesian-optimization loop over the free
bias parameters, restricted to the calibration surrogate's feasible
region, maximizing snr - lambda_power * power_cost as the search
objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ...core.job import checkpoint
from ...core.tandem_sweep import SweepResult
from ._bayes import AskTellOptimizer
from .calibrate import AmpCalibration
from .context import AmpTuneContext


@dataclass
class AmpTuneTrial:
    """One evaluated (or attempted) bias point during the search."""
    params: Dict[str, float]
    snr: float | None
    power_cost: float | None
    objective: float | None
    feasible: bool
    result: SweepResult | None

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        if not self.feasible:
            return f"AmpTuneTrial({params_str}, infeasible, result={self.result})"
        return (f"AmpTuneTrial({params_str}, snr={self.snr:.4g}, "
                f"objective={self.objective:.4g})")


@dataclass
class AmpTuneResult:
    """Outcome of amp_optimize_snr: the best point found, plus the full
    trial history for diagnosis/logging."""
    best_params: Dict[str, float] | None
    best_snr: float | None
    best_objective: float | None
    trials: List[AmpTuneTrial] = field(default_factory=list)

    def __str__(self) -> str:
        if self.best_params is None:
            return f"AmpTuneResult(no feasible trial found, {len(self.trials)} trials)"
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in self.best_params.items())
        return (
            f"AmpTuneResult(best_snr={self.best_snr:.4g}, "
            f"best_objective={self.best_objective:.4g}, {params_str}, "
            f"{len(self.trials)} trials)"
        )


def _random_candidate(
    rng: np.random.Generator,
    bounds: Dict[str, tuple[float, float]],
) -> Dict[str, float]:
    return {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in bounds.items()}


def amp_optimize_snr(
    ctx: AmpTuneContext,
    calibration: AmpCalibration,
    max_resample: int = 20,
) -> AmpTuneResult:
    """
    Ask/tell Bayesian-optimization search for the bias point maximizing
    snr - lambda_power * power_cost(weights), restricted to points the
    calibration surrogate marks feasible (transistor VGS window and every
    stage's max_power, as hard constraints). amp.move_to()'s live safety
    net remains active throughout as defense in depth against anything
    the surrogate got wrong.

    The first ctx.n_init trials are drawn uniformly at random (rejecting
    surrogate-infeasible draws); the remaining trials come from the
    Bayesian optimizer's acquisition function, with the same
    surrogate-feasibility rejection applied to its suggestions.

    Parameters
    ----------
    ctx : AmpTuneContext
    calibration : AmpCalibration
        From a prior amp_calibrate(ctx) call.
    max_resample : int, default=20
        Maximum surrogate-infeasible candidates to reject per iteration
        before giving up on that iteration (checking feasibility against
        the calibration surrogate is cheap -- no hardware is touched
        until a feasible candidate is found).

    Returns
    -------
    AmpTuneResult
    """
    bounds = ctx.require_bounds()
    rng = np.random.default_rng(ctx.random_state)
    optimizer = AskTellOptimizer(bounds=bounds, random_state=ctx.random_state)

    result = AmpTuneResult(best_params=None, best_snr=None, best_objective=None)
    n_init = max(0, min(ctx.n_init, ctx.n_iter))

    for i in range(ctx.n_iter):
        checkpoint()

        candidate = None
        for attempt in range(max_resample):
            if i < n_init:
                proposal = _random_candidate(rng, bounds)
            elif attempt == 0:
                proposal = optimizer.ask()
            else:
                # optimizer.ask() optimizes the acquisition function
                # deterministically given its current state; calling it
                # again here without an intervening tell() would just
                # return the same (rejected) point. Fall back to random
                # sampling for the rest of this iteration's resample
                # budget instead of burning it on repeats.
                proposal = _random_candidate(rng, bounds)
            if calibration.is_feasible(proposal):
                candidate = proposal
                break
        if candidate is None:
            # This iteration's resample budget didn't turn up a feasible
            # point, but that doesn't mean no feasible point exists
            # elsewhere in the search space -- the acquisition function's
            # exploration changes as more points get told, so later
            # iterations may well land somewhere productive. Skip rather
            # than abort; feasibility checks are pure computation (no
            # hardware touched), so this costs nothing but CPU time even
            # in the worst case of every remaining iteration failing too.
            ctx.warn(
                f"amp_optimize_snr: no feasible candidate found after "
                f"{max_resample} resamples at iteration {i}; skipping."
            )
            continue

        sweep_result = ctx.amp.move_to(**candidate)
        if sweep_result != SweepResult.SUCCEEDED:
            # The live safety net rejected a candidate the surrogate
            # thought was feasible. Record it and move on without
            # registering an objective value -- we never actually
            # reached this point, so there is nothing real to tell the
            # optimizer.
            ctx.warn(
                "amp_optimize_snr: live safety net rejected a "
                f"surrogate-feasible candidate {candidate} at iteration "
                f"{i} (result={sweep_result}); skipping."
            )
            result.trials.append(AmpTuneTrial(
                params=candidate, snr=None, power_cost=None,
                objective=None, feasible=False, result=sweep_result,
            ))
            continue

        snr = ctx.snr_estimator()
        power_cost = ctx.amp.power_cost(ctx.power_cost_weights)
        objective = snr - ctx.lambda_power * power_cost

        optimizer.tell(candidate, objective)
        result.trials.append(AmpTuneTrial(
            params=candidate, snr=snr, power_cost=power_cost,
            objective=objective, feasible=True, result=sweep_result,
        ))

        if result.best_objective is None or objective > result.best_objective:
            result.best_params = candidate
            result.best_snr = snr
            result.best_objective = objective

    if result.best_params is not None:
        ctx.log(f"amp_optimize_snr: moving to the best point found: {result}")
        ctx.amp.move_to(**result.best_params)

    return result


if __name__ == '__main__':
    """
    Self-test against a synthetic AmplifierLike stand-in -- no real
    hardware, no HEMTCommonSource even -- to validate the
    constrained-search logic before ever touching a real amplifier. This
    is also a regression check on two real bugs found this way during
    development: (1) a resample-exhaustion abort that
    used to give up on the *entire remaining search* instead of just
    that iteration, and (2) optimizer.ask() being asked repeatedly
    without an intervening tell(), which just returns the same
    (rejected) point every time -- see the comments above.
    """

    from .calibrate import amp_calibrate

    class SyntheticAmp:
        """VGS == VG (grounded source); ID = max(VDD, 0) * 1e-3, VDS ==
        VDD; all dissipation on stage '1K'. Safety: VGS in [-0.6, 0.0],
        power <= 5 mW."""
        def __init__(self):
            self._VG, self._VDD = -0.3, 1.0
            self.vgs_min, self.vgs_max = -0.6, 0.0
            self.max_power = 5e-3

        def free_channels(self):
            return {'VG': None, 'VDD': None}

        def VGS(self) -> float:
            return self._VG

        def ID(self) -> float:
            return max(self._VDD, 0.0) * 1e-3

        def VDS(self) -> float:
            return self._VDD

        def dissipated_power_by_stage(self) -> Dict[str, float]:
            return {'1K': self.ID() * self.VDS()}

        def power_cost(self, weights: Dict[str, float] | None = None) -> float:
            by_stage = self.dissipated_power_by_stage()
            if weights is None:
                return sum(by_stage.values())
            return sum(w * by_stage.get(k, 0.0) for k, w in weights.items())

        def check_safe(self) -> bool:
            if not (self.vgs_min <= self.VGS() <= self.vgs_max):
                return False
            return self.dissipated_power_by_stage()['1K'] <= self.max_power

        def move_to(self, panic_behavior='zero', min_wait=None, **targets):
            start = {'VG': self._VG, 'VDD': self._VDD}
            new = {**start, **targets}
            self._VG, self._VDD = new['VG'], new['VDD']
            if not self.check_safe():
                self._VG, self._VDD = start['VG'], start['VDD']
                return SweepResult.PANICKED
            return SweepResult.SUCCEEDED

    def check(label: str, cond: bool):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        assert cond, label

    # SNR peaks at VG=-0.2 and increases monotonically with VDD, so the
    # true unconstrained optimum would push VDD to the edge of its box
    # bound -- the power cap (not the box bound) is what should actually
    # end up binding.
    amp = SyntheticAmp()
    ctx = AmpTuneContext(
        amp=amp,
        snr_estimator=lambda: -(amp.VGS() + 0.2) ** 2 * 50.0 + 5.0 * amp.VDS(),
        bounds={'VG': (-0.6, 0.0), 'VDD': (0.0, 3.0)},
        n_init=6, n_iter=25, random_state=0,
    )
    calibration = amp_calibrate(ctx, numpoints=6)
    check('calibration found some safe grid points',
          any(p.safe for p in calibration.points))

    result = amp_optimize_snr(ctx, calibration)
    print(result)
    check('found a best point', result.best_params is not None)
    check('best point respects the VGS window',
          amp.vgs_min <= result.best_params['VG'] <= amp.vgs_max)
    check('best point respects the power cap', amp.power_cost() <= amp.max_power)
    check('search pushed VDD toward the power-cap boundary (not the box edge)',
          result.best_params['VDD'] > 1.7)
    check('search found VG close to the true optimum (-0.2)',
          abs(result.best_params['VG'] - (-0.2)) < 0.15)

    # Every trial marked feasible must actually be safe on the amp -- the
    # surrogate + live move_to() safety net combination should never
    # mis-report an unsafe point as feasible.
    for t in result.trials:
        if t.feasible:
            vg, vdd = t.params['VG'], t.params['VDD']
            vgs_ok = amp.vgs_min <= vg <= amp.vgs_max
            power_ok = (max(vdd, 0.0) * 1e-3) * vdd <= amp.max_power
            check(f'feasible trial {t.params} is actually safe', vgs_ok and power_ok)

    print("\nAll checks passed.")
