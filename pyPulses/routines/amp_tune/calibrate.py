"""
pyPulses.routines.amp_tune.calibrate
=====================================
amp_calibrate: a cheap electrical-characterization pass over the free bias
parameters, reading only ID/VGS/dissipated_power_by_stage (no SNR calls).
Builds a grid-based surrogate (AmpCalibration) that amp_optimize_snr uses
to cheaply pre-filter candidates for feasibility before ever touching
hardware with a (slow, noisy) SNR measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List

import numpy as np

from ...core.job import checkpoint
from ...core.tandem_sweep import SweepResult
from .context import AmpTuneContext


@dataclass
class AmpCalibrationPoint:
    """One measured (or attempted) point in the calibration grid."""
    params: Dict[str, float]
    ID: float | None
    VGS: float | None
    power_by_stage: Dict[str, float]
    safe: bool

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        status = "safe" if self.safe else "UNSAFE"
        return f"AmpCalibrationPoint({params_str}, {status})"


@dataclass
class AmpCalibration:
    """
    Grid of measured operating points, plus cheap lookup helpers
    (nearest-neighbor in raw parameter units -- deliberately simple) used
    by amp_optimize_snr to test feasibility/power for a candidate bias
    point without touching hardware.
    """
    points: List[AmpCalibrationPoint] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        n_safe = sum(1 for p in self.points if p.safe)
        return f"AmpCalibration({n_safe}/{len(self.points)} safe points)"

    def nearest(self, params: Dict[str, float]) -> AmpCalibrationPoint:
        """Nearest-neighbor lookup in the calibration grid (Euclidean, in
        raw parameter units)."""
        if not self.points:
            raise ValueError("AmpCalibration has no points to search.")
        target = np.array([params[name] for name in self.param_names])
        best, best_d = None, np.inf
        for p in self.points:
            v = np.array([p.params[name] for name in self.param_names])
            d = float(np.sum((v - target) ** 2))
            if d < best_d:
                best, best_d = p, d
        return best

    def is_feasible(self, params: Dict[str, float]) -> bool:
        return self.nearest(params).safe

    def power_cost(self,
        params: Dict[str, float],
        weights: Dict[str, float] | None = None,
    ) -> float:
        by_stage = self.nearest(params).power_by_stage
        if weights is None:
            return sum(by_stage.values())
        return sum(w * by_stage.get(name, 0.0) for name, w in weights.items())


def amp_calibrate(ctx: AmpTuneContext, numpoints: int = 5) -> AmpCalibration:
    """
    Sweep ctx.free_params over a numpoints^N grid within ctx.bounds,
    reading only the electrical operating point (no SNR calls) at each
    point.

    Every grid point is reached via amp.move_to(), so the existing live
    safety net (transistor VGS window + per-stage power caps, via
    check_safe()) remains active throughout: a grid point outside the
    safe region reverts rather than damaging hardware, and is recorded as
    infeasible rather than aborting the whole calibration sweep. Because
    the sweep reverts on panic, the grid point actually visited is *not*
    the requested target in that case -- no electrical readings are
    fabricated for a point the amp never actually reached.

    Parameters
    ----------
    ctx : AmpTuneContext
    numpoints : int, default=5
        Points per free parameter; the grid has numpoints**len(free_params)
        points in total, so keep this modest for more than ~2 free params.

    Returns
    -------
    AmpCalibration
    """
    bounds = ctx.require_bounds()
    ctx.log(f"amp_calibrate: sweeping {ctx.free_params} over a "
             f"{numpoints}^{len(ctx.free_params)} grid.")

    axes = {name: np.linspace(*bounds[name], numpoints) for name in ctx.free_params}
    calibration = AmpCalibration(param_names=list(ctx.free_params))

    for combo in product(*(axes[name] for name in ctx.free_params)):
        checkpoint()
        target = dict(zip(ctx.free_params, (float(v) for v in combo)))

        sweep_result = ctx.amp.move_to(**target)

        if sweep_result != SweepResult.SUCCEEDED:
            calibration.points.append(AmpCalibrationPoint(
                params=target, ID=None, VGS=None, power_by_stage={}, safe=False,
            ))
            continue

        safe = ctx.amp.check_safe()
        try:
            id_ = ctx.amp.ID()
            vgs = ctx.amp.VGS()
            power_by_stage = ctx.amp.dissipated_power_by_stage()
        except ValueError:
            # No current source configured at all (neither VD nor IDS) --
            # nothing electrical can be verified for this point.
            id_ = vgs = None
            power_by_stage = {}
            safe = False

        calibration.points.append(AmpCalibrationPoint(
            params=target, ID=id_, VGS=vgs, power_by_stage=power_by_stage, safe=safe,
        ))

    n_safe = sum(1 for p in calibration.points if p.safe)
    ctx.log(f"amp_calibrate: {n_safe}/{len(calibration.points)} grid points safe.")
    return calibration
