"""
pyPulses.routines.cap.prep_amp
================================
prep_amp: cap-specific glue wiring cap_noise_probe into the general
SNR-tuning framework (routines/amp_tune).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import CapContext
    from .noise_probe import CapNoiseProbeResult

from .noise_probe import cap_noise_probe
from ..amp_tune import AmpTuneContext, AmpTuneResult, amp_calibrate, amp_optimize_snr
from ...devices.HEMT import AmplifierLike

from typing import Dict, Tuple
import logging
import numpy as np


def prep_amp(
    cap_ctx: 'CapContext',
    amp: AmplifierLike,
    bounds: Dict[str, Tuple[float, float]],
    N: int = 15,
    method: str = 'three_point',
    n_refine: int = 1,
    cal_samples: int = 3,
    cal_wait: float | None = None,
    refine_samples: int = 1,
    refine_wait: float | None = None,
    lambda_power: float = 0.0,
    lambda_nongauss: float = 0.0,
    power_cost_weights: Dict[str, float] | None = None,
    n_init: int = 8,
    n_iter: int = 40,
    calibration_numpoints: int = 5,
    random_state: int | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[AmpTuneContext, AmpTuneResult]:
    """
    Tune `amp`'s bias to minimize the capacitance noise floor (and
    non-Gaussian noise) reported by `cap_noise_probe` against `cap_ctx`,
    subject to `amp`'s hard safety window and per-stage power caps, plus
    a soft power penalty (lambda_power).

    The search objective is built entirely in this function's closure --
    the general amp_tune framework (AmpTuneContext.snr_estimator) is an
    opaque callable specifically so cap-specific terms like
    lambda_nongauss stay local here rather than becoming a generic
    AmpTuneContext field. amp_calibrate/amp_optimize_snr from
    routines/amp_tune are used unmodified.

    Parameters
    ----------
    cap_ctx : CapContext
        The capacitance-bridge context to probe (via cap_noise_probe) at
        each candidate bias point. Its cap_filter/extrapolator are
        untouched by this process (see cap_noise_probe).
    amp : AmplifierLike
        The amplifier to tune (e.g. a HEMTCommonSource).
    bounds : dict
        {bias_channel_name: (min, max)} search bounds -- see
        AmpTuneContext.bounds.
    N : int, default=15
        Repeated reads per cap_noise_probe() call -- see its docstring.
    method : {'three_point', 'two_point'}, default='three_point'
        Which local calibration cap_noise_probe uses at each candidate --
        see its docstring. Three-point is the bridge's usual off-balance
        calibration method.
    n_refine : int, default=1
        Number of cap_balance_refine() correction steps cap_noise_probe
        applies before taking its N reads -- keeps the probe centered on
        the true balance point rather than the raw calibration's
        estimate, for a more faithful noise assessment. 0 disables it.
    cal_samples, refine_samples : int
        Passed through to cap_noise_probe's own `cal_samples`/
        `refine_samples`.
    cal_wait, refine_wait : float, optional
        Passed through to cap_noise_probe's own `cal_wait`/`refine_wait`.
        None uses real-hardware-tuned defaults; override for a faster
        synthetic/no-hardware setup.
    lambda_power : float, default=0.0
        Soft power penalty coefficient, same semantics as
        AmpTuneContext.lambda_power. 0.0 disables it.
    lambda_nongauss : float, default=0.0
        Soft penalty coefficient on excess (positive) kurtosis of the
        capacitance noise probe -- catches amp settings whose noise
        variance looks fine but which show slow, non-Gaussian gain jumps
        (e.g. from trap states). 0.0 disables it.
    power_cost_weights : dict, optional
        Passed through to amp.power_cost() inside the objective.
    n_init, n_iter, random_state, logger
        Passed through to AmpTuneContext.
    calibration_numpoints : int, default=5
        Passed through to amp_calibrate()'s `numpoints` -- the
        electrical-feasibility grid has numpoints**len(free_params)
        points. Coarser than this can under-resolve the feasible region
        near a sharp constraint boundary (amp_calibrate's surrogate is a
        nearest-neighbor lookup over this grid); increase it if the
        search seems to stop well short of a hard cap it should be able
        to approach.

    Returns
    -------
    (AmpTuneContext, AmpTuneResult)
        The context that was built (for inspection/reuse) and the
        optimization result. On return, `amp` is sitting at
        `result.best_params` if one was found (see amp_optimize_snr).
    """
    def snr_estimator() -> float:
        probe = cap_noise_probe(
            cap_ctx, N=N, method=method, n_refine=n_refine,
            cal_samples=cal_samples, cal_wait=cal_wait,
            refine_samples=refine_samples, refine_wait=refine_wait,
            logger=logger,
        )
        return (
            -np.log(max(probe.std_Cex, 1e-300))
            - lambda_nongauss * max(0.0, probe.kurtosis_Cex)
        )
        # Only positive (leptokurtic/heavy-tailed) excess kurtosis is
        # penalized -- platykurtic isn't the failure mode this targets.
        # log-scale on std_Cex: it can span orders of magnitude across
        # candidate bias points, same rationale as the power penalty's
        # own objective composition.

    amp_ctx = AmpTuneContext(
        amp=amp,
        snr_estimator=snr_estimator,
        bounds=bounds,
        power_cost_weights=power_cost_weights,
        lambda_power=lambda_power,
        n_init=n_init,
        n_iter=n_iter,
        random_state=random_state,
        logger=logger,
    )
    calibration = amp_calibrate(amp_ctx, numpoints=calibration_numpoints)
    result = amp_optimize_snr(amp_ctx, calibration)
    return amp_ctx, result
