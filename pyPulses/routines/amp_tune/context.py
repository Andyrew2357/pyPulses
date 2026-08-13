"""
pyPulses.routines.amp_tune.context
===================================
AmpTuneContext: the general SNR-tuning framework's context object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import logging

from ...devices.HEMT import AmplifierLike


@dataclass
class AmpTuneContext:
    """
    Context for the general SNR-tuning framework (amp_calibrate +
    amp_optimize_snr). Holds the amplifier being tuned, the
    experiment-specific SNR estimator, and the search configuration.

    Parameters
    ----------
    amp : AmplifierLike
        The amplifier to tune. Any object satisfying the AmplifierLike
        protocol (devices/HEMT.py) works -- this framework is not
        specific to HEMTCommonSource.
    snr_estimator : Callable[[], float]
        Experiment-specific SNR measurement. Takes no arguments and
        returns a single scalar to maximize; the caller owns all the
        details of how that number is obtained (averaging, lock-in
        reads, etc.).
    free_params : list of str, optional
        Which of amp's bias channels to search over. Defaults to
        whichever channels amp.free_channels() reports as sweepable.
    bounds : dict, optional
        {param_name: (min, max)} search bounds. Required before calling
        amp_calibrate or amp_optimize_snr (see require_bounds()) -- there
        is no safe default, since it depends entirely on the specific
        hardware's travel range and the transistor's safe window.
    power_cost_weights : dict, optional
        {stage_name: weight} passed to amp.power_cost(), used both as a
        diagnostic and as the search objective's soft power penalty (see
        lambda_power).
    lambda_power : float, default=0.0
        Trade-off coefficient: objective = snr - lambda_power *
        amp.power_cost(power_cost_weights). 0.0 means power plays no role
        in the search objective, though the hard per-stage caps
        amp.check_safe() enforces still apply regardless of this value.
    n_init : int, default=8
        Number of purely-random (not acquisition-driven) feasible
        candidates amp_optimize_snr evaluates before switching to the
        Bayesian optimizer's own suggestions.
    n_iter : int, default=40
        Total number of bias points amp_optimize_snr evaluates, including
        the n_init random ones.
    random_state : int, optional
        Seed for both the random-candidate draws and the Bayesian
        optimizer, for reproducibility.
    logger : Logger, optional
    """

    amp: AmplifierLike
    snr_estimator: Callable[[], float]

    free_params: List[str] | None = None
    bounds: Dict[str, Tuple[float, float]] | None = None

    power_cost_weights: Dict[str, float] | None = None
    lambda_power: float = 0.0

    n_init: int = 8
    n_iter: int = 40
    random_state: int | None = None

    logger: logging.Logger | None = None

    def __post_init__(self):
        if self.free_params is None:
            self.free_params = list(self.amp.free_channels().keys())

    """Logging"""

    def log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(*args, **kwargs)

    """Validation"""

    def require_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Validate that self.bounds covers every entry in self.free_params,
        and return the bounds restricted to just those entries (in
        free_params order). Raises ValueError otherwise.
        """
        if self.bounds is None:
            raise ValueError(
                "AmpTuneContext.bounds must be set before calibrating or "
                "optimizing -- there is no safe default search range."
            )
        missing = [p for p in self.free_params if p not in self.bounds]
        if missing:
            raise ValueError(
                f"AmpTuneContext.bounds is missing entries for: {missing}"
            )
        return {name: self.bounds[name] for name in self.free_params}
