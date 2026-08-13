"""
Thin wrapper isolating the bayes_opt (bayesian-optimization PyPI package)
ask/tell API behind an internal interface, so optimize.py has a single
place to change if the underlying library is ever swapped.
"""

from __future__ import annotations

from typing import Dict, Tuple

from bayes_opt import BayesianOptimization


class AskTellOptimizer:
    """
    Minimal ask/tell wrapper around bayes_opt.BayesianOptimization. Bounds
    are fixed at construction; call ask() to get the next candidate,
    evaluate its objective externally (a live hardware measurement, in
    this framework's case), then tell() the observed value.

    bayes_opt's own constraint support is not used here -- hard
    feasibility (transistor safety window, per-stage power caps) is
    enforced by the caller via the calibration surrogate before any
    candidate reaches ask()/tell(), so this wrapper stays a plain
    unconstrained box-bounded optimizer.
    """

    def __init__(self,
        bounds: Dict[str, Tuple[float, float]],
        random_state: int | None = None,
    ):
        self._opt = BayesianOptimization(
            f=None,
            pbounds=bounds,
            random_state=random_state,
            verbose=0,
            allow_duplicate_points=True,
        )

    def ask(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self._opt.suggest().items()}

    def tell(self, params: Dict[str, float], objective: float) -> None:
        self._opt.register(params=params, target=float(objective))

    @property
    def best(self) -> Dict[str, object] | None:
        """{'params': {...}, 'target': ...} for the best point told so far,
        or None if nothing has been told yet."""
        return self._opt.max
