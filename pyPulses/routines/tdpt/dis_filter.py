from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import TDPTContext

from .config import (
    HEURISTIC_NOISE_MULT,
    STRONG_UPDATE_R_MULT,
    LARGE_RESET_P_MULT,
    RELIABILITY_FACTOR,
)
import numpy as np

def _extrap(x: np.ndarray, order: int) -> float:
    coeff = np.polyfit(np.arange(len(x)), x, min(order, len(x) - 1))
    poly = np.poly1d(coeff)
    return float(poly(len(x)))

class DischargeFilter():
    """
    Manages a scalar Kalman filter for dMdW, the sensitivity of the
    post-discharge slope to discharge pulse width.

    This is a trivial 1D filter. The state is just dMdW. The measurement
    model is a direct observation of dMdW computed from consecutive
    (W, M) pairs during the balance loop. Both the balance change and
    excitation change propagators are identity on the state — they only
    differ in the process noise added to P.
    """

    def __init__(self, ctx: 'TDPTContext | None' = None):

        self.ctx = ctx

        self.is_positive_initialized = False
        self.is_negative_initialized = False
        self._positive_init_params: dict | None = None
        self._negative_init_params: dict | None = None

        self.use_heuristic_process_noise = True
        self._previous_dMdW: float | None = None

        self.balance_change_Q: float | None = None
        self.excitation_change_Q: float | None = None

        self.dMdW: float | None = None
        self.P: float | None = None

        self._home_W: float = None

    def add_context(self, ctx: 'TDPTContext'):
        self.ctx = ctx
        ctx.dis_filter = self

    def _initialize(self, init_params: dict):
        self.balance_change_Q    = init_params['balance_change_Q']
        self.excitation_change_Q = init_params['excitation_change_Q']
        self.dMdW                = init_params['dMdW']
        self._previous_dMdW      = self.dMdW
        self.P                   = init_params['P']
        self._home_W             = init_params['W0']

    def positive_initialize(self):
        if self.is_positive_initialized:
            return
        if self._positive_init_params is None:
            raise RuntimeError(
                "DischargeFilter has no positive initialization parameters."
            )
        self._initialize(self._positive_init_params)
        self.is_negative_initialized = False
        self.is_positive_initialized = True

    def negative_initialize(self):
        if self.is_negative_initialized:
            return
        if self._negative_init_params is None:
            raise RuntimeError(
                "DischargeFilter has no negative initialization parameters."
            )
        self._initialize(self._negative_init_params)
        self.is_positive_initialized = False
        self.is_negative_initialized = True

    def set_positive_init_params(self,
        bal_change_Q: float,
        exc_change_Q: float,
        dMdW: float,
        P: float,
    ):
        self._positive_init_params = {
            'balance_change_Q':    bal_change_Q,
            'excitation_change_Q': exc_change_Q,
            'dMdW':                dMdW,
            'P':                   P,
        }

    def set_negative_init_params(self,
        bal_change_Q: float,
        exc_change_Q: float,
        dMdW: float,
        P: float,
    ):
        self._negative_init_params = {
            'balance_change_Q':    bal_change_Q,
            'excitation_change_Q': exc_change_Q,
            'dMdW':                dMdW,
            'P':                   P,
        }

    """Kalman Filter Functionality"""

    def predict(self, balance_change: bool = True):
        """
        Kalman prediction step. Both propagators are identity on the
        state, so this only adds process noise to P.

        For balance changes, heuristic noise is used if enabled:
          Q = HEURISTIC_NOISE_MULT * ((dMdW - prev_dMdW)^2 + dMdW^2)
        This makes the process noise scale with how much dMdW has been
        changing, which is more adaptive than a fixed Q.

        For excitation changes, the fixed excitation_change_Q is always
        used since we have no heuristic basis for excitation noise.
        """
        if balance_change:
            if self.use_heuristic_process_noise:
                self.P += HEURISTIC_NOISE_MULT * (
                    (self.dMdW - self._previous_dMdW) ** 2 + self.dMdW ** 2
                )
            else:
                self.P += self.balance_change_Q
        else:
            self.P += self.excitation_change_Q

    def update(self, o_dMdW: float, R: float):
        """
        Kalman update step. Direct observation of dMdW.

        Parameters
        ----------
        o_dMdW : float
            Observed dMdW computed from consecutive (W, M) pairs:
            o_dMdW = (M_new - M_prev) / (W_new - W_prev)
        R : float
            Variance of o_dMdW.
        """
        self._previous_dMdW = self.dMdW
        K          = self.P / (self.P + R)
        self.dMdW += K * (o_dMdW - self.dMdW)
        self.P    *= (1 - K)

    def strong_update(self, o_dMdW: float, R: float | None = None):
        self._previous_dMdW = self.dMdW
        if R is None:
            R = STRONG_UPDATE_R_MULT * self.dMdW ** 2
        K          = self.P / (self.P + R)
        self.dMdW += K * (o_dMdW - self.dMdW)
        self.P    *= (1 - K)

    def reset_uncertainty(self, mult: float = LARGE_RESET_P_MULT):
        """
        Reset P to according to a multiple of dM/dW^2
        """
        self.P = mult * self.dMdW ** 2

    def hard_reset_dMdW(self, dMdW: float):
        self.dMdW           = dMdW
        self._previous_dMdW = dMdW
        self.P              = LARGE_RESET_P_MULT * dMdW ** 2

    """Accessors"""

    def get_dMdW(self) -> float:
        return self.dMdW

    def is_reliable(self) -> bool:
        """
        Returns True if P is small enough relative to dMdW^2 that
        the slope estimate can be trusted for a Newton step.
        """
        return self.P < RELIABILITY_FACTOR * self.dMdW ** 2

    """Serialization"""

    def _serialize_state(self) -> dict:
        return {
            'DischargeFilter_kwargs': dict(
                is_positive_initialized  = self.is_positive_initialized,
                is_negative_initialized  = self.is_negative_initialized,
                _positive_init_params    = self._positive_init_params,
                _negative_init_params    = self._negative_init_params,
                use_heuristic_process_noise = self.use_heuristic_process_noise,
                _previous_dMdW           = self._previous_dMdW,
                balance_change_Q         = self.balance_change_Q,
                excitation_change_Q      = self.excitation_change_Q,
                dMdW                     = self.dMdW,
                P                        = self.P,
            )
        }

    def _deserialize_state(self, state: dict):
        self._positive_init_params  = state.get('_positive_init_params')
        self._negative_init_params  = state.get('_negative_init_params')
        self.is_positive_initialized = state.get('is_positive_initialized', False)
        self.is_negative_initialized = state.get('is_negative_initialized', False)

        if self.is_positive_initialized:
            self.is_positive_initialized = False
            self.positive_initialize()
        elif self.is_negative_initialized:
            self.is_negative_initialized = False
            self.negative_initialize()

        # Overwrite with saved live state
        self.use_heuristic_process_noise = state.get(
            'use_heuristic_process_noise', True
        )
        self._previous_dMdW  = state.get('_previous_dMdW')
        self.balance_change_Q  = state.get('balance_change_Q')
        self.excitation_change_Q = state.get('excitation_change_Q')
        self.dMdW            = state.get('dMdW')
        self.P               = state.get('P')

    def __str__(self) -> str:
        return (
            f"dM/dW = {self.dMdW:.5e} +- {np.sqrt(self.P):.5e}\n"
        )


class DischargeExtrapolator():
    """
    Maintains histories of W0 (discharge pulse width balance points)
    and Y2/X2 (discharge amplitude ratio balance points) across calls,
    and extrapolates the next predicted value from each history using a
    polynomial fit of configurable order and support window.
    """

    def __init__(self,
        extrap_support: int = 7,
        extrap_order: int   = 2,
        W_min: float | None = None,
        W_max: float | None = None,
    ):
        self.extrap_support = extrap_support
        self.extrap_order   = extrap_order
        self.W_min          = W_min
        self.W_max          = W_max

        self.W0_history:    list[float] = []
        self.ratio_history: list[float] = []   # Y2/X2 history

    def push_W0(self, W0: float):
        self.W0_history.append(float(W0))

    def push_Y2(self, Y2: float, X2: float):
        self.ratio_history.append(float(Y2 / X2))

    def extrapolate_W0(self) -> float | None:
        """
        Extrapolate the next W0 from history. Returns None if no
        history is available, in which case the caller should fall
        back to the current W setting.
        """
        if len(self.W0_history) == 0:
            return None
        self.W0_history = self.W0_history[-self.extrap_support:]
        # This meanders way too much if you fit to anything other than a constant
        # For safety and to avoid persistent railing, I just take a mean
        W0 = float(np.mean(self.W0_history))

        if self.W_min is not None and self.W_max is not None:
            W0 = float(np.clip(W0, self.W_min, self.W_max))
        return W0

    def extrapolate_Y2(self, X2: float) -> float | None:
        """
        Extrapolate the next Y2 from the Y2/X2 ratio history,
        then multiply by X2 to recover the absolute Y2 value.
        Returns None if no history is available, in which case
        the caller should fall back to -Cac * X2.
        """
        if len(self.ratio_history) == 0:
            return None
        self.ratio_history = self.ratio_history[-self.extrap_support:]
        ratio = _extrap(self.ratio_history, self.extrap_order)
        return float(ratio * X2)

    def clear(self):
        self.W0_history    = []
        self.ratio_history = []

    """Serialization"""

    def _serialize_state(self) -> dict:
        return {
            'DischargeExtrapolator_kwargs': dict(
                extrap_support = self.extrap_support,
                extrap_order   = self.extrap_order,
                W_min          = self.W_min,
                W_max          = self.W_max,
                W0_history     = [float(v) for v in self.W0_history],
                ratio_history  = [float(v) for v in self.ratio_history],
            )
        }

    def _deserialize_state(self, state: dict):
        self.extrap_support = state.get('extrap_support', self.extrap_support)
        self.extrap_order   = state.get('extrap_order',   self.extrap_order)
        self.W_min          = state.get('W_min',          self.W_min)
        self.W_max          = state.get('W_max',          self.W_max)
        self.W0_history     = [float(v) for v in state.get('W0_history',    [])]
        self.ratio_history  = [float(v) for v in state.get('ratio_history', [])]

    def __str__(self) -> str:
        return (
            f"W0 history:    {[float(v) for v in self.W0_history]}\n"
            f"Y2/X2 history: {[float(v) for v in self.ratio_history]}\n"
        )