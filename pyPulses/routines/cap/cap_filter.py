from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import CapContext

import numpy as np
from .config import PROCESS_NOISE_COEFF

def _extrap(x: list[float], order: int) -> float:
    arr   = np.array(x)
    coeff = np.polyfit(np.arange(len(arr)), arr, min(order, len(arr) - 1))
    return float(np.poly1d(coeff)(len(arr)))

def _rth_cov_to_xy_cov(Q_rth: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Transform a covariance matrix from polar to Cartesian coordinates.

    Parameters
    ----------
    Q_rth : (2, 2) ndarray
        Covariance in (r, theta) space.
    v : (2,) ndarray
        Current state vector [X, Y] at which to evaluate the Jacobian.

    Returns
    -------
    (2, 2) ndarray
        Covariance in (X, Y) space.
    """
    r  = np.sqrt(v[0] ** 2 + v[1] ** 2)
    th = np.arctan2(v[1], v[0])
    c  = np.cos(th)
    s  = np.sin(th)
    J  = np.array([[c, -r * s],
                   [s,  r * c]])
    return J @ Q_rth @ J.T

class CapFilter():
    """
    Scalar Kalman filter for the complex bridge gain A = (X, Y).

    The state vector is A = [X, Y]^T, where X + iY is the effective
    small-signal gain of the capacitance bridge: given a step dv = (dvx, dvy)
    in Vstd, the predicted change in lock-in reading is

        dL = [[X, -Y], [Y, X]] @ dv

    Process model
    -------------
    State is expected to be roughly constant. Process noise is constructed
    assuming uncorrelated fractional errors in the modulus and phase of A:

        Q = PROCESS_NOISE_COEFF^2 * J @ diag(|A|^2, (2π)^2) @ J^T

    where J is the polar-to-Cartesian Jacobian evaluated at the current A.

    Measurement model
    -----------------
    Direct observation of dL given a known dv:

        z = [[X, -Y], [Y, X]] @ dv
        H = [[dvx, -dvy], [dvy, dvx]]
    """

    def __init__(self, ctx: 'CapContext | None' = None):
        self.ctx = ctx

        self._A: np.ndarray | None = None   # state vector, shape (2,)
        self._P: np.ndarray | None = None   # covariance, shape (2, 2)

        self._init_params: dict | None = None

        # Full 2x2 sensitivity matrix from three-point balance, if available.
        # K maps (dVc, dVr) -> (dLX, dLY): K = [[Kc1, Kr1], [Kc2, Kr2]].
        # Set by cap_initialize_filter when initialized from ThreePointBalanceResult.
        self.K_matrix: np.ndarray | None = None

    def add_context(self, ctx: 'CapContext'):
        self.ctx = ctx
        ctx.cap_filter = self

    """Initialization"""

    def set_init_params(self,
        A: np.ndarray,
        P: np.ndarray,
    ):
        """
        Store initialization parameters. Call initialize() afterwards to
        actually set the live filter state.

        Parameters
        ----------
        A : (2,) ndarray
            Initial complex gain estimate [X, Y].
        P : (2, 2) ndarray
            Initial covariance matrix.
        """
        self._init_params = {
            'A': np.asarray(A, dtype=float).reshape(2),
            'P': np.asarray(P, dtype=float).reshape(2, 2),
        }

    def initialize(self):
        """Apply stored init params to the live filter state."""
        if self._init_params is None:
            raise RuntimeError(
                "CapFilter has no initialization parameters. "
                "Call set_init_params() first."
            )
        self._A = self._init_params['A'].copy()
        self._P = self._init_params['P'].copy()

    """Kalman mechanics"""

    def _process_noise(self) -> np.ndarray:
        """
        Build the process noise matrix Q at the current state A.
        """
        Q_rth = np.diag([
            self._A[0] ** 2 + self._A[1] ** 2,
            (2 * np.pi) ** 2,
        ])
        return PROCESS_NOISE_COEFF ** 2 * _rth_cov_to_xy_cov(Q_rth, self._A)

    def predict(self):
        """
        Kalman prediction step (identity state transition, additive Q).
        Call before each measurement.
        """
        self._P = self._P + self._process_noise()

    def update(self,
        dL: np.ndarray,
        R:  np.ndarray,
        dv: np.ndarray,
    ):
        """
        Kalman update step. Direct observation of dL = H @ A.

        Parameters
        ----------
        dL : (2,) ndarray
            Observed change in lock-in reading [dLX, dLY].
        R : (2, 2) ndarray
            Measurement noise covariance of dL.
        dv : (2,) ndarray
            Vstd step [dvx, dvy] that caused the observed dL.
        """
        dvx, dvy = float(dv[0]), float(dv[1])
        H = np.array([[dvx, -dvy],
                      [dvy,  dvx]])

        S     = H @ self._P @ H.T + R
        K     = self._P @ H.T @ np.linalg.inv(S)
        innov = np.asarray(dL, dtype=float).reshape(2) - H @ self._A

        self._A = self._A + K @ innov
        self._P = (np.eye(2) - K @ H) @ self._P

    """Accessors"""

    def get_A(self) -> np.ndarray:
        """Return the current gain estimate as a (2,) array [X, Y]."""
        return self._A.copy()

    def get_P(self) -> np.ndarray:
        """Return the current covariance matrix as a (2, 2) array."""
        return self._P.copy()

    def get_A_complex(self) -> complex:
        """Return the current gain estimate as a complex number X + iY."""
        return complex(self._A[0], self._A[1])

    """Serialization"""

    def _serialize_state(self) -> dict:
        def _arr(a):
            return a.tolist() if isinstance(a, np.ndarray) else a
 
        init = None
        if self._init_params is not None:
            init = {k: _arr(v) for k, v in self._init_params.items()}
 
        return {
            'CapFilter_kwargs': dict(
                _init_params = init,
                A            = _arr(self._A),
                P            = _arr(self._P),
                K_matrix     = _arr(self.K_matrix),
            )
        }
 
    def _deserialize_state(self, state: dict):
        raw_init = state.get('_init_params')
        if raw_init is not None:
            self._init_params = {
                'A': np.array(raw_init['A']),
                'P': np.array(raw_init['P']),
            }
        else:
            self._init_params = None
 
        if state.get('A') is not None:
            self._A = np.array(state['A'])
        if state.get('P') is not None:
            self._P = np.array(state['P'])
        K = state.get('K_matrix')
        self.K_matrix = np.array(K) if K is not None else None

    def __str__(self) -> str:
        X, Y = self._A
        P    = self._P
        return (
            f"A = {X:.5e} + {Y:.5e}i\n"
            f"     Covariance: ┏                          ┓\n"
            f"                 ┃ {f"{P[0,0]:.5e}":<12}{f"{P[0,1]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{P[1,0]:.5e}":<12}{f"{P[1,1]:.5e}":>12} ┃\n"
            f"                 ┗                          ┛"
        )

class CapExtrapolator:
    """
    Maintains a rolling history of balance points (Vb_x, Vb_y) in Vstd
    space and extrapolates the next predicted balance point via a
    polynomial fit of configurable order and support window.
    """

    def __init__(self,
        support: int = 7,
        order:   int = 2,
        Vstd_range: float | None = None,
    ):
        self.support    = support
        self.order      = order
        self.Vstd_range = Vstd_range

        self._x_hist: list[float] = []
        self._y_hist: list[float] = []

    def push(self, Vbx: float, Vby: float):
        """Record a balance point."""
        self._x_hist.append(float(Vbx))
        self._y_hist.append(float(Vby))

    def extrapolate(self) -> tuple[float, float] | None:
        """
        Extrapolate the next balance point. Returns None if no history.
        Result is clamped to Vstd_range if set.
        """
        if not self._x_hist:
            return None

        xh = self._x_hist[-self.support:]
        yh = self._y_hist[-self.support:]

        Vbx = _extrap(xh, self.order)
        Vby = _extrap(yh, self.order)

        if self.Vstd_range is not None:
            r = np.sqrt(Vbx ** 2 + Vby ** 2)
            if r > self.Vstd_range:
                scale = self.Vstd_range / r
                Vbx  *= scale
                Vby  *= scale

        return float(Vbx), float(Vby)

    def clear(self):
        self._x_hist = []
        self._y_hist = []

    """Serialization"""
    
    def _serialize_state(self) -> dict:
        return {
            'CapExtrapolator_kwargs': dict(
                support    = self.support,
                order      = self.order,
                Vstd_range = self.Vstd_range,
                x_hist     = list(self._x_hist),
                y_hist     = list(self._y_hist),
            )
        }

    def _deserialize_state(self, state: dict):
        self.support    = state.get('support',    self.support)
        self.order      = state.get('order',      self.order)
        self.Vstd_range = state.get('Vstd_range', self.Vstd_range)
        self._x_hist    = [float(v) for v in state.get('x_hist', [])]
        self._y_hist    = [float(v) for v in state.get('y_hist', [])]

    def __str__(self) -> str:
        return (
            f"Vbx history: {self._x_hist[-self.support:]}\n"
            f"Vby history: {self._y_hist[-self.support:]}\n"
        )