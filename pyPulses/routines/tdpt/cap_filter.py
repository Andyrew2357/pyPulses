from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import TDPTContext

from ...utils.kalman import kalman

import numpy as np

"""
================================================================================
Capacitance Filter
================================================================================
"""

class CapacitanceFilter():
    """
    Manages a Kalman filter for G, Cac, and dCac, where:
      G    — effective gain from balance point to scope
      Cac  — AC capacitance ratio at the current operating point
      dCac — change in Cac upon an excitation change

    State vector: x = [G, Cac, dCac]^T

    Process models:
      balance_change:    state unchanged, P += balance_change_Q
      excitation_change: Cac += dCac,     P += excitation_change_Q
                         Jacobian has F[1,2] = 1 (dCac feeds into Cac)

    Measurement model:
      z = G * (Yamp + Cac * Xamp)
      which equals zero at the exact balance point Yamp = -Cac * Xamp.
    """

    def __init__(self, ctx: 'TDPTContext | None' = None):

        self.ctx = ctx

        self.is_positive_initialized = False
        self.is_negative_initialized = False
        self._positive_init_params: dict | None = None
        self._negative_init_params: dict | None = None

        self.kfilter: kalman | None = None
        self.balance_change_Q: np.ndarray | None = None
        self.excitation_change_Q: np.ndarray | None = None

        # Tracked excitation state, updated by TDPTContext.X1/Y1 accessors
        self.Xamp: float | None = None
        self.Yamp: float | None = None

    def add_context(self, ctx: 'TDPTContext'):
        self.ctx = ctx
        ctx.cap_filter = self

    def _initialize(self, init_params: dict):
        self.balance_change_Q    = init_params['balance_change_Q']
        self.excitation_change_Q = init_params['excitation_change_Q']
        self.kfilter = kalman(
            f_F = self._f_F,
            h_H = self._h_H,
            x   = init_params['x_init'],
            P   = init_params['P_init'],
        )

    def positive_initialize(self):
        if self.is_positive_initialized:
            return
        if self._positive_init_params is None:
            raise RuntimeError(
                "CapacitanceFilter has no positive initialization parameters."
            )
        self._initialize(self._positive_init_params)
        self.is_negative_initialized = False
        self.is_positive_initialized = True

    def negative_initialize(self):
        if self.is_negative_initialized:
            return
        if self._negative_init_params is None:
            raise RuntimeError(
                "CapacitanceFilter has no negative initialization parameters."
            )
        self._initialize(self._negative_init_params)
        self.is_positive_initialized = False
        self.is_negative_initialized = True

    def set_positive_init_params(self,
        bal_change_Q: np.ndarray,
        exc_change_Q: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
    ):
        self._positive_init_params = {
            'balance_change_Q':    bal_change_Q.reshape((3, 3)),
            'excitation_change_Q': exc_change_Q.reshape((3, 3)),
            'x_init':              x.reshape((3, 1)), # [G, Cac, dCac]
            'P_init':              np.diag(P.flatten()).reshape((3, 3)),
        }

    def set_negative_init_params(self,
        bal_change_Q: np.ndarray,
        exc_change_Q: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
    ):
        self._negative_init_params = {
            'balance_change_Q':    bal_change_Q.reshape((3, 3)),
            'excitation_change_Q': exc_change_Q.reshape((3, 3)),
            'x_init':              x.reshape((3, 1)), # [G, Cac, dCac]
            'P_init':              np.diag(P.flatten()).reshape((3, 3)),
        }

    """Kalman Filter Functionality"""

    def _f_F(self, X: np.ndarray, balance_change: bool = True):
        """
        Process model.

        Balance change: identity on state, Q = balance_change_Q.
        Excitation change: Cac += dCac, Q = excitation_change_Q,
                           F[1,2] = 1 reflects dCac coupling into Cac.

        State ordering: [G, Cac, dCac] — indices 0, 1, 2.
        """
        F = np.eye(3)
        if balance_change:
            Q = self.balance_change_Q
        else:
            X[1, 0] += X[2, 0] # Cac += dCac
            Q = self.excitation_change_Q
            F[1, 2] = 1.0 # dCac feeds into Cac in Jacobian
        return X, F, Q

    def _h_H(self, X: np.ndarray, Xamp: float, Yamp: float):
        """
        Measurement model.

        The observed capacitive jump at the current balance point is:
          z = G * (Yamp + Cac * Xamp)

        This is zero at exact balance (Yamp = -Cac * Xamp).

        State ordering: [G, Cac, dCac] — indices 0, 1, 2.
        """
        G, Cac, dCac = X.flatten()
        z = np.array([[G * (Yamp + Cac * Xamp)]])
        H = np.array([[Yamp + Cac * Xamp, G * Xamp, 0.0]])
        return z, H

    def predict(self, balance_change: bool = True):
        """
        Kalman prediction step.

        Called with balance_change=False once at the start of a new
        phase space point (excitation change), then with
        balance_change=True before each control correction within
        the balance loop.
        """
        self.kfilter.predict(balance_change)

    def update(self,
        dQ: float,
        R: float,
        Xamp: float | None = None,
        Yamp: float | None = None,
    ):
        """
        Kalman update step.

        Parameters
        ----------
        dQ : float
            Observed capacitive jump at the start of the excitation pulse.
        R : float
            Variance of dQ.
        Xamp : float, optional
            X excitation amplitude. Uses tracked self.Xamp if not provided.
        Yamp : float, optional
            Y balance amplitude. Uses tracked self.Yamp if not provided.
        """
        if Xamp is None:
            Xamp = self.Xamp
        if Yamp is None:
            Yamp = self.Yamp

        pG, pCac, _ = self.kfilter.x.flatten()
        self.kfilter.update(np.array([[dQ]]), np.array([[R]]), Xamp, Yamp)
        G, Cac, _ = self.kfilter.x.flatten()

        # Guard against unphysical sign changes in G or Cac.
        # These cannot occur from a smooth physical change and indicate
        # a filter update that has gone badly wrong.
        if pCac * Cac < 0:
            if self.ctx:
                self.ctx.log(
                    f"WARNING: Ignoring requested sign change in Cac "
                    f"({pCac:.5e} → {Cac:.5e})."
                )
            self.kfilter.x[1, 0] = pCac

        if pG * G < 0:
            if self.ctx:
                self.ctx.log(
                    f"WARNING: Ignoring requested sign change in G "
                    f"({pG:.5e} → {G:.5e})."
                )
            self.kfilter.x[0, 0] = pG

    """Accessors"""

    def get_G(self) -> float:
        return float(self.kfilter.x[0, 0])

    def get_Cac(self) -> float:
        return float(self.kfilter.x[1, 0])

    def get_state(self) -> tuple[float, float, float]:
        return tuple(self.kfilter.x.flatten())

    def widen_uncertainty(self, factor: float):
        """
        Scale P by factor. Used by map-level health monitor to force
        re-learning after a corrective intervention.
        """
        self.kfilter.P *= factor

    def reset_uncertainty(self, P: np.ndarray):
        """Hard reset of P to a specified matrix."""
        self.kfilter.P = P.copy()

    """Serialization"""

    def _serialize_state(self) -> dict:
        def _to_list(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

        def _params_to_list(params):
            if params is None:
                return None
            return {k: _to_list(v) for k, v in params.items()}

        return {
            'CapacitanceFilter_kwargs': dict(
                is_positive_initialized  = self.is_positive_initialized,
                is_negative_initialized  = self.is_negative_initialized,
                _positive_init_params    = _params_to_list(self._positive_init_params),
                _negative_init_params    = _params_to_list(self._negative_init_params),
                balance_change_Q         = _to_list(self.balance_change_Q),
                excitation_change_Q      = _to_list(self.excitation_change_Q),
                Xamp                     = self.Xamp,
                Yamp                     = self.Yamp,
                kfilter_x                = self.kfilter.x.tolist() if self.kfilter else None,
                kfilter_P                = self.kfilter.P.tolist() if self.kfilter else None,
            )
        }

    def _deserialize_state(self, state: dict):
        def _to_array(val):
            if val is None:
                return None
            return np.array(val)

        def _restore_params(params):
            if params is None:
                return None
            array_keys = {'balance_change_Q', 'excitation_change_Q', 'x_init', 'P_init'}
            return {k: _to_array(v) if k in array_keys else v
                    for k, v in params.items()}

        self.is_positive_initialized = state.get('is_positive_initialized', False)
        self.is_negative_initialized = state.get('is_negative_initialized', False)
        self._positive_init_params   = _restore_params(state.get('_positive_init_params'))
        self._negative_init_params   = _restore_params(state.get('_negative_init_params'))

        # Re-initialize from init_params to restore kfilter structure
        # and Q matrices, then overwrite with saved live state.
        if self.is_positive_initialized:
            self.is_positive_initialized = False
            self.positive_initialize()
        elif self.is_negative_initialized:
            self.is_negative_initialized = False
            self.negative_initialize()

        if self.kfilter is not None:
            if state.get('kfilter_x') is not None:
                self.kfilter.x = np.array(state['kfilter_x'])
            if state.get('kfilter_P') is not None:
                self.kfilter.P = np.array(state['kfilter_P'])

        # Overwrite Q matrices with saved live values in case they
        # differ from initialization defaults.
        if state.get('balance_change_Q') is not None:
            self.balance_change_Q = np.array(state['balance_change_Q'])
        if state.get('excitation_change_Q') is not None:
            self.excitation_change_Q = np.array(state['excitation_change_Q'])

        self.Xamp = state.get('Xamp')
        self.Yamp = state.get('Yamp')

    def __str__(self) -> str:
        G, Cac, dCac = self.kfilter.x.flatten()
        PG, PCac, PdCac = self.kfilter.P.diagonal()
        P = self.kfilter.P
        return (
            f"G = {G:.5e} +- {np.sqrt(PG):.5e}\n"
            f"Cac = {Cac:.5e} +- {np.sqrt(PCac):.5e}\n"
            f"dCac = {dCac:.5e} +- {np.sqrt(PdCac):.5e}\n"
            f"     Covariance: ┏                                      ┓\n"
            f"                 ┃ {f"{P[0,0]:.5e}":<12}{f"{P[0,1]:.5e}":>12}{f"{P[0,2]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{P[1,0]:.5e}":<12}{f"{P[1,1]:.5e}":>12}{f"{P[1,2]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{P[2,0]:.5e}":<12}{f"{P[2,1]:.5e}":>12}{f"{P[2,2]:.5e}":>12} ┃\n"
            f"                 ┗                                      ┛"
        )