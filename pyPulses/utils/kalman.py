import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class kalman:
    """
    Implementation of a Kalman Filter compatible with the nonlinear EKF variety. 
    A Kalman filter is intended to optimally converge on the true state by 
    calculating the optimal mixing weights of extrapolated states (there must be 
    a model for the process) and new measurements. The extended Kalman filter 
    (EKF) is not optimal due to the linearization, and it tends to underestimate 
    covariance. It may also diverge if the measurement and process noise are not 
    set appropriately. 

    Extraptolation Equations
        State Extrapolation:        x_{k|k-1} = f(x_{k|k}, u_k)
        Covariance Extrapolation:   P_{k|k-1} = F_k P_{k-1|k-1} (F_k)^T + Q_{k-1}

    Update Equations
        Measurement Residual:       y_k = z_k - h(x_{k|k-1})
        Residual Covariance:        S_k = H_k P_{k|k-1} (H_k)^T + R_k
        Kalman Gain:                K_k = P_{k|k-1} (H_k)^T (S_k)^{-1}
        State Prediction:           x_{k|k} = x_{k|k-1} + K_k y_k
        Covariance Prediction:      P_{k|k} = (I - K_k H_k) P_{k|k-1}

    Where F_k and H_k are defined by:
        State Transition Matrix:    F_k = (df/dx)|_{x_{k|k-1}, u_{k}}
        Observation Matrix:         H_k = (dh/dx)|_{x{k|k-1}}

    Other variables:
        Estimated State at k:           x_{k|k}
        Extrapolated State at k-1:      x_{k|k-1}
        Est. Covariance Matrix at k:    P_{k|k}
        Est. Covariance Extra. at k:    P_{k|k-1}
        Process noise:                  Q_{k-1}

    Parameters
    ----------
    f_F : Callable
        (x, u) -> (x, F, Q)
    h_H : Callable
        (x) -> (z_pred, H)
    x : np.ndarray
        state
    P : np.ndarray
        covariance matrix
    """
    f_F : Callable[..., 
            Tuple[np.ndarray, np.ndarray, np.ndarray]]  # (x, u) --> (x, F, Q)
    h_H : Callable[...,
            Tuple[np.ndarray, np.ndarray]]              # x --> (z_pred, H)
    x   : np.ndarray    # state
    P   : np.ndarray    # covariance matrix

    def predict(self, *args):
        """
        Predict the future state updating x and P accordingly.
        
        Parameters
        ----------
        *args
            args passed to `f_F` (ie. `f_F(self.x, *args)`).
        """

        self.x, F, Q = self.f_F(self.x, *args)  # extrapolate x, compute F, Q
        self.P = F @ self.P @ F.T + Q           # extrapolate covariance

    def update(self, z: np.ndarray, R:np.ndarray, *args):
        """
        Optimally mix measured and predicted states.
        
        Parameters
        ----------
        z : np.ndarray
            measurement.
        R : np.ndarray
            covariance matrix of the measurement.
        *args
            args passed to `h_H` (ie. `h_H(self.x, *args)`).
        """

        z_pred, H = self.h_H(self.x, *args) # compute z_pred, H
        y = z - z_pred                      # measurement residual (y)
        S = H @ self.P @ H.T + R            # residual covariance (S)
        K = self.P @ H.T @ inv(S)           # Kalman gain (K)
        
        self.x += K @ y                 # update state
        self.P -= K @ H @ self.P        # update covariance
