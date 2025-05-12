
from ..devices import sr865a
from ..utils import kalman

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections import deque

def extrap(x: np.ndarray, order: int) -> float:
    coeff = np.polyfit(np.arange(len(x)), x, min(order, len(x) - 1))        
    poly = np.poly1d(coeff)
    return poly(len(x))

class KFilter():
    def __init__(self, support: int, order: int, sensitivity_increment: int, 
                 A: np.ndarray, P: np.ndarray):
        
        self.order = order

        # History of balance points in terms of r and theta for Vstd
        self.r_hist     = deque(maxlen = support)
        self.theta_hist = deque(maxlen = support)
        
        # optimal lock-in sensitivity and input range
        self.lockin_sensitivity = None
        self.lockin_input_range = None

        # we periodically increase the sensitivity
        self.use_count = 0
        self.sensitivity_increment = sensitivity_increment

        """
        the state vector x is (R,Θ)
        R is the modulus of the effective gain of the bridge setup.
        Θ is the effective phase offset of the bridge setup.
        Together, these characterize the complex effective gain of the bridge.

        Given a step (dr, dθ) on Vstd, the response on the lockin will be:
                                    
                                (R * dr, Θ + dθ)

        f(x, u) is the predicted evolution of the state (x). In this case, we
        expect minimal variation in the effective gain, so our model for state
        evolution is f(x) = x. We also need the Jacobian, F = 1. F must also
        return a process noise covariance matrix. In this case we assume an
        uncertainty in R of 0.01*R and in Θ of 0.01*2pi, which we take to be
        uncorrelated. This reflects the expected variability in the gain as we
        change Vstd, which should be small.

        h(x, v) is the predicted change in lock-in reading given a step
        dv = dVstd = (dr, dθ). This is given by (R * dr, Θ + dθ), as discussed.
        The Jacobian is H = ((R, 0), (0, 1)).

        For clarity, I label the state vector as A and the extra arguments for h
        as dv. I also rename f and h to predicted_gain and lockin_response.
        """
        
        def predicted_gain(A):
            return A, np.eye(2), (0.01 * np.array([[A[0], 0], [0, 2*np.pi]]))**2

        def lockin_response(A, dv):
            return np.array([A[0]*dv[0], A[1] + dv[1]]), \
                    np.array([[A[0], 0], [0, 1]])

        self.kalman = kalman(
            f_F = predicted_gain,
            h_H = lockin_response,
            x = A,
            P = P
        )
    
    def predict(self):
        self.kalman.predict()

    def update(self):
        self.kalman.update()

    def append(self, r, theta):
        self.r_hist.append(r)
        self.theta_hist.append(theta)

    def extrapolate(self):
        return extrap(self.r_hist, self.order), \
                extrap(self.theta_hist, self.order)

@dataclass
class KapBridge():
    lockin          : sr865a
    set_Vstd        : Callable[[float], Any]
    set_phase       : Callable[[float], Any]
    Vstd_range      : float
    averages        : int = 100
    sample_rate     : float = 0.1
    sync_sampling   : bool = False
    min_tries       : int = 0
    max_tries       : int = 10
    order           : int = 2
    support         : int = 3
    errt            : float = 1e-6
    sensitivity_increment: float = 10
    logger          : Optional[object] = None

    def __post_init__(self):
        self.kfilter = dict[Any, KFilter]

        self.sample_rate = self.lockin.setup_data_acquisition(
            buffer_size     = self.averages, 
            sample_rate     = self.sample_rate, 
            sync_sampling   = self.sync_sampling
        )

    def add_filter(self):
        pass

    def balance(self, filter_key):
        r_b, theta_b = None, None # figure this out

        if self.filter_key != filter_key:
            if not filter_key in self.kfilter:

                # self.logger.info(f"Initializing filter {filter_key}")
                # self.kfilter[filter_key] = KFilter(
                #     self.support, 
                #     self.sensitivity_increment,
                #     A = ,
                #     P =
                # )

                # Need to perform a raw balance.
                pass
            else:
                if self.logger:
                    self.logger.info(f"Restoring filter {filter_key}")
                pass
        
        else:

            # Calculate a projected balance point
            self.kfilter[filter_key].append(r_b, theta_b)
            r_b, theta_b = self.kfilter[filter_key].extrapolate()

        self.filter_key = filter_key

        if self.logger:
            self.logger.info(
                f"Projected r = {r_b:.7f}, theta = {np.degrees(theta_b):.5f}"
            )

        if r_b > self.Vstd_range or r_b < 0:
            r_b = min(self.Vstd_range, max(0, r_b))
            if self.logger:
                self.logger.info(f"Truncated to r = {r_b:.7f}")

        # periodically increment the sensitivity and decrement the input range
        self.kfilter[filter_key].use_count +=1
        if self.kfilter[filter_key].use_count % self.kfilter[filter_key] == 0:
            self.lockin.increment_sensitivity()
            self.lockin.decrement_input_range()
        
        # iteratively move towards the expected balance point until error is met
        for iter in range(self.max_tries):
            
            self.set_Vstd(r_b)
            self.set_phase(np.degrees(theta_b))

            if self.logger:
                self.logger.info(
                    f"Set r = {r_b:.7f}, theta = {np.degrees(theta_b):.5f}"
                )

            # NEED TO FIGURE OUT HOW THIS RETURNS FIRST
            # lock-in reading and covariance
            L, P = self.lockin.get_average()

            if self.logger:
                self.logger.info(f"Measured")
