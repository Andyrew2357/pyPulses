# NEED TO ALTER THIS BECAUSE THE PERIODIC DEGREE ARGUMENT IN THE KALMAN FILTER
# IS A PROBLEM.


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
    def __init__(self, support: int, order: int, sens_increment: int, 
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
        self.sens_increment = sens_increment

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
            return A, np.eye(2), (0.01 * np.diag([A[0], 360]))**2
            # process noise covariance matrix is assumed to be 
            # diag(0.01*R, 3.6 degrees)^2

        def lockin_response(A, dv):
            return np.array([A[0]*dv[0], A[1] + dv[1]]), \
                    np.array([[A[0], 0], [0, 1]])

        self.kalman = kalman(
            f_F = predicted_gain,
            h_H = lockin_response,
            x = A,
            P = P
        )

    def append(self, r, theta):
        self.r_hist.append(r)
        self.theta_hist.append(theta)

    def extrapolate(self):
        return extrap(self.r_hist, self.order), \
                extrap(self.theta_hist, self.order)

@dataclass
class KapBridgeBalance():
    success : bool
    r_b     : float
    th_b    : float
    r_m     : float
    th_m    : float
    P       : np.ndarray
    errt    : float
    iter    : int
    prev_r  : float
    prev_th : float

@dataclass
class KapBridge():
    lockin          : sr865a                    # lock-in amplifier
    set_Vstd        : Callable[[float], Any]    # setter for Vstd amplitude
    set_phase       : Callable[[float], Any]    # setter for Vstd phase
    Vstd_range      : float                     # range of Vstd in volts
    buffer_size     : int = 5                   # amount of data to take in kB
    sample_rate     : float = 512               # sample rate in Hz
    min_tries       : int = 0                   # min tries for balancing
    max_tries       : int = 10                  # max tries for balancing
    order           : int = 2                   # order of fit for extrapolation
    support         : int = 3                   # support for extrapolation
    erroff          : float = 0.0               # error offset for balance
    errmult         : float = 2.0               # error multiplier for balance  
    sens_increment  : float = 10                # push sensitivity and input 
                                                # range every few points
    logger          : Optional[object] = None   # logger

    def __post_init__(self):
        self.kfilter = dict[Any, KFilter]

        self.sample_rate = self.lockin.setup_data_acquisition(
            buffer_size     = self.buffer_size,
            config = 'RT', 
            sample_rate     = self.sample_rate,
        )

    def add_filter(self):
        pass

    def balance(self, filter_key):
        r_b, th_b = None, None # figure this out

        if self.filter_key != filter_key:
            if not filter_key in self.kfilter:
                pass
            else:
                if self.logger:
                    self.logger.info(f"Restoring filter {filter_key}")
                pass
        
        ################################################################
        else:

            # Calculate a projected balance point
            r_b, th_b = self.kfilter[filter_key].extrapolate()

        self.filter_key = filter_key

        if self.logger:
            self.logger.info(
                f"Projected r = {r_b:.7f}, theta = {th_b:.2f}"
            )

        if r_b > self.Vstd_range or r_b < 0:
            r_b = min(self.Vstd_range, max(0, r_b))
            if self.logger:
                self.logger.info(f"Truncated to r = {r_b:.7f}")

        # periodically increment the sensitivity and input range
        self.kfilter[filter_key].use_count +=1
        if self.kfilter[filter_key].use_count % \
            self.kfilter[filter_key].sens_increment == 0:
            self.lockin.increment_sensitivity()
            self.lockin.increment_input_range()
        
        # iteratively move towards the expected balance point until error is met
        first_guess = True
        errt = None
        for iter in range(self.max_tries):
            
            self.set_Vstd(r_b)
            self.set_phase(th_b)

            if self.logger:
                self.logger.info(
                    f"Set r = {r_b:.7f}, theta = {th_b:.2f}"
                )

            # lock-in reading and covariance
            L, P = self.lockin.get_average(auto_rescale = True)
            r_m, th_m = L

            if self.logger:
                self.logger.info(
                  f"Lock-in reading: R = {L[0]:.8f} V, theta = {L[1]} degrees\n"
                + f"     Covariance: [{P[0,0]:.9f}, {P[0,1]:.9f}]\n"
                + f"                 [{P[1,0]:.9f}, {P[1,1]:.9f}]"
                )

            if not first_guess:
                # We gain more information about the effective gain the bridge
                # setup as we take more measurements. Here we update the Kalman
                # filter with this new information.
                
                # calculate the observed change in the lock-in reading in polar
                # coordinates
                sm, cm = np.sin(np.deg2rad(th_m)), np.cos(np.deg2rad(th_m))
                sp, cp = np.sin(np.deg2rad(prev_th_m)), \
                            np.cos(np.deg2rad(prev_th_m))
                
                dr = np.sqrt(r_m*r_m + prev_r_m*prev_r_m - \
                             r_m*prev_r_m*(sm*cp + sp*cm))
                dth = np.degrees(np.atan2(r_m*sm - prev_r_m*sm, 
                                          r_m*cm - prev_r_m*cm))
                
                # measurement
                z = np.array([dr, dth])
                # measurement covariance matrix
                # first need to calculate a jacobian for this polar vector 
                # addition thing.
                J = # figure this out...
                R = # J.T @ diag(P, prev_P) @ J
                
            first_guess = False

            if self.logger:
                self.logger.info(
                   "Prior to Kalman Prediction Step\n"
                + f"Effective gain: R = {self.kfilter[filter_key].kalman.x[0]},\n"
                + f"            Theta = {self.kfilter[filter_key].kalman.x[1]} degrees\n"
                + f"       Covariance = {self.kfilter[filter_key].kalman.P}"
                )

            self.kfilter[filter_key].kalman.predict()

            if self.logger:
                self.logger.info(
                   "Predicted Bridge Gain\n"
                + f"Effective gain: R = {self.kfilter[filter_key].kalman.x[0]},\n"
                + f"            Theta = {self.kfilter[filter_key].kalman.x[1]} degrees\n"
                + f"       Covariance = {self.kfilter[filter_key].kalman.P}"
                )

            r_g, th_g = self.kfilter[filter_key].kalman.x
            dr = r_m/r_g
            dth = th_m - th_g

            # store the old state
            prev_r_b    = r_b
            prev_th_b   = th_b
            prev_r_m    = r_m
            prev_th_m   = th_m
            prev_P      = P 

            # calculate the new projected balance point
            sb, cb = np.sin(np.deg2rad(th_b)), np.cos(np.deg2rad(th_b))
            sd, cd = np.sin(np.deg2rad(dth)), np.cos(np.deg2rad(dth))
            r_b = np.sqrt(r_b*r_b + dr*dr - r_b*dr*(sb*cd + sd*cb))
            th_b = np.degrees(np.atan2(r_b*sb - dr*sd, r_b*cb - dr*cd))

            if self.logger:
                self.logging.info(
                    f"Corrected balance at r = {r_b:.7f}, theta = {th_b:.2f}\n"
                  + f"              (relative change of {100*dr/prev_r_b:.3f})"
                )

            # if the lock-in reading was sufficiently small, terminate.
            # by sufficiently small, we mean that it is indistinguishable from
            # 0 within the uncertainty of the measurement. We have functionality
            # to include a multiplier and offset to the error threshold. By
            # default, the offset is 0.0, and the multiplier is 2.0.
            
            # This threshold is really set by the first measurement we take, and
            # later we mix it slightly with subsequent measurements. 
            
            errmult = self.erroff + self.errmult * np.sqrt(P[0,0])
            if errt is None:
                errt = errmult + self.erroff
            else:
                errt = 0.95*errt + 0.05*errmult

            if (r_m < errt) and (iter + 1 >= self.min_tries):
                if self.logger:
                    self.logger.info(
                        f"Balanced on iteration {iter}\n"
                      + f"  Error Tolerance: {errt:.9f} V\n"
                      + f"  Lock-in Reading: {r_m:.9f} V"
                    )

                # append the new balance point to the historical data
                self.kfilter[filter_key].append(r_b, th_b)
                return KapBridgeBalance(
                    success = True,
                    r_b     = r_b,
                    th_b    = th_b,
                    r_m     = r_m,
                    th_m    = th_m,
                    P       = P,
                    errt    = errt,
                    iter    = iter,
                    prev_r  = prev_r_m,
                    prev_th = prev_th_m
                )
            
            else:
                if self.logger:
                    self.logger.info(
                        f"Not balanced on iteration {iter}\n"
                      + f"  Error Tolerance: {errt:.9f} V\n"
                      + f"  Lock-in Reading: {r_m:.9f} V\n"
                      + f"Continuing to iteration {iter + 1}..."
                    )
        
        # if we reach this point, we have not balanced after max_tries
        else:
            if self.logger:
                self.logger.info(
                    f"Failed to balance after {self.max_tries} tries"
                )
            return KapBridgeBalance(
                success = False,
                r_b     = r_b,
                th_b    = th_b,
                r_m     = r_m,
                th_m    = th_m,
                P       = P,
                errt    = errt,
                iter    = iter,
                prev_r  = prev_r_m,
                prev_th = prev_th_m
            )