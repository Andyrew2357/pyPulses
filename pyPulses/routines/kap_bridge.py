"""
Balance a capacitance bridge using a Kalman filter. The game is basically to
represent the balance matrix by a complex gain describing the lock-in response
as a function of Vstd (understood as a complex number described by its amplitude
and phase). Most of the time, we can directly guess the balance point by
extrapolating from previous points. When this is not the case, we can step to
the correct balance point by calculating what it should be based on the off-
balance signal and the complex gain. Supposing this balance point is still 
inadequate, we take further iterations to get to the true balance condition. As
we take these steps and measure at different points, we gain information about
the effective complex gain and update our estimate of the gain accordingly (this
is where the Kalman filter comes in).
"""

from ..devices import sr865a, ad9854
from ..utils import kalman
from .cap_bridge import balanceCapBridge, BalanceCapBridgeConfig

import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
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
        self.x_hist = deque(maxlen = support)
        self.y_hist = deque(maxlen = support)
        
        # optimal lock-in sensitivity and input range
        self.lockin_sensitivity = None
        self.lockin_input_range = None
        
        # excitation size
        self.Vex = None

        # we periodically increase the sensitivity
        self.use_count = 0
        self.sens_increment = sens_increment

        """
        the state vector x is (X, Y)
        X + iY is the complex effective gain of the capacitance bridge.

        Given a step (dx, dy) on Vstd, the response on the lockin will be:
                                    
                            (X*dx - Y*dy, Y*dx + X*dy)

        f(x, u) is the predicted evolution of the state (x). In this case, we
        expect minimal variation in the effective gain, so our model for state
        evolution is f(x) = x. We also need the Jacobian, F = 1. F must also
        return a process noise covariance matrix. In this case we assume
        uncorrelated errors in the modulus and phase of the complex gain. If the
        error in phase is dθ and the error in modulus is dR, we can represent
        the covariance in cartesian coordinates by transforming basis under the
        Jacobian. In this case:

        Q = J Q_p J^T, J = ((cosθ,-rsinθ), (sinθ,rcosθ)), Q_p = diag(dR^2, dθ^2)

        I assume dR = 0.01*R and dθ = 0.01*2pi

        h(x, v) is the predicted change in lock-in reading given a step
        dv = dVstd = (dx, dy). This is given by (X*dx - Y*dy, Y*dx + X*dy), as 
        discussed. The Jacobian is H = ((dx, -dy), (dy, dx)).

        For clarity, I label the state vector as A and the extra arguments for h
        as dv. I also rename f and h to predicted_gain and lockin_response.
        """
        
        def predicted_gain(A):
            r = np.sum(A**2)
            s, c = A[1]/r, A[0]/r
            J = np.array([[c, -r*s], [s, r*c]])
            Q = J @ np.diag([0.01*r, 0.01*(2*np.pi)]) @ J.T
            return A, np.eye(2), Q

        def lockin_response(A, dv):
            dx, dy = dv
            H = np.array([[dx, -dy], [dy, dx]])
            return H @ A, H

        self.kalman = kalman(
            f_F = predicted_gain,
            h_H = lockin_response,
            x = A,
            P = P
        )

    def append(self, x, y):
        self.x_hist.append(x)
        self.y_hist.append(y)

    def extrapolate(self):
        return extrap(self.x_hist, self.order), extrap(self.y_hist, self.order)

"""Describes a balance point reached after balancing the bridge."""
@dataclass
class KapBridgeBalance():
    success : bool          # whether balance was successful
    x_b     : float         # estimated true balance point (x component)
    y_b     : float         # estimated true balance point (y component)
    x_m     : float         # final achieved lock-in reading (x component)
    y_m     : float         # final achieved lock-in reading (y component)
    R       : np.ndarray    # covariance matrix for lock-in reading
    A       : np.ndarray    # effective complex gain of the capacitance bridge
    P       : np.ndarray    # covariance of the complex gain
    errt    : float         # error threshold used for this point
    iter    : int           # terminating iteration
    prev_x_b: float         # Vstd when last measurement was taken (x component)
    prev_y_b: float         # Vstd when last measurement was taken (y component)

@dataclass
class KapBridge():
    lockin          : sr865a                    # lock-in amplifier
    acbox           : ad9854                    # ac box
    Vstd_range      : float                     # range of Vstd in volts
    Vex_range       : float                     # range of Vex in volts
    acbox_channels  : Tuple[Tuple[int, str],    # (excitation, standard) channel
                            Tuple[int, str]
                    ] = ((1, 'X'), (2, 'X'))
    buffer_size     : int = 1                   # amount of data to take in kB
    sample_rate     : float = 100               # sample rate in Hz
    min_tries       : int = 0                   # min tries for balancing
    max_tries       : int = 10                  # max tries for balancing
    order           : int = 2                   # order of fit for extrapolation
    support         : int = 3                   # support for extrapolation
    erroff          : float = 0.0               # error offset for balance
    errmult         : float = 2.0               # error multiplier for balance  
    sens_increment  : float = 10                # push sensitivity and input 
                                                # range every few points
    Cstd            : float = 1.0               # capacitance of the reference
    raw_samples     : int = 50                  # samples for a raw balance
    raw_wait        : float = 0.1               # wait time for a raw balance 
    logger          : Optional[object] = None   # logger

    def __post_init__(self):
        self.filter_key = None
        self.kfilter: Dict[Any, KFilter] = {}

        self.sample_rate = self.lockin.setup_data_acquisition(
            buffer_size     = self.buffer_size,
            config = 'XY', 
            sample_rate     = self.sample_rate,
        )

        exc_, std_      = self.acbox_channels
        self.set_Vex    = lambda x: self.acbox.set_amplitude(*exc_, x)
        self.set_Vstd   = lambda x: self.acbox.set_amplitude(*std_, x)
        self.set_phase  = lambda x: self.acbox.set_phase(exc_[0], x)
        self.get_Vex    = lambda: self.acbox.get_amplitude(*exc_)
        self.get_Vstd   = lambda: self.acbox.get_amplitude(*std_)

    def add_filter(self, filter_key):
        """
        Add a KFilter to self.kfilter
        For this, we need to perform a raw balance measurement.
        """

        if self.logger:
            self.logger.info(f"Adding new filter {filter_key}...")
            self.logger.info("Performing a raw balance of the bridge")

        # perform a raw balance measurement
        balance_config = BalanceCapBridgeConfig(
            Vstd_range  = self.Vstd_range,
            Vex         = self.Vex_range,
            samples     = self.raw_samples,
            wait        = self.raw_wait
        )

        raw_balance = balanceCapBridge(
            balance_config,
            set_Vex     = self.set_Vex,
            get_Vex     = self.get_Vex,
            set_Vstd    = self.set_Vstd,
            get_Vstd    = self.get_Vstd,
            set_Vstd_ph = self.set_phase,
            get_XY      = self.lockin.get_xy
        )

        # we establish a gain tensor from the cap_bridge parameters
        # this also gives us our balance point guess.
        Kc1, Kc2, Kr1, Kr2, y_b, x_b = raw_balance.balance_matrix
        x_b = -x_b # subtleties related to how the balance matrix is designed

        # approximately map this to an effective complex gain
        # chosen representation of C in terms of 2x2 matrices is ((X,-Y),(Y, X))
        A = np.array([(Kr2 - Kc1)/2, -(Kr1 + Kc2)/2])

        # covariance matrix for complex gain is just assumed to look 
        # like small uncorrelated errors in modulus and phase. See
        # the filter for an explanation for how I convert this to
        # Cartesian coordinates.
        r = np.sum(A**2)
        s, c = A[1]/r, A[0]/r
        J = np.array([[c, -r*s], [s, r*c]])
        P = J @ np.diag([0.01*r, 0.01*(2*np.pi)]) @ J.T
        
        self.kfilter[filter_key] = KFilter(
            support         = self.support, 
            order           = self.order, 
            sens_increment  = self.sens_increment, 
            A               = A,
            P               = P
        )
        
        self.kfilter[filter_key].append(x_b, y_b)
        self.kfilter[filter_key].lockin_input_range = \
                                                self.lockin.get_input_range()
        self.kfilter[filter_key].lockin_sensitivity = \
                                                self.lockin.get_sensitivity()
        self.kfilter[filter_key].Vex = self.get_Vex()

        if self.logger:
            self.logger.info("Raw balance result:")
            self.logger.info(raw_balance)
            self.logger.info(f"Effective gain: {A[0]} + i{A[1]}")
            self.logger.info(f"Balance point: {x_b:.8f} V, {y_b:.8f} V")

    def balance(self, filter_key) -> KapBridgeBalance:
        x_b, y_b = None, None

        if self.filter_key != filter_key:
            if self.filter_key is not None:
                # save the gain and sensitivity settings for the current filter
                self.kfilter[self.filter_key].lockin_input_range = \
                                                self.lockin.get_input_range()
                self.kfilter[self.filter_key].lockin_sensitivity = \
                                                self.lockin.get_sensitivity()

            # make a new filter
            if not filter_key in self.kfilter:                
                self.add_filter(filter_key)

            # restore an existing filter with a different key
            else:
                if self.logger:
                    self.logger.info(f"Restoring filter {filter_key}")
                
                self.lockin.set_input_range(
                    self.kfilter[filter_key].lockin_input_range)
                self.lockin.set_sensitivity(
                    self.kfilter[filter_key].lockin_sensitivity)

        # calculate a projected balance point
        x_b, y_b = self.kfilter[filter_key].extrapolate()

        self.filter_key = filter_key

        if self.logger:
            self.logger.info(
                f"Projected x = {x_b:.7f} V, y = {y_b:.7f} V"
            )

        r_b = np.sqrt(x_b*x_b + y_b*y_b)
        sb, cb = y_b/r_b, x_b/r_b
        th_b = np.degrees(np.atan2(y_b, x_b))
        if r_b > self.Vstd_range:
            r_b = min(self.Vstd_range, r_b)
            x_b = r_b*cb
            y_b = r_b*sb
            if self.logger:
                self.logger.info(
                    f"Truncated to x = {x_b:.7f} V, y = {y_b:.7f} V")

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
            L, R = self.lockin.get_average(auto_rescale = True)
            x_m, y_m = L

            if self.logger:
                self.logger.info(
                  f"Lock-in reading: x = {L[0]:.8f} V, y = {L[1]:.8f} V\n"
                + f"     Covariance: [{R[0,0]:.9f}, {R[0,1]:.9f}]\n"
                + f"                 [{R[1,0]:.9f}, {R[1,1]:.9f}]"
                )

            if not first_guess:
                # We gain more information about the effective gain the bridge
                # setup as we take more measurements. Here we update the Kalman
                # filter with this new information.
                
                # measured change in the lock-in reading
                dLx, dLy = x_m - prev_x_m, y_m - prev_y_m
                # change in Vstd
                dvx, dvy = x_b - prev_x_b, y_b - prev_y_b

                # calculating the covariance matrix for z
                R_change = R + prev_R

                dL = np.array([dLx, dLy])
                dv = np.array([dvx, dvy])
                self.kfilter[filter_key].kalman.update(dL, R_change, dv)

            first_guess = False

            if self.logger:
                self.logger.info(
                    "Prior to Kalman Prediction Step\n"
                    + f"Effective gain: \n"
                    + f"    x = {self.kfilter[filter_key].kalman.x[0]}\n"
                    + f"    y = {self.kfilter[filter_key].kalman.x[1]}\n"
                    + f"  Cov = {self.kfilter[filter_key].kalman.P}"
                )

            self.kfilter[filter_key].kalman.predict()

            if self.logger:
                self.logger.info(
                    "Predicted Bridge Gain\n"
                    + f"Effective gain: \n"
                    + f"    x = {self.kfilter[filter_key].kalman.x[0]}\n"
                    + f"    y = {self.kfilter[filter_key].kalman.x[1]}\n"
                    + f"  Cov = {self.kfilter[filter_key].kalman.P}"
                )

            # based on the gain, determine how far we are off balance
            x_g, y_g = self.kfilter[filter_key].kalman.x
            Z = np.array([[x_g, -y_g], [y_g, x_g]])
            dx, dy = inv(Z) @ L

            # store the old state
            prev_x_b    = x_b
            prev_y_b    = y_b
            prev_x_m    = x_m
            prev_y_m    = y_m
            prev_R      = R 
            prev_r_b    = r_b

            # calculate the new projected balance point
            x_b -= dx
            y_b -= dy
            r_b = np.sqrt(x_b*x_b + y_b*y_b)

            if self.logger:
                self.logger.info(
                    f"Corrected balance at x = {x_b:.7f} V, y = {y_b:.7f} V\n"
                  + f"            (relative change of {100*r_b/prev_r_b:.3f})"
                )

            # if the lock-in reading was sufficiently small, terminate.
            # by sufficiently small, we mean that it is indistinguishable from
            # 0 within the uncertainty of the measurement. We have functionality
            # to include a multiplier and offset to the error threshold. By
            # default, the offset is 0.0, and the multiplier is 2.0.
            
            # This threshold is really set by the first measurement we take, and
            # later we mix it slightly with subsequent measurements. 
            
            r_m = np.sqrt(x_m*x_m + y_m*y_m)
            m = L.reshape((-1, 1))
            r_m_err = np.sum(np.sqrt((m.T @ R @ m)/r_m))

            errmult = self.erroff + self.errmult * r_m_err
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
                self.kfilter[filter_key].append(x_b, y_b)
                return KapBridgeBalance(
                    success = True,
                    x_b     = x_b,
                    y_b     = y_b,
                    x_m     = x_m,
                    y_m     = y_m,
                    R       = R,
                    A       = self.kfilter[filter_key].kalman.x,
                    P       = self.kfilter[filter_key].kalman.P,
                    errt    = errt,
                    iter    = iter,
                    prev_x_b= prev_x_b,
                    prev_y_b= prev_y_b
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
                x_b     = x_b,
                y_b     = y_b,
                x_m     = x_m,
                y_m     = y_m,
                R       = R,
                A       = self.kfilter[filter_key].kalman.x,
                P       = self.kfilter[filter_key].kalman.P,
                errt    = errt,
                iter    = iter,
                prev_x_b= prev_x_b,
                prev_y_b= prev_y_b
            )
    
    def measure_capacitance(self, filter_key) -> bool:
        self.balance_state = self.balance(filter_key)
        return self.balance_state.success

    def get_param(self, field: str) -> Union[bool, float, int, np.ndarray]:
        return getattr(self.balance_state, field)

    def get_Cex(self) -> float:
        return -self.Cstd * self.balance_state.x_b / \
                            self.kfilter[self.filter_key].Vex

    def get_Closs(self) -> float:
        # Note that this one is lacking a negative sign to maintain consistency
        # with the outputs of cap_bridge
        return self.Cstd * self.balance_state.y_b / \
                            self.kfilter[self.filter_key].Vex
