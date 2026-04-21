from .balance_parameter import balanceError, balanceKnob

from ..core.job import checkpoint

import time
import logging
import numpy as np
from typing import Callable, List, Tuple

class lstsqBalance():
    def __init__(self, 
        controls: List[balanceKnob], 
        error_parms: List[balanceError],
        pre_measurement_callback: Callable | None = None,
        post_measurement_callback: Callable | None = None,
        settle_time: float = 0.0,
        v0: float = 0.5,
        logger: logging.Logger | None = None
    ):
        
        self.controls = controls
        self.error_parms = error_parms

        self.v0 = v0
        
        self.P = len(controls)
        self.M = len(self.error_parms)

        self.controls = controls
        self.error_parms = error_parms

        self._pre_measurement_callback = pre_measurement_callback
        def callback(*args, **kwargs):
            time.sleep(settle_time)
            if self._pre_measurement_callback is not None:
                self._pre_measurement_callback(*args, **kwargs)

        self.pre_callback = callback
        self.post_callback = post_measurement_callback
        self.logger = logger

        self.N = 0
        self._X = []
        self._Y = []
        self._Ydev = []
        self._W = []
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._x0: np.ndarray | None = None
        self._x0_set: np.ndarray | None = None
        self._yerr: np.ndarray | None = None

        self.test_points = []
        self.good = False

    def log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)

    def reset(self):
        self.N = 0
        self._X = []
        self._Y = []
        self._Ydev = []
        self._W = []
        self._A = None
        self._b = None
        self._x0 = None
        self._x0_set = None
        self._yerr = None
        self.test_points = []
        self.good = False

    def push(self, x: np.ndarray, y: List[Tuple[float, float]], w: float = 1.0):

        self.N+=1
        yv, yd = zip(*y)
        print(yv, yd)
        self._X.append(x)
        self._Y.append(yv)
        self._Ydev.append(yd)

        if self.good and self._x0 is not None:
            dx = x - self._x0
            dr = np.dot(dx, dx)
            w = self.v0 #+ dr
        else:
            w = 1.0

        self._W.append(w)

    def set_good(self, g: bool = True):
        self.good = g

    def prepare_test_points(self):

        # 0000
        # 1111
        # 1000
        # 0100

        for i in range(self.P + 1):
            if i == 0:
                self.test_points.append(np.zeros(self.P, dtype=bool))
            elif i == 1:
                self.test_points.append(np.ones(self.P, dtype=bool))
            else:
                p = np.zeros(self.P, dtype=bool)
                p[i - 2] = True
                self.test_points.append(p)

        self.log(f"Prepared test points:\n{self.test_points}")

    def set_test_point(self, point: np.ndarray):
        self.log(f"Setting test point: {point}")
        for x, p in zip(self.controls, point):
            x.set_bool(p)

    def get_control_vals(self) -> np.ndarray:
        return np.array([c.get_val() for c in self.controls], dtype=float)

    def measure_error_parms(self) -> np.ndarray:
        checkpoint()

        self.log(f"Measuring error parameters...")
        self.pre_callback(self)
        res = [P() for P in self.error_parms]
        self.log(f"Measured error parameters: {res}")
        if self.post_callback is not None:
            self.post_callback(self, res)
        return res

    def measure_test_points(self):
        for point in self.test_points:
            self.set_test_point(point)
            self.push(self.get_control_vals(), self.measure_error_parms())

    def calculate_response_matrix(self, rcond: float = 1e-12):
        self.log("Calculating response matrix...")

        Xhist = np.asarray(self._X)
        Yhist = np.asarray(self._Y)

        self.log(f"X history:\n{Xhist}\nY history:\n{Yhist}\nWeights:\n{self._W}")

        Xaug = np.hstack([Xhist, np.ones((self.N, 1))])
        Wsqrt = np.sqrt(self._W)[:, None]

        Xw = Xaug*Wsqrt
        Yw = Yhist*Wsqrt
        coeffs, *_ = np.linalg.lstsq(Xw, Yw, rcond=rcond)
        self._A = coeffs[:-1, :].T
        self._b = coeffs[-1, :]

        self.log(f"Response matrix:\nA:\n{self._A}\nb:\n{self._b}")

    def calculate_balance_point(self, rcond: float = 1e-12):
        x_balance, *_ = np.linalg.lstsq(self._A, -self._b, rcond=rcond)
        if self.good and self._yerr is not None:
            dx, *_ = np.linalg.lstsq(self._A, -self._yerr)
            self._x0 = self._x0_set + dx
            self.log(
                f"Balance is 'good', refining...\n   Set point = {self._x0_set}"
                f"\n    Error = {self._yerr}\n  Balance point = {self._x0}"
            )
        else:
            x_balance, *_ = np.linalg.lstsq(self._A, -self._b, rcond=rcond)
            self._x0 = x_balance
            self.log(
                "Balance is not 'good', using raw balance point calculation..."
            )

        self.log(f"Calculated balance point: {self._x0}")

    def get_current_balance_point(self) -> np.ndarray | None:
        return self._x0.copy()

    def set_balance_point(self, x0: np.ndarray):
        for x, p in zip(self.controls, x0):
            x.set_val(p)
        self._x0_set = None
        self._yerr = None

    def step_to_balance(self):

        self.log(f"Stepping to balance point: {self._x0}")

        for x, p in zip(self.controls, self._x0):
            x.set_val(p)
            x.update_guess(p)
        self._x0_set = self.get_control_vals()

        self.log(f"Stepped to balance point: {self._x0_set}")

    def balance(self):

        self.log("Starting balance procedure...")
        
        if len(self.test_points) == 0:
            self.prepare_test_points()

        self.measure_test_points()
        self.calculate_response_matrix()
        self.calculate_balance_point()
        self.step_to_balance()

    def refine(self):
        
        self.log("Refining balance point...")

        self.step_to_balance()
        y = self.measure_error_parms()
        self._yerr = np.array([v for v, _ in y], dtype=float)
        self.push(self._x0_set, y)
        self.calculate_response_matrix()
        self.calculate_balance_point()
        self.step_to_balance()

    def update_guesses(self):
        for x, g in zip(self.controls, self._x0):
            x.guess = g

class bracket1d():
    def __init__(self, min_x: float, max_x: float, padding: float):
        self.min_x = min_x
        self.max_x = max_x
        self.padding = padding

        self.low = _bound1d(False)
        self.high = _bound1d(True)

        # most recent measurement
        self.x: float | None = None
        self.y: float | None = None
        self.dydx: float | None = None

        self.hitting_max_rail = False
        self.hitting_min_rail = False

    def clear(self):
        self.low.clear()
        self.high.clear()
        self.x = None
        self.y = None
        self.dydx = None
        self.hitting_max_rail = False
        self.hitting_min_rail = False

    def update(self, x: float, y: float, dydx: float | None):
        """
        Whenever we take a new measurement, update the brackets and latest
        measurement accordingly.
        """

        self.low.update(x, y, dydx)
        self.high.update(x, y, dydx)
        self.x = x
        self.y = y
        self.dydx = dydx

    def iterate(self, dydx: float) -> Tuple[float | None, bool]:
        """
        We take Newton's Method-like steps until we have brackets on the low
        and high side, at which point we perform a Secant Method-like iteration.

        Returns
        -------
        float
            New predicted x
        bool
            Whether the prediction is thought to be good
        """

        # If neither bound is available, we can't iterate
        if self.low.x is None and self.high.x is None:
            self.hitting_min_rail = False
            self.hitting_max_rail = False
            return None, False
        
        # If only the high bound is available
        if self.low.x is None:
            xn = self.high.x - self.high.y / dydx

            # check if we're butting up against the boundaries
            self.hitting_min_rail = ((self.high.x <= self.min_x + self.padding) and (dydx > 0))
            self.hitting_max_rail = ((self.high.x >= self.max_x - self.padding) and (dydx < 0))
            return xn, True

        if self.high.x is None:
            xn = self.low.x - self.low.y / dydx

            # check if we're butting up against the boundaries
            self.hitting_min_rail = ((self.low.x <= self.min_x + self.padding) and (dydx < 0))
            self.hitting_max_rail = ((self.low.x >= self.max_x - self.padding) and (dydx > 0))
            return xn, True                

        # If both bounds are available, we use a secant method
        self.hitting_min_rail = False
        self.hitting_max_rail = False
        xn = (self.low.x * self.high.y - self.high.x * self.low.y) / (self.high.y - self.low.y)
        return xn, True
    
    def span(self) -> float | None:
        if self.low.x is None or self.high.x is None:
            return None
        return abs(self.low.x - self.high.x)

class _bound1d:
    def __init__(self, 
        high: bool, 
        x: float | None = None, 
        y: float | None = None, 
        dydx: float | None = None
    ):
        self.high = high
        self.x = x
        self.y = y
        self.dydx = dydx

    def clear(self):
        self.x = None
        self.y = None

    def update(self, x: float, y: float, dydx: float | None):
        
        # If this is the wrong kind of measurement for this bracket, no-op
        if self.high == (y <= 0):
            return
        
        # If this measurement improves the bound, replace it
        if (self.y is None) or (self.high == (y <= self.y)):
            self.x = x
            self.y = y
        
        # This condition may warrant some explanation. The condition 
        # ((dydx > 0) == (x < self.x)) is naturally interpreted as follows: 
        # In the high case: True if updating this bound moves the high side in 
        # the right direction based on the slope.
        # In the low case: False if updating this bound moves the low side in 
        # the right direction based on the slope.
        # So a simple way of asking whether we move in the right direction is to
        # ask if this condition is equal to high.
        elif (self.high == ((dydx > 0) == (x < self.x))):
            self.x = x
            self.y = y
