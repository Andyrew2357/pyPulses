"""
This class implements a modified version of the Brent-Dekker method for root
finding. This is likely more sophisticated than our purposes demand, but it may
help in reducing the required number of function calls (costly measurements) to
converge on a balance point.
"""

from .rootfinder import RootFinderState, RootFinderStatus
from typing import Tuple

class BrentSolver:
    """
    Class implementation of a modified version of the Brent-Dekker method for 
    root finding.
    """
    def __init__(self, xa: float, xb: float,
                 search_range: Tuple[float, float] = None,
                 x_tolerance: float         = 1e-6,
                 y_tolerance: float         = 1e-6,
                 max_iter: int              = 100,
                 max_reps: int              = 6,
                 max_coll: int              = 3):
        """
        Parameters
        ----------
        xa, xb : float
            first and second initial guesses
        search_range : tuple of float, optional
            (min, max) defining search range
        x_tolerance : float, default=1e-6
            convergence tolerance for x.
        y_tolerance : float, default=1e-6
            convergence tolerance for y (regardless of x status).
        max_iter : int, default=100
            maximum iterations allowed
        max_reps : int, default=6
            maximum allowed repetitions before we assume a cycle
        max_coll : int, default=3
            maximum allowed collisions with the boundary
        """

        self.search_range = search_range
        if search_range:
            min_x, max_x = search_range
            self.xa = min(max(xa, min_x), max_x)
            self.xb = min(max(xb, min_x), max_x)

            self.min_x, self.max_x = search_range
        else:
            self.xa = xa
            self.xb = xb

        self.max_iter = max_iter
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance

        # Internal state
        self.fa = None
        self.fb = None

        # Previous guessed point
        self.xc = None
        self.fc = None

        # Previous-previous guessed point
        self.xd = None

        self.past_points = set()
        self.reps = 0
        self.max_allowed_reps = max_reps

        self.boundary_collisions = 0
        self.max_allowed_collisions = max_coll

        self.min_f_seen = float('inf')
        self.iterations = 0

        # Initialize state
        self.state      = RootFinderState(
            status      = RootFinderStatus.NEEDS_EVALUATION,
            point       = self.xa,
            root        = None,
            iterations  = 0,
            message     = "Awaiting initial function evaluation",
            best_value  = float('inf')
        )
       
        self._evaluation_step = 0  # Tracks which initial point we're evaluating

        self._bracketed = False
        self._prev_step_bisect = True

    def update(self, f_value: float) -> RootFinderState:
        """
        Provide a function evaluation and get the next state.

        Parameters
        ----------
        f_value : float

        Returns
        -------
        RootFinderState
        """

        # Handle initial evaluations
        if self._evaluation_step == 0:
            self.fa = f_value
            self._evaluation_step = 1
            self.min_f_seen = abs(f_value)
           
            if abs(f_value) < self.y_tolerance:
                self.state = RootFinderState(
                    status      = RootFinderStatus.CONVERGED,
                    point       = self.xa,
                    root        = self.xa,
                    iterations  = 0,
                    message     = "Root found at initial point xa",
                    best_value  = self.min_f_seen
                )
            else:
                self.state = RootFinderState(
                    status      = RootFinderStatus.NEEDS_EVALUATION,
                    point       = self.xb,
                    root        = None,
                    iterations  = 0,
                    message     = "Need second initial evaluation",
                    best_value  = self.min_f_seen
                )
           
            return self.state
           
        if self._evaluation_step == 1:
            self.fb = f_value
            self._evaluation_step = 2
            self.min_f_seen = min(self.min_f_seen, abs(f_value))
           
            if abs(f_value) < self.y_tolerance:
                self.state = RootFinderState(
                    status      = RootFinderStatus.CONVERGED,
                    point       = self.xb,
                    root        = self.xb,
                    iterations  = 0,
                    message     = "Root found at initial point xb",
                    best_value  = self.min_f_seen
                )
                return self.state

            # Add points to past_points
            self.past_points.add(self.xa)
            self.past_points.add(self.xb)

        self.min_f_seen = min(self.min_f_seen, abs(f_value))
        self.iterations += 1

        # Check to see if we have converged
        if self.min_f_seen < self.y_tolerance:
            self.state = RootFinderState(
                    status      = RootFinderStatus.CONVERGED,
                    point       = self.state.point,
                    root        = self.state.point,
                    iterations  = self.iterations,
                    message     = f"Root found within tolerance",
                    best_value  = self.min_f_seen
            )
            return self.state
        
        # Check if we've reached max iterations
        if self.iterations > self.max_iter:
            self.state = RootFinderState(
                status      = RootFinderStatus.MAX_ITERATIONS,
                point       = self.state.point,
                root        = None,
                iterations  = self.max_iter,
                message     = "Reached the maximum number of iterations.",
                best_value  = self.min_f_seen
            )
            return self.state

        self.xd = self.xc
        self.xc, self.fc = self.xb, self.fb

        if self._bracketed:
            # Proceed by Brent-Dekker Iteration
            if self.fa*f_value < 0:
                self.xb, self.fb = self.state.point, f_value
            else:
                self.xa, self.fa = self.state.point, f_value

            # Swap labels so that |fb| < |fa|
            if abs(self.fa) < abs(self.fb):
                self.xa, self.xb = self.xb, self.xa
                self.fa, self.fb = self.fb, self.fa

            # Check if we have converged
            if abs(self.xa - self.xb) < self.x_tolerance:
                self.state = RootFinderState(
                    status      = RootFinderStatus.CONVERGED,
                    point       = self.state.point,
                    root        = self.state.point,
                    iterations  = self.iterations,
                    message     = f"Root found within tolerance",
                    best_value  = self.min_f_seen
                )
                return self.state

            s = self._brent_iteration(f_value)
        
        else:
            # Case if we haven't yet entered Brent-Dekker iterations
            self.xb, self.fb = self.xa, self.fa
            self.xa, self.fa = self.state.point, f_value

            # Check if we are bracketed
            if self.fa*self.fb < 0:
                self._bracketed = True
                # Proceed by Brent-Dekker Setup and Iteration
                
                # Swap labels so that |fb| < |fa|
                if abs(self.fa) < abs(self.fb):
                    self.xa, self.xb = self.xb, self.xa
                    self.fa, self.fb = self.fb, self.fa
                self.xc, self.fc = self.xa, self.fa
                s = self._brent_iteration(f_value)
            
            elif  self.fa*self.fb > 0:
                # Proceed by Unbracketed Iteration
                s = self._unbracketed_iteration(f_value)

        # Truncate to the search range and check for collisions
        if self.search_range:
            if abs(s - self.min_x) < self.x_tolerance \
                or abs(s - self.max_x) < self.x_tolerance:
                self.boundary_collisions += 1

            if self.boundary_collisions > self.max_allowed_collisions:
                self.state = RootFinderState(
                    status      = RootFinderStatus.NO_ROOT_LIKELY,
                    point       = s,
                    root        = None,
                    iterations  = self.iterations,
                    message     = "No root is likely in the given search range",
                    best_value  = self.min_f_seen
                )
                return self.state

            s = max(self.min_x, min(self.max_x, s))
            

        # Check for cycling
        if s in self.past_points:
            self.reps += 1
        if self.reps > self.max_allowed_reps:
            self.state = RootFinderState(
                status      = RootFinderStatus.CYCLING,
                point       = s,
                root        = None,
                iterations  = self.iterations,
                message     = "Cycling encountered during root finding",
                best_value  = self.min_f_seen
            )
            return self.state
        self.past_points.add(s) 

        self.state = RootFinderState(
            status      = RootFinderStatus.NEEDS_EVALUATION,
            point       = s,
            root        = None,
            iterations  = self.iterations,
            message     = "Need another evaluation",
            best_value  = self.min_f_seen
        )
        return self.state
        
    def _brent_iteration(self, f_value: float) -> float:
        """Perform an iteration using the Brent-Dekker method."""

        if self.fa != self.fc and self.fb != self.fc:
            # Attempt Inverse Quadratic Interpolation
            s  = self.xa * self.fb * self.fc / ((self.fa - self.fb) * (self.fa - self.fc))
            s += self.xb * self.fa * self.fc / ((self.fb - self.fa) * (self.fb - self.fc))
            s += self.xc * self.fa * self.fb / ((self.fc - self.fa) * (self.fc - self.fb))
        
        else:
            # Attempt Secant Method
            s = self.xb - self.fb * (self.xb - self.xa) / (self.fb - self.fa)

        m = (3 * self.xa + self.xb) / 4
        if (not ((m <= s <= self.xb) or (self.xb <= s < m))) \
            or (self._prev_step_bisect and abs(s - self.xb) >= abs(self.xb - self.xc) / 2) \
            or (not self._prev_step_bisect and abs(s - self.xb) >= abs(self.xc - self.xd) / 2) \
            or (self._prev_step_bisect and abs(self.xb - self.xc) < self.x_tolerance) \
            or (not self._prev_step_bisect and abs(self.xc - self.xd) < self.x_tolerance):

            # Fall back to Bisection if any of these conditions hold
            s = 0.5 * (self.xa + self.xb)
            self._prev_step_bisect = True

        else:
            self._prev_step_bisect = False

        return s
    
    def _unbracketed_iteration(self, f_value) -> float:
        """
        Perform an iteration without any bracket. This proceeds either with IQI 
        or Secant Method.
        """

        if self.fa != self.fc and self.fb != self.fc:
            # Attempt Inverse Quadratic Interpolation
            s  = self.xa * self.fb * self.fc / ((self.fa - self.fb) * (self.fa - self.fc))
            s += self.xb * self.fa * self.fc / ((self.fb - self.fa) * (self.fb - self.fc))
            s += self.xc * self.fa * self.fb / ((self.fc - self.fa) * (self.fc - self.fb))
        
        else:
            # Attempt Secant Method
            s = self.xb - self.fb * (self.xb - self.xa) / (self.fb - self.fa)

        return s
         