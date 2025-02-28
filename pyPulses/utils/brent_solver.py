"""
This class implements a modified version of the Brent-Dekker method for root
finding. This is likely more sophisticated than our purposes demand, but it may
help in reducing the required number of function calls (costly measurements) to
converge on a balance point.
"""

from .rootfinder import RootFinderState, RootFinderStatus
from typing import Tuple, Optional
import numpy as np

class BrentSolver:
    def __init__(self, x0: float, x1: float,
                 search_range: Optional[Tuple[float, float]] = None,
                 max_iter: int = 100,
                 tolerance: float = 1e-6,
                 max_step: Optional[float] = None):
        """
        Initialize the Brent solver with external function evaluation.
       
        Args:
            x0: First initial guess
            x1: Second initial guess
            search_range: Optional tuple of (min, max) defining search range
            max_iter: Maximum iterations allowed
            tolerance: Convergence tolerance
            max_step: Maximum allowed step size
        """
        self.search_range = search_range
        if search_range:
            min_x, max_x = search_range
            self.x0 = min(max(x0, min_x), max_x)
            self.x1 = min(max(x1, min_x), max_x)
        else:
            self.x0 = x0
            self.x1 = x1
           
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_step = max_step
       
        # Internal state
        self.x2 = self.x0
        self.f0 = None
        self.f1 = None
        self.f2 = None
       
        self.past_points = set()
        self.points_near_boundary = 0
        self.min_f_seen = float('inf')
        self.iterations = 0
       
        # Initialize state
        self.state = RootFinderState(
            status = RootFinderStatus.NEEDS_EVALUATION,
            point = self.x0,
            root = None,
            iterations = 0,
            message = "Awaiting initial function evaluation",
            best_value = float('inf')
        )
       
        self._evaluation_step = 0  # Tracks which initial point we're evaluating
       
    def update(self, f_value: float) -> RootFinderState:
        """
        Provide a function evaluation and get the next state.
        """
        # Handle initial evaluations
        if self._evaluation_step == 0:
            self.f0 = f_value
            self.f2 = f_value
            self._evaluation_step = 1
            self.min_f_seen = abs(f_value)
           
            if abs(f_value) < self.tolerance:
                self.state = RootFinderState(
                    status = RootFinderStatus.CONVERGED,
                    point = self.x0,
                    root = self.x0,
                    iterations = 0,
                    message = "Root found at initial point x0",
                    best_value = abs(f_value)
                )
            else:
                self.state = RootFinderState(
                    status = RootFinderStatus.NEEDS_EVALUATION,
                    point = self.x1,
                    root = None,
                    iterations = 0,
                    message = "Need second initial evaluation",
                    best_value = self.min_f_seen
                )
           
            return self.state
           
        if self._evaluation_step == 1:
            self.f1 = f_value
            self._evaluation_step = 2
            self.min_f_seen = min(self.min_f_seen, abs(f_value))
           
            if abs(f_value) < self.tolerance:
                self.state = RootFinderState(
                    status = RootFinderStatus.CONVERGED,
                    point = self.x1,
                    root = self.x1,
                    iterations = 0,
                    message = "Root found at initial point x1",
                    best_value = abs(f_value)
                )
            else:
                # Add points to past_points
                self.past_points.add(self.x0)
                self.past_points.add(self.x1)
           
            return self.state
           
        # Main iteration
        if self._evaluation_step >= 2:
            self.iterations += 1
           
            # Sort points so x1 has smallest function value
            if abs(self.f0) < abs(self.f1):
                self.x0, self.x1 = self.x1, self.x0
                self.f0, self.f1 = self.f1, self.f0
               
            # Update minimum function value seen
            self.min_f_seen = min(self.min_f_seen, abs(f_value))
           
            # Check convergence
            if abs(f_value) < self.tolerance:
                self.state = RootFinderState(
                    status = RootFinderStatus.CONVERGED,
                    point = self.x1,
                    root = self.x1,
                    iterations = self.iterations,
                    message = "Root found within tolerance",
                    best_value = abs(f_value)
                )
                return self.state

            # Set a default step using bisection as fallback
            default_step = 0.5 * (self.x0 - self.x1)
               
            # Calculate next step
            try:
                # Check if denominators are close to zero
                if abs(self.f0 - self.f1) < 1e-10 or abs(self.f2 - self.f1) < 1e-10 or abs(self.f2 - self.f0) < 1e-10:
                    raise ZeroDivisionError("Denominator too small")
               
                # Try IQI (Inverse Quadratic Interpolation) first
                q11 = (self.x1 - self.x0) * self.f1 / (self.f0 - self.f1)
                q21 = (self.x1 - self.x2) * self.f1 / (self.f2 - self.f1)
                d = (self.x1 - self.x2) * self.f0 / (self.f2 - self.f0)
                step = self.f1 * (q11 + q21 - d)
               
                # Check if step resulted in NaN or infinity
                if np.isnan(step) or np.isinf(step):
                    raise ValueError("Step is NaN or infinity")
               
                # Fall back to secant if step is too large
                if self.max_step and abs(step) > self.max_step:
                    if abs(self.f1 - self.f0) < 1e-10:
                        raise ZeroDivisionError("Secant denominator too small")
                   
                    step = -self.f1 * (self.x1 - self.x0) / (self.f1 - self.f0)
                   
                    # Check if secant step resulted in NaN or infinity
                    if np.isnan(step) or np.isinf(step):
                        raise ValueError("Secant step is NaN or infinity")
                   
            except (ZeroDivisionError, ValueError):
                # Use bisection as ultimate fallback
                step = default_step
                   
            # Apply max step size if specified
            if self.max_step:
                step = max(min(step, self.max_step), -self.max_step)
               
            # Calculate new point
            new_x = self.x1 + step
           
            # Sanity check the new point - if it's NaN, use bisection
            if np.isnan(new_x):
                if self.search_range:
                    min_x, max_x = self.search_range
                    new_x = 0.5 * (min_x + max_x)
                else:
                    new_x = 0.5 * (self.x0 + self.x1)
           
            # Handle search range if specified
            if self.search_range:
                min_x, max_x = self.search_range
                if new_x < min_x:
                    new_x = min_x
                    self.points_near_boundary += 1
                elif new_x > max_x:
                    new_x = max_x
                    self.points_near_boundary += 1
                   
                # Check if we're repeatedly hitting boundaries with no improvement
                if self.points_near_boundary > 5 and self.min_f_seen > self.tolerance:
                    self.state = RootFinderState(
                        status = RootFinderStatus.NO_ROOT_LIKELY,
                        point = new_x,
                        root = None,
                        iterations = self.iterations,
                        message = f"No root likely in range [{min_x}, {max_x}]",
                        best_value = self.min_f_seen
                    )
                    return self.state
                   
            # Check for cycling
            if new_x in self.past_points:
                if self.min_f_seen > self.tolerance:
                    self.state = RootFinderState(
                        status = RootFinderStatus.CYCLING,
                        point = new_x,
                        root = None,
                        iterations = self.iterations,
                        message = "Search cycling with no root found",
                        best_value = self.min_f_seen
                    )
                    return self.state
                   
                # Take a small random step if cycling
                new_x = self.x0 + np.random.uniform(-0.1, 0.1) * (self.x1 - self.x0)
               
            # Update points for next iteration - CRITICAL CODE
            self.x2, self.f2 = self.x0, self.f0  # Old x0 becomes x2
            self.x0, self.f0 = self.x1, self.f1  # Old x1 becomes x0
            self.x1 = new_x                      # New point becomes x1
            self.f1 = None                       # Clear f1 as it needs evaluation
            self.past_points.add(new_x)
           
            # Check max iterations
            if self.iterations >= self.max_iter:
                self.state = RootFinderState(
                    status = RootFinderStatus.MAX_ITERATIONS,
                    point = new_x,
                    root = None,
                    iterations = self.iterations,
                    message = f"Failed to converge in {self.max_iter} iterations",
                    best_value = self.min_f_seen
                )
                return self.state
               
            # Request evaluation at new point
            self.state = RootFinderState(
                status = RootFinderStatus.NEEDS_EVALUATION,
                point = new_x,
                root = None,
                iterations = self.iterations,
                message = "Need function evaluation at new point",
                best_value = self.min_f_seen
            )
           
        return self.state