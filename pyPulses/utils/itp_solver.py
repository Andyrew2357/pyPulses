# Note: I've opted to use a modified Brent-Dekker method instead for our
# purposes. The usage of that solver is slightly different from this one, so if
# one wishes to use this for balancing procedures, they should modify it to
# comply with the standard set by brent_solver.py
"""
This class implements a modified version of the ITP method for root finding.
This is likely more sophisticated than our purposes demand, but it may help in
reducing the required number of function calls (costly measurements) to 
converge on a balance point.
"""

from math import ceil, log2, copysign
from typing import Optional

class ITPsolver:
    def __init__(self, eps:float, xa: float, xb: float, ya: float, yb: float,
                 kap1: Optional[float] = 0.1, kap2: Optional[float] = 2, 
                 n0: Optional[float] = 1):
        
        # input validation
        assert( eps > 0, 
            f"bracketITP Error: Expected a positive value for eps, got {eps}."
        )
        assert( xa < xb, 
            f"bracketITP Error: Invalid bracketing interval {xa} >= {xb}."
        )
        assert( ya*yb < 0, 
            f"bracketITP Error: Invalid bracketing interval {ya}*{yb} >= 0."
        )

        # we flip the function sign internally if ya > yb, because our method
        # assumes ya < yb.
        self.eta = copysign(1, yb - ya)

        self.eps:   float = eps             # target error bound
        self.xa:    float = xa              # bracket lower bound
        self.xb:    float = xb              # bracket upper bound
        self.ya:    float = ya * self.eta   # function value at xa
        self.yb:    float = yb * self.eta   # function value at xb
        
        # tunable hyperparameters for ITP
        self.kap1:  float = kap1    # kap1 > 0, usually ~0.1
        self.kap2:  float = kap2    # kap2 in [1, 1 + golden_ratio)
        self.n0:    float = n0      # n0 > 0, usually ~1

        # preprocessed parameters
        self.n12 = ceil(log2((xb - xa) / (2 * eps)))
        self.nmax = n0 + self.n12
        self.j = 0

    def update_xITP(self) -> float:
        """Determine the next bracket bound for ITP iteration"""
        
        # midpoint of the bracketed interval
        x12 = (self.xa + self.xb) / 2.
        # maximum step from x12 to xITP
        r = self.eps * (2**(self.nmax - self.j)) - (self.xb - self.xa) / 2

        # Interpolation (regula falsi point)
        xf = (self.yb * self.xa - self.ya * self.xb) / (self.yb - self.ya)
        # maximum step from xf to xt
        delta = self.kap1 * (self.xb - self.xa)**(self.kap2)

        # Truncation (perturb the estimator towards the center)
        sig = copysign(1, x12 - xf)
        xt = xf + sig * delta if delta <= abs(x12 - xf) else x12

        # Projection (project estimator to the minmax interval)
        self.xITP = xt if abs(xt - x12) <= r else x12 - sig * r

    def update_bracket(self, yITP):
        """
        Provided a function evaluation at xITP, update the bracket. Return True
        if the target error bound is met.
        """

        # fix the sign of y to make the bracket increasing
        yITP *= self.eta

        if yITP > 0:
            self.xb = self.xITP
            self.yb = yITP
        elif yITP < 0:
            self.xa = self.xITP
            self.ya = yITP
        else:
            self.xa = self.xb = self.xITP

        self.j += 1
        # have we attained the target error bound?
        return (self.xb - self.xa) <= self.eps
