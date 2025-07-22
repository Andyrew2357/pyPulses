import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

class BalancePredictor(ABC):
    @abstractmethod
    def predict0(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def predict1(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def update(self, *args, **kwargs): ...

class DummyPred(BalancePredictor):
    """
    Dummy predictor that always returns the same values for initial and second 
    guess.
    """
    def __init__(self, x0: float, x1: float):
        """
        Parameters
        ----------
        x0, x1 : float
            first and second guesses
        """
        self.x0 = x0
        self.x1 = x1

    def predict0(self, *args, **kwargs) -> float:
        return self.x0
    
    def predict1(self, *args, **kwargs) -> float:
        return self.x1
    
    def update(self, *args, **kwargs):
        pass

class DummyPredFunc(BalancePredictor):
    """
    Dummy predictor that excepts arbitrary functions for its first and second
    guesses.
    """
    def __init__(self, f0: Callable[[Any], float], f1: Callable[[Any], float]):
        """
        Parameters
        ----------
        f0, f1 : Callable
            functions used for first and second guess respectively
        """
        self.predict0 = f0
        self.predict1 = f1

    def update(self, *args, **kwargs):
        pass

class ExtrapPred1d(BalancePredictor):
    """
    Predict future balance points by performing a polynomial fit on the last few.
    The user must provide functions for default0 and default1.
    """
    def __init__(self, support: int, order: int, 
                 default0: Callable[..., float], default1: Callable[..., float]):
        """
        Parameters
        ----------
        support : int
            number of previous points to use for extrapolation
        order : int
            order of polynomial to use for extrapolation
        default0, default1 : Callable
            functions for making uninformed first and second guesses.
        """
        if order >= support:
            raise RuntimeError(
                f"order exceeds the maximum supported ({order} >= {support})."
            ) 
        self.support    = support
        self.order      = order
        self.default0   = default0
        self.default1   = default1

        self.X = np.zeros(support, dtype = float)
        self.Y = np.zeros(support, dtype = float)
        self.seen: int = 0

    def predict0(self, x: float) -> float:
        if self.seen == 0:
            return self.default0(x)
        
        coeff = np.polyfit(self.X[:self.seen], self.Y[:self.seen], 
                           min(self.order, self.seen - 1))
        poly = np.poly1d(coeff)
        return poly(x)
    
    def predict1(self, x: float, p: Tuple[float, float]) -> float:
        if self.seen == 0:
            return self.default1(x, p, self)
        
        xp, yp = self.get_last()
        return yp
    
    def update(self, x: float, y: float):
        """
        Add a point to the record of previous points

        Parameters
        ----------
        x, y : float
        """
        self.X[1:] = self.X[0:-1]
        self.X[0] = x

        self.Y[1:] = self.Y[0:-1]
        self.Y[0] = y

        if self.seen < self.support:
            self.seen += 1

    def reset(self):
        """Reset the predictor."""
        self.seen = 0

    def get_last(self) -> Tuple[float, float]:
        """
        Get the previous added point
        
        Returns
        -------
        x, y : float
        """
        return self.X[0], self.Y[0]

# I want to significantly alter how this works in the future.
"""
Predict future balance points for an N dimensional phase space. 

The indices run in order of most frequent change (e.g. for a 2d map, the row 
index will correspond to the sweeping direction, and the column index will 
correspond to the stepping direction). Ideally, all balancing parameters will be
updated with their extrapolated balance point continuously, without necessarily
having to balance each one; provided all these predictions are reasonably good,
this makes extrapolation feasible, since there should be no large jumps in any
balance parameter. It is also recommended to use lower order polynomials for 
extrapolation to avoid overfitting, since this also leads to overcorrecting
balance parameters. 

To predict the balance parameter at a new point in phase space, we attempt to
extrapolate along each direction in phase space for which we have previous
points, first prioritizing directions with the most previous points (up to some
support argument) and then prioritizing directions with the best polynomial 
fits. The logic of this is that we want to extrapolate using directions for
which we have the most information and which are best represented by the
extrapolation model. For this reason, it is very important to avoid overfitting.

For our first guess of a balance parameter, we use the highest priority 
direction. For the second, we use the next highest priority direction.
"""

import numpy as np
from typing import Callable, Tuple

class ExtrapPredNd(BalancePredictor):
    """Predict future balance points for an N dimensional phase space."""
    def __init__(self,
                 pspace_shape   : Tuple[int, ...],      # phase space shape 
                 support        : Tuple[int, ...],      # support
                 order          : Tuple[int, ...],      # polynomial order
                 default0       : Callable[..., float], # default first guess
                 default1       : Callable[..., float], # default second guess
                 axes           : Tuple[np.ndarray, ...]# phase space axes
                 ):
        """
        Parameters
        ----------
        pspace_shape : tuple of int
            shape of the phase space.
        support : tupe of int
            number of previous points to use for extrapolation in each dimension.
        order : tuple of int
            order of the polynomial to use for extrapolation in each dimension.
        default0, default1 : Callable
            functions for making uninformed first and second guesses.
        axes : tuple of np.ndarray
            axes in each parameter of the phase space
        """
        for n, x in zip(pspace_shape, axes):
            if not n == x.size:
                raise IndexError("Provided axes must match pspace_shape.")
        self.axes = axes

        # Previous established balance points (inf if not initialized)
        self.balance_history = np.full(shape = pspace_shape, 
                                       fill_value = np.inf)

        # Where are we in the scan
        self.pointer = np.zeros(len(pspace_shape), dtype = int)
        
        self.support    = support
        self.order      = order
        self.ndim       = len(pspace_shape)

        self.default0 = default0
        self.default1 = default1

        self.z1_guess = None

    def predict0(self, p: Tuple[float, ...]) -> float:
        if np.all(self.pointer == 0):
            return self.default0(p)
        
        xcuts = {}
        zcuts = {}
        max_cut_len = 0
        n_max_cut = 0
        next_max_cut_len = 0
        
        for d in range(self.ndim):
            pd = np.array(
            [np.sum(np.where(self.axes[d] == p[d])) 
                                for d in range(self.ndim)]
            )
            id_end = pd[d]

            n = min(self.support[d], pd[d])
            
            xcut = []
            zcut = []
            for id in range(id_end - n, id_end):
                pd[d] = id
                z = self.balance_history[*pd]
                if np.isinf(z):
                    continue

                xcut.append(self.axes[d][id])
                zcut.append(z)

            xcuts[d] = np.array(xcut.copy())
            zcuts[d] = np.array(zcut.copy())
            
            if len(xcut) > max_cut_len:
                next_max_cut_len = max_cut_len
                max_cut_len = len(xcut)
                n_max_cut = 0
            elif len(xcut) == max_cut_len:
                n_max_cut += 1

        if max_cut_len == 0:
            return self.default0(p)
        
        z0_guess = None
        z0_guess_perf = np.inf
        self.z1_guess = None
        z1_guess_perf = np.inf

        z0_d = None

        for d in range(self.ndim):
            if len(xcuts[d]) < max_cut_len:
                if next_max_cut_len == 0: 
                    continue
                if not (n_max_cut == 1 and len(xcuts[d]) == next_max_cut_len):
                    continue

                coeff = np.polyfit(xcuts[d], zcuts[d], 
                                   min(self.order[d], next_max_cut_len - 1))
                
                poly = np.poly1d(coeff)
                zpred = poly(p[d])

                perf = np.sum((poly(xcuts[d]) - zcuts[d])**2) / next_max_cut_len

                if perf < z1_guess_perf:
                    z1_guess_perf = perf
                    self.z1_guess = zpred
            
            else:
                coeff = np.polyfit(xcuts[d], zcuts[d], 
                                   min(self.order[d], max_cut_len - 1))
                
                poly = np.poly1d(coeff)
                zpred = poly(p[d])

                perf = np.sum((poly(xcuts[d]) - zcuts[d])**2) / max_cut_len

                if perf < z0_guess_perf:
                    z1_guess_perf = z0_guess_perf
                    z0_guess_perf = perf

                    self.z1_guess = z0_guess
                    z0_guess = zpred

                elif perf < z1_guess_perf:
                    z1_guess_perf = perf
                    self.z1_guess = zpred

        return z0_guess

    def predict1(self, p: Tuple[float, ...], meas: Tuple[float, ...]):
        if self.z1_guess is None:
            return self.default1(p, meas, self)
        
        return self.z1_guess

    def update(self, p: Tuple[float, ...], z: float):
        """
        Add a new point to the record of previous points

        Parameters
        ----------
        p : tuple of float
            location in phase space
        z : float
            value at that point in phase space
        """
        self.pointer = np.array(
            [np.sum(np.where(self.axes[d] == p[d])) 
                                for d in range(self.ndim)]
        )
        self.balance_history[*self.pointer] = z
        