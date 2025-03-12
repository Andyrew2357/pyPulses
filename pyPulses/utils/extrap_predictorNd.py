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

class ExtrapPredNd():
    def __init__(self,
                 pspace_shape   : Tuple[int, ...],      # phase space shape 
                 support        : Tuple[int, ...],      # support
                 order          : Tuple[int, ...],      # polynomial order
                 default0       : Callable[..., float], # default first guess
                 default1       : Callable[..., float], # default second guess
                 axes           : Tuple[np.ndarray, ...]# phase space axes
                 ):
        
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

    def predict0(self, p: Tuple[float, ...]) -> float:
        if np.all(self.pointer == 0):
            return self.default0(p)
        
        xcuts = {}
        zcuts = {}
        max_cut_len = 0
        n_max_cut = 0
        next_max_cut_len = 0
        for d in range(self.ndim):
            pd = self.pointer.copy()
            id_end = pd[d]

            n = min(self.order[d], pd[d])
            
            xcut = []
            zcut = []
            for id in range(id_end - n, id_end):
                pd[d] = id

                z = self.balance_history[*pd]
                if np.isinf(z):
                    continue

                xcut.append(self.axes[d][id])
                zcut.append(z)

            xcuts[d] = np.array(xcut)
            zcuts[d] = np.array(zcut)
            
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

        for d in range(self.ndim):
            if len(xcuts[d]) < max_cut_len:
                if not (n_max_cut == 1 and len(xcuts[d]) == next_max_cut_len):
                    continue

                coeff = np.polyfit(xcut[d], zcut[d], 
                                   min(self.order[d], next_max_cut_len))
                
                poly = np.poly1d(coeff)
                zpred = poly(p[d])

                perf = np.sum((poly(xcuts[d]) - zcuts[d])**2) / next_max_cut_len

                if perf < z1_guess_perf:
                    z1_guess_perf = perf
                    self.z1_guess = zpred
            
            else:
                coeff = np.polyfit(xcut[d], zcut[d], 
                                   min(self.order[d], max_cut_len))
                
                poly = np.poly1d(coeff)
                zpred = poly(p[d])

                perf = np.sum((poly(xcuts[d]) - zcuts[d])**2) / max_cut_len

                if perf < z0_guess_perf:
                    z1_guess_perf = z0_guess_perf
                    z0_guess_perf = perf

                    self.z1_guess = z0_guess
                    z0_guess = zpred

        return z0_guess

    def predict1(self, p: Tuple[float, ...], meas: Tuple[float, ...]):
        if self.z1_guess is None:
            return self.default1(p, meas, self)
        
        return self.z1_guess

    def update(self, p, z):
        self.pointer = np.array(
            [np.sum(np.where(self.axes[d] == p[d])) 
                                for d in range(self.ndim)]
        )
        self.balance_history[*self.pointer] = z
        