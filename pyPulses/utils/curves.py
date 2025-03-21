"""
Utility functions for dealing with curves. This includes interpolation,
inversion, smoothing, etc.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np

def prune_unique(x: Union[np.ndarray, list], 
                 y: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicate x values from a curve"""
    x = np.array(x)
    y = np.array(y)

    idx = np.unique(x, return_index=True)[1]
    return x[idx], y[idx]

def prune_sort(x: Union[np.ndarray, list],
                    y: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate x values and sort so that x is strictly increasing
    """
    x = np.array(x)
    y = np.array(y)
    xn, yn = prune_unique(x, y)
    idx = np.argsort(xn)
    return xn[idx], yn[idx]

def pchip(xs: Union[np.ndarray, list], 
          ys: Union[np.ndarray, list]
          ) -> Callable[[float], Tuple[float, float]]:
    """
    Implementation of shape preserving PCHIP (Piecewise Cubic Hermite 
    Interpolating Polynomial) interpolation. Notably, this interpolation method
    preserves the monotonicity of the underlying data.

    It takes in xt and yt, the x and y values of the data points, and returns a
    function that maps x values to interpolated y and dy/dx values.

    See wikipedia.org/wiki/Monotone_cubic_interpolation for the algorithm and
    JavaScript implementation on which this is based.
    """
    xs = np.array(xs, dtype = float)
    ys = np.array(ys, dtype = float)
    
    length = len(xs)
    if length != len(ys):
        raise ValueError("xt and yt must have the same length")
    
    if length == 0:
        return lambda x: [0.0, 0.0]
    
    if length == 1:
        return lambda x: [ys[0], 0.0]
    
    # sort xs and ys by xs
    sorted_indices = np.argsort(xs)
    xs = xs[sorted_indices]
    ys = ys[sorted_indices]

    # get the slopes between points
    dys = np.diff(ys)
    dxs = np.diff(xs)
    ms  = dys / dxs

    # get degree 1 coefficients
    c1s = np.zeros(length)
    c1s[0] = ms[0]

    for i in range(len(dxs) - 1):
        m = ms[i]
        m_next = ms[i + 1]

        if m * m_next <= 0:
            c1s[i + 1] = 0.0
        else:
            dx = dxs[i]
            dx_next = dxs[i + 1]
            common = dx + dx_next
            c1s[i + 1] = 3.0 * common / ((common + dx_next) / m + (common + dx) / m_next)

    c1s[-1] = ms[-1]

    # get degree 2 and 3 coefficients
    c2s = np.zeros(length - 1)
    c3s = np.zeros(length - 1)

    for i in range(length - 1):
        c1 = c1s[i]
        m_ = ms[i]
        inv_dx = 1.0 / dxs[i]
        common_ = c1 + c1s[i + 1] - m_ - m_

        c2s[i] = (m_ - c1 - common_) * inv_dx
        c3s[i] = common_ * inv_dx * inv_dx

    # return interpolant function
    def interp(x):
        # handle out-of-bounds x values
        if xs < xs[0]:
            i = 0
        elif x > xs[-1]:
            i = length - 2
        else:
            # Binary search to find the interval that contains x
            i = np.searchsorted(xs, x, side = 'right') - 1

        # If x is one of the original points
        if x == xs[i]:
            return ys[i], c1s[i]
        
        # perform interpolation
        diff = x - xs[i]
        rval = ys[i] + diff * (c1s[i] + diff * (c2s[i] + diff * c3s[i]))
        dval = c1s[i] + diff * (2.0 * c2s[i] + diff * 3.0 * c3s[i])

        return rval, dval
    
    return interp
