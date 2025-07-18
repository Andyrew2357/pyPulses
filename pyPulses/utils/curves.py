"""
Utility functions for dealing with curves. This includes interpolation,
inversion, smoothing, etc.
"""

from typing import Callable, Tuple
import numpy as np

def prune_sort(x: np.ndarray | list, y: np.ndarray | list
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove non, monotonic values in x so that x is strictly increasing or 
    decreasing. Then sort so that x is increasing. Whether we assume decreasing 
    or increasing is determined by the first two elements.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    x, y : array-like
    """
    x = np.array(x)
    y = np.array(y)
    xn = x.copy()
    yn = y.copy()
    s = 1 if x[0] < x[1] else -1
    
    idx = [0, 1]
    xp = xn[1]
    for i in range(2, xn.size):
        if s*xn[i] > s*xp:
            idx.append(i)
            xp = xn[i]

    return xn[idx], yn[idx]

def pchip(xs: np.ndarray | list, ys: np.ndarray | list
          ) -> Callable[[float], Tuple[float, float]]:
    """
    Implementation of shape preserving PCHIP (Piecewise Cubic Hermite 
    Interpolating Polynomial) interpolation. Notably, this interpolation method
    preserves the monotonicity of the underlying data.

    See wikipedia.org/wiki/Monotone_cubic_interpolation for the algorithm and
    JavaScript implementation on which this is based.

    Parameters
    ----------
    xs, ys : array-like

    Returns
    -------
    f : Callable
        function maps (x) -> (y, dy/dx)
    """

    # calculate the interpolation parameters
    xs, ys, c1s, c2s, c3s = pchip_params(xs, ys)
    return pchip_interp_from_params(xs, ys, c1s, c2s, c3s)

def pchip_params(xs: np.ndarray | list, ys: np.ndarray | list
                ) -> Tuple[np.ndarray, ...]:
    """
    Implementation of shape preserving PCHIP (Piecewise Cubic Hermite 
    Interpolating Polynomial) interpolation. Notably, this interpolation method
    preserves the monotonicity of the underlying data.

    See wikipedia.org/wiki/Monotone_cubic_interpolation for the algorithm and
    JavaScript implementation on which this is based.

    Parameters
    ----------
    xs, ys : array-like

    Returns
    -------
    xs, ys, c1s, c2s, c3s : np.ndarray
        input samples and coefficients required for piecewise interpolation.
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

    return xs, ys, c1s, c2s, c3s

def pchip_interp_from_params(xs, ys, c1s, c2s, c3s):
    """
    Interpolation functiom compatible with numpy arrays that returns both
    the interpolated value and interpolated derivative

    Parameters
    ----------
    xs, ys, c1s, c2s, c3s : np.ndarray
        sample points and interpolation coefficients.
    
    Returns
    -------
    f : Callable
        f maps (x) -> (y, dy/dx).
    """
    length = len(xs)

    @np.vectorize
    def interp(x):
        # handle out-of-bounds x values
        if x < xs[0]:
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

def pchip_rval_from_params(xs, ys, c1s, c2s, c3s):
    """
    Interpolation function compatible with numpy arrays
    
    Parameters
    ----------
    xs, ys, c1s, c2s, c3s : np.ndarray
        sample points and interpolation coefficients.
    
    Returns
    -------
    f : Callable
        f maps (x) -> (y).
    """

    length = len(xs)
    @np.vectorize
    def interp(x):
        # handle out-of-bounds x values
        if x < xs[0]:
            i = 0
        elif x > xs[-1]:
            i = length - 2
        else:
            # Binary search to find the interval that contains x
            i = np.searchsorted(xs, x, side = 'right') - 1

        # If x is one of the original points
        if x == xs[i]:
            return ys[i]
        
        # perform interpolation
        diff = x - xs[i]
        rval = ys[i] + diff * (c1s[i] + diff * (c2s[i] + diff * c3s[i]))

        return rval
    
    return interp

def pchip_dval_from_params(xs, ys, c1s, c2s, c3s):
    """
    Derivative interpolation function compatible with numpy arrays
    
    Parameters
    ----------
    xs, ys, c1s, c2s, c3s : np.ndarray
        sample points and interpolation coefficients.
    
    Returns
    -------
    f : Callable
        f maps (x) -> (dy/dx).
    """
    
    length = len(xs)
    @np.vectorize
    def interp(x):
        # handle out-of-bounds x values
        if x < xs[0]:
            i = 0
        elif x > xs[-1]:
            i = length - 2
        else:
            # Binary search to find the interval that contains x
            i = np.searchsorted(xs, x, side = 'right') - 1

        # If x is one of the original points
        if x == xs[i]:
            return c1s[i]
        
        # perform interpolation
        diff = x - xs[i]
        dval = c1s[i] + diff * (2.0 * c2s[i] + diff * 3.0 * c3s[i])

        return dval
    
    return interp
