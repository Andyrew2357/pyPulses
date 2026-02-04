"""
Utility functions for dealing with curves. This includes interpolation,
inversion, smoothing, etc.
"""

from typing import Callable, List, Tuple
import numpy as np
import bisect

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

class MonotonicPiecewiseLinear():
    def __init__(self, x_breaks, y_breaks):
        """
        Parameters
        ----------
        x_breaks : array, shape (n,)
            Breakpoints in x (must be sorted).
        y_breaks : array, shape (n,)
            Function values at the breakpoints (same length as x_breaks).
        """
        if len(x_breaks) != len(y_breaks):
            raise ValueError("x_breaks and y_breaks must have same length")
        self.xb = np.asarray(x_breaks)
        self.yb = np.asarray(y_breaks)

    def __call__(self, x):
        """Forward evaluation"""
        x = np.asarray(x)
        idx = np.searchsorted(self.xb, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.xb) - 2)

        x0, x1 = self.xb[idx], self.xb[idx+1]
        y0, y1 = self.yb[idx], self.yb[idx+1]

        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def inverse(self, y):
        """Inverse evaluation"""
        y = np.asarray(y)
        idx = np.searchsorted(self.yb, y, side="right") - 1
        idx = np.clip(idx, 0, len(self.yb) - 2)

        x0, x1 = self.xb[idx], self.xb[idx+1]
        y0, y1 = self.yb[idx], self.yb[idx+1]

        return x0 + (x1 - x0) * (y - y0) / (y1 - y0)

"""Generic Integration tools (repurposed from gap extraction calculations)"""

def pad_z(a: np.ndarray) -> np.ndarray:
    """Pad zeros along the final axis"""

    pad_width = [(0, 0)] * (a.ndim - 1) + [(1, 0)]
    return np.pad(a, pad_width, mode = 'constant', constant_values = 0.0)

def integrate_trapz(x: np.ndarray, f: np.ndarray, mask_nans: bool = False
                    ) -> np.ndarray:
    """
    Vectorized trapezoidal integration along the last axis.
    If 'mask_nans' is True, segments with NaNs are skipped.
    """
    f0 = f[..., :-1]
    f1 = f[..., 1:]
    dx = np.diff(x, axis=-1)

    if mask_nans:
        mask = np.isnan(f0) | np.isnan(f1)
        integrand = np.where(mask, 0.0, 0.5 * (f0 + f1) * dx)
    else:
        integrand = 0.5 * (f0 + f1) * dx

    return np.cumsum(integrand, axis=-1)

def integrate_trapz_padded(x: np.ndarray, f: np.ndarray, 
                           mask_nans: bool = False) -> np.ndarray:
    """Like '_integrate_trapz', but prepends a zero for shape alignment."""
    return pad_z(integrate_trapz(x, f, mask_nans))

def squared_integral(dx: np.ndarray, f: np.ndarray, mask_nans: bool = False
                     ) -> np.ndarray:
    """
    Compute cummulated sum of square of trapezoidal integrals along the last 
    axis.
    """
    f0 = f[..., :-1]
    f1 = f[..., 1:]

    if mask_nans:
        mask = np.isnan(f0) | np.isnan(f1)
        integrand = np.where(mask, 0.0, 0.5 * (f0 + f1) * dx)
    else:
        integrand = 0.5 * (f0 + f1) * dx

    return np.cumsum(integrand**2, axis = -1)