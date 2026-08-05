"""
Vendor-neutral lock-in channel adapters and the covariance-estimation helper
they build on. `resolve()` on lock-in drivers (SRS SR8xx/SR850, AMETEK/Signal
Recovery 7280, ...) returns instances of these to expose input sensitivity
and averaged X/Y readings as ScalarChannelAdapter-compatible objects.
"""

from __future__ import annotations

import numpy as np

from ..channel_adapter import ScalarChannelAdapter
from .base import LockinLike


def series_correlated_covariance(samples: np.ndarray, L: int = None) -> np.ndarray:
    """
    Provide a covariance estimate that reflects serial correlations present in
    the underlying data. The measurements are assumed to be steady-state.

    Parameters
    ----------
    samples : ndarray
        Shape (N, d) where N is the number of samples and d is the dimension of
        the data.
    L : int, optional
        Maximum autocorrelation length to use

    Returns
    -------
    cov : ndarray
    """

    if samples.ndim != 2:
        raise ValueError("`samples` should be a 2D array")

    N, _ = samples.shape
    v = samples - samples.mean(0)
    if L is None:
        L = max(10, int(N**(1 / 3)))
    L = min(N - 1, L)

    nfft = 1 << (2 * N - 1).bit_length()
    Vf = np.fft.rfft(v, n=nfft, axis=0)
    cross_spec = np.einsum('fk, fj->fkj', Vf, np.conj(Vf)) / N
    full_corr = np.fft.irfft(cross_spec, n=nfft, axis=0)
    Gam = full_corr[:L + 1]

    w = 1.0 - np.arange(L + 1) / (L + 1.0)
    S0 = Gam[0].real
    for k in range(1, L + 1):
        Gk = Gam[k].real
        S0 += w[k] * (Gk + Gk.T)
    return S0/N


class sensitivity_channel(ScalarChannelAdapter):
    """
    ScalarChannel for lock-in input sensitivity in volts.

    Parameters
    ----------
    parent : LockinLike
    lockin_scale : float
        The scale factor used by the accompanying lockin_call channel
        (e.g. 1e6 if readings are in µV). Sensitivity targets derived
        from lock-in readings must be divided by this before being passed
        to input_sensitivity().
    """
    def __init__(self, parent: LockinLike, lockin_scale: float = 1.0):
        super().__init__(parent, 'input_sensitivity')
        self.lockin_scale = lockin_scale

    def get_output(self) -> float:
        return self._parent.input_sensitivity() * self.lockin_scale

    def set_output(self, value: float):
        self._parent.input_sensitivity(value / self.lockin_scale)


class lockin_channel():
    """
    LockInChannel for <lockin>.get_average / get_average_series_correlated.

    Parameters
    ----------
    parent : LockinLike
    accessor : str
    scale : float
        1.0 for raw volts, 1e6 for microvolts. Applied as mean*scale,
        cov*(scale**2) so that units are consistent throughout.
    series_corr : bool
        If True, use get_average_series_correlated; else get_average.
    """
    def __init__(self, parent: LockinLike, accessor: str, scale: float, series_corr: bool):
        self._parent      = parent
        self._accessor    = accessor
        self.scale        = scale
        self._series_corr = series_corr

    def format_ref(self):
        return self._parent, self._accessor

    def __call__(self):
        if self._series_corr:
            mean, cov = self._parent.get_average_series_correlated()
        else:
            mean, cov = self._parent.get_average()
        s = self.scale
        return mean * s, cov * (s * s)
