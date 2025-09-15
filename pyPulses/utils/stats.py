import numpy as np

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
