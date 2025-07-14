"""
These are tools for extracting gaps from capacitance data (conversion to 
compressibility and proper integration/error propagation)

It is suggested that the user pass common arguments in a dictionarry à la **kwargs.
Typically this will looks like

{
'cb'        : <bottom gate capacitance per unit area in F-m^-2>
              (Note that this value has no effect on the calculated gap, since 
              it factors into both n and cq in ways that cancel each other.)
'gamma'     : <top gate capacitance / bottom gate capacitance>
'var_gamma  : <variance in gamma>,
'omega'     : <angular frequency>
}

See my notes on gap extraction for a clearer explanation of the math involved.
"""

import numpy as np
import os
from typing import List, Tuple
from ..utils import curves

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
    """Like 'integrate_trapz', but prepends a zero for shape alignment."""
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

def ztan_inv(z: float) -> float: ...
def ztan_invp(z: float) -> float: ...

def get_ztan_inv(xrange = (0, 2), N = 1000):
    inv_coeff_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), r'ztan_inv_coeff.npy'
    )

    global ztan_inv, ztan_invp

    if not os.path.isfile(inv_coeff_path):
        xs = np.linspace(*xrange, N)
        ztans = (np.sin(2*xs) - np.sinh(2*xs))/(np.sin(2*xs) + np.sinh(2*xs))
        ztans[0] = 0
        pchip_coeff = curves.pchip_params(*curves.prune_sort(ztans, xs))
        pchip_coeff = np.array([np.array(a) for a in pchip_coeff], 
                               dtype = object)
        ztan_inv  = curves.pchip_rval_from_params(*pchip_coeff)
        ztan_invp = curves.pchip_dval_from_params(*pchip_coeff)
        np.save(inv_coeff_path, pchip_coeff, allow_pickle = True)

    else:
        pchip_coeff = np.load(inv_coeff_path, allow_pickle = True)
        ztan_inv  = curves.pchip_rval_from_params(*pchip_coeff)
        ztan_invp = curves.pchip_dval_from_params(*pchip_coeff)

get_ztan_inv()

DERIVS_OUT = Tuple[float, float, float, float, float, float]

@np.vectorize
def lf_model(chi_r: float, chi_i: float, chi_g: float, chi_b: float, 
             cb: float = 1.0, gamma: float = 1.0, 
             **kwargs) -> Tuple[float, float]:
    """
    Back out the quantum capacitance and AR from raw capacitance data using a
    low frequency model.
    """
    
    if np.isnan(chi_r) or np.isnan(chi_i):
        return np.nan, np.nan

    return cb * (1 + gamma) * (chi_g - chi_r) / (chi_r - chi_b), 0

@np.vectorize
def lf_model_deriv(chi_r: float, chi_i: float, chi_g: float, chi_b: float,
                   cb: float = 1.0, gamma: float = 1.0, **kwargs
                   ) -> DERIVS_OUT:
    """
    Back out the quantum capacitance and AR from raw capacitance data using a
    low frequency model. Return derivatives with respect to uncertain quantities.
    """

    cq, AR = lf_model(chi_r, chi_i, chi_g, chi_b, cb, gamma, **kwargs)
    cT = cb * (1 + gamma)
    dcq_dchi_r = cT*(chi_b - chi_g) / (chi_r - chi_b)**2
    dcq_dchi_i = 0
    dcq_dchi_g = cT/(chi_b - chi_r)
    dcq_dchi_b = cq/(chi_r - chi_b)

    return cq, AR, dcq_dchi_r, dcq_dchi_i, dcq_dchi_g, dcq_dchi_b

@np.vectorize
def tl_model(chi_r: float, chi_i: float, 
             chi_g: float, chi_b: float, 
             cb: float = 1.0, gamma: float = 1.0, omega: float = 0.0,
             **kwargs) -> Tuple[float, float]:
    """
    Back out the quantum capacitance and AR from raw capacitance data using a
    transmission line model. 
    """

    if np.isnan(chi_r) or np.isnan(chi_i):
        return np.nan, np.nan

    # calculate Z = (χ_g - χ)/(χ_g - χ_b) in terms of real and imaginary parts 
    # (X and Y)
    X = (chi_g - chi_r) / (chi_g - chi_b)
    #(Y is flipped to correct for the time evolution convention of the lock-in)
    Y = chi_i / (chi_g - chi_b)
    if abs(Y) < 1e-2:
        return lf_model(chi_r, chi_i, chi_g, chi_b, cb, gamma, **kwargs)
    
    # calculate the z-tangent Y/X
    ztan = Y / X

    # must complex conjugate everything by flipping R if we get a tangent with 
    # the wrong sign (the transmission line model doesn't really permitthis, but 
    # it's conceivable it shows up anyway).
    if ztan > 0:
        sign_R = -1
        ztan = -ztan
    else:
        sign_R = 1

    # invert ztan = (sin(2x) - sinh(2x))/(sin(2x) + sinh(2x))
    x = ztan_inv(ztan)
    # calculate the real and imaginary parts of tanh(x)/x
    if abs(x) < 1e-3:
        F_re = 1
        F_im = 0
    else:
        denom = 2*x*(np.cos(2*x) + np.cosh(2*x))
        F_re = (np.sin(2*x) + np.sinh(2*x)) / denom
        F_im = (np.sin(2*x) - np.sinh(2*x)) / denom

    # calculate cq using the determined x
    cT = cb * (1 + gamma)
    cq = cT * (Y*(Y - F_im) - X*(X - F_re)) / ((X - F_re)**2 + (Y - F_im)**2)

    # back out AR from x = (ωARcq(ct+cb)/8(cq+ct+cb))^1/2                
    AR = sign_R*(8*(cq + cT)/(omega*cq*cT))*x**2

    return cq, AR

@np.vectorize
def tl_model_deriv(chi_r: float, chi_i: float, 
                   chi_g: float, chi_b: float, 
                   cb: float = 1.0, gamma: float = 1.0, 
                   omega: float = 0.0, delta: float = 1e-8,
                   **kwargs) -> DERIVS_OUT:
    """
    Back out the quantum capacitance and AR from raw capacitance data using a
    transmission line model. Return derivatives with respect to uncertain
    quantities.
    """

    cq, AR = tl_model(chi_r, chi_i, chi_g, chi_b, cb, gamma, omega, **kwargs)

    dcq_dchi_r = (tl_model(chi_r + delta, chi_i, chi_g, chi_b, 
                           cb, gamma, omega, **kwargs) - cq)[0] / delta
    dcq_dchi_i = (tl_model(chi_r, chi_i + delta, chi_g, chi_b, 
                           cb, gamma, omega, **kwargs) - cq)[0] / delta
    dcq_dchi_g = (tl_model(chi_r, chi_i, chi_g + delta, chi_b, 
                           cb, gamma, omega, **kwargs) - cq)[0] / delta
    dcq_dchi_b = (tl_model(chi_r, chi_i, chi_g, chi_b + delta, 
                           cb, gamma, omega, **kwargs) - cq)[0] / delta

    return cq, AR, dcq_dchi_r, dcq_dchi_i, dcq_dchi_g, dcq_dchi_b

def gap(vt: List[float], vb: List[float], cq: List[float], cb: float = 1.0, 
        gamma: float = 1.0, mask_nans: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the change in chemical potential at all points along the domain
    """

    vt = np.array(vt)
    vb = np.array(vb)
    cq = np.array(cq)

    mt = (1 + cq/ (gamma * cb))**-1
    mb = (1 + cq / cb)**-1

    return integrate_trapz_padded(vt, mt, mask_nans) + \
            integrate_trapz_padded(vb, mb, mask_nans)

def gap_unc(vt: List[float], vb: List[float], cq: List[float], 
            dcq_dchi_r: list[float], dcq_dchi_i: List[float], 
            dcq_dchi_g: List[float], dcq_dchi_b: List[float],
            var_chi_r: float, var_chi_i: float,
            var_chi_g: float, var_chi_b: float,
            cb: float = 1.0, gamma: float = 1.0, var_gamma: float = 0.0,
            mask_nans: bool = False, **kwargs) -> np.ndarray:
    """
    Assign an uncertainty in the change of chemical potential at all points 
    along the domain.
    """

    vt = np.array(vt)
    vb = np.array(vb)
    cq = np.array(cq)
    dcq_dchi_r = np.array(dcq_dchi_r)
    dcq_dchi_i = np.array(dcq_dchi_i)
    dcq_dchi_g = np.array(dcq_dchi_g)
    dcq_dchi_b = np.array(dcq_dchi_b)

    mt2 = (1 + cq / (gamma * cb))**-2
    mb2 = (1 + cq / cb)**-2
    
    dmt_dchi_r = -dcq_dchi_r * mt2 / (gamma * cb)
    dmt_dchi_i = -dcq_dchi_i * mt2 / (gamma * cb)
    dmt_dchi_g = -dcq_dchi_g * mt2 / (gamma * cb)
    dmt_dchi_b = -dcq_dchi_b * mt2 / (gamma * cb)

    dmb_dchi_r = -dcq_dchi_r * mb2 / cb
    dmb_dchi_i = -dcq_dchi_i * mb2 / cb
    dmb_dchi_g = -dcq_dchi_g * mb2 / cb
    dmb_dchi_b = -dcq_dchi_b * mb2 / cb

    dmb_dgamma = cq * mb2 / (cb * gamma**2)

    gamma_part = var_gamma * integrate_trapz(vb, dmb_dgamma, mask_nans)**2
    chi_g_part = var_chi_g * (integrate_trapz(vt, dmt_dchi_g, mask_nans) + \
                              integrate_trapz(vb, dmb_dchi_g, mask_nans))**2
    chi_b_part = var_chi_b * (integrate_trapz(vt, dmt_dchi_b, mask_nans) + \
                              integrate_trapz(vb, dmb_dchi_b, mask_nans))**2
    
    # note, the square being inside for this one is intentional. See my document 
    # for notes on how this works out (note I am ignoring covariance terms here). 
    # If you want to include both real and imaginary variances in chi, as I do 
    # here, the covariance terms will also have real and imaginary cross terms. 
    # The whole thing becomes a mess, and I don't think that the covariance terms 
    # do much in practice, so I ignore them here.

    dx_vt = np.diff(vt, axis = -1)
    dx_vb = np.diff(vb, axis = -1)

    chi_re_part = var_chi_r * (
       squared_integral(dx_vt, dmt_dchi_r, mask_nans) + \
       squared_integral(dx_vb, dmb_dchi_r, mask_nans)
    )
    
    chi_im_part = var_chi_i * (
        squared_integral(dx_vt, dmt_dchi_i, mask_nans) + \
        squared_integral(dx_vb, dmb_dchi_i, mask_nans)    
    )

    return pad_z(np.sqrt(gamma_part + chi_g_part + 
                         chi_b_part + chi_re_part + chi_im_part))

def mu(vt: List[float], vb: List[float], 
       chi_r: List[float], chi_i: List[float],
       chi_g: float, chi_b: float,
       var_chi_r: float = 0.0, var_chi_i: float = 0.0,
       var_chi_g: float = 0.0, var_chi_b: float = 0.0,
       cb: float = 1.0, gamma: float = 1.0, var_gamma: float = 0.0,
       omega: float = 0.0, mask_nans: bool = False, 
       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate jump in chemical potential and estimated uncertainty for all
    points along the domain.
    """

    mode = kwargs.get('mode', 'tl')
    match mode:
        case 'tl':
            res = tl_model_deriv(chi_r, chi_i, chi_g, chi_b, cb, gamma, omega, 
                                 **kwargs)
        case 'lf':
            res = lf_model_deriv(chi_r, chi_i, chi_g, chi_b, cb, gamma, **kwargs)
        case _:
            raise ValueError(f"Unrecognized mode: {mode}")
    
    cq, Ar, dcq_dchi_r, dcq_dchi_i, dcq_dchi_g, dcq_dchi_b  = res
    mu = gap(vt, vb, cq, cb, gamma, mask_nans, **kwargs)
    mu_unc = gap_unc(vt, vb, cq, 
                     dcq_dchi_r, dcq_dchi_i, dcq_dchi_g, dcq_dchi_b, 
                     var_chi_r, var_chi_i, var_chi_g, var_chi_b, 
                     cb, gamma, var_gamma, mask_nans, **kwargs)
    
    return mu, mu_unc

def phase_correct(chi_r_uncor: np.ndarray, chi_i_uncor: np.ndarray, 
                  X_spur: float, Y_spur: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct for unequal phase rotations between the excitation and standard
        lines. This manifests as a rotation of the in and out of phase signals,
        which we correct for by providing a spurious signal to zero out
        """

        theta = np.arctan2(Y_spur, X_spur)
        chi_re = np.cos(theta)*chi_r_uncor + np.sin(theta)*chi_i_uncor
        chi_im = -np.sin(theta)*chi_r_uncor + np.cos(theta)*chi_i_uncor
        return chi_re, chi_im
