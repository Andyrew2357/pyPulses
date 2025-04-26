import numpy as np
import os
from typing import Optional
from ..utils import curves

class GapExtractor():
    def __init__(self, ct: Optional[float], cb: Optional[float], 
                 chi_g: Optional[float], chi_b: Optional[float], 
                 omega: Optional[float]):
        self.ct = ct
        self.cb = cb
        self.chi_g = chi_g
        self.chi_b = chi_b
        self.omega = omega

        self.inv_coeff_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), r'ztan_inv_coeff.npy'
        )

        if not os.path.isfile(self.inv_coeff_path):
            self.recalculate_ztan_inv()
        else:
            pchip_coeff = np.load(self.inv_coeff_path, allow_pickle = True)
            self.ztan_inv  = curves.pchip_rval_from_params(*pchip_coeff)
            self.ztan_invp = curves.pchip_dval_from_params(*pchip_coeff)


    def recalculate_ztan_inv(self, xrange = (0, 2), N = 1000):
        xs = np.linspace(*xrange, N)
        ztans = (np.sin(2*xs) - np.sinh(2*xs))/(np.sin(2*xs) + np.sinh(2*xs))
        ztans[0] = 0
        pchip_coeff = curves.pchip_params(*curves.prune_sort(ztans, xs))
        pchip_coeff = np.array([np.array(a) for a in pchip_coeff], dtype = object)
        self.ztan_inv  = curves.pchip_rval_from_params(*pchip_coeff)
        self.ztan_invp = curves.pchip_dval_from_params(*pchip_coeff)
        np.save(self.inv_coeff_path, pchip_coeff, allow_pickle = True)

    def calc_cq_AR(self, chi_re, chi_im = 0, mode = 'tl_model'):
        """
        Back out the quantum capacitance and AR (defined by the transmission 
        line model) from raw capacitance data.
        """

        match mode:
            # transmission line model (intermediate frequencies)
            case 'tl_model':
                
                # calculate Z = (χ_g - χ)/(χ_g - χ_b) in terms of real
                # and imaginary parts (X and Y)
                X = (self.chi_g - chi_re) / (self.chi_g - self.chi_b)
                #(Y is flipped to correct for the time evolution convention of 
                # the lock-in)
                Y = chi_im / (self.chi_g - self.chi_b)
                
                # calculate the z-tangent Y/X
                ztan = Y / X

                # must complex conjugate everything by flipping R if we get a 
                # tangent with the wrong sign (the transmission line model 
                # doesn't really permitthis, but it's conceivable it shows up 
                # anyway).
                if ztan > 0:
                    sign_R = -1
                    ztan = -ztan
                else:
                    sign_R = 1

                # invert ztan = (sin(2x) - sinh(2x))/(sin(2x) + sinh(2x))
                x = self.ztan_inv(ztan)
                # calculate the real and imaginary parts of tanh(x)/x
                if abs(x) < 1e-3:
                    F_re = 1
                    F_im = 0
                else:
                    denom = 2*x*(np.cos(2*x) + np.cosh(2*x))
                    F_re = (np.sin(2*x) + np.sinh(2*x)) / denom
                    F_im = (np.sin(2*x) - np.sinh(2*x)) / denom

                # calculate cq using the determined x
                cT = self.ct + self.cb
                cq = cT*(Y*(Y - F_im) - X*(X - F_re)) / \
                     ((X - F_re)**2 + (Y - F_im)**2)

                # back out AR from x = (ωARcq(ct+cb)/8(cq+ct+cb))^1/2                
                AR = sign_R*(8*(cq + cT)/(self.omega*cq*cT))*x**2

                print(f"chi_re={chi_re}, chi_im={chi_im}, chi_g={self.chi_g}, chi_b={self.chi_b}, ct={self.ct}, cb={self.cb}, cq={cq}, AR={AR}, x={x}, F_re={F_re}, F_im={F_im}")

            # low frequency limit of the transmission line calculation
            case 'low_freq':

                cq = (self.ct + self.cb)*(self.chi_g - chi_re) / \
                                         (chi_re - self.chi_b)
                AR = 0

            case _:
                raise ValueError(f"{mode} is not a valid mode for calc_cq_AR")

        return cq, AR
