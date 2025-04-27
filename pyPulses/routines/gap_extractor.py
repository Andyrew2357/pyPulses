import numpy as np
import os
from typing import Optional
from ..utils import curves

class GapExtractor():
    def __init__(self, ct   : Optional[float] = None, 
                 cb         : Optional[float] = None, 
                 chi_g      : Optional[float] = None, 
                 chi_b      : Optional[float] = None, 
                 omega      : Optional[float] = None,
                 var_chi_g  : Optional[float] = None, 
                 var_chi_b  : Optional[float] = None):
        self.ct = ct
        self.cb = cb
        self.chi_g = chi_g
        self.chi_b = chi_b
        self.omega = omega

        self.var_chi_g = var_chi_g
        self.var_chi_b = var_chi_b

        self.q = 1.60217663e-19

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

    def calc_cq_AR(self, chi_re, chi_im = 0, mode = 'tl_model', derivatives = False):
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

                if derivatives:

                    # the pseudo-analytic version of this doesn't seem to work
                    # for some reason, so I'm just doing it numerically.
                    if False:
                        # see the document I wrote to understand what's happening here

                        if abs(x) < 1e-3:
                            dF_re_dX = dF_im_dX = 0
                            dF_re_dY = dF_im_dY = 0
                        else:
                            gp = self.ztan_invp(ztan)
                            dx_dX = -(Y/X**2)*gp
                            dx_dY = (1/X)*gp

                            Fp_re = (1 - F_re)/x + 4*x*F_re*F_im
                            Fp_im = (2/denom)*(np.cos(2*x) - np.cosh(2*x)) - \
                                                    F_im/x + 4*x*F_im*F_im

                            dF_re_dX = Fp_re*dx_dX
                            dF_im_dX = Fp_im*dx_dX
                            dF_re_dY = Fp_re*dx_dY
                            dF_im_dY = Fp_im*dx_dY

                        D = cT/((X - F_re)**2 + (Y - F_im)**2)
                        r = 2*cq/cT
                        a = 1 - r
                        b = 1 + r
                        c = r - 2

                        dcq_dX = D*(a*(X*dF_re_dX + F_re) + b*Y*dF_im_dX + c*X + \
                                                r*(F_re*dF_re_dX + F_im*dF_im_dX))
                        dcq_dY = -D*(a*(Y*dF_im_dY + F_im) + b*X*dF_re_dY + c*Y + \
                                                r*(F_re*dF_re_dY + F_im*dF_im_dY))


                        d = 1/(self.chi_b - self.chi_g)

                        dX_dchi_r = d
                        dY_dchi_r = 0
                        dX_dchi_i = 0
                        dY_dchi_i = d

                        dX_dchi_g = -d - d*d*(self.chi_g - chi_re)
                        dY_dchi_g = d*d*chi_im
                        dX_dchi_b = d*d*(self.chi_g - chi_re)
                        dY_dchi_b = -d*d*chi_im

                        dcq_dchi_r = dcq_dX*dX_dchi_r + dcq_dY*dY_dchi_r
                        dcq_dchi_i = dcq_dX*dX_dchi_i + dcq_dY*dY_dchi_i
                        dcq_dchi_g = dcq_dX*dX_dchi_g + dcq_dY*dY_dchi_g
                        dcq_dchi_b = dcq_dX*dX_dchi_b + dcq_dY*dY_dchi_b

                    delta = 1e-6
                    dcq_dchi_r = (self.calc_cq_AR(chi_re + delta, chi_im, 
                        mode = 'tl_model', derivatives = False)[0] - cq)/delta
                    
                    dcq_dchi_i = (self.calc_cq_AR(chi_re, chi_im + delta, 
                        mode = 'tl_model', derivatives = False)[0] - cq)/delta
                    
                    chi_g = self.chi_g
                    self.chi_g += delta
                    dcq_dchi_g = (self.calc_cq_AR(chi_re, chi_im, 
                        mode = 'tl_model', derivatives = False)[0] - cq)/delta
                    self.chi_g = chi_g

                    chi_b = self.chi_b
                    self.chi_b += delta
                    dcq_dchi_b = (self.calc_cq_AR(chi_re, chi_im, 
                        mode = 'tl_model', derivatives = False)[0] - cq)/delta
                    self.chi_b = chi_b

            # low frequency limit of the transmission line calculation
            case 'low_freq':

                cq = (self.ct + self.cb)*(self.chi_g - chi_re) / \
                                         (chi_re - self.chi_b)
                AR = 0

                if derivatives:
                    cT = self.ct + self.cb
                    dcq_dchi_r = cT*(self.chi_b - self.chi_g)/(chi_re - self.chi_b)**2
                    dcq_dchi_i = 0
                    dcq_dchi_g = cT/(self.chi_b - chi_re)
                    dcq_dchi_b = cq/(chi_re - self.chi_b)

            case _:
                raise ValueError(f"{mode} is not a valid mode for calc_cq_AR")

        if derivatives:
            return cq, AR, (dcq_dchi_r, dcq_dchi_i, dcq_dchi_g, dcq_dchi_b)

        return cq, AR

    def calc_gap(self, vt, vb, cq):
        """Integrate to determine the gap in eV"""

        mt = (1 + cq/self.ct)**-1
        mb = (1 + cq/self.cb)**-1

        return self.q*(np.trapezoid(mt, vt) + np.trapezoid(mb, vb))
    
    def calc_gap_uncertainty(self, vt, vb, cq, derivatives, cov_chi):
        """
        Calculate the uncertainty for an integrated gap using error propagation
        """

        # derivatives = dcq_dchi_r, dcq_dchi_i, dcq_dchi_g, dcq_dchi_b
        mt2 = (1 + cq/self.ct)**-2
        mb2 = (1 + cq/self.cb)**-2
        dmt_dchi_r, dmt_dchi_i, dmt_dchi_g, dmt_dchi_b = [-d*mt2/self.ct for d in derivatives]
        dmb_dchi_r, dmb_dchi_i, dmb_dchi_g, dmb_dchi_b = [-d*mb2/self.cb for d in derivatives]
        chi_g_part = self.var_chi_g*(np.trapezoid(dmt_dchi_g, vt) + np.trapezoid(dmb_dchi_g, vb))**2
        chi_b_part = self.var_chi_b*(np.trapezoid(dmt_dchi_b, vt) + np.trapezoid(dmb_dchi_b, vb))**2

