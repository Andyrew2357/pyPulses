"""
This is a reference aware predictor for balance points that directly predicts
future device resistance. The intended usage is to systematically set the 
reference resistance slightly larger than the device resistance and balance
primarily using pulse heights.
"""

# THIS IS NOT FULLY IMPLEMENTED / READY TO BE USED YET. DO NOT USE.

from ..utils.curves import pchip, prune_sort
from ..utils.balance_predictor import ExtrapPredNd, DummyPred
from typing import Tuple
import numpy as np

class RPredictor:
    def __init__(self, Vg: np.ndarray, Rref: np.ndarray, 
                 default_R: float, axes: Tuple[np.ndarray, ...]):
        self.Vg     = Vg
        self.Rref   = Rref
        self.RtoVg  = pchip(*prune_sort(Rref, Vg))
        self.VgtoR  = pchip(Vg, Rref)

        self.Rpredictor = ExtrapPredNd(
            support     = 5,
            order       = 3,
            default0    = lambda *args, **kwargs: default_R,
            default1    = lambda *args, **kwargs: default_R,
            axes        = axes
        )

        self.VyPred = DummyPred(
            f0  = self.predict_Vy0,
            f1  = self.predict_Vy1
        )

        self.VgPred = DummyPred(
            f0  = self.predict_Vg0,
            f1  = self.predict_Vg1
        )


    def predict_Vg0(self, Vx: float, *p: Tuple[float, ...]) -> float:
        Rp = self.Rpredictor.predict0()
        dRp = 0.1*Rp
        Vg_guess = self.RtoVg(Rp + dRp)
        self.Vy_guess = (1 + dRp / Rp) * Vx 
        return Vg_guess

    # THERE HAVE TO BE MUCH BETTER WAYS TO DO THIS
    def predict_Vg1(self, Vx: float, *p: Tuple[float, ...]) -> float:
        Rp = self.Rpredictor.predict1()
        dRp = 0.1*Rp
        Vg_guess = self.RtoVg(Rp + dRp)
        self.Vy_guess = (1 + dRp / Rp) * Vx 
        return Vg_guess

    def predict_Vy0(self, Vx: float, *p: Tuple[float, ...]) -> float:
        return self.Vy_guess

    # FIGURE OUT A NON-STUPID WAY TO DO THIS
    def predict_Vy1(self, Vx: float, *p: Tuple[float, ...]) -> float:
        pass

    def update(self, Vgref: float, Vy: float, 
               Vx: float, *p: Tuple[float, ...]):
        R_measured = self.VgtoR(Vgref) * (Vx / Vy)
        self.Rpredictor.update((Vx, *p), R_measured)
