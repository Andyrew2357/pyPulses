"""
This is a reference aware predictor for balance points that directly predicts
future device resistance. The intended usage is to systematically set the 
reference resistance slightly larger than the device resistance and balance
primarily using pulse heights.
"""

# THIS IS NOT FULLY IMPLEMENTED / READY TO BE USED YET. DO NOT USE.

from .curves import pchip, prune_sort
from .extrap_predictorNd import ExtrapPredNd
from .extrap_predictor1d import ExtrapPred1d
from .dummy_predictor import DummyPred
from typing import Tuple
import numpy as np

class RPredictor:
    def __init__(self, Vg: np.ndarray, Rref: np.ndarray, 
                 default_R: float, axes: Tuple[np.ndarray, ...]):
        self.Vg     = Vg
        self.Rref   = Rref
        self.RtoVg  = pchip(*prune_sort(Rref, Vg))

        self.Rpredictor = ExtrapPredNd(
            support     = 5,
            order       = 3,
            default0    = lambda *args, **kwargs: default_R,
            default1    = lambda *args, **kwargs: default_R,
            axes        = axes
        )

        self.VyPred = ExtrapPred1d(

        )
        self.VgPred = self.VgPredictor()

