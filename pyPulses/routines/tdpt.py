from ..utils import kalman
from ..devices.pulse_pair import pulsePair
from ..devices.wfatd import wfAverager, wfBalance
from ..thread_job import _checkpoint

import logging
from dataclasses import dataclass
from collections import deque
import numpy as np

from typing import Callable, Tuple

def extrap(x: np.ndarray, order: int) -> float:
    coeff = np.polyfit(np.arange(len(x)), x, min(order, len(x) - 1))        
    poly = np.poly1d(coeff)
    return poly(len(x))

class WFilter():
    def __init__(self, support: int, order: int, dMdW: float, P: float):
        self.order = order
        self.W_hist = deque(maxlen = support)
        
        self.Q_balance_change: float
        self.Q_excitation_change: float # NEED TO INITIALIZE THESE!!!!!!
        self.kalman = kalman(
            f_F = self.f_F,
            h_H = self.h_H,
            x = dMdW,
            P = P,
        )

    def f_F(self, dMdW: float, balance_change: bool) -> Tuple[float, float, float]:
        Q = self.Q_balance_change if balance_change else self.Q_excitation_change
        return dMdW, 1, Q
    
    def h_H(self, dMdW) -> Tuple[float, float]:
        return dMdW, 1 

    def append(self, W: float):
        self.W_hist.append(W)

    def extrapolate(self) -> float:
        return extrap(self.W_hist, self.order)

class CFilter(kalman):
    def __init__(self, G: float, Cac: float, P: np.ndarray):
        super().__init__(
            f_F = self.f_F,
            h_H = self.h_H,
            x = np.array([G, Cac, 0]).reshape(-1, 1)
        )

        self.Q_balance_change: np.ndarray
        self.Q_excitation_change: np.ndarray

    def f_F(self, X: np.ndarray, dVx: float = 0.0):
        X[1] += X[2] * dVx
        F = np.eye(3)
        F[1, 2] += dVx
        Q = self.Q_balance_change if dVx == 0 else self.Q_excitation_change
        return X, F, Q

    def h_H(self, X: np.ndarray, VY: float, VX: float):
        z = X[0] * (VY - X[1] * VX)
        H = np.array([VY - X[1] * VX, -X[0] * VX, 0]).reshape(1, -1)
        return z, H

@dataclass
class TDPTBalanceResult():
    status: bool

@dataclass
class TDPTContext():
    exc: pulsePair
    dis: pulsePair
    pulse_height_res: float
    averager: wfAverager
    Cac_balance: wfBalance   # Ordinarily wfJump
    C_error_thresh: float

    W_balance: wfBalance     # Ordinarily wfSlope
    min_W: float
    max_W: float
    W_res: float
    W_error_thresh: float

    settle_time: float = 0.1
    max_tries: int = 10
    order: int = 3
    support: int = 5
    iteration_callback: Callable = None
    logger: logging.Logger = None

    def __post_init__(self):
        self.wfilter: WFilter | None = None
        self.cfilter: CFilter | None = None
        self.prev_result: TDPTBalanceResult | None = None

    def add_filters(self, G: float, Cac: float, PCac: np.ndarray, dMdW: float, PdMdW: float):
        self.wfilter = WFilter(self.support, self.order, dMdW, PdMdW)
        self.cfilter = CFilter(G, Cac, PCac)
        
    def clear_filters(self):
        self.wfilter = None
        self.cfilter = None

    def predict_Cac(self, dVx: float = 0.0):
        self.cfilter.predict(dVx)

    def update_Cac(self, z: float, R: float, VY: float, VX: float):
        self.cfilter.update(z, R, VY, VX)

    def get_Cac(self) -> float:
        Cac = self.cfilter.x[0]
        if Cac < 0:
            self.log(
                f"WARNING: Cac estimate is currently negative ({Cac:.5e}).\n"
                "Ignoring sign change."
            )
            Cac = - Cac
        return Cac

    def predict_dMdW_balance_change(self):
        self.wfilter.kalman.predict(True)

    def predict_dMdW_excitation_change(self):
        self.wfilter.kalman.predict(False)

    def update_dMdW(self, dMdW: float, R: float):
        self.wfilter.kalman.update(dMdW, R)

    def get_dMdW(self) -> float:
        return self.wfilter.kalman.x

    def extrapolate_W(self) -> float:
        return self.wfilter.extrapolate()

    def append_W(self, W: float):
        self.wfilter.append(W)

    def log(self, *args):
        if self.logger:
            self.logger.info(*args)

def _limit_change(v, old_v, max_dv) -> Tuple[float, bool]:
    if v > old_v + max_dv:
        return old_v + max_dv, False
    elif v < old_v - max_dv:
        return old_v - max_dv, False
    return v, True

def _initialBalanceTDCS(ctx: TDPTContext) -> Tuple[float, float]:
    pass

def balanceTDCS(ctx: TDPTContext) -> TDPTBalanceResult:
    if ctx.wfilter is None or ctx.cfilter is None:
        _initialBalanceTDCS(ctx)
    
    Xexc = ctx.exc.X()
    Xdis = ctx.dis.X()
    pYb = ctx.exc.Y()
    pYdis = ctx.dis.Y()
    pWb = ctx.dis.W()
    pR = np.inf
    R = None
    for itr in range(ctx.max_tries):
        _checkpoint()
        if ctx.iteration_callback:
            ctx.iteration_callback(itr, ctx)

        # Predict excpected balance parameters
        Cac = ctx.get_Cac()
        dMdW = ctx.get_dMdW()
        Yb = -Cac * Xexc
        Ydis = -Cac * Xdis

        if itr != 0:
            # For refinements, we want to limit the changes
            temp_Yb = Yb
            temp_Ydis = Ydis
            Yb, Yb_change_success = _limit_change(Yb, pYb, ctx.max_refinement_dYexc)
            Ydis, Ydis_change_success = _limit_change(Ydis, pYdis, ctx.max_refinement_dYdis)
            if not Yb_change_success:
                ctx.log(f"Requested change in Yb ({temp_Yb - pYb:.5e}) "
                        "exceeded the value allowed for refinements!\n")
            if not Ydis_change_success:
                ctx.log(f"Requested change in Ydis ({temp_Ydis - pYdis:.5e}) "
                        "exceeded the value allowed for refinements!\n")
        
        # Set the expected balance parameters    
        ctx.log(
            f"Moving to predicted balance point:\n"
            f"  Cac = {Cac:.5e}, dMdW = {Cac:.5e}\n"
            f"  Yexc = {Yb:.5e}\n"
            f"  Ydis = {Ydis:.5e}\n"
            f"     W = {Wb:.5e}"
        )
        ctx.exc.Y(Yb)
        ctx.dis.Y(Ydis)
        ctx.dis.W(Wb)
        Yb = ctx.exc.Y()
        Ydis = ctx.dis.Y()
        Wb = ctx.dis.W()

        # Filters acknowledge the change in balance parameters
        ctx.predict_dMdW_balance_change()
        ctx.predict_Cac()

        # Take time domain curves
        ctx.averager.take_curve()
        dQ, RdQ = ctx.Cac_balance()
        M, RM = ctx.W_balance()

        # Kalman updates based on the measurement
        ctx.update_Cac(dQ, RdQ, Yb, Xexc)
        if itr != 0:
            # If this is not the first iteration, we update dM/dW
            dM = M - pM
            dW = Wb - pWb
            o_dMdW = dM / dW
            R = (RM + pRM) / dW**2 + (ctx.W_res * o_dMdW / dW)**2
            ctx.update_dMdW(o_dMdW, R)

        # Store the old parameters
        pCac = Cac
        pdMdW = dMdW
        pYb = Yb
        pYdis = Ydis
        pWb = Wb

        #Update the new predicted balance parameters
        Cac = ctx.get_Cac()
        dMdW = ctx.get_dMdW()
        Yb = Cac * Xexc
        Ydis = Cac * Xdis

        # W undergoes bracketed root finding
        Wb -= M / dMdW

        # Check for termination

        # THIS IS NOT COMPLETE - NEED TO FIGURE OUT TERMINATION CONDITIONS, LOGIC FOR DISCHARGE ROOT FINDING, ETC.

        # We limit changes based on ctx parameters and bracketing
        
        pdQ = dQ
        pM = M
        pRdQ = RdQ
        pRM = RM
        pR = R
    else:
        pass