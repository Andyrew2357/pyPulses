from .wf_balance import wfBalanceKnob, wfBalanceCorrelated, wfBalanceResult
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
class TDPTBalanceParms():
    Cac: float
    dMdW: float
    Yb: float
    Ydis: float
    Wb: float
    dQ: float
    RdQ: float
    M: float
    RM: float

@dataclass
class TDPTFilterParms():
    Cfilter_x: np.ndarray
    Cfilter_P: np.ndarray
    Wfilter_x: float
    Wfilter_P: float

@dataclass
class TDPTBalanceResult():
    status      : bool
    parms       : TDPTBalanceParms
    prev_parms  : TDPTBalanceParms | None
    filter_parms: TDPTFilterParms

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

    def getFilterParms(self) -> TDPTFilterParms:
        return TDPTFilterParms(
            Cfilter_x = self.cfilter.x.flatten(),
            Cfilter_P = self.cfilter.P,
            Wfilter_x = self.wfilter.kalman.x,
            Wfilter_P = self.wfilter.kalman.P,
        )

    def log(self, *args):
        if self.logger:
            self.logger.info(*args)

def _limit_change(v, old_v, max_dv) -> Tuple[float, bool]:
    if v > old_v + max_dv:
        return old_v + max_dv, False
    elif v < old_v - max_dv:
        return old_v - max_dv, False
    return v, True

@dataclass
class TDPTInitialResult():
    C_success: bool
    W_success: bool
    bal_success: bool
    C_err: float
    W_err: float
    A: np.ndarray
    b: np.ndarray
    x0: np.ndarray

def initialBalanceTDPT(
    ctx: TDPTContext, 
    min_Cac: float = 0.0, 
    max_Cac: float = 2.0,
    C_err_thresh: float | None = None,
    W_err_thresh: float | None = None,
    reps: int = 2
) -> bool:
    
    if C_err_thresh is None:
        C_err_thresh = ctx.C_error_thresh
    if W_err_thresh is None:
        W_err_thresh = ctx.W_error_thresh

    Xexc = ctx.exc.X()
    Xdis = ctx.dis.X()

    def Cac(C: float | None = None):
        if C is None:
            return ctx.exc.Y() / Xexc
        Yb = -C * Xexc
        Ydis = -C * Xdis
        ctx.exc.Y(Yb)
        ctx.dis.Y(Ydis)

    Cac_knob = wfBalanceKnob(min_Cac, max_Cac, Cac)
    W_knob = wfBalanceKnob(ctx.min_W, ctx.max_W, ctx.dis.W)

    C_ac_s = []
    G_s = []
    dMdW_s = []
    A_s = []
    b_s = []
    x0_s = []
    for i in range(reps):
        bal = wfBalanceCorrelated(
            knobs = [Cac_knob, W_knob], 
            balances = [ctx.Cac_balance, ctx.W_balance],
            averager = ctx.averager
        )
        A_s.append(bal.A)
        b_s.append(bal.b)
        x0_s.append(bal.x0)
        C_ac_s.append(bal.x0[0])
        G_s.append(-bal.A[0, 0] / Xexc)
        dMdW_s.append(bal.A[1, 1])

    A = np.array(A_s).mean(0)
    b = np.array(b_s).mean(0)
    x0 = np.array(x0_s).mean(0)
    C_ac_s = np.array(C_ac_s)
    G_s = np.array(G_s)
    dMdW_s = np.array(dMdW_s)

    C_ac = C_ac_s.mean()
    W = x0[1]
    G = G_s.mean()
    C_stack = np.vstack([G_s - G, C_ac_s - C_ac])
    PCac_small = C_stack @ C_stack.T
    PCac = np.block([[PCac_small, 0], [0, np.inf]]) / reps

    dMdW = dMdW_s.mean()
    PdMdW = dMdW_s.std()**2

    Cac(C_ac)
    ctx.dis.W(W)
    ctx.averager.take_curve()
    Cac_err = ctx.Cac_balance()
    Dis_err = ctx.W_balance()
    ctx.add_filters(G, C_ac, PCac, dMdW, PdMdW)
    return TDPTInitialResult(
        C_success = Cac_err < C_err_thresh,
        W_success = Dis_err < W_err_thresh,
        bal_success = bal.success,
        C_err = Cac_err,
        W_err = Dis_err,
        A = A,
        b = b,
        x0 = x0,
    )

def balanceTDPT(ctx: TDPTContext) -> TDPTBalanceResult:
    if ctx.wfilter is None or ctx.cfilter is None:
        raise RuntimeError(
            "TDPTContext must be initialized before it is used to balance"
        )

    # Fixed Pulse Heights
    Xexc = ctx.exc.X()
    Xdis = ctx.dis.X()

    # Previous Settings
    pYb = ctx.exc.Y()
    pYdis = ctx.dis.Y()
    pWb = ctx.dis.W()
    Wb = pWb

    # Previous Measurements
    pdQ = None
    pRdQ = None
    pM = None
    pRM = None

    # Measurement Uncertainties
    pR = np.inf
    R = None

    # Discharge Pulse Width Bracketing
    W_low = ctx.min_W
    W_high = ctx.max_W

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

        # Update the bracketing on W; We infer which bound needs to be updated  ################### NEED TO FIX THIS / RECONSIDER
        # based on the sign of dM/dW (inferred from Xdis) and M.                ################### BROADLY THERE ARE SIGN CHANGE REQUESTS THAT I SHOULD NOT RESPECT...
        if M * Xdis > 0:
            W_high = min(W_high, Wb)
        else:
            W_low = max(W_low, Wb)

        ctx.log(f"New W bracketing: [{W_low:.5e}, {W_high:.5e}]")

        # Kalman updates based on the measurement
        ctx.update_Cac(dQ, RdQ, Yb, Xexc)
        if itr != 0:
            # If this is not the first iteration, we update dM/dW
            dM = M - pM
            dW = Wb - pWb
            o_dMdW = dM / dW
            R = (RM + pRM) / dW**2 + (ctx.W_res * o_dMdW / dW)**2

            # We warn if there is a sign inconsistent with the sign of Xdis
            if Xdis * o_dMdW < 0:
                ctx.log(f"Inconsistent signs of discharge pulse and slope change.") ############## NEED TO FIX THIS / RECONSIDER (DO I NEED TO EXPLICITLY CORRECT FOR THIS??)

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

        # We limit changes based on bracketing
        if Wb > W_high:
            ctx.log(
                f"Extrapolated discharge balance point (W = {Wb:.5e}) "
                "is too long; Truncating..."
            )
            Wb = W_high

        if Wb < W_low:
            ctx.log(
                f"Extrapolated discharge balance point (W = {Wb:.5e}) "
                "is too short; Truncating..."
            )
            Wb = W_low

        # Check for termination
        exc_sat = abs(dQ) < ctx.C_error_thresh
        dis_sat = abs(M) < ctx.W_error_thresh or (W_high - W_low) < ctx.W_res

        if exc_sat and dis_sat:
            # Append the latest W balance point to the context
            ctx.append_W(Wb)

            return TDPTBalanceResult(
                status = True,
                parms = TDPTBalanceParms(
                    Cac = Cac,
                    dMdW = dMdW,
                    Yb = Yb,
                    Ydis = Ydis,
                    Wb = Wb,
                    dQ = dQ,
                    RdQ = RdQ,
                    M = M,
                    RM = RM,
                ),
                prev_parms = TDPTBalanceParms(
                    Cac = pCac,
                    dMdW = pdMdW,
                    Yb = pYb,
                    Ydis = pYdis,
                    Wb = pWb,
                    dQ = pdQ,
                    RdQ = pRdQ,
                    M = pM,
                    RM = pRM,
                ) if itr != 0 else None,
                filter_parms = ctx.getFilterParms(),
            )
        
        pdQ = dQ
        pM = M
        pRdQ = RdQ
        pRM = RM
        pR = R
    else:
        return TDPTBalanceResult(
            status = False,
            parms = TDPTBalanceParms(
                Cac = Cac,
                dMdW = dMdW,
                Yb = Yb,
                Ydis = Ydis,
                Wb = Wb,
                dQ = dQ,
                RdQ = RdQ,
                M = M,
                RM = RM,
            ),
            prev_parms = TDPTBalanceParms(
                Cac = pCac,
                dMdW = pdMdW,
                Yb = pYb,
                Ydis = pYdis,
                Wb = pWb,
                dQ = pdQ,
                RdQ = pRdQ,
                M = pM,
                RM = pRM,
            ) if itr != 0 else None,
            filter_parms = ctx.getFilterParms(),
        )