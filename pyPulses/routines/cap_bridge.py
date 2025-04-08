"""
Implementation of capacitance routines for pyPulses, based on Sergio de la 
Barrera's smartysweep implementations in MATLAB.
"""

from dataclasses import dataclass
import numpy as np
import time
from typing import Any, Callable, Optional, Tuple

"""Configuration dataclass for balanceCapBridge function"""
@dataclass
class BalanceCapBridgeConfig:
    small_step  : Optional[Tuple[float, float]] = (0.01, 0.01)  # small step in Vex/Vstd
    large_step  : Optional[Tuple[float, float]] = (0.94, 0.94)  # large step in Vex/Vstd
    Vex         : Optional[float] = None    # Vex to use
    Vstd_range  : Optional[float] = None    # Vstd to use
    Vex_gain    : Optional[float] = 1       # gain associated with Vex
    Vstd_gain   : Optional[float] = 1       # gain associated with Vstd
    Cstd        : Optional[float] = 1       # standard capacitor Cstd
    wait        : Optional[float] = 10      # time to wait after changing voltage
    samples     : Optional[int] = 100       # number of averages to use  
    logger      : Optional[object] = None   # logger
    callback    : Optional[Callable[[int, np.ndarray, np.ndarray], Any]] = None
    ignore_warning: Optional[bool] = False

    def __str__(self):
        s  = f"small_step: {self.small_step:.5f}"
        s += f"small_step: {self.large_step:.5f}"
        s += f"       Vex: {self.Vex:.5f} V_rms"
        s += f"Vstd_range: {self.Vstd_range:.5f} V_rms"
        s += f"  Vex_gain: {self.Vex_gain:.5f}"
        s += f" Vstd_gain: {self.Vstd_gain:.5f}"
        s += f"      Cstd: {self.Cstd:.5f}"
        s += f"      wait: {self.wait:.5f} s"
        s += f"   samples: {self.samples}"
        return s

"""Output for balanceCapBridge function, also used for CapBridge object"""
@dataclass
class CapBridgeBalance:
    status          : bool
    balance_matrix  : Optional[np.ndarray] = None
    Cex             : Optional[float] = None
    Closs           : Optional[float] = None
    Vc0Vex          : Optional[float] = None
    Vr0Vex          : Optional[float] = None
    R               : Optional[float] = None
    phase           : Optional[float] = None
    error           : Optional[Tuple[float, float]] = None
    Vex             : Optional[float] = None
    Cstd            : Optional[float] = None

    def __str__(self):
        s  = f"        status: {'balanced' if self.status else 'unbalanced'}\n"
        s += f"balance_matrix: " + \
                ''.join([f"{e:.5f}   " for e in self.balance_matrix]) + '\n'
        s += f"           Cex: {self.Cex:.5f}\n"
        s += f"         Closs: {self.Closs:.5f}\n"
        s += f"     Vc0 / Vex: {self.Vc0Vex:.5f}\n"
        s += f"     Vr0 / Vex: {self.Vr0Vex:.5f}\n"
        s += f"             R: {self.R:.5f}\n"
        s += f"         phase: {self.phase:.5f}\n"
        s += f"       error_X: {self.error[0]:.5f}\n"
        s += f"       error_Y: {self.error[1]:.5f}\n"
        s += f"           Vex: {self.Vex:.5f}\n"
        s += f"          Cstd: {self.Cstd:.5f}"
        return s

"""Balances the capacitance bridge. Run once before capacitance measurements"""
def balanceCapBridge(C          : BalanceCapBridgeConfig,
                     set_Vex    : Optional[Callable[[float], Any]],
                     get_Vex    : Callable[[], float],
                     set_Vstd   : Optional[Callable[[float], Any]],
                     get_Vstd   : Callable[[], float],
                     set_Vstd_ph: Callable[[float], Any],
                     get_X      : Callable[[], float],
                     get_Y      : Callable[[], float]) -> CapBridgeBalance:
    
    # if an excitation amplitude is provided, set it
    if C.Vex is not None:
        set_Vex(C.Vex/C.Vex_gain)

    # if a standard amplitude is provided, set it
    if C.Vstd_range is not None:
        set_Vstd(C.Vstd_range/C.Vstd_gain)

    C.Vex = get_Vex()
    C.Vstd_range = get_Vstd()

    # log the passed arguments
    if C.logger:
        C.logger.info("Attempting to balance capacitor bridge with parameters:")
        C.logger.info(C)

    vc, vr      = C.small_step    
    dvc, dvr    = C.large_step
    
    if not C.ignore_warning:
        Vhi = C.Vstd_range * max(np.sqrt(vr^2 + (vc + dvc)^2), 
                                 np.sqrt(vc^2 + (vr + dvr)^2))
        if Vhi > 2:
            msg  = f"WARNING: Balance Cancelled.\n"
            msg += f"Standard voltage will reach {Vhi:.5f} V.\n"
            msg += f"To override warning, run with ignore_warning = True."
            if C.logger:
                C.logger.warning(msg)
            else:
                print(msg)
            return CapBridgeBalance(status = False)

    # measure off-balance voltage components at three points 
    # (using fraction of range)
    Vcs = np.array([vc, vc, vc + dvc]) * C.Vstd_range   # rms voltages
    Vrs = np.array([vr, vr + dvr, vr]) * C.Vstd_range
    L = np.zeros((2, 3, C.samples))

    if C.logger:
        C.logger.info("Balancing...")
    
    for n in range(3):
        R = np.sqrt(Vcs(n)^2 + Vrs(n)^2)
        phase = 180 - np.arctan2(Vrs(n), Vcs(n))    # 4-quadrant tangent function
        set_Vstd(R/C.Vstd_gain)
        set_Vstd_ph(phase)
        time.sleep(C.wait)
        for m in range(C.samples):
            L[0, n, m] = get_X()
            L[1, n, m] = get_Y()

            if C.callback:
                C.callback(n * C.samples + m, 
                           np.ndarray([L[0, n, m]]), 
                           np.ndarray([L[1, n, m]]))

    L = np.mean(L, axis = 2)

    # convert remaining fractional voltages to real voltage units
    Vr  = C.Vstd_range * vr
    Vc  = C.Vstd_range * vc
    dVr = C.Vstd_range * dvr
    dVc = C.Vstd_range * dvc

    # the algorithmic part; see Ashoori thesis
    Kr1 = (L[1,2] - L[1,1]) / dVr   # real voltage units (K's are dimensionless)
    Kc1 = (L[1,3] - L[1,1]) / dVc
    Kr2 = (L[2,2] - L[2,1]) / dVr
    Kc2 = (L[2,3] - L[2,1]) / dVc
    P   = 1 / (1 - (Kc1 * Kr2) / (Kr1 * Kc2))
    Vr0 = Vr + (P / Kr1) * ((Kc1 / Kc2) * L[2,1] - L[1,1]) # all rms voltages
    Vc0 = Vc + (P / Kc2) * ((Kr2 / Kr1) * L[1,1] - L[2,1])

    # calculate device capacitance (rms voltages)
    Cex     = C.Cstd * Vc0 / C.Vex
    Closs   = C.Cstd * Vr0 / C.Vex

    # Vc0/Vex and Vr0/Vex
    Vc0Vex = Vc0/C.Vex
    Vr0Vex = Vr0/C.Vex

    balance_matrix = np.array([Kc1, Kc2, Kr1, Kr2, Vc0, Vr0])
    phase = 180 - np.arctan2(Vr0, Vc0)  # 4-quadrant tangent function
    set_Vstd(R / C.Vstd_gain)
    set_Vstd_ph(phase)
    R = np.sqrt(Vc0**2 + Vr0**2)

    out = CapBridgeBalance(
        status          = False,
        balance_matrix  = balance_matrix,
        Cex             = Cex,
        Closs           = Closs,
        Vc0Vex          = Vc0Vex,
        Vr0Vex          = Vr0Vex,
        R               = R,
        phase           = phase,
        Vex             = C.Vex,
        Cstd            = C.Cstd
    )
    
    # set excitation to balance point
    if R > C.Vstd_range:
        msg  = f"WARNING: Balanced Vstd ({R} V) is outside available range."
        msg += f"To proceed, run with higher Vstd_range or  lower Vex."
        if C.logger:
            C.logger.warning(msg)
        else:
            print(msg)

        if C.logger:
            C.logger.info(out)

        return out
    
    else:
        time.sleep(C.wait)
        if C.logger:
            C.logger.info("Balanced.")

    out.error = (get_X(), get_Y())
    out.status = True

    if C.logger:
        C.logger.info(out)

    return out

"""
Capacitance bridge object for taking off-balance measurements.
Generally, measure_capacitance should be called as a pre-callback for a 
parameter sweep, with the getters being used as the measured parameters
"""
class CapBridge():
    def __init__(self, balance: CapBridgeBalance, get_X: Callable[[], float],
                 get_Y: Callable[[], float], logger = None):
        if not balance.status:
            if logger:
                logger.warning("WARNING: Provided balance was unsuccessful.") 
                logger.warning("         Results may be unreliable.")
            else:
                print("WARNING: Provided balance was unsuccessful.")
                print("         Results may be unreliable.")

        self.Kc1, self.Kc2, self.Kr1, self.Kr2, self.Vc0, self.Vr0 = balance.balance_matrix
        self.get_X = get_X
        self.get_Y = get_Y
        self.Vex   = balance.Vex
        self.Cstd  = balance.Cstd

    def XYtoC(self, X: float, Y: float) -> Tuple[float, float]:
        """Convert lock-in measurement to capacitance."""
        L1prime = X
        L2prime = Y
        detK = (self.Kc1 * self.Kr2 - self.Kr1 * self.Kc2)
        Vr0prime = self.Vr0 + (self.Kc2 * L1prime - self.Kc1 * L2prime) / detK
        Vc0prime = self.Vc0 + (self.Kr1 * L2prime - self.Kr2 * L1prime) / detK

        Cex   = self.Cstd * Vc0prime / self.Vex
        Closs = self.Cstd * Vr0prime / self.Vex

        return Cex, Closs

    def measure_capacitance(self):
        """Pull data from the lock-in and convert to capacitance."""
        Xoffbal = self.get_X()
        Yoffbal = self.get_Y()
        self.Xoffbal = Xoffbal
        self.Yoffbal = Yoffbal
        self.Cex, self.Closs = self.XYtoC(Xoffbal, Yoffbal)

    def get_Xoffbal(self) -> float:
        """Query the measured in-phase signal"""
        if self.Xoffbal is not None:
            return self.Xoffbal
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")
        
    def get_Yoffbal(self) -> float:
        """Query the measured out-of-phase signal"""
        if self.Yoffbal is not None:
            return self.Yoffbal
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")

    def get_Cex(self) -> float:
        """Query C_ex"""
        if self.Cex is not None:
            return self.Cex
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")

    def get_Closs(self) -> float:
        """Query C_loss"""
        if self.Closs is not None:
            return self.Closs
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")
