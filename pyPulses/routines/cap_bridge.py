"""
Implementation of capacitance routines for pyPulses, based on Sergio de la 
Barrera's smartysweep implementations in MATLAB.
"""

from dataclasses import dataclass
import numpy as np
import time
from typing import Any, Callable, Tuple

"""Configuration dataclass for balanceCapBridge function"""
@dataclass
class BalanceCapBridgeConfig:
    """
    Config for `balanceCapBridge`.
    
    Attributes
    ----------
    time_const : float
        lock-in time constant (informs wait times).
    small_step : Tuple[float, float], default=(0.01, 0.01)
        small step in Vex, Vstd, as proportion of range.
    large_step : Tuple[float, float], default=(0.94, 0.94)
        large step in Vex, Vstd as a proportion of range.
    Vex : float, optional
    Vstd_range : float, optional
    Vex_gain : float, default=1.0
        gain associated with the excitation line.
    Vstd_gain : float, default=1.0
        gain associated with the standard line.
    Cstd : float, default=1.0
        Standard capacitor.
    wait : float, default=3.0
        seconds to wait after changing signal.
    samples : int, default=100
        number of samples to take at each point.
    logger : Logger, default=None
    callback : Callable, default=None
        Callback during balance procedure (usually used for plotting).
    ignore_warning : bool, default=False
        override that lets Vstd exceed its allowed range.
    """
    time_const  : float                                 # time constant for lock-in
    small_step  : Tuple[float, float] = (0.01, 0.01)    # small step in Vex/Vstd
    large_step  : Tuple[float, float] = (0.94, 0.94)    # large step in Vex/Vstd
    Vex         : float = None                          # Vex to use
    Vstd_range  : float = None                          # Vstd to use
    Vex_gain    : float = 1                             # gain associated with Vex
    Vstd_gain   : float = 1                             # gain associated with Vstd
    Cstd        : float = 1                             # standard capacitor Cstd
    wait        : float = 3                             # time to wait after changing voltage
    samples     : int = 100                             # number of averages to use  
    logger      : object = None                         # logger
    callback    : Callable[[int, np.ndarray, np.ndarray], Any] = None
    ignore_warning: bool = False

    def __str__(self):
        s  = f"small_step: {self.small_step[0]:.5e}, {self.small_step[1]:.5e}\n"
        s += f"large_step: {self.large_step[0]:.5e}, {self.large_step[1]:.5e}\n"
        s += f"       Vex: {self.Vex:.5e} V_rms\n"
        s += f"Vstd_range: {self.Vstd_range:.5e} V_rms\n"
        s += f"  Vex_gain: {self.Vex_gain:.5e}\n"
        s += f" Vstd_gain: {self.Vstd_gain:.5e}\n"
        s += f"      Cstd: {self.Cstd:.5e}\n"
        s += f"      wait: {self.wait:.5e} s\n"
        s += f"   samples: {self.samples:.5e}"
        return s

"""Output for balanceCapBridge function, also used for CapBridge object"""
@dataclass
class CapBridgeBalance:
    """
    Result of `balanceCapBridge`.

    Attributes
    ----------
    status : bool
        whether balance was successful.
    balance_matrix : np.ndarray
    Cex : float
        capacitance.
    Closs : float
        loss.
    Vc0Vex : float
        balanced Vc in units of Vex.
    Vr0Vex : float
        balanced Vr in units of Vex.
    R : float
        magnitude of the signal when balanced
    phase : float
        phase of the signal when balanced.
    error : Tuple[float, float]
        X and Y signals when at the balance point.
    Vex : float
        excitation voltage used.
    Cstd : float
        standard capacitance.
    """
    status          : bool
    balance_matrix  : np.ndarray = None
    Cex             : float = None
    Closs           : float = None
    Vc0Vex          : float = None
    Vr0Vex          : float = None
    R               : float = None
    phase           : float = None
    error           : Tuple[float, float] = None
    Vex             : float = None
    Cstd            : float = None

    def __str__(self):
        s  = f"        status: {'balanced' if self.status else 'unbalanced'}\n"
        s += f"balance_matrix: " + \
                ''.join([f"{e:.5e}   " for e in self.balance_matrix]) + '\n'
        s += f"           Cex: {self.Cex:.5e}\n"
        s += f"         Closs: {self.Closs:.5e}\n"
        s += f"     Vc0 / Vex: {self.Vc0Vex:.5e}\n"
        s += f"     Vr0 / Vex: {self.Vr0Vex:.5e}\n"
        s += f"             R: {self.R:.5e}\n"
        s += f"         phase: {self.phase:5e}\n"
        if self.error is not None:
            s += f"       error_X: {self.error[0]:.5e}\n"
            s += f"       error_Y: {self.error[1]:.5e}\n"
        s += f"           Vex: {self.Vex:.5e}\n"
        s += f"          Cstd: {self.Cstd:.5e}"
        return s

def balanceCapBridge(C          : BalanceCapBridgeConfig,
                     set_Vex    : Callable[[float], Any] | None,
                     get_Vex    : Callable[[], float],
                     set_Vstd   : Callable[[float], Any] | None,
                     get_Vstd   : Callable[[], float],
                     set_Vstd_ph: Callable[[float], Any],
                     get_XY      : Callable[[], Tuple[float, float]],
                    ) -> CapBridgeBalance:
    """
    Balances the capacitance bridge. Run once before capacitance measurements.

    Parameters
    ----------
    C : BalanceCapBridgeConfig
    set_Vex : Callable or None
        setter for Vex
    get_Vex : Callable
        getter for Vex
    set_Vstd : Callable or None
        setter for Vstd
    get_Vstd : Callable
        getter for Vstd
    set_Vstd_ph : Callable
        setter for Vstd phase relative to Vex
    get_XY : Callable
        getter for X and Y signals from the lock-in

    Returns
    -------
    CapBridgeBalance
    """
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
        Vhi = C.Vstd_range * max(np.sqrt(vr**2 + (vc + dvc)**2), 
                                 np.sqrt(vc**2 + (vr + dvr)**2))
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
        R = np.sqrt(Vcs[n]**2 + Vrs[n]**2)
        phase = 180 - np.degrees(np.arctan2(Vrs[n], Vcs[n])) # 4-quadrant tangent function
        set_Vstd(R/C.Vstd_gain)
        set_Vstd_ph(phase)
        time.sleep(C.wait)

        for m in range(C.samples):

            time.sleep(3*C.time_const)
            x, y = get_XY()
            L[0, n, m] = x
            L[1, n, m] = y

            if C.callback:
                C.callback(
                    n * C.samples + m, np.array([n * C.samples + m]), 
                    L[:, n, m].flatten()
                )

    L = np.mean(L, axis = 2)

    # convert remaining fractional voltages to real voltage units
    Vr  = C.Vstd_range * vr
    Vc  = C.Vstd_range * vc
    dVr = C.Vstd_range * dvr
    dVc = C.Vstd_range * dvc

    # the algorithmic part; see Ashoori thesis
    Kr1 = (L[0,1] - L[0,0]) / dVr   # real voltage units (K's are dimensionless)
    Kc1 = (L[0,2] - L[0,0]) / dVc
    Kr2 = (L[1,1] - L[1,0]) / dVr
    Kc2 = (L[1,2] - L[1,0]) / dVc
    P   = 1 / (1 - (Kc1 * Kr2) / (Kr1 * Kc2))
    Vr0 = Vr + (P / Kr1) * ((Kc1 / Kc2) * L[1,0] - L[0,0]) # all rms voltages
    Vc0 = Vc + (P / Kc2) * ((Kr2 / Kr1) * L[0,0] - L[1,0])

    # calculate device capacitance (rms voltages)
    Cex     = C.Cstd * Vc0 / C.Vex
    Closs   = C.Cstd * Vr0 / C.Vex

    # Vc0/Vex and Vr0/Vex
    Vc0Vex = Vc0/C.Vex
    Vr0Vex = Vr0/C.Vex

    balance_matrix = np.array([Kc1, Kc2, Kr1, Kr2, Vc0, Vr0])
    phase = 180 - np.degrees(np.arctan2(Vr0, Vc0)) # 4-quadrant tangent function
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
        msg  = f"WARNING: Balanced Vstd ({R:.5e} V) is outside available range."
        msg += f"To proceed, run with higher Vstd_range or  lower Vex."
        if C.logger:
            C.logger.warning(msg)
        else:
            print(msg)

        if C.logger:
            C.logger.info(out)

        return out
    
    else:
        set_Vstd(R / C.Vstd_gain)
        set_Vstd_ph(phase)

        time.sleep(C.wait)
        if C.logger:
            C.logger.info("Balanced.")

    out.error = get_XY()
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
    """
    Capacitance bridge object for taking off-balance measurements.
    Generally, measure_capacitance should be called as a pre-callback for a 
    parameter sweep, with the getters being used as the measured parameters
    """
    def __init__(self, balance: CapBridgeBalance, 
                 get_XY: Callable[[], Tuple[float, float]], logger = None):
        """
        Parameters
        ----------
        balance : CapBridgeBalance
        get_XY : Callable
            getter for X and Y signals from the lock-in.
        logger : Logger, optional
        """
        if not balance.status:
            if logger:
                logger.warning("WARNING: Provided balance was unsuccessful.") 
                logger.warning("         Results may be unreliable.")
            else:
                print("WARNING: Provided balance was unsuccessful.")
                print("         Results may be unreliable.")

        self.Kc1, self.Kc2, self.Kr1, self.Kr2, self.Vc0, self.Vr0 = balance.balance_matrix
        self.get_XY = get_XY
        self.Vex   = balance.Vex
        self.Cstd  = balance.Cstd

    def XYtoC(self, X: float, Y: float) -> Tuple[float, float]:
        """
        Convert lock-in measurement to capacitance.
        
        Parameters
        ----------
        X, Y : float

        Returns
        -------
        Cex, Closs : float
        """
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
        Xoffbal, Yoffbal = self.get_XY()
        self.Xoffbal = Xoffbal
        self.Yoffbal = Yoffbal
        self.Cex, self.Closs = self.XYtoC(Xoffbal, Yoffbal)

    def get_Xoffbal(self) -> float:
        """
        Query the measured in-phase signal
        
        Returns
        -------
        X : float
        """
        if self.Xoffbal is not None:
            return self.Xoffbal
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")
        
    def get_Yoffbal(self) -> float:
        """
        Query the measured out-of-phase signal
        
        Returns
        -------
        Y : float
        """
        if self.Yoffbal is not None:
            return self.Yoffbal
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")

    def get_Cex(self) -> float:
        """
        Returns
        -------
        Cex : float
        """
        if self.Cex is not None:
            return self.Cex
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")

    def get_Closs(self) -> float:
        """
        Returns
        -------
        Closs : float
        """
        if self.Closs is not None:
            return self.Closs
        else:
            raise RuntimeWarning("Attempted to 'get' before measurement.")
