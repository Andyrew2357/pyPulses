import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

@dataclass
class TwoPointCapBalance:
    status  : bool
    A       : complex
    V0      : complex
    Vi      : complex
    Li      : complex
    Vf      : complex
    Lf      : complex
    Lunc    : list
    P       : np.ndarray

def balanceCapBridgeTwoPoint(
    Vstd: Callable[[float | None], Any], 
    Theta: Callable[[float | None], Any],
    Vout: Callable[[], complex | List[float] | Tuple[List[float], np.ndarray]],
    Vstd_range: float,
    dVstd: complex,
    settle_time: float = 1.0,
    fudge: float = 1.0
):

    Lunc = []
    def set_Vstd(x: complex):
        mag = abs(x)
        phs = np.degrees(np.angle(x)) % 360
        Vstd(mag)
        Theta(phs)

    def get_Vout():
        L = Vout()
        if isinstance(L, complex):
            return L
        elif isinstance(L[1], np.ndarray):
            Lunc.append(L[1])
            return L[0][0] + 1j * L[0][1]
        else:
            return L[0] + 1j * L[1]

    V = Vstd()
    Theta_ = Theta()
    Vi = V * (np.cos(np.deg2rad(Theta_)) + 1j * np.sin(np.deg2rad(Theta_)))
    Li = get_Vout()
    Vf = Vi + dVstd
    set_Vstd(Vf)
    V = Vstd()
    Theta_ = Theta()
    Vf = V * (np.cos(np.deg2rad(Theta_)) + 1j * np.sin(np.deg2rad(Theta_)))
    time.sleep(settle_time)
    Lf = get_Vout()
    A = (Lf - Li) / dVstd
    V0 = Vi - Li/A

    balanced = abs(V0) <= Vstd_range
    if balanced:
        set_Vstd(V0)

    if len(Lunc) > 0:
        dV_mat = np.array([[dVstd.real, -dVstd.imag], [dVstd.imag, dVstd.real]])
        P = fudge * dV_mat.T @ (Lunc[0] + Lunc[1]) @ dV_mat / abs(dVstd)**4
    else:
        P = None

    return TwoPointCapBalance(
        status  = balanced,
        A       = A,
        V0      = V0,
        Vi      = Vi,
        Li      = Li,
        Vf      = Vf,
        Lf      = Lf,
        Lunc    = Lunc,
        P       = P,
    )
