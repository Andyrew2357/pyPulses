from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import CapContext

from dataclasses import dataclass
import numpy as np

@dataclass
class CapMeasureResult:
    """
    Result of a single off-balance capacitance measurement.

    The bridge is held at the most recent balance point. The lock-in
    reading is converted to (Cex, Closs) using the cached complex gain A.
    The filter is not updated.

    Attributes
    ----------
    Cex : float
        Measured capacitance.
    Closs : float
        Measured loss.
    LX : float
        Raw lock-in X reading.
    LY : float
        Raw lock-in Y reading.
    A : (2,) ndarray
        Complex gain [X, Y] used for the conversion (snapshot at call time).
    """
    Cex:   float
    Closs: float
    LX:    float
    LY:    float
    A:     np.ndarray

    def __str__(self) -> str:
        return (
            f"CapMeasureResult: Cex={self.Cex:.5e}  Closs={self.Closs:.5e}  "
            f"LX={self.LX:.5e}  LY={self.LY:.5e}"
        )


def cap_measure(ctx: 'CapContext', use_matrix: bool = False) -> CapMeasureResult:
    """
    Take a single off-balance capacitance measurement.

    Reads the lock-in and converts the reading to (Cex, Closs) using the
    complex gain A currently stored in the Kalman filter. The filter state
    is **not** updated — this function is purely passive.

    The bridge must have been balanced (via `cap_balance` or an initialization
    routine) before calling this function. The Vstd hardware is not touched.

    Parameters
    ----------
    ctx : CapContext
    use_matrix : bool, default False
        If True and a K_matrix is stored on the filter (set when initialized
        from a three-point balance), use the full 2x2 matrix inversion instead
        of the compressed complex gain A. Falls back to A if no matrix is stored.

    Returns
    -------
    CapMeasureResult
    """
    
    if ctx.cap_filter is None:
        raise RuntimeError(
            "cap_measure called before filter initialization. "
            "Call cap_initialize_filter() first."
        )

    mean, _ = ctx.lockin_call()
    LX, LY = float(mean[0]), float(mean[1])

    V_now = ctx._Vstd_now
    if V_now is None:
        V_now = ctx.get_Vstd_complex()
    A = ctx.cap_filter.get_A()
    L = np.array([LX, LY])

    if use_matrix and ctx.cap_filter.K_matrix is not None:
        # Full 2x2 matrix inversion
        #   V0_eff = V_now + K^{-1} @ L
        Kc1, Kr1 = ctx.cap_filter.K_matrix[0]
        Kc2, Kr2 = ctx.cap_filter.K_matrix[1]
        det = Kc1 * Kr2 - Kr1 * Kc2
        dV = np.array([
            (Kr2 * LX - Kr1 * LY) / det,
            (-Kc2 * LX + Kc1 * LY) / det,
        ])
    else:
        # Compressed complex gain inversion:
        #   V0_eff = V_now + A^{-1} @ L
        X, Y = float(A[0]), float(A[1])
        absA2 = X**2 + Y**2
        dV = np.array([(X * LX + Y * LY) / absA2, (-Y * LX + X * LY) / absA2])

    V0_eff = np.array([V_now.real, V_now.imag]) + dV
    Cex = ctx.Cstd * V0_eff[0] / ctx.Vex
    Closs = ctx.Cstd * V0_eff[1] / ctx.Vex

    measure_result = CapMeasureResult(
        Cex = Cex,
        Closs = Closs,
        LX = LX,
        LY = LY,
        A = A,
    )

    ctx.log(measure_result)
    return measure_result