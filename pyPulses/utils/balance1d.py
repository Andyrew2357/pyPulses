
from .rootfinder import RootFinderState, RootFinderStatus
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

@dataclass
class BalanceConfig:
    set_x           : Callable[[float], Any]
    get_y           : Callable[[], float]
    predictor       : object
    rootfinder      : Callable[..., object]
    y_tolerance     : float
    x_tolerance     : float
    search_range    : Tuple
    max_iter        : int
    max_step        : Optional[float]
    logger          : Optional[object]

def balance1d(p: float, C: BalanceConfig) -> bool:

    if C.logger:
        C.logger.info("=" * 80)
        C.logger.info(f"1D Balance Procedure: p = {p}")
    
    x0 = C.predictor.predict(p)
    if C.logger:
        C.logger.info(f"Predicted balance at x0 = {p}")

    min_x, max_x = C.search_range
    if not min_x <= x0 <= max_x:
        xt = min(max_x, max(min_x, x0))
        if C.logger:
            C.logger.info(
                f"x0 = {x0} is out of range ({min_x}, {max_x}). Truncating to {xt}."
            )
        
        x0 = xt

    C.set_x(x0)
    y0 = C.get_y()
    if abs(y0) <= C.y_tolerance:
        if C.logger:
            C.logger.info(
                f"Measured y0 = {y0} meets the specified tolerance ({C.y_tolerance})!"
            )
            C.logger.info("Balancing terminated successfully.")
        
        return True
    
    elif C.logger:
        C.logger.info(f"Measured y0 = {y0}.")

    x1 = C.predictor(p, (x0, y0))
    if C.logger:
        C.logger.info(f"New Predicted balance at x1 = {p}")

    if not min_x <= x1 <= max_x:
        xt = min(max_x, max(min_x, x1))
        if C.logger:
            C.logger.info(
                f"x1 = {x1} is out of range ({min_x}, {max_x}). Truncating to {xt}."
            )
        
        x1 = xt
    
    C.set_x(x1)
    y1 = C.get_y()
    if abs(y1) <= C.y_tolerance:
        if C.logger:
            C.logger.info(
                f"Measured y1 = {y1} meets the specified tolerance ({C.y_tolerance})!"
            )
            C.logger.info("Balancing terminated successfully.")
        
        return True
    
    elif C.logger:
        C.logger.info(f"Measured y0 = {y0}.")

    Solver = C.rootfinder(x0, x1, C.search_range, C.max_iter, 
                          C.x_tolerance, C.max_step)
    
    if C.logger:
        C.logger.info("Entering rootfinding mainloop")

    state = Solver.state
    while state.status == RootFinderStatus.NEEDS_EVALUATION:

        C.set_x(state.point)
        y = C.get_y()
        state = Solver.update(y)

        if C.logger:
            C.logger.info("-"*40)
            C.logger.info(f"Iteration {state.iterations}:")
            C.logger.info(f"  Evaluated at x = {state.point:.6f}")
            C.logger.info(f"  Status    : {state.status}")
            C.logger.info(f"  Message   : {state.message}")
            C.logger.info(f"  Best value: {state.best_value:.2e}")
        
        if state.root is not None:
            if C.logger:
                C.logger.info(f"  Root found: {state.root:.6f}")
                C.logger.info("Balancing terminated successfully")
            
            return True
        
    if C.logger:
        C.logger.warning("Balancing terminated unsucessfully.")
    
    return False
