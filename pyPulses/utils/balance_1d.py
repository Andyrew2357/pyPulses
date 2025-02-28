"""
This is a general algorithm for abstract balancing. It uses a predictor object
to inform its initial guesses and undergoes a numerical root finding procedure
to converge on a balance point if its first two guesses are insufficient.
"""

from .rootfinder import RootFinderState, RootFinderStatus
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

@dataclass
class BalanceConfig:
    set_x           : Callable[[float], Any]    # Set independent parameter
    get_y           : Callable[[], float]       # Function we are driving to 0
    predictor       : object                    # Helps make initial guess
    rootfinder      : Callable[..., object]     # Numerical root finder class
    y_tolerance     : float                     # Acceptable y error for guesses
    search_range    : Tuple                     # Range of values x can take
    max_iter        : int                       # Max root finder iterations
    max_step        : Optional[float]           # Max step size for root finding
    logger          : Optional[object]          # Logger

def balance1d(p: float, C: BalanceConfig) -> RootFinderState:

    if C.logger:
        C.logger.info("=" * 80)
        C.logger.info(f"1D Balance Procedure: p = {p}")
    
    # Make the initial guess, x0
    x0 = C.predictor.predict0(p)
    if C.logger:
        C.logger.info(f"Predicted balance at x0 = {x0}")

    # Truncate the guess if it's out of range
    min_x, max_x = C.search_range
    if not min_x <= x0 <= max_x:
        xt = min(max_x, max(min_x, x0))
        if C.logger:
            C.logger.info(
                f"x0 = {x0} is out of range ({min_x}, {max_x}). Truncating to {xt}."
            )
        
        x0 = xt

    # Take a measurement at x0. If y0 is small enough, terminate successfully
    C.set_x(x0)
    y0 = C.get_y()
    if abs(y0) <= C.y_tolerance:
        if C.logger:
            C.logger.info(
                f"Measured y0 = {y0} meets the specified tolerance ({C.y_tolerance})!"
            )
            C.logger.info("Balancing terminated successfully.")
        
        return RootFinderState(
            status      = RootFinderStatus.CONVERGED,
            point       = x0,
            root        = x0,
            iterations  = 0,
            message     = "Terminated on the first guess.",
            best_value  = y0,
        )
    
    elif C.logger:
        C.logger.info(f"Measured y0 = {y0}.")

    # Make a second guess, informed by the first, x1
    x1 = C.predictor.predict1(p, (x0, y0))
    if C.logger:
        C.logger.info(f"New Predicted balance at x1 = {x1}")

    # Truncate the guess if it's out of range
    if not min_x <= x1 <= max_x:
        xt = min(max_x, max(min_x, x1))
        if C.logger:
            C.logger.info(
                f"x1 = {x1} is out of range ({min_x}, {max_x}). Truncating to {xt}."
            )
        
        x1 = xt
    
    # Take a measurement at x1. If y1 is small enough, terminate successfully
    C.set_x(x1)
    y1 = C.get_y()
    if abs(y1) <= C.y_tolerance:
        if C.logger:
            C.logger.info(
                f"Measured y1 = {y1} meets the specified tolerance ({C.y_tolerance})!"
            )
            C.logger.info("Balancing terminated successfully.")
        
        return RootFinderState(
            status      = RootFinderStatus.CONVERGED,
            point       = x1,
            root        = x1,
            iterations  = 0,
            message     = "Terminated on the second guess.",
            best_value  = y1
        )
    
    elif C.logger:
        C.logger.info(f"Measured y1 = {y1}.")

    # If our initial guesses weren't good enough, move to root finding
    Solver = C.rootfinder(x0, x1, C.search_range, C.max_iter, 
                          C.y_tolerance, C.max_step)
    
    if C.logger:
        C.logger.info("Entering rootfinding mainloop")

    state = Solver.state
    Solver.update(y0)
    Solver.update(y1)
    while state.status == RootFinderStatus.NEEDS_EVALUATION:

        # Set x to the guessed point
        C.set_x(state.point)
        # Measure at that point
        y = C.get_y()
        # Update the solver with the new measurement
        state = Solver.update(y)

        if C.logger:
            C.logger.info("-"*40)
            C.logger.info(state)
        
        if state.root is not None:
            if C.logger:
                C.logger.info(f"  Root found: {state.root:.6f}")
                C.logger.info("Balancing terminated successfully")
            
            return state
    
    # If we didn't terminate in the root finder main loop,
    # we encountered an error of some kind
    if C.logger:
        C.logger.warning("Balancing terminated unsucessfully.")
        match state.status:
            case RootFinderStatus.CYCLING:
                C.logger.warning("Cycling encountered during root finding.")
            case RootFinderStatus.MAX_ITERATIONS:
                C.logger.warning("Max iterations used during root finding.")
            case RootFinderStatus.NO_ROOT_LIKELY:
                C.logger.warning("No root is likely within the search range.")
            case _:
                C.logger.warning("Misc. error encountered during root finding.")
        
    return state
