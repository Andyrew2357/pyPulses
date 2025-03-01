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
    search_range    : Tuple                     # Range of values x can take
    x_tolerance     : float                     # Acceptable x error for guesses
    y_tolerance     : float                     # Acceptable y error for guesses
    max_iter        : int                       # Max root finder iterations
    max_reps        : int                       # Guard against cycles
    max_coll        : int                       # Max boundary collisions
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

    # Take a measurement at x0.
    C.set_x(x0)
    y0 = C.get_y()
    # Make a second guess, informed by the first, x1
    x1 = C.predictor.predict1(p, (x0, y0))

    Solver = C.rootfinder(xa = x0, xb = x1, 
        search_range    = C.search_range,  
        x_tolerance     = C.x_tolerance,
        y_tolerance     = C.y_tolerance,
        max_iter        = C.max_iter, 
        max_reps        = C.max_reps,
        max_coll        = C.max_coll
    )

    state = Solver.update(y0)

    # If y0 is small enough, terminate successfully
    if state.status == RootFinderStatus.CONVERGED:
        if C.logger:
            C.logger.info(
                f"Measured y0 = {y0} meets the specified tolerance ({C.y_tolerance})!"
            )
            C.logger.info("Balancing terminated successfully.")
        return state
    
    elif C.logger:
        C.logger.info(f"Measured y0 = {y0}.")

    # If there is no convergence on the first guess, proceed with the second
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
        # We have to let the solver know that this has changed as well
        Solver.xb = x1

    # Take a measurement at x1.
    C.set_x(x1)
    y1 = C.get_y()
    state = Solver.update(y1)

    # If y1 is small enough, terminate successfully
    if state.status == RootFinderStatus.CONVERGED:
        if C.logger:
            C.logger.info(
                f"Measured y1 = {y1} meets the specified tolerance ({C.y_tolerance})!"
            )
            C.logger.info("Balancing terminated successfully.")
        return state
    
    elif C.logger:
        C.logger.info(f"Measured y1 = {y1}.")

    # If our initial guesses weren't good enough, move to root finding
    if C.logger:
        C.logger.info("Entering rootfinding mainloop")
    
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
        
    if state.status == RootFinderStatus.CONVERGED:
        if C.logger:
            C.logger.info(f"  Root found: {state.root:.6f}")
            C.logger.info("Balancing terminated successfully")
        
        return state
    
    # If we didn't terminate in the root finder main loop,
    # we encountered an error of some kind
    if C.logger:
        C.logger.warning("Balancing terminated unsuccessfully.")
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
