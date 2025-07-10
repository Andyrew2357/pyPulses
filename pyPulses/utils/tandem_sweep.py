"""
Smart function for sweeping multiple parameters simultaneously. The number of
steps is determined by whichever parameter is the bottleneck. For costly
setters, the user can provide min_step as a key word argument, which avoids
unnecessarily granular sweeping (this is especially important for things like
the magnet). Usually, this is not an issue, because we rarely sweep highly
dissimilar parameters simultaneously; it's mostly there for param_sweep_measure.
"""

from .getsetter import getSetter

from typing import Any, Callable, List
import numpy as np
from math import ceil
import time

def tandemSweep(setters: List[Callable[[float], Any]], 
                start: List[float], end: List[float], 
                wait: float,
                max_step: List[float | None],  
                min_step: List[float | None] = None,
                tolerance: List[float | None] = None,
                handle_exceptions: bool = True) -> bool:
    """
    Sweeps multiple parameters smoothly while respecting their maximum step
    sizes. Returns a boolean indicating success or failure.
    """
    
    N = len(setters)
    if len(start) != N or len(end) != N or len(max_step) != N:
        raise IndexError(
            "Mismatch between number of setters and other arguments."
        )

    if min_step is None: 
        min_step = np.zeros(N)
    elif len(min_step) != N:
        raise IndexError(
            "Mismatch between number of setters and other arguments."
        )
    
    if tolerance is None: 
        tolerance = np.zeros(N)
    elif len(tolerance) != N:
        raise IndexError(
            "Mismatch between number of setters and other arguments."
        )

    min_step = np.array([0 if m is None else m for m in min_step])
    max_step = np.array([np.inf if m is None else m for m in max_step])
    tolerance = np.array([0 if m is None else m for m in tolerance])

    # min_step should never be smaller than the tolerance
    min_step = np.maximum(min_step, tolerance)

    if np.any(max_step - min_step < 0):
        raise ValueError("'min_step' values cannot exceed 'max_step' values.")
    if np.any(min_step < 0):
        raise ValueError("'min_step' values cannot be negative.")
    
    start = np.array(start)
    end = np.array(end)

    # Determine the number of steps to take. This is determined by the 'weakest
    # link', the variable the ramp that needs the most steps respecting its
    # maximum step size.
    M = ceil(np.max(np.abs(end - start) / max_step))

    # Perform the sweep, setting each value and waiting between steps
    # It will only set a parameter to a new value if the change exceeds the
    # minimum step size for that parameter (if none is provided, that step
    # size is 0). This is to prevent repeatedly calling costly setters for
    # meaningless changes.
    prev_settings = start.copy()
    for i in range(M):
        new_settings = start + i * (end - start) / M

        # determine which parameters are close enough not to set
        similar = np.abs(new_settings - prev_settings) <= min_step
        new_settings[similar] = prev_settings[similar]
        
        for i in range(N):
            if prev_settings[i] != new_settings[i]:
                try:
                    success = setters[i](new_settings[i])
                    
                    # Some setters may indicate success or failure by returning
                    # a boolean. If failure is encountered, raise an exception.
                    if type(success) == bool:
                        assert success
                
                except:
                    # Handle exceptions when setting; Attempt to sweep back to
                    # start if we encounter an error.
                    print(
                        "tandemSweep encountered an error when setting: "
                        + setters[i].__name__
                    )
                    if handle_exceptions:
                        print("Attempting to sweep back to start...")
                        tandemSweep(setters, prev_settings, start,
                                    max_step    = max_step,
                                    wait        = wait,
                                    min_step    = min_step,
                                    handle_exceptions = False)
                        return False
                    else:
                        raise RuntimeError(
                            "tandemSweep unable to resolve error whilst setting"
                        )

            prev_settings[i] = new_settings[i]
        
        time.sleep(wait)

    # Make sure we actually make it to the target by the end 
    # (to within tolerance).
    for i in range(N):
        if np.abs(prev_settings[i] - end[i]) > tolerance[i]:
            setters[i](end[i])
    
    return True

def ezTandemSweep(parms: List[dict], target: List[float], wait: float, 
                  handle_exceptions: bool = True) -> bool:
    """
    Wrapper for more human syntax. Pass parameters as a list of dictionaries 
    describing their behavior. Automatically gets the start values, by requiring 
    users to provide the getters.
    
    Recognized fields for parms elements:
    'f'         : <getsetter (function)> 
                    (must provide either 'f' or 'get' and 'set')
    'get'       : <getter (function)> (Ignored if 'f' is provided)
    'set'       : <setter (function)> (Ignored if 'f' is provided)
    'min_step'  : <minimum step size (float)> (optional)
    'max_step'  : <maximum step size (float)> (optional)
    'tolerance' : <tolerance (float)> (optional)
    """

    for P in parms:
        if not 'f' in P:
            P['f'] = getSetter(P['get'], P['set'])

    setters = [P['f'] for P in parms]
    max_step = [P.get('max_step') for P in parms]
    min_step = [P.get('min_step') for P in parms]
    tolerance = [P.get('tolerance') for P in parms]
    start = [G() for G in setters]

    return tandemSweep(setters, start, target, 
                       wait, max_step, min_step, tolerance, 
                       handle_exceptions)
