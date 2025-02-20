from typing import Any, Callable, List, Optional, Tuple
from math import ceil
import time

def tandemSweep(wait: float,
                *sweeps: List[Tuple[
                    Callable[[float], Any], # setter
                    float,                  # starting value
                    float,                  # target value
                    Optional[float]         # max_step
                ]]):
    """
    Sweeps multiple parameters smoothly while respecting their maximum step
    sizes. It takes arguments of the form:
    wait_time, 
    (setter1, start1, target1, max_step1), 
    (setter2, start2, target2, max_step2),
    ...
    """
    
    # Repackage the input into an 'instructions' list of tuples like:
    # (setter, start, target, max_step = None)
    instructions = []
    for sweep in sweeps:
        if len(sweep) < 4:
            instructions.append((*sweep, None)) # max_step = None if not given
        else:
            instructions.append(sweep)

    # Determine the number of steps to take. This is determined by the 'weakest
    # link', the variable the ramp that needs the most steps respecting its
    # maximum step size.
    min_steps = 0
    for _, start, target, max_step in instructions:
        if max_step:
            min_steps = max(ceil((target - start) / max_step), min_steps)

    # Perform the sweep, setting each value and waiting between steps
    for step in range(1, min_steps + 1):
        for setter, start, target, _ in instructions:
            setter(start + step * (target - start) / min_steps)
        time.sleep(wait)
