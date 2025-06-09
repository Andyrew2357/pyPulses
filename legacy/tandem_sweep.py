# largely obsolete (old version of tandemSweep)
from typing import Any, Callable, List, Optional, Tuple
from math import ceil
import time

def tandemSweep(wait: float,
            *sweeps: List[Tuple[
                Callable[[float], Any], # setter
                float,                  # starting value
                float,                  # target value
                Optional[float]         # max_step
            ]], **kwargs):
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

    min_step = getattr(kwargs, 'min_step', [0]*len(instructions))

    # Determine the number of steps to take. This is determined by the 'weakest
    # link', the variable the ramp that needs the most steps respecting its
    # maximum step size.
    min_steps = 0
    for _, start, target, max_step in instructions:
        if max_step:
            min_steps = max(ceil(abs(target - start) / max_step), min_steps)

    # Perform the sweep, setting each value and waiting between steps
    # It will only set a parameter to a new value if the change exceeds the
    # minimum step size for that parameter (if none is provided, that step
    # size is 0). This is to prevent repeatedly calling costly setters for
    # meaningless changes.
    prev_settings = [start for _, start, _, _ in instructions]
    for step in range(1, min_steps):
        time.sleep(wait)

        new_settings = [start + step * (target - start) / min_steps
                        for _, start, target, _ in instructions]
        for i in range(len(instructions)):
            if abs(new_settings[i] - prev_settings[i]) <= min_step[i]:
                new_settings[i] = prev_settings[i]

        for i in range(len(instructions)):
            if not prev_settings[i] == new_settings[i]:
                instructions[i][0](new_settings[i])
            prev_settings[i] = new_settings[i]

    # Make sure we actually make it to the target by the end.
    time.sleep(wait)
    final_settings = [stop for _, _, stop, _ in instructions]
    for i in range(len(instructions)):
        if not final_settings[i] == prev_settings[i]:
            instructions[i][0](final_settings[i])
