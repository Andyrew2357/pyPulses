import numpy as np
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

@dataclass
class BiasSweepConfig:
    measurement     : Union[            # getter(s) for measurement
                            Tuple[Callable[[], float], ...], 
                            Callable[[], float]
                        ]
    time_per_point  : float             # wait time per point
    npoints         : int               # number of points
    sweep           : Union[            # parameters to sweep
                            List[
                                Tuple[
                                    Callable[[float], Any], # setter for parameter
                                    float,                  # starting value
                                    float                   # ending value
                                ]
                            ],
                            Tuple[
                                Callable[[float], Any],     # setter for parameter
                                float,                      # starting value
                                float                       # ending value
                            ]
                        ]
    fname           : Optional[float]   # path to output file
    measured_name   : Optional[         # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ]
    swept_name      : Optional[         # names of swept parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ]
    logger          : Optional[object]  # logger

def biasSweep(C: BiasSweepConfig) -> np.ndarray:
    
    """
    Sweeps any number of parameters and measures any number of parameters at 
    each point.
    
        measurement     : Getter(s) for measured parameter(s) 
                            (callable or tuple of callables)
            
    """

    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    if (C.logger or C.fname) and not (C.swept_name and C.measured_name):
        raise ValueError(
            "swept_name and measured_name are required for file or console logging."
        )

    if not type(C.sweep[0]) == tuple:
        C.sweep = (C.sweep)

    if not type(C.measurement) in [tuple, list]:
        C.measurement = (C.measurement)
    
    if C.swept_name and C.measured_name:
        if type(C.swept_name) == str:
            C.swept_name = (C.swept_name)
        
        if type(C.measured_name) == str:
            C.measured_name = (C.measured_name)

        if not (len(C.swept_name) == len(C.sweep) and \
                len(C.measured_name) == len(C.measurement)):
            raise ValueError("Dimension mismatch in parameters and names.")

    # result array
    result = np.full(shape = (C.npoints, len(C.measurement)), 
                     fill_value = np.nan)

    # write the header for the output file
    if C.fname:
        with open(C.fname, 'w') as f:
            msg = f"{''.join(f"{p}, " for p in C.swept_name)}"
            msg += f"{''.join(f"{p}, " for p in C.measured_name)[:-2]}\n"
            f.write(msg)

    # log the swept parameters
    if C.logger:
        C.logger.info(
            f"Sweeping: {''.join(f"{p}, " for p in C.swept_name)[:-2]}"
        )
        C.logger.info(
            f"Measuring: {''.join(f"{p}, " for p in C.measured_name)[:-2]}")

    # move everything to the start of the sweep
    for setter, start, _ in C.sweep:
        setter(start)

    # step through each bias point
    for n in range(C.npoints):
        biases = [start + (stop - start) * n / (C.npoints - 1) 
                      for _, start, stop in C.sweep]
        # log the bias point coordinates
        if C.logger:
            C.logger.info(
                f"Biases: {''.join([f"{x:.5f}, " for x in biases])[:-2]}"
            )

        # move to the bias point, wait some time, then measure
        for setter, start, stop in C.sweep:
            setter(start + (stop - start) * n / (C.npoints - 1))
        
        time.sleep(C.time_per_point)
        for i in range(len(C.measurement)):
            result[n, i] = C.measurement[i]()
        
        # write the result to the ouput file if provided
        if C.fname:
            with open(C.fname, 'a') as f:
                msg = f"{''.join([f"{x}, " for x in biases])}"
                msg += f"{''.join(f"{r}, " for r in result[n])[:-2]}\n"
                f.write(msg)

        # log the result
        if C.logger:
            msg = f"Result ({n+1}/{C.npoints}) = "
            msg += f"{''.join(f"{r:.5f}, " for r in result[n])[:-2]}"
            C.logger.info(msg)

    return result
