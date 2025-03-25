"""Various functions for sweeping parameters and measuring at each step."""

import numpy as np
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

"""
SweepMeasureCutConfig
sweepMeasureCut
(replaces biasSweep from a previous version)

Sweep parameters linearly from one point in parameter space to another, taking
measurements at each point.

    measurement     : Getter(s) for measured parameter(s) 
                        (callable or tuple of callables)
    time_per_point  : Time to wait before taking a measurement at each point
    npoints         : Number of points to take
    sweep           : tuple or list of tuples describing the sweep in each
                      dimension of parameter space. These tuples take the form:
                        (setter, starting value, ending value)
    fname           : Optional; path to output file
    measured_name   : string or tuple of strings naming the measured parameters
                      for purposes of logging or writing the ouput
    swept_name      : string or tuple of strings naming the swept parameters for
                      purposes of logging or writing the ouput
    logger          : Optional; logger object
    pre_callback    : Optional; generic callback function called before the
                      measurement is taken at each point. The function passes
                      back the index of the current point in the sweep, and a 1d
                      array of the swept parameters at that point.
    post_callback   : Optional; generic callback function called after the 
                      result is updated at each point. The function passes back
                      the index of the current point in the sweep, a 1d array of
                      the swept paramters at that point, and a 1d array of 
                      measurement results at that point for use by the callback
                      function.
"""

@dataclass
class SweepMeasureCutConfig:
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
    pre_callback    : Optional[Callable[[int, np.ndarray], Any]]
    post_callback   : Optional[Callable[[int, np.ndarray, np.ndarray], Any]]

def sweepMeasureCut(C: SweepMeasureCutConfig) -> np.ndarray:
    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    if (C.logger or C.fname) and not (C.swept_name and C.measured_name):
        raise ValueError(
            "swept_name and measured_name are required for file or console logging."
        )

    if not type(C.sweep[0]) == tuple:
        C.sweep = (C.sweep,)

    if not type(C.measurement) in [tuple, list]:
        C.measurement = (C.measurement,)
    
    if C.swept_name and C.measured_name:
        if type(C.swept_name) == str:
            C.swept_name = (C.swept_name,)
        
        if type(C.measured_name) == str:
            C.measured_name = (C.measured_name,)

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

        if C.pre_callback:
            C.pre_callback(n, np.array(biases))
        
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

        if C.post_callback:
            C.post_callback(n, np.array(biases), result[n,:].flatten())

    return result

"""
SweepMeasureConfig
sweepMeasure

Generalization of sweepMeasureCut. Sweep parameters to each point provided in 
the array 'points', taking measurements at each step.

    measurement     : Getter(s) for measured parameter(s) 
                        (callable or tuple of callables)
    sweep           : Setter(s) for swept parameter(s)
                        (callable or tuple of callables)
    points          : ndarray of shape (number of points, number of swept
                      parameters) listing the points in parameter space at which
                      to take measurements.
    time_per_point  : Time to wait before taking a measurement at each point
    fname           : Optional; path to output file
    measured_name   : string or tuple of strings naming the measured parameters
                      for purposes of logging or writing the ouput
    swept_name      : string or tuple of strings naming the swept parameters for
                      purposes of logging or writing the ouput
    logger          : Optional; logger object
    pre_callback    : Optional; generic callback function called before the
                      measurement is taken at each point. The function passes
                      back the index of the current point in the sweep, and a 1d
                      array of the swept parameters at that point.
    post_callback   : Optional; generic callback function called after the 
                      result is updated at each point. The function passes back
                      the index of the current point in the sweep, a 1d array of
                      the swept paramters at that point, and a 1d array of 
                      measurement results at that point for use by the callback
                      function.
"""

@dataclass
class SweepMeasureConfig:
    measurement     : Union[            # getter(s) for measurement
                            Tuple[Callable[[], float], ...], 
                            Callable[[], float]
                        ]
    sweep           : Union[            # setter(s) for parameters to sweep
                            Tuple[Callable[[float], Any]],
                            Callable[[float], Any]
                        ]
    points          : np.ndarray        # points at which to measure. Has shape
                                        # (number of distinct points,
                                        #  dimensions of the parameter space)
    time_per_point  : float             # wait time per point
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
    pre_callback    : Optional[Callable[[int, np.ndarray], Any]]
    post_callback   : Optional[Callable[[int, np.ndarray, np.ndarray], Any]]

def sweepMeasure(C: SweepMeasureConfig) -> np.ndarray:
    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    if (C.logger or C.fname) and not (C.swept_name and C.measured_name):
        raise ValueError(
            "swept_name and measured_name are required for file or console logging."
        )

    if not type(C.sweep) in [tuple, list]:
        C.sweep = (C.sweep,)

    if not type(C.measurement) in [tuple, list]:
        C.measurement = (C.measurement,)
    
    if C.swept_name and C.measured_name:
        if type(C.swept_name) == str:
            C.swept_name = (C.swept_name,)
        
        if type(C.measured_name) == str:
            C.measured_name = (C.measured_name,)

        if not (len(C.swept_name) == len(C.sweep) and \
                len(C.measured_name) == len(C.measurement)):
            raise ValueError("Dimension mismatch in parameters and names.")

    # If points is flat, reshape it to a column vector
    if C.points.ndim == 1:
        C.points = C.points.reshape(-1, 1)

    npoints, sweep_dim = C.points.shape
    if not sweep_dim == len(C.sweep):
        raise ValueError("points.shape does not match the number of setters.")

    # result array
    result = np.full(shape = (npoints, len(C.measurement)), 
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
    for d, setter in enumerate(C.sweep):
        setter(C.points[0, d])

    # step through each bias point
    for n in range(npoints):
        biases = C.points[n, :]
        # log the bias point coordinates
        if C.logger:
            C.logger.info(
                f"Biases: {''.join([f"{x:.5f}, " for x in biases])[:-2]}"
            )

        # move to the bias point, wait some time, then measure
        for d, setter in enumerate(C.sweep):
            setter(C.points[n, d])

        if C.pre_callback:
            C.pre_callback(n, biases.flatten())
        
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
            msg = f"Result ({n+1}/{npoints}) = "
            msg += f"{''.join(f"{r:.5f}, " for r in result[n])[:-2]}"
            C.logger.info(msg)

        if C.post_callback:
            C.post_callback(n, biases.flatten(), result[n,:].flatten())

    return result

"""
SweepMeasureProductConfig
sweepMeasureProduct

Sweep parameters over the product space of provided axes, taking measurements at
each point.

    measurement     : Getter(s) for measured parameter(s) 
                        (callable or tuple of callables)
    sweep           : Setters for swept parameters
                        (tuple of callables)
    axes            : Tuple of ndarrays representing the axes for each parameter
    time_per_point  : Time to wait before taking a measurement at each point
    fname           : Optional; path to output file
    measured_name   : string or tuple of strings naming the measured parameters
                      for purposes of logging or writing the ouput
    swept_name      : tuple of strings naming the swept parameters for purposes
                      of logging or writing the ouput
    logger          : Optional; logger object
    pre_callback    : Optional; generic callback function called before the
                      measurement is taken at each point. The function passes
                      back the index of the current point in the sweep, and a 1d
                      array of the swept parameters at that point.
    post_callback   : Optional; generic callback function called after the 
                      result is updated at each point. The function passes back
                      the index of the current point in the sweep, a 1d array of
                      the swept paramters at that point, and a 1d array of 
                      measurement results at that point for use by the callback
                      function.
"""

@dataclass
class SweepMeasureProductConfig:
    measurement     : Union[            # getter(s) for measurement
                            Tuple[Callable[[], float], ...], 
                            Callable[[], float]
                        ]
    sweep           : Tuple[            # setter(s) for parameters to sweep
                            Callable[[float], Any]
                        ]
    axes            : Tuple[np.ndarray] # axes over which to sweep parameters
    time_per_point  : float             # wait time per point
    fname           : Optional[float]   # path to output file
    measured_name   : Optional[         # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ]
    swept_name      : Optional[         # names of swept parameters
                            Tuple[str, ...]
                        ]
    logger          : Optional[object]  # logger
    pre_callback    : Optional[Callable[[int, np.ndarray], Any]]
    post_callback   : Optional[Callable[[int, np.ndarray, np.ndarray], Any]]

def sweepMeasureProduct(C: SweepMeasureProductConfig) -> np.ndarray:

    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    if (C.logger or C.fname) and not (C.swept_name and C.measured_name):
        raise ValueError(
            "swept_name and measured_name are required for file or console logging."
        )

    if not type(C.sweep) in [tuple, list]:
        C.sweep = (C.sweep,)

    if not type(C.measurement) in [tuple, list]:
        C.measurement = (C.measurement,)
    
    if C.swept_name and C.measured_name:
        if type(C.swept_name) == str:
            C.swept_name = (C.swept_name,)
        
        if type(C.measured_name) == str:
            C.measured_name = (C.measured_name,)

        if not (len(C.swept_name) == len(C.sweep) and \
                len(C.measured_name) == len(C.measurement)):
            raise ValueError("Dimension mismatch in parameters and names.")


    if not len(C.axes) == len(C.sweep):
        raise ValueError("axes does not match the number of setters.")

    # result array
    result = np.full(shape = (len(C.measurement), *(ax.size for ax in C.axes)),
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

    # step through each bias point
    for ind in itertools.product(*(range(ax.size) for ax in C.axes)):
        biases = np.array([C.axes[d][ind[d]] 
                           for d in range(len(C.sweep))])
        
        # log the bias point coordinates
        if C.logger:
            C.logger.info(
                f"Biases: {''.join([f"{x:.5f}, " for x in biases])[:-2]}"
            )

        # move to the bias point, wait some time, then measure
        for d, setter in enumerate(C.sweep):
            setter(C.axes[d][ind[d]])

        if C.pre_callback:
            C.pre_callback(ind, biases.flatten())
        
        time.sleep(C.time_per_point)
        for i in range(len(C.measurement)):
            result[i, *ind] = C.measurement[i]()
        
        # write the result to the ouput file if provided
        if C.fname:
            with open(C.fname, 'a') as f:
                msg = f"{''.join([f"{x}, " for x in biases])}"
                msg += f"{''.join(f"{r}, " for r in result[:, *ind])[:-2]}\n"
                f.write(msg)

        # log the result
        if C.logger:
            msg = f"Result = "
            msg += f"{''.join(f"{r:.5f}, " for r in result[:, *ind])[:-2]}"
            C.logger.info(msg)

        if C.post_callback:
            C.post_callback(ind, biases.flatten(), result[:, *ind].flatten())

    return result
