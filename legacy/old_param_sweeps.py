# pyPulses.utils
# # Deprecated - This is all implemented with far better templating in param_sweep_measure.py
"""Various functions for sweeping parameters and measuring at each step."""

import numpy as np
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, IO, List, Optional, Tuple, Union

from ..pyPulses.utils.tandem_sweep import tandemSweep

def set_swept_params(setters    : List[Callable[[float], Any]], 
                     prev       : np.ndarray,
                     targets    : np.ndarray, 
                     ramp_wait  : Optional[float] = None, 
                     ramp_steps : Optional[List[float]] = None,
                     ramp_kwargs: Optional[dict] = None):
    """Utility function to smoothly integrate tandemSweep functionality."""

    if ramp_wait is not None:
        tandemSweep(
            wait = ramp_wait,
            sweeps = [
                (setters[i], prev[i], targets[i], ramp_steps[i])
                for i in range(len(setters))
            ],
            **ramp_kwargs
        )
    else:
        for i in range(len(setters)):
            setters[i](targets[i])

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
    file            : Optional; path to output file or a file-like object
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
    ramp_wait       : Optional; wait time if using tandem sweep (important for
                      safely sweeping in some circumstances)
    ramp_step       : Optional; list of maximum step sizes for each parameter if
                      using controlled tandem sweeping. Otherwise, this should
                      be handled by passing the dedicated sweep function for
                      that parameter (this only matters for parameters that need
                      to be swept simultaneously to avoid issue, such as top and
                      bottom gates for VdW systems)
    ramp_kwargs     : Optional; keyword arguments for tandem sweep (particularly
                      important for 'min_step', which tells the sweep to ignore
                      trivial changes in a swept parameter that cost unecessary
                      time.)
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
    file            : Optional[Union[IO, str]] = None       # output file-like object or string
    measured_name   : Optional[             # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    swept_name      : Optional[         # names of swept parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    logger          : Optional[object] = None   # logger
    pre_callback    : Optional[Callable[[int, np.ndarray], Any]] = None
    post_callback   : Optional[Callable[[int, np.ndarray, np.ndarray], Any]] = None
    
    # If tandem sweeping is desired, need to provide these
    ramp_wait       : Optional[float] = None        # wait time between steps when ramping
    ramp_steps      : Optional[List[float]] = None  # maximum step sizes
    ramp_kwargs     : Optional[dict] = None         # keyword arguments for tandem sweep

def sweepMeasureCut(C: SweepMeasureCutConfig) -> np.ndarray:
    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions

    if type(C.file) == str:
        with open(C.file, 'w') as f:
            fname = C.file
            C.file = f
            res = sweepMeasureCut(C)
            C.file = fname
            return res

    if (C.logger or C.file) and not (C.swept_name and C.measured_name):
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
    if C.file:
        msg = f"{''.join(f"{p}, " for p in C.swept_name)}"
        msg += f"{''.join(f"{p}, " for p in C.measured_name)[:-2]}\n"
        C.file.write(msg)

    # log the swept parameters
    if C.logger:
        C.logger.info(
            f"Sweeping: {''.join(f"{p}, " for p in C.swept_name)[:-2]}"
        )
        C.logger.info(
            f"Measuring: {''.join(f"{p}, " for p in C.measured_name)[:-2]}"
        )

    # move everything to the start of the sweep
    for setter, start, _ in C.sweep:
        setter(start)

    setters = [s[0] for s in C.sweep]
    prev = [start for _, start, stop in C.sweep]

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
        set_swept_params(setters, prev, biases, 
                         C.ramp_wait, C.ramp_steps, C.ramp_kwargs)
        prev = biases

        if C.pre_callback:
            C.pre_callback(n, np.array(biases))
        
        time.sleep(C.time_per_point)
        for i in range(len(C.measurement)):
            result[n, i] = C.measurement[i]()
        
        # write the result to the ouput file if provided
        if C.file:
            msg = f"{''.join([f"{x}, " for x in biases])}"
            msg += f"{''.join(f"{r}, " for r in result[n])[:-2]}\n"
            C.file.write(msg)

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
    file            : Optional; path to output file or a file-like object
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
    ramp_wait       : Optional; wait time if using tandem sweep (important for
                      safely sweeping in some circumstances)
    ramp_step       : Optional; list of maximum step sizes for each parameter if
                      using controlled tandem sweeping. Otherwise, this should
                      be handled by passing the dedicated sweep function for
                      that parameter (this only matters for parameters that need
                      to be swept simultaneously to avoid issue, such as top and
                      bottom gates for VdW systems)
    ramp_kwargs     : Optional; keyword arguments for tandem sweep (particularly
                      important for 'min_step', which tells the sweep to ignore
                      trivial changes in a swept parameter that cost unecessary
                      time.)
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
    file            : Optional[Union[IO, str]] = None   # output file-like
    measured_name   : Optional[         # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    swept_name      : Optional[         # names of swept parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    logger          : Optional[object] = None   # logger
    pre_callback    : Optional[Callable[[int, np.ndarray], Any]] = None
    post_callback   : Optional[Callable[[int, np.ndarray, np.ndarray], Any]] = None

    # If tandem sweeping is desired, need to provide these
    ramp_wait       : Optional[float] = None        # wait time between steps when ramping
    ramp_steps      : Optional[List[float]] = None  # maximum step sizes
    ramp_kwargs     : Optional[dict] = None         # keyword arguments for tandem sweep

def sweepMeasure(C: SweepMeasureConfig) -> np.ndarray:
    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    
    if type(C.file) == str:
        with open(C.file, 'w') as f:
            fname = C.file
            C.file = f
            res = sweepMeasure(C)
            C.file = fname
            return res
    
    if (C.logger or C.file) and not (C.swept_name and C.measured_name):
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
    if C.file:
        msg = f"{''.join(f"{p}, " for p in C.swept_name)}"
        msg += f"{''.join(f"{p}, " for p in C.measured_name)[:-2]}\n"
        C.file.write(msg)

    # log the swept parameters
    if C.logger:
        C.logger.info(
            f"Sweeping: {''.join(f"{p}, " for p in C.swept_name)[:-2]}"
        )
        C.logger.info(
            f"Measuring: {''.join(f"{p}, " for p in C.measured_name)[:-2]}"
        )

    # move everything to the start of the sweep
    for d, setter in enumerate(C.sweep):
        setter(C.points[0, d])

    prev = C.points[0, :]

    # step through each bias point
    for n in range(npoints):
        biases = C.points[n, :]
        # log the bias point coordinates
        if C.logger:
            C.logger.info(
                f"Biases: {''.join([f"{x:.5f}, " for x in biases])[:-2]}"
            )

        # move to the bias point, wait some time, then measure
        set_swept_params(C.sweep, prev, biases, 
                         C.ramp_wait, C.ramp_steps, C.ramp_kwargs)
        prev = biases

        if C.pre_callback:
            C.pre_callback(n, biases.flatten())
        
        time.sleep(C.time_per_point)
        for i in range(len(C.measurement)):
            result[n, i] = C.measurement[i]()
        
        # write the result to the ouput file if provided
        if C.file:
            msg = f"{''.join([f"{x}, " for x in biases])}"
            msg += f"{''.join(f"{r}, " for r in result[n])[:-2]}\n"
            C.file.write(msg)

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
    file            : Optional; path to output file or a file-like object
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
    space_mask      : Optional; called before moving to a new point to determine
                      whether we take a point there.
    ramp_wait       : Optional; wait time if using tandem sweep (important for
                      safely sweeping in some circumstances)
    ramp_step       : Optional; list of maximum step sizes for each parameter if
                      using controlled tandem sweeping. Otherwise, this should
                      be handled by passing the dedicated sweep function for
                      that parameter (this only matters for parameters that need
                      to be swept simultaneously to avoid issue, such as top and
                      bottom gates for VdW systems)
    ramp_kwargs     : Optional; keyword arguments for tandem sweep (particularly
                      important for 'min_step', which tells the sweep to ignore
                      trivial changes in a swept parameter that cost unecessary
                      time.)
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
    file            : Optional[Union[IO, str]] = None   # output file-like
    measured_name   : Optional[         # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    swept_name      : Optional[         # names of swept parameters
                            Tuple[str, ...]
                        ] = None
    logger          : Optional[object] = None   # logger
    pre_callback    : Optional[Callable[[np.ndarray, np.ndarray], Any]] = None
    post_callback   : Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Any]] = None
    space_mask      : Optional[Callable[[np.ndarray, np.ndarray], bool]] = None

    # If tandem sweeping is desired, need to provide these
    ramp_wait       : Optional[float] = None        # wait time between steps when ramping
    ramp_steps      : Optional[List[float]] = None  # maximum step sizes
    ramp_kwargs     : Optional[dict] = None         # keyword arguments for tandem sweep

def sweepMeasureProduct(C: SweepMeasureProductConfig) -> np.ndarray:
    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    
    if type(C.file) == str:
        with open(C.file, 'w') as f:
            fname = C.file
            C.file = f
            res = sweepMeasureProduct(C)
            C.file = fname
            return res
    
    if (C.logger or C.file) and not (C.swept_name and C.measured_name):
        raise ValueError(
            "swept_name and measured_name are required for file or console logging."
        )

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
    if C.file:
        msg = f"{''.join(f"{p}, " for p in C.swept_name)}"
        msg += f"{''.join(f"{p}, " for p in C.measured_name)[:-2]}\n"
        C.file.write(msg)

    # log the swept parameters
    if C.logger:
        C.logger.info(
            f"Sweeping: {''.join(f"{p}, " for p in C.swept_name)[:-2]}"
        )
        C.logger.info(
            f"Measuring: {''.join(f"{p}, " for p in C.measured_name)[:-2]}"
        )

    prev = np.array([C.axes[d][*[0]*len(C.axes)] for d in range(len(C.sweep))])

    # step through each bias point
    for ind in itertools.product(*(range(ax.size) for ax in C.axes)):
        biases = np.array([C.axes[d][ind[d]] 
                           for d in range(len(C.sweep))])
        
        if (C.space_mask is not None) and (not C.space_mask(ind, biases)):
            continue
        
        # log the bias point coordinates
        if C.logger:
            C.logger.info(
                f"Biases: {''.join([f"{x:.5f}, " for x in biases])[:-2]}"
            )

        # move to the bias point, wait some time, then measure
        set_swept_params(C.sweep, prev, biases, 
                         C.ramp_wait, C.ramp_steps, C.ramp_kwargs)
        prev = biases

        if C.pre_callback:
            C.pre_callback(ind, biases.flatten())
        
        time.sleep(C.time_per_point)
        for i in range(len(C.measurement)):
            result[i, *ind] = C.measurement[i]()
        
        # write the result to the ouput file if provided
        if C.file:
            msg = f"{''.join([f"{x}, " for x in biases])}"
            msg += f"{''.join(f"{r}, " for r in result[:, *ind])[:-2]}\n"
            C.file.write(msg)

        # log the result
        if C.logger:
            msg = f"Result = "
            msg += f"{''.join(f"{r:.5f}, " for r in result[:, *ind])[:-2]}"
            C.logger.info(msg)

        if C.post_callback:
            C.post_callback(ind, biases.flatten(), result[:, *ind].flatten())

    return result

"""
SweepMeasureParallelepipedConfig
sweepMeasureParallelepiped

Sweep parameters over the product space of provided axes, taking measurements at
each point.

    measurement     : Getter(s) for measured parameter(s) 
                        (callable or tuple of callables)
    sweep           : Setters for swept parameters
                        (tuple of callables)
    origin          : starting corner of the parallelepiped
    endpoints       : ndarray for the other corners of the parallelepiped, the
                      rows each being a respective endpoints. The fastest swept
                      direction is last.
    shape           : list representing the number of points along each axis
    time_per_point  : Time to wait before taking a measurement at each point
    file            : Optional; path to output file or a file-like object
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
    space_mask      : Optional; called before moving to a new point to determine
                      whether we take a point there.
    ramp_wait       : Optional; wait time if using tandem sweep (important for
                      safely sweeping in some circumstances)
    ramp_step       : Optional; list of maximum step sizes for each parameter if
                      using controlled tandem sweeping. Otherwise, this should
                      be handled by passing the dedicated sweep function for
                      that parameter (this only matters for parameters that need
                      to be swept simultaneously to avoid issue, such as top and
                      bottom gates for VdW systems)
    ramp_kwargs     : Optional; keyword arguments for tandem sweep (particularly
                      important for 'min_step', which tells the sweep to ignore
                      trivial changes in a swept parameter that cost unecessary
                      time.)
"""

@dataclass
class SweepMeasureParallelepipedConfig:
    measurement     : Union[            # getter(s) for measurement
                            Tuple[Callable[[], float], ...], 
                            Callable[[], float]
                        ]
    sweep           : Tuple[            # setter(s) for parameters to sweep
                            Callable[[float], Any]
                        ]
    origin          : List[float]       # starting corner of the parallelepiped
    endpoints       : np.ndarray        # other corners (fastest direction last)
    shape           : List[int]         # number of points for each axis
    time_per_point  : float             # wait time per point
    file            : Optional[Union[IO, str]] = None   # output file-like
    measured_name   : Optional[         # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    swept_name      : Optional[         # names of swept parameters
                            Tuple[str, ...]
                        ] = None
    logger          : Optional[object] = None   # logger
    pre_callback    : Optional[Callable[[np.ndarray, np.ndarray], Any]] = None
    post_callback   : Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Any]] = None
    space_mask      : Optional[Callable[[np.ndarray, np.ndarray], bool]] = None

    # If tandem sweeping is desired, need to provide these
    ramp_wait       : Optional[float] = None        # wait time between steps when ramping
    ramp_steps      : Optional[List[float]] = None  # maximum step sizes
    ramp_kwargs     : Optional[dict] = None         # keyword arguments for tandem sweep

def sweepMeasureParallelepiped(C: SweepMeasureParallelepipedConfig) -> np.ndarray:
    # For parameters that can be either single values or tuples, wrap the single
    # values in a tuple and validate input dimensions
    
    if type(C.file) == str:
        with open(C.file, 'w') as f:
            fname = C.file
            C.file = f
            res = sweepMeasureParallelepiped(C)
            C.file = fname
            return res
    
    if (C.logger or C.file) and not (C.swept_name and C.measured_name):
        raise ValueError(
            "swept_name and measured_name are required for file or console logging."
        )

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

    Ncorners, Nsetters = C.endpoints.shape

    if not len(C.shape) == Ncorners:
        raise ValueError("shape does not match the rows in endpoints.")
    
    if not len(C.origin) == Nsetters:
        raise ValueError("origin does not match columns in endpoints.")
    
    if not len(C.sweep) == Nsetters:
        raise ValueError("sweep does not match columns in endpoints.")
    
    C.origin = np.array(C.origin)
    
    # result array
    result = np.full(shape = (len(C.measurement), *C.shape), 
                     fill_value = np.nan)

    # write the header for the output file
    if C.file:
        msg = f"{''.join(f"{p}, " for p in C.swept_name)}"
        msg += f"{''.join(f"{p}, " for p in C.measured_name)[:-2]}\n"
        C.file.write(msg)

    # log the swept parameters
    if C.logger:
        C.logger.info(
            f"Sweeping: {''.join(f"{p}, " for p in C.swept_name)[:-2]}"
        )
        C.logger.info(
            f"Measuring: {''.join(f"{p}, " for p in C.measured_name)[:-2]}"
        )

    A = (C.endpoints - C.origin.reshape(1, -1)).T
    prev = C.origin

    for ind in itertools.product(*(range(p) for p in C.shape)):
        norm_coords = np.array([ind[i] / (C.shape[i] - 1) 
                                for i in range(len(C.shape))])
        biases = C.origin + (A @ norm_coords)

        if (C.space_mask is not None) and (not C.space_mask(ind, biases)):
            continue
        
        # log the bias point coordinates
        if C.logger:
            C.logger.info(
                f"Biases: {''.join([f"{x:.5f}, " for x in biases])[:-2]}"
            )

        # move to the bias point, wait some time, then measure
        set_swept_params(C.sweep, prev, biases, 
                         C.ramp_wait, C.ramp_steps, C.ramp_kwargs)
        prev = biases

        if C.pre_callback:
            C.pre_callback(ind, biases.flatten())
        
        time.sleep(C.time_per_point)
        for i in range(len(C.measurement)):
            result[i, *ind] = C.measurement[i]()
        
        # write the result to the ouput file if provided
        if C.file:
            msg = f"{''.join([f"{x}, " for x in biases])}"
            msg += f"{''.join(f"{r}, " for r in result[:, *ind])[:-2]}\n"
            C.file.write(msg)

        # log the result
        if C.logger:
            msg = f"Result = "
            msg += f"{''.join(f"{r:.5f}, " for r in result[:, *ind])[:-2]}"
            C.logger.info(msg)

        if C.post_callback:
            C.post_callback(ind, biases.flatten(), result[:, *ind].flatten())

    return result
