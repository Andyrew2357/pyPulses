"""Various functions for sweeping parameters and measuring at each step."""

import numpy as np
import itertools
import datetime
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, IO, List, Optional, Tuple, Union

from .tandem_sweep import tandemSweep

"""
ParamSweepMeasureConfig

Arguments that are shared between all parameter sweeps.

    getters         : Getter(s) for measured parameter(s) 
                        (callable or tuple of callables)
    setters         : Setters for swept parameters; must act as getters when
                      called without an argument
                        (tuple of callables)
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
    retain_return   : Optional; whether to hold on to measured values and return
                      as a numpy array. This can be memory intensive for large
                      scans, so make sure to set it False in those cases. By
                      default it is True.
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
    timestamp       : Optional; whether to include a timestamp for each point 
                      (Note this is not returned, only written to files/passed
                      to callbacks)
"""

@dataclass(kw_only=True)
class ParamSweepMeasure:
    getters         : Union[                                                    # getter(s) for measurement
                            Tuple[Callable[[], float], ...], 
                            Callable[[], float]
                        ] = ()
    setters         : Union[                                                    # setter(s) for swept parameters
                            Tuple[Callable[[Any], Any], ...],                   # setters must also be able to 'get'
                            Callable[[Any], Any]                                # the parameter if called without an
                        ] = ()                                                  # argument
    time_per_point  : Optional[float] = 0
    file            : Optional[Union[IO, str]] = None                           # output file-like object or string
    measured_name   : Optional[                                                 # names of measured parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    swept_name      : Optional[                                                 # names of swept parameters
                            Union[
                                Tuple[str, ...],
                                str
                            ]
                        ] = None
    retain_return   : Optional[bool] = True                                     # whether we return results (can be memory intensive)
    logger          : Optional[object] = None                                   # logger
    pre_callback    : Optional[                                                 # callback before measurement
                            Callable[
                                [Union[int, np.ndarray], 
                                 np.ndarray, Optional[datetime.datetime]], 
                            Any]
                        ] = None
    post_callback   : Optional[                                                 # callback after measurement
                            Callable[
                                [Union[int, np.ndarray], 
                                np.ndarray, np.ndarray, 
                                Optional[datetime.datetime]], 
                            Any]
                        ] = None
    space_mask      : Optional[                                                 # mask for which points to take
                            Callable[
                                [Union[int, np.ndarray], 
                                np.ndarray], 
                            bool]
                        ] = lambda *args: True
                                                                                # If tandem sweeping is desired, need to provide these
    ramp_wait       : Optional[float] = None                                    # wait time between steps when ramping
    ramp_steps      : Optional[List[float]] = None                              # maximum step sizes
    ramp_kwargs     : Optional[dict] = None                                     # keyword arguments for tandem sweep
    timestamp       : Optional[bool] = True                                     # whether to include a timestamp for each point
    return_home     : Optional[bool] = True                                     # whether to return to the starting point after sweeping

    def __post_init__(self):
        """Input validation"""

        if (self.file or self.logger) and not \
            (self.swept_name and self.measured_name):
            raise ValueError(
                "swept_name and measured_name are required for file or logging."
            )

        if not type(self.setters) in [tuple, list]:
            self.setters = (self.setters,)

        if not type(self.getters) in [tuple, list]:
            self.getters = (self.getters,)

        if self.swept_name and self.measured_name:
            if type(self.swept_name) == str:
                self.swept_name = (self.swept_name,)

            if type(self.measured_name) == str:
                self.measured_name = (self.measured_name,)

            if (len(self.swept_name) != len(self.setters)) or \
                (len(self.measured_name) != len(self.getters)):
                raise ValueError("Dimension mismatch in parameters and names.")
            
        if not hasattr(self, 'npoints'):
            self.npoints = 0

    """
    It is possible to add and multiply sweeps to obtain more complicated
    behavior. Addition corresponds to concatenation, wheras multiplication
    places the 'inner' sweep (the right factor) into the callback of the 'outer'
    sweep (left factor). The resulting sweep also inherits the return_home
    behavior of the left object. As of right now, the file writing and
    indexing behavior is not analogous to the behavior exhibited by other
    sweeps. I may change how this functions in the future
    """

    def __barebones(self):
        res = deepcopy(self)
        res.logger          = None
        res.retain_return   = False
        return res
            
    def __add_compat__(self, other) -> bool:
        if not isinstance(other, ParamSweepMeasure):
            return False
        
        return True # should maybe be more stringent
             
    def __add__(self, other):
        if not self.__add_compat__(other):
            raise SyntaxError(
                f"Operation '+' not supported between {self} and {other}."
            )
        
        first   = deepcopy(self)
        second  = deepcopy(other)
        return_home = first.return_home
        first.return_home = False
        first.return_home = False

        sum = ParamSweepMeasure(return_home = return_home)
        def run(self):
            first.run()
            second.run()
            if self.return_home:
                set_swept_params(second, second.home)
                set_swept_params(first, first.home)
        sum.run = run
        sum.npoints = first.npoints + second.npoints

        return sum

    def __mul_compat__(self, other) -> bool:
        if not isinstance(other, ParamSweepMeasure):
            return False
        
        return True # Should maybe be more stringent

    def __mul__(self, other):
        if not self.__mul_compat__(other):
            raise SyntaxError(
                f"Operation '*' not supported between {self} and {other}."
            )

        outer = self.__barebones()
        inner = deepcopy(other)
        return_home = outer.return_home
        inner.return_home = True
        
        outer_pre_callback = outer.pre_callback
        def pre_callback(*args, **kwargs):
            if outer_pre_callback:
                outer_pre_callback(*args, **kwargs)
            inner.run()

        outer.pre_callback = pre_callback
        outer.npoints *= inner.npoints

        return outer

"""Utility functions to avoid redundancy"""

def safe_file_handling(paramSweep: Callable[[ParamSweepMeasure], 
                                            Optional[np.ndarray]]):
    def file_handling_paramSweep(C: ParamSweepMeasure):
        if type(C.file) == str:
            with open(C.file, 'a') as f:
                fname = C.file
                C.file = f
                res = paramSweep(C)
                C.file = fname
                return res
            
        return paramSweep(C)
        
    return file_handling_paramSweep

def set_swept_params(C: ParamSweepMeasure, targets: np.ndarray):
    """Utility function to smoothly integrate tandemSweep functionality."""

    prev = np.array([setter() for setter in C.setters])
    if C.ramp_wait is not None:
        tandemSweep(C.setters, prev, targets, 
                    C.ramp_steps, C.ramp_wait, **C.ramp_kwargs)
    else:
        for i in range(len(C.setters)):
            if prev[i] != targets[i]:
                C.setters[i](targets[i])

def measure_at_point(C: ParamSweepMeasure, ind: Union[int, np.ndarray],
                     targets: np.ndarray) -> np.ndarray:
    """Move to a given point in parameter space, and measure there."""

    # If we are outside the mask, don't measure
    if not C.space_mask(ind, targets):
        return np.full(len(C.getters), fill_value = np.nan)

    # log the target coordinates    
    if C.logger:
        C.logger.info(
            f"Targets: {''.join([f"{x:.5f}, " for x in targets])[:-2]}"
        )

    # move to the target point
    set_swept_params(C, targets)

    # wait for things to settle
    time.sleep(C.time_per_point)

    if C.timestamp:
        now = datetime.datetime.now()

    # pre-measurement callback
    if C.pre_callback:
        if C.timestamp:
            C.pre_callback(ind, targets, now)
        else:
            C.pre_callback(ind, targets)

    # measure the desired parameters
    measured = np.array([get() for get in C.getters])

    # write the result to an output file if provided
    if C.file:
        msg  = f"{''.join([f"{x}, " for x in targets])}"
        msg += f"{''.join(f"{r}, " for r in measured)[:-2]}\n"
        if C.timestamp:
            msg = f"{now}, " + msg
        C.file.write(msg)

    # log the result
    if C.logger:
        C.logger.info(
            f"Measured: {''.join(f"{r:.5f}, " for r in measured)[:-2]}"
        )

    # post-measurement callback
    if C.post_callback:
        if C.timestamp:
            C.post_callback(ind, targets, measured, now)
        else:
            C.post_callback(ind, targets, measured)

    return measured

def initialize_sweep(C: ParamSweepMeasure):
    """
    Start off a sweep by writing to an output file, logging, and moving to the
    starting point.
    """

    # write the header for the output file
    if C.file:
        msg = f"{''.join(f"{p}, " for p in C.swept_name)}"
        msg += f"{''.join(f"{p}, " for p in C.measured_name)[:-2]}\n"
        if C.timestamp:
            msg = "timestamp, " + msg
        C.file.write(msg)

    # log the swept parameters
    if C.logger:
        C.logger.info(
            f"Sweeping: {''.join(f"{p}, " for p in C.swept_name)[:-2]}"
        )
        C.logger.info(
            f"Measuring: {''.join(f"{p}, " for p in C.measured_name)[:-2]}"
        )

    set_swept_params(C, C.home)

def cleanup_after_sweep(C: ParamSweepMeasure):
    """
    Clean up after a sweep is done (includes things like returning to the 
    starting point if return_home is True).
    """

    if C.return_home:
        set_swept_params(C, C.home)

"""
SweepMeasure

Generalization of SweepMeasureCut. Sweep parameters to each point provided in 
the array 'points', taking measurements at each step.

    points  : ndarray of shape (number of points, number of swept
              parameters) listing the points in parameter space at which
              to take measurements.
"""

@dataclass(kw_only=True)
class SweepMeasure(ParamSweepMeasure):
    points: np.ndarray  # points at which to measure, 
                        # shape = (#points, #swept params)

    def __post_init__(self):
        super().__post_init__()

        # if points is flat, reshape it to a column vector
        if self.points.ndim == 1: 
            self.points = self.points.reshape(-1, 1)

        self.npoints = self.points.shape[0]
        self.home = self.points[0, :].copy()

    @safe_file_handling
    def run(self) -> Optional[np.ndarray]:

        npoints, nswept = self.points.shape
        nmeasured = len(self.getters)
        if self.retain_return:
            result = np.full(shape = (npoints, nmeasured), fill_value = np.nan)

        initialize_sweep(self)

        # step through each bias point
        for n in range(npoints):
            targets = self.points[n, :]
            
            measured = measure_at_point(self, n, targets)
            if self.retain_return:
                result[n, :] = measured

        cleanup_after_sweep(self)

        if self.retain_return:
            return result

"""
SweepMeasureCut
(replaces biasSweep from a previous version)

Sweep parameters linearly from one point in parameter space to another, taking
measurements at each point.

    numpoints : Number of points to take
    start     : Starting points in parameter space
    end       : Ending points in parameter space
"""

@dataclass(kw_only=True)
class SweepMeasureCut(ParamSweepMeasure):
    numpoints: int                                   # number of points to take
    start    : Union[float, np.ndarray, List[float]] # starting parameters
    end      : Union[float, np.ndarray, List[float]] # ending parameters

    def __post_init__(self):
        super().__post_init__()

        # check the dimensionality of start, end, and setters match
        if (len(self.start) != len(self.setters)) or \
            (len(self.end) != len(self.setters)):
            raise ValueError(
                "start and end dimensions to not match the number of setters."
            )
        
        self.start = np.array(self.start)
        self.end = np.array(self.end)
        self.npoints = self.numpoints
        self.home = self.start.copy()

    @safe_file_handling
    def run(self) -> Optional[np.ndarray]:

        nmeasured = len(self.getters)
        if self.retain_return:
            result = np.full(shape = (self.points, nmeasured), fill_value = np.nan)

        initialize_sweep(self)

        # step through each bias point
        for n in range(self.numpoints):
            targets = self.start + (self.end - self.start)*n/(self.numpoints - 1)
            
            measured = measure_at_point(self, n, targets)
            if self.retain_return:
                result[n, :] = measured

        cleanup_after_sweep(self)

        if self.retain_return:
            return result

"""
SweepMeasureProduct

Sweep parameters over the product space of provided axes, taking measurements at
each point.

    axes: Tuple of ndarrays representing the axes for each parameter
"""

@dataclass(kw_only=True)
class SweepMeasureProduct(ParamSweepMeasure):
    axes: Tuple[np.ndarray] # axes over which to sweep parameters

    def __post_init__(self):
        super().__post_init__()

        # check that axes match the number of setters
        if len(self.axes) != len(self.setters):
            raise ValueError("axes does not match the number of setters.")
        
        self.npoints = np.prod([len(a) for a in self.axes])
        self.home = np.array([self.axes[i][0] for i in range(len(self.axes))])
        
    @safe_file_handling
    def run(self) -> Optional[np.ndarray]:

        space_dim = [len(a) for a in self.axes]
        if self.retain_return:
            result = np.full(shape = (*space_dim, len(self.getters)), 
                            fill_value = np.nan)

        initialize_sweep(self)

        #step through each bias point
        for ind in itertools.product(*[range(d) for d in space_dim]):
            targets = np.array([self.axes[d][ind[d]] 
                                for d in range(len(self.axes))])

            measured = measure_at_point(self, ind, targets)
            if self.retain_return:
                result[*ind, :] = measured

        cleanup_after_sweep(self)

        if self.retain_return:
            return result

"""
SweepMeasureParallelepiped

Sweep parameters over the product space of provided axes, taking measurements at
each point.

    origin      : starting corner of the parallelepiped
    endpoints   : ndarray for the other corners of the parallelepiped, the
                  rows each being a respective endpoints. The fastest swept
                  direction is last.
    shape       : list representing the number of points along each axis
"""

@dataclass(kw_only=True)
class SweepMeasureParallelepiped(ParamSweepMeasure):
    origin      : List[float]   # starting corner of the parallelepiped
    endpoints   : np.ndarray    # other corners (fastest direction last)
    shape       : List[int]     # number of points for each axis

    def __post_init__(self):
        super().__post_init__()

        # check that all of the dimensions line up
        ncorners, nsetters = self.endpoints.shape

        if not len(self.shape) == ncorners:
            raise ValueError("shape does not match the rows in endpoints.")
        
        if not len(self.origin) == nsetters:
            raise ValueError("origin does not match columns in endpoints.")
        
        if not len(self.setters) == nsetters:
            raise ValueError("sweep does not match columns in endpoints.")
        
        self.origin = np.array(self.origin)
        self.npoints = np.prod(self.shape)
        self.home = self.origin.copy()

    @safe_file_handling
    def run(self) -> Optional[np.ndarray]:

        if self.retain_return:
            result = np.full(shape = (*self.shape, len(self.getters)), 
                            fill_value = np.nan)
            
        initialize_sweep(self)

        # step through each point
        A = (self.endpoints - self.origin.reshape(1, -1)).T
        for ind in itertools.product(*(range(d) for d in self.shape)):
            norm_coords = np.array([ind[i] / (self.shape[i] - 1) 
                                    for i in range(len(self.shape))])
            targets = self.origin + (A @ norm_coords)

            measured = measure_at_point(self, ind, targets)
            if self.retain_return:
                result[*ind, :] = measured

        cleanup_after_sweep(self)

        if self.retain_return:
            return result
