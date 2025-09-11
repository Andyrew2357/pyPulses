"""Various functions for sweeping parameters and measuring at each step."""

import numpy as np
import itertools
import datetime
import time
import os
import logging
from copy import deepcopy
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from .tandem_sweep import tandemSweep
from .getsetter import getSetter
from ..plotting.recorder import SweepRecorder
from ..thread_job import _checkpoint, ThreadJob

def _get_data_str(coords: np.ndarray, data: np.ndarray, 
                 now: datetime.datetime | None) -> str:
    """Pretty formatting for output data"""
    coord_cols = "".join(f'\t{val:12g}' for val in coords)
    data_cols = "".join(f'\t{val:12g}' for val in data)
    cols = coord_cols + data_cols
    return f"{str(now):<24}{cols}" if now else cols.lstrip()

COMMON_ARGS = [
    'measurements', 
    'coordinates', 
    'time_per_point', 
    'file_prefix',
    'points_per_file', 
    'starting_fnum', 
    'retain_return', 
    'logger', 
    'pre_callback', 
    'post_callback',
    'space_mask', 
    'ramp_wait', 
    'ramp_kwargs', 
    'timestamp'
]

@dataclass(kw_only=True)
class ParamSweepMeasure:
    """
    Base class for all parameter sweeps.

    We define an algebra over ParamSweepMeasure object consisting of 
    concatenation and multiplication (somewhat like an outer product).
    
    For both addition and multiplication, the resulting sweep inherits its 
    common properties like logging, file_prefix, etc. from the left operand.

    For addition, the concatenation must be between sweeps that differ in at
    most the outermost dimension. The left operance comes before the right, and
    concatenation occurs along the outermost axis (axis 0).

    For multiplication, the left operand forms the outer loop, and the right
    operance forms the inner loop. We perform a linear inclusion map from the 
    coordinate space of both sweeps into a larger space that includes all the
    coordinates (there can be overlap). Any given set of coordinates in the
    output sweep is then calculated by taking the sum of the inner and outer
    coordinates under the inclusion map.

    Attributes
    ----------
    measurements : dict or list of dicts
        Measured parameters in the form: 
            {'f': getter, 'name': parameter name or list of column names if
            the return is a tuple, 'lazy': boolean flag indicating whether
            this is a lazy measurement.}
        The 'f' attribute is required and must be a callable that returns a 
        float. The 'name' and 'lazy' attribute are optional.
    coordinates : dict or list of dicts
        Swept parameters in the form:
            {
            'f': getter, 'name': parameter name, 
            'min_step': minimum step to take when sweeping,
            'max_step': maximum step to take when sweeping,
            'tolerance': error tolerance when deciding if 
                         parameter needs to be set at all
            }
        The 'f' attribute is required and must be a callable that returns a 
        float when called without an argument and sets the parameter when called 
        with a float as its argument. Alternatively, the user can provide 'get' 
        and 'set' attributes that behave appropriately, and the program will 
        package them into an 'f' attribute itself. The name attribute is 
        optional and should be a string. The other arguments are floats and 
        should be self explanatory by looking at tandemSweep.
    time_per_point : float
        Time to wait before taking a measurement at each point.
    file_prefix : str, optional
        Prefix to use for output files.
    points_per_file : int, optional
        Number of points to include in a single file.
    starting_fnum : int, default=0
        Starting file number; The full name takes the form:
            f'{file_prefix} ({n:05}).dat'
    logger : Logger, optional
    pre_callback : Callable, optional
        Generic callback function called before the measurement is taken at each 
        point. The function passes back the index of the current point in the 
        sweep, and a 1d array of the swept parameters at that point.
    post_callback : Callback, optional
        Generic callback function called after the result is updated at each 
        point. The function passes back the index of the current point in the 
        sweep, a 1d array of the swept paramters at that point, and a 1d array 
        of measurement results at that point for use by the callback function.
    space_mask : Callable, optional
        Called before moving to a new point to determine whether we take a point 
        there.
    retain_return : bool, default=True
        Whether to hold on to measured values and return as a numpy array. This 
        can be memory intensive for large scans, so make sure to set it False in 
        those cases.
    skip_points : int, default=0
        Number of points at the beginning of the scan to skip (Can be useful
        when restarting a partially completed scan that was interrupted).
    timestamp : bool, default=True
        Whether to include a timestamp for each point (Note this is not returned, 
        only written to files and passed to callbacks).
    ramp_wait : float, optional 
        Wait time if using tandem sweep (important for safely sweeping in some 
        circumstances).
    ramp_checkpoints : bool, default=False
        Whether to include `thread_job` checkpoints when ramping.
    ramp_kwargs : dict, optional
        Keyword arguments for tandem sweep.
    plot_fields : str or List[str | int], optional
        If 'all', will plot all coordinates and measurements. If a list, it will
        plot the parameters identified by index or name.
    plot_kwargs : dict, optional
        Additional arguments passed when creating the `SweepRecorder` for live
        plotting.
    cleanup : Callable
        Cleanup called after the measurement has finished.
    """
    measurements    : Dict[str, Any] | List[Dict[str, Any]]                     # measured variables
    coordinates     : Dict[str, Any] | List[Dict[str, Any]]                     # swept variables
    time_per_point  : float = 0.
    file_prefix     : str = None                                                # string prefix for output file
    points_per_file : int = None                                                # number of points per output file
    starting_fnum   : int = 1                                                   # starting file number
    retain_return   : bool = True                                               # whether we return results (can be memory intensive)
    logger          : logging.Logger = None                                     # logger
    pre_callback    : Callable[                                                 # callback before measurement
                        [np.ndarray, np.ndarray, datetime.datetime], Any
                        ] = None
    post_callback   : Callable[                                                 # callback after measurement
                        [np.ndarray, np.ndarray, np.ndarray, datetime.datetime], 
                        Any
                        ] = None
    space_mask      : Callable[[int | np.ndarray, np.ndarray], bool] \
                        = lambda *args: True                                    # mask for which points to take
    skip_points     : int = 0                                                   # points to skip (useful when restarting an interrupted scan)
    timestamp       : bool = True                                               # whether to include a timestamp for each point
    
    # If tandem sweeping is desired, need to provide these
    ramp_wait       : float = None                                              # wait time between steps when ramping
    ramp_checkpoints: bool = False                                              # whether to include thread_job checkpoints when ramping
    ramp_kwargs     : dict = None                                               # keyword arguments for tandem sweep

    # provide these if you want to plot
    plot_fields     : str | List[int | str]                                     # fields to plot while taking sweep
    plot_kwargs     : dict = None                                               # keyword arguments passed to sweep recorder

    # cleanup steps to perform after the sweep
    cleanup         : Callable[[], Any] = None

    def __post_init__(self):
        """Input validation"""

        if isinstance(self.measurements, dict):
            self.measurements = (self.measurements,)

        measurement_widths = []
        self.unwrapped_measurements = []
        for v in self.measurements:
            cols = v.get('name')
            if cols is None or isinstance(cols, str):
                measurement_widths.append(1)
                self.unwrapped_measurements.append(v)
            else:
                measurement_widths.append(len(cols))
                for c in cols:
                    temp = deepcopy(v)
                    temp['name'] = c
                    self.unwrapped_measurements.append(temp)
        self.num_measurement_cols = sum(measurement_widths)

        i = 0
        self.normal_measurements = []
        self.lazy_measurements = []
        for v, m in zip(self.measurements, measurement_widths):
            f = v.get('f', v.get('get'))
            if f is None:
                raise ValueError(f'Invalid getter for measured variable: {v}.')
            
            if v.get('lazy', False):
                self.lazy_measurements.append((i, m, f))
            else:
                self.normal_measurements.append((i, m, f))
            i += m
        
        if isinstance(self.coordinates, dict):
            self.coordinates = (self.coordinates,)
        self.coord_name = [v.get('name', 'unnamed') for v in self.coordinates]

        temp_data_name = [v.get('name', 'unnamed') for v in self.measurements]
        self.data_name = []
        for n in temp_data_name:
            if isinstance(n, str):
                self.data_name.append(n)
            else:
                self.data_name.extend(n)
        
        for v in self.coordinates:
            if 'get' and 'set' in v:
                v['f'] = getSetter(v['get'], v['set'])

        self.coord_vars = [v.get('f', None) for v in self.coordinates]

        self.ramp_max_step = [v.get('max_step', None) for v in self.coordinates]
        self.ramp_min_step = [v.get('min_step', None) for v in self.coordinates]
        self.ramp_tolerance = [v.get('tolerance', 0.) for v in self.coordinates]

        if not self.ramp_kwargs:
            self.ramp_kwargs = {}

        self.written_points = 0
        self.npoints = 0
        self.dim = 0
        self.dimensions = ()

    def run_threaded(self) -> np.ndarray | None:
        self.job = ThreadJob(self.run)
        def cleanup(*_):
            del self.job
            self.job = None
        self.job.on_finish = cleanup
        self.job.on_stop = cleanup
        self.job.on_error = cleanup
        self.job.start_with_controls()
        return self.job

    def run(self) -> np.ndarray | None:
        """
        Run the parameter sweep.
        
        Returns
        -------
        dataset : np.ndarray or None
            Only returned if `retain_return` is True.
        """

        if self.retain_return:
            result = np.full(shape = (*self.dimensions, self.num_measurement_cols),
                             fill_value = np.nan, dtype = float)

        start_time = time.time()
        points_taken = 0
        try:
            if self.plot_fields is not None:
                sr = self._get_live_plotter()
                sr.show()
            
            self._log_header()

            for idx, target in self._iterator():
                
                points_taken += 1
                if points_taken < self.skip_points:
                    continue
                _checkpoint()

                measured_vars = self.measure_at_point(idx, target)

                if self.plot_fields is not None:
                    sr.update(target, measured_vars)

                if self.retain_return:
                    result[*idx,:] = measured_vars

                self._log_time_remaining(points_taken, start_time)

            if self.plot_fields is not None:
                sr.draw()

        finally:
            self._close_file_logger()
            if self.cleanup:
                self.cleanup()

        if self.retain_return:
            return result
        
    def preview(self, coords: List[int | str], use_mask = True, **kwargs):
        """
        Plot a preview of the path swept out by the sweep <= 3 dimensions

        Parameters
        ----------
        coords : list of int or str
            Swept parameters to use as the axes for the plot (we project to 
            these axes from the higher dimensional space).
        use_mask : bool, default=True
            Whether to apply the `space_mask` during the simulation.
        """

        if len(coords) > 3 or len(coords) < 1:
            raise ValueError("Invalid 'coord_indices'.")
        
        coord_indices = []
        for c in coords:
            if isinstance(c, int):
                coord_indices.append(c)
            else:
                for i, v in self.coordinates:
                    if v.get('name', '') == c:
                        coord_indices.append(i)
                        break
                else:
                    raise ValueError(f"{c} is not a coordinate.")

        if use_mask:
            points = np.full(shape = (self.npoints, len(coord_indices)), 
                             fill_value=np.nan)
            i = 0
            for idx, coord in self._iterator():
                if self.space_mask(idx, coord):
                    points[i, :] = coord[coord_indices]
                    i += 1
        else:
            points = np.empty(shape = (self.npoints, len(coord_indices)), 
                            dtype = float)
            for i, (idx, coord) in enumerate(self._iterator()):
                points[i, :] = coord[coord_indices]

        defaults = {'marker': 'o', 'color': 'r', 'alpha': 0.5}
        kwargs = {**defaults, **kwargs}

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize = (12, 12))
        match len(coord_indices):
            case 1:
                ax = fig.add_subplot(111)
                ax.plot(points[:,0], **kwargs)
                ax.set_title(self.coord_name[coord_indices[0]])
            case 2:
                ax = fig.add_subplot(111)
                ax.plot(points[:,0], points[:,1], **kwargs)
                ax.set_xlabel(self.coord_name[coord_indices[0]])
                ax.set_ylabel(self.coord_name[coord_indices[1]])
            case 3:
                from mpl_toolkits.mplot3d import Axes3D
                ax = fig.add_subplot(111, projection='3d')
                ax.plot3D(points[:,0], points[:,1], points[:,2], **kwargs)
                ax.set_xlabel(self.coord_name[coord_indices[0]])
                ax.set_ylabel(self.coord_name[coord_indices[1]])
                ax.set_zlabel(self.coord_name[coord_indices[2]])
                
        return points, (fig, ax)

    def _iterator(self):
        raise NotImplementedError(
            "Default ParamSweepMeasure has no iterator."
        )

    def get_coordinates(self) -> np.ndarray:
        """
        Get the current settings of swept parameters

        Returns
        -------
        np.ndarray
        """
        return np.array([P() for P in self.coord_vars])

    def set_coordinates(self, target_coords: np.ndarray):
        """
        Set swept parameters (integrated with tandemSweep)
        
        Parameters
        ----------
        target_coords : np.ndarray
        """

        previous_coords = self.get_coordinates()
        if self.ramp_wait is None:
            for i, P in enumerate(self.coord_vars):
                if previous_coords[i] != target_coords[i]:
                    P(target_coords[i])

        else:
            tandemSweep(self.coord_vars, previous_coords, target_coords,
                        self.ramp_wait, self.ramp_max_step, self.ramp_min_step,
                        self.ramp_tolerance, 
                        ignore_checkpoints = not self.ramp_checkpoints, 
                        **self.ramp_kwargs)
            
    def get_measured_vars(self) -> np.ndarray:
        M = np.full(self.num_measurement_cols, fill_value = np.nan)

        if self.lazy_measurements:
            with ThreadPoolExecutor(max_workers = len(self.lazy_measurements)) as ex:
                
                # thread the lazy measurements
                futures = [ex.submit(f) for _, _, f in self.lazy_measurements]

                # take any non-lazy measurements while waiting for the others
                for i, m, f in self.normal_measurements:
                    M[i: i + m] = f()

                # collect lazy measurement results
                for (i, m, _), p in zip(self.lazy_measurements, futures):
                    M[i: i + m] = p.result()

        else:
            for i, m, f in self.normal_measurements:
                M[i: i + m] = f()
                
        return M

    def measure_at_point(self, idx: np.ndarray, target_coords: np.ndarray
                         ) -> np.ndarray:
        """
        Move to a given point in parameter space and measure there
        
        Parameters
        ----------
        idx : int
            index within the sweep
        target_coords : np.ndarray
        
        Returns
        -------
        np.ndarray
        """

        # If the target coords are outside the mask, terminate.
        if not self.space_mask(idx, target_coords):
            return np.full(self.num_measurement_cols, fill_value = np.nan)

        # Get the time if we want timestamped data
        now = datetime.datetime.now() if self.timestamp else None

        # Move to the target coordinates.
        self.set_coordinates(target_coords)

        # Wait for things to settle
        time.sleep(self.time_per_point)

        # Pre-measurement Callback
        if self.pre_callback:
            self.pre_callback(idx, target_coords, now)

        # Measure the desired parameters
        measured_vars = self.get_measured_vars()

        # log to an output file and logger if desired
        self._log_data(target_coords, measured_vars, now)

        # Post-measurement Callback
        if self.post_callback:
            self.post_callback(idx, target_coords, measured_vars, now)

        return measured_vars
    
    def _get_live_plotter(self):
        variables = [*self.coordinates, *self.unwrapped_measurements]
        if self.plot_fields == 'all':
            plot_fields = variables
        else:
            plot_fields = []
            for i, v in variables:
                if i in self.plot_fields or \
                    v.get('name', '') in self.plot_fields:
                    plot_fields.append(v)
                
        plot_kwargs = self.plot_kwargs or {}
        return SweepRecorder(plot_fields, **plot_kwargs)

    def _get_header_str(self):
        coord_cols = "".join(f'\t{col:>12}' for col in self.coord_name)
        data_cols = "".join(f'\t{col:>12}' for col in self.data_name)
        cols = coord_cols + data_cols
        return f"{'#Timestamp':<24}{cols}" if self.timestamp else cols.lstrip()

    def _log_header(self):
        if not self.logger:
            return
        header_str = self._get_header_str()
        self._log('info', header_str)

    def _log_data(self, coords: np.ndarray, data: np.ndarray, 
                  now: datetime.datetime | None):
        if not self.file_prefix and not self.logger:
            return
        data_str = _get_data_str(coords, data, now)
        self._log('info', data_str)
        self._log_file(data_str)

    def _log_time_remaining(self, points_taken, start_time):
        if not self.logger or not hasattr(self, 'npoints'):
            return
        taken = points_taken - self.skip_points
        npoints = self.npoints - self.skip_points
        pcomplete = taken/npoints
        time_taken = time.time() - start_time
        pcomplete_inv = 1 if pcomplete == 0 else 1/pcomplete
        remaining = int(time_taken * (pcomplete_inv - 1))
        hrs = remaining // 3600
        remaining -= hrs * 3600
        mns = remaining // 60
        scs = remaining - mns * 60
        self._log('info', 
            f"Taken {taken}/{npoints} ({100*pcomplete:.2f}%); "
            f"Est. {hrs}:{mns:02d}:{scs:02d} remaining."
        )

    def _prep_file_handler(self, fname):
        if hasattr(self, 'file_handler'):
            del self.file_handler
        self.file_handler = logging.FileHandler(fname)
        self.file_handler.setLevel(logging.INFO)
        self.file_logger.addHandler(self.file_handler)
        header_str = self._get_header_str()
        self.file_logger.info(header_str)

    def _log_file(self, msg):
        if not self.file_prefix:
            return
        
        # If we don't already have a logger, get one and add the first handler        
        if not hasattr(self, 'file_logger'):
            fname = f"{self.file_prefix} ({self.starting_fnum:05}).dat"
            self.file_logger = logging.getLogger(repr(self))
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.propagate = False
            self._prep_file_handler(fname)

        if self.points_per_file is None:
            self.file_logger.info(msg)
            return

        # If we have written enough points, move to a new file  
        if self.written_points != 0 and\
            self.written_points % self.points_per_file == 0:
            fnum = self.starting_fnum + \
                    self.written_points // self.points_per_file
            fname = f"{self.file_prefix} ({fnum:05}).dat"

            self.file_handler.flush()
            self.file_logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self._prep_file_handler(fname)

        self.file_logger.info(msg)
        self.written_points += 1

    def _close_file_logger(self):
        if not hasattr(self, 'file_logger'):
            return
        
        self.file_handler.flush()
        self.file_logger.removeHandler(self.file_handler)
        self.file_handler.close()
        del self.file_handler
        self.file_handler = None
        self.file_logger.handlers.clear()
        del self.file_logger
        self.file_logger = None
        self.written_points = 0

    def _log(self, type, msg):
        if not self.logger:
            return
        
        match type:
            case 'info':
                self.logger.info(msg)
            case 'debug':
                self.logger.debug(msg)
            case 'warn':
                self.logger.warning(msg)
            case 'error':
                self.logger.error(msg)

    def _common_dict(self) -> dict:
        return {arg: getattr(self, arg) for arg in COMMON_ARGS}

    def __add__(self, other):
        """
        Return the sum (concatenation) of two parameter sweeps.
        They are concatenated along the outermost axis.
        """
        assert isinstance(other, ParamSweepMeasure)

        kwargs = self._common_dict()
        
        if self.dim != other.dim:
            raise ValueError(
                "Forbidden dimension mismatch between operands."
            )

        if self.coordinates != other.coordinates:
            raise ValueError(
                "Cannot add parameter sweeps with different coordinates."
            )
        
        # Check that all but the first dimension matches in the two sweeps
        if not all(self.dim[i] == other.dim[i] for i in range(1, self.dim)):
            raise ValueError(
                "Operands must match in all dimensions but the first."
            )
        
        offset = self.dimensions[0]
        def iterate():
            for idx, coord in self._iterator():
                yield idx, coord

            for idx, coord in other._iterator():
                idx[0] += offset
                yield idx, coord

        concat = SweepMeasureArbitraryIterator(iterate = iterate, **kwargs)
        concat.npoints = self.npoints + other.npoints
        concat.dimensions = self.dimensions.copy()
        concat.dimensions[0] += other.dimensions[0]
        concat.dim = self.dim

        return concat

    def __mul__(self, other):
        """Return the product of two parameter sweeps."""
        assert isinstance(other, ParamSweepMeasure)

        kwargs = self._common_dict()

        # We need to provide inclusion maps from the coord_vars of the original
        # spaces to the new space. 

        coordinates = []
        for coord in self.coordinates:
            if coord in coordinates:
                continue
            coordinates.append(coord)
        for coord in other.coordinates:
            if coord in coordinates:
                continue
            coordinates.append(coord)
        
        N = len(coordinates)
        M1 = len(self.coordinates)
        M2 = len(other.coordinates)
        
        INC1 = np.zeros((N, M1))
        for i in range(M1):
            for j in range(N):
                if coordinates[j]['f'] == self.coordinates[i]['f']:
                    INC1[j, i] += 1
                    break

        INC2 = np.zeros((N, M2))
        for i in range(M2):
            for j in range(N):
                if coordinates[j]['f'] == other.coordinates[i]['f']:
                    INC2[j, i] += 1
                    break

        def iterate():
            for idx1, coords1 in self._iterator():
                for idx2, coords2 in other._iterator():
                    idx = np.concatenate([idx1, idx2])
                    coords = INC1 @ coords1 + INC2 @ coords2
                    yield idx, coords.flatten()

        kwargs['coordinates'] = coordinates
        product = SweepMeasureArbitraryIterator(iterate = iterate, **kwargs)
        product.npoints = self.npoints * other.npoints
        product.dimensions = np.concatenate([self.dimensions, other.dimensions])
        product.dim = self.dim + other.dim

        return product

@dataclass(kw_only=True)
class SweepMeasureArbitraryIterator(ParamSweepMeasure):
    """
    Sweep parameters over the product space of provided axes, taking 
    measurements at each point.

    Attributes
    ----------
    iterate : Callable
        What iterator to use for parameter sweep; yields tuples of index and 
        coordinate values.
    """
    iterate: Callable

    def __post_init__(self):
        super().__post_init__()
        self._iterator = self.iterate

################################################################################

@dataclass(kw_only=True)
class SweepMeasure(ParamSweepMeasure):
    """
    Generalization of SweepMeasureCut. Sweep parameters to each point provided 
    in the array 'points', taking measurements at each step.

    Attributes
    ----------
    points : ndarray 
        Shape (number of points, number of swept parameters) listing the points 
        in parameter space at which to take measurements.
    """
    points: np.ndarray  # points at which to measure, 
                        # shape = (#points, #swept params)

    def __post_init__(self):
        super().__post_init__()

        # if points is flat, reshape it to a column vector
        if self.points.ndim == 1: 
            self.points = self.points.reshape(-1, 1)

        if self.points.shape[1] != len(self.coord_vars):
            raise ValueError(
                "start and end dimensions to not match the number of coord vars."
            )

        self.npoints = self.points.shape[0]
        self.dim = 1
        self.dimensions = self.npoints * np.array([1], dtype = int)

    def _iterator(self):
        for i, pnt in enumerate(self.points):
            yield np.array([i]), pnt

@dataclass(kw_only=True)
class SweepMeasureCut(ParamSweepMeasure):
    """
    Sweep parameters linearly from one point in parameter space to another, 
    taking measurements at each point.

    Attributes
    ----------
    numpoints : int
        Number of points to take.
    start : float or array-like
        Starting points in parameter space.
    end : float or array-like
        Ending points in parameter space.
    """
    numpoints: int                              # number of points to take
    start    : float | np.ndarray | List[float] # starting parameters
    end      : float | np.ndarray | List[float] # ending parameters

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.start, float):
            self.start = (self.start,)
        if isinstance(self.end, float):
            self.end = (self.end,)

        # check the dimensionality of start, end, and coord vars match
        if (len(self.start) != len(self.coord_vars)) or \
            (len(self.end) != len(self.coord_vars)):
            raise ValueError(
                "start and end dimensions to not match the number of coord vars."
            )
        
        self.start = np.array(self.start)
        self.end = np.array(self.end)
        self.npoints = self.numpoints
        self.dim = 1
        self.dimensions = self.npoints * np.array([1], dtype = int)

    def _iterator(self):
        for i in range(self.numpoints):
            yield np.array([i]), \
                self.start + (self.end - self.start) * i / (self.numpoints - 1)

@dataclass(kw_only=True)
class SweepMeasureProduct(ParamSweepMeasure):
    """
    Sweep parameters over the product space of provided axes, taking 
    measurements at each point.

    Attributes
    ----------
    axes: tuple of ndarrays 
        Axes for each parameter
    """
    axes: Tuple[np.ndarray] # axes over which to sweep parameters

    def __post_init__(self):
        super().__post_init__()

        # check that axes match the number of coord vars
        if len(self.axes) != len(self.coord_vars):
            raise ValueError("axes does not match the number of setters.")
        
        self.dimensions = np.array([len(a) for a in self.axes], dtype = int)
        self.dim = len(self.axes)
        self.npoints = self.dimensions.prod()

    def _iterator(self):
        for idx in itertools.product(*[range(d) for d in self.dimensions]):
            yield np.array(idx), \
                np.array([self.axes[d][idx[d]] for d in range(self.dim)])

@dataclass(kw_only=True)
class SweepMeasureParallelepiped(ParamSweepMeasure):
    """
    Sweep parameters over a parallelepiped in parameter space, taking 
    measurements at each point.

    Attributes
    ----------
    origin : list of float
        starting corner of the parallelepiped.
    endpoints : ndarray 
        Other corners of the parallelepiped, the rows each being a respective 
        endpoints. The fastest swept direction is last.
    shape : list of int
        Number of points along each axis.
    """
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
        
        if not len(self.coord_vars) == nsetters:
            raise ValueError("sweep does not match columns in endpoints.")
        
        self.origin = np.array(self.origin)
        self.npoints = np.prod(self.shape)
        self.dim = ncorners
        self.dimensions = np.array(self.shape, dtype = int)

    def _iterator(self):
        A = (self.endpoints - self.origin.reshape(1, -1)).T
        for idx in itertools.product(*(range(d) for d in self.shape)):
            idx = np.array(idx)
            yield idx, (self.origin + \
                            (A @ (idx / (self.dimensions - 1)))).flatten()
