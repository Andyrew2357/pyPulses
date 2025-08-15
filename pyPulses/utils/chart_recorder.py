from .param_sweep_measure import SweepMeasureArbitraryIterator
from ..thread_job import ThreadJob

import logging
import datetime
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

@dataclass
class ChartRecorder():
    measurements    : Dict[str, Any] | List[Dict[str, Any]]                     # measured variables
    time_per_point  : float = 0.1
    file_prefix     : str = None                                                # string prefix for output file
    points_per_file : int = None                                                # number of points per output file
    starting_fnum   : int = 1                                                   # starting file number
    logger          : logging.Logger = None                                     # logger
    pre_callback    : Callable[                                                 # callback before measurement
                        [datetime.datetime, np.ndarray, np.ndarray], Any
                        ] | \
                    Callable[[np.ndarray, np.ndarray], Any] = None
    post_callback   : Callable[                                                 # callback after measurement
                        [datetime.datetime, np.ndarray, np.ndarray, np.ndarray], 
                        Any
                        ] | \
                    Callable[[np.ndarray, np.ndarray, np.ndarray], Any] = None
    timestamp       : bool = True                                               # whether to include a timestamp for each point

    # Recorder arguments
    line_kwargs     : dict = None
    twinx           : List[Tuple[str, str]]  = None
    twin_axes       : List[Tuple[str, ...]] = None
    master_var      : str = None
    max_cols        : int = 5
    width           : int = None
    height          : int = None
    draw_interval   : float = 0.2               
        
    def __post_init__(self):
        def indefinite_iterator():
            i = 0
            while True:
                yield np.array([i]), np.array([])

        self._dummy_sweep = SweepMeasureArbitraryIterator(
            measurements = self.measurements,
            coordinates = [],
            time_per_point = self.time_per_point,
            file_prefix = self.file_prefix,
            points_per_file = self.points_per_file,
            retain_return = False,
            logger = self.logger,
            pre_callback = self.pre_callback,
            post_callback = self.post_callback,
            timestamp = self.timestamp,
            plot_fields = 'all',
            plot_kwargs = {
                'line_kwargs': self.line_kwargs,
                'twinx'      : self.twinx,
                'twin_axes'  : self.twin_axes,
                'master_var' : self.master_var,
                'max_cols'   : self.max_cols,
                'width'      : self.width,
                'height'     : self.height,
                'draw_interval': self.draw_interval
            },
            iterate = indefinite_iterator
        )

    def run(self):
        self.job = ThreadJob(self._dummy_sweep.run)
        def cleanup(*_):
            del self.job
            self.job = None
        self.job.on_finish = cleanup
        self.job.on_stop = cleanup
        self.job.on_error = cleanup
        self.job.start_with_controls()
        return self.job
