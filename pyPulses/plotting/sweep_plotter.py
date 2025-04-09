"""
A class for dynamically plotting data from parameter sweeps.
Designed to work with the sweepMeasureCut and related functions as a callback to
visualize data as it's being collected.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import display, clear_output
from typing import List, Optional, Tuple
from collections import deque

class SweepPlotter:    
    def __init__(self, 
                 swept_names    : List[str], 
                 measured_names : List[str],
                 plot_layout    : str = 'grid',
                 update_interval: int = 1,
                 figsize        : Tuple[int, int] = (12, 8),
                 display_mode   : str = 'jupyter',
                 max_history    : Optional[int] = None,
                 total_points   : Optional[int] = None):
        """
        Initialize the SweepPlotter.
        
        Parameters:
        -----------
        swept_names     : List[str]
            Names of the swept parameters
        measured_names  : List[str]
            Names of the measured parameters
        plot_layout     : str, optional
            Layout style for plots ('grid' or 'matrix'). 
            'grid' shows each measurement vs each swept parameter.
            'matrix' shows all pairs of measurements and swept parameters.
        update_interval : int, optional
            Update the plot every N points
        figsize         : tuple, optional
            Figure size (width, height) in inches
        display_mode    : str, optional
            'jupyter' for Jupyter notebook display or 'window' for normal matplotlib window
        total_points    : int, optional
            Total number of points in the sweep. If provided, a progress bar will be shown.
            If None, no progress bar will be displayed.
        """
        self.swept_mask         = [n is not None for n in swept_names]
        self.measured_mask      = [n is not None for n in measured_names]
        self.swept_names        = [n for n in swept_names if n is not None]
        self.measured_names     = [n for n in measured_names if n is not None]
        self.plot_layout        = plot_layout
        self.update_interval    = update_interval
        self.figsize            = figsize
        self.display_mode       = display_mode
        self.n_points_total     = total_points
        self.show_progress      = total_points is not None
        
        # Data storage
        self.swept_data = deque(maxlen = max_history)
        self.measured_data = deque(maxlen = max_history)
        
        # Create figure and axes based on layout
        self._setup_figure()
        
    def _setup_figure(self):
        """Create the figure and axes based on the selected layout."""
        self.fig = plt.figure(figsize = self.figsize)
        
        n_swept = len(self.swept_names)
        n_measured = len(self.measured_names)
        
        if self.plot_layout == 'grid':
            # Create a grid with measurements as rows and swept parameters as 
            # columns
            if self.show_progress:
                # Add a row for the progress bar
                self.gs = GridSpec(n_measured + 1, n_swept, 
                                   height_ratios = [0.2] + [3] * n_measured)
                
                # Create a special axes for the progress bar
                self.progress_ax = self.fig.add_subplot(self.gs[0, :])
                self.progress_ax.set_title("Progress: 0%")
                self.progress_ax.set_xlim(0, 100)
                self.progress_ax.set_ylim(0, 1)
                self.progress_ax.get_yaxis().set_visible(False)
                self.progress_bar = self.progress_ax.barh(0.5, 0, height = 1, 
                                                          color = 'green')[0]
                
                # Adjust the starting row for the data plots
                start_row = 1
            else:
                # No progress bar, just the data plots
                self.gs = GridSpec(n_measured, n_swept)
                start_row = 0
            
            # Create the grid of plot axes
            self.axes = {}
            for i, measured in enumerate(self.measured_names):
                for j, swept in enumerate(self.swept_names):
                    ax = self.fig.add_subplot(self.gs[i+start_row, j])
                    ax.set_xlabel(swept)
                    ax.set_ylabel(measured)
                    ax.grid(True)
                    self.axes[(swept, measured)] = ax
            
        elif self.plot_layout == 'matrix':
            # Create a matrix layout where each parameter (swept and measured) 
            # is plotted against every other
            all_params = self.swept_names + self.measured_names
            n_params = len(all_params)
            
            if self.show_progress:
                # Create a matrix with an extra row for the progress bar
                self.gs = GridSpec(n_params + 1, n_params)
                
                # Create a special axes for the progress bar spanning the top
                self.progress_ax = self.fig.add_subplot(self.gs[0, :])
                self.progress_ax.set_title("Progress")
                self.progress_ax.set_xlim(0, 100)
                self.progress_ax.set_ylim(0, 1)
                self.progress_ax.set_xlabel("Percent Complete")
                self.progress_ax.get_yaxis().set_visible(False)
                self.progress_bar = self.progress_ax.barh(0.5, 0, height = 1, 
                                                          color = 'green')[0]
                
                # Adjust the starting row for the data plots
                start_row = 1
            else:
                # No progress bar
                self.gs = GridSpec(n_params, n_params)
                self.fig.suptitle("Parameter Sweep", fontsize=16)
                start_row = 0
            
            # Create the matrix of plot axes
            self.axes = {}
            for i, param1 in enumerate(all_params):
                for j, param2 in enumerate(all_params):
                    if i != j:  # Skip diagonals
                        ax = self.fig.add_subplot(self.gs[i+start_row, j])
                        ax.set_xlabel(param2)
                        ax.set_ylabel(param1)
                        ax.grid(True)
                        self.axes[(param2, param1)] = ax
        
        self.fig.tight_layout(rect=[0, 0, 1, 1])
        
    def update_callback(self, index: int, swept_values: np.ndarray, 
                                        measured_values: np.ndarray):
        """
        Callback function to update plots during a sweep.
        
        This function is designed to be used as the post_callback in 
        sweepMeasureCut.
        
        Parameters:
        -----------
        index           : int
            Current point index
        swept_values    : np.ndarray
            Values of the swept parameters at this point
        measured_values : np.ndarray
            Values of the measured parameters at this point
        """
        # Store the data
        self.swept_data.append(swept_values[self.swept_mask])
        self.measured_data.append(measured_values[self.measured_mask])
        
        # If total points wasn't specified and we're using a progress bar,
        # try to infer it from the indices
        if self.n_points_total is None:
            # Can infer total points if we're using an index that starts at 0
            # and increments by 1, but we'd need additional heuristics to be sure
            pass
        
        # Update plots at specified intervals or when we get the last point
        if index % self.update_interval == 0 or \
            (self.n_points_total is not None and \
            index == self.n_points_total - 1):
            self._update_plots(index)
    
    def _update_plots(self, current_index: int):
        """Update all plots with current data."""
        # Convert stored data to numpy arrays for easier slicing
        swept_array = np.array(self.swept_data)
        measured_array = np.array(self.measured_data)
        
        # Update progress bar and title if we're showing progress
        if self.show_progress:
            progress_percent = (current_index + 1) / self.n_points_total * 100
            self.progress_bar.set_width(progress_percent)
            self.progress_ax.set_title(
                f"Parameter Sweep Progress: {progress_percent:.1f}%"
                )
        
        # Update data plots based on layout
        if self.plot_layout == 'grid':
            for i, measured_name in enumerate(self.measured_names):
                for j, swept_name in enumerate(self.swept_names):
                    ax = self.axes[(swept_name, measured_name)]
                    ax.clear()
                    ax.plot(swept_array[:, j], measured_array[:, i], 'o-')
                    ax.set_xlabel(swept_name)
                    ax.set_ylabel(measured_name)
                    ax.grid(True)
        
        elif self.plot_layout == 'matrix':
            all_names = self.swept_names + self.measured_names
            all_data = np.hstack((swept_array, measured_array))
            
            for i, name1 in enumerate(all_names):
                for j, name2 in enumerate(all_names):
                    if i != j and (name2, name1) in self.axes:
                        ax = self.axes[(name2, name1)]
                        ax.clear()
                        ax.plot(all_data[:, j], all_data[:, i], 'o-')
                        ax.set_xlabel(name2)
                        ax.set_ylabel(name1)
                        ax.grid(True)
        
        # Display the updated figure
        if self.display_mode == 'jupyter':
            display(self.fig)
            clear_output(wait=True)
        else:
            self.fig.canvas.draw()
            plt.pause(0.01)
    
    def reset(self):
        """Reset the plotter to start a new sweep."""
        self.swept_data = []
        self.measured_data = []
        plt.close(self.fig)
        self._setup_figure()
    
    def save_figure(self, filename: str):
        """Save the current figure to a file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the collected data as numpy arrays."""
        return np.array(self.swept_data), np.array(self.measured_data)
