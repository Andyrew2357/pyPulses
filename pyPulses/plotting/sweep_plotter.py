"""
This is a class for dynamically plotting data from parameter sweeps using Bokeh.
Uses asynchronous updates to minimize impact on measurement time.
"""

import numpy as np
from collections import deque
from typing import List, Tuple
import threading
import time
import queue

from IPython.display import display

# Bokeh imports
from bokeh.plotting import figure, output_notebook
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, Div, HoverTool
from bokeh.palettes import Category10

from jupyter_bokeh.widgets import BokehModel

class SweepPlotter:
    def __init__(self, 
                 swept_names    : List[str], 
                 measured_names : List[str],
                 plot_layout    : str = 'grid',
                 update_interval: float = 0.1, # Time-based throttling (seconds)
                 plot_width     : int = 350, 
                 plot_height    : int = 300,
                 max_history    : int = None,
                 total_points   : int = None):
        """
        Initialize the SweepPlotter.
        
        Parameters:
        -----------
        swept_names     : List[str]
            Names of the swept parameters
        measured_names  : List[str]
            Names of the measured parameters
        plot_layout     : str, optional
            Layout style for plots ('grid' or 'matrix')
        update_interval : float, optional
            Minimum time between plot updates (seconds)
        plot_width      : int, optional
            Width of each plot in pixels
        plot_height     : int, optional
            Height of each plot in pixels
        max_history     : int, optional
            Maximum number of data points to keep (for memory management)
        total_points    : int, optional
            Total number of points in the sweep (for progress indicator)
        """
        # Clean up parameter names and create masks
        self.swept_names = [n for n in swept_names if n is not None]
        self.measured_names = [n for n in measured_names if n is not None]
        self.swept_mask = [n is not None for n in swept_names]
        self.measured_mask = [n is not None for n in measured_names]
        
        # Configuration
        self.plot_layout = plot_layout
        self.update_interval = update_interval
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.total_points = total_points
        self.show_progress = total_points is not None
        
        # Data storage
        self.swept_data = deque(maxlen=max_history)
        self.measured_data = deque(maxlen=max_history)
        
        # Async update mechanism
        self.update_queue = queue.Queue()
        self.last_update_time = 0
        self.handle = None
        self.plots_initialized = False
        
        # Initialize Bokeh for Jupyter
        try:
            output_notebook(hide_banner=True)
        except:
            pass

        # Create data sources and plots
        self._create_data_sources()
        self._setup_plots()
        
        # Start the update thread
        self.stop_thread = False
        self.update_thread = threading.Thread(target=self._update_thread_function)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _create_data_sources(self):
        """Create Bokeh data sources for plots."""
        self.sources = {}
        self.last_points = {}
        
        if self.plot_layout == 'grid':
            # Grid layout: each measurement vs each swept parameter
            for swept in self.swept_names:
                for measured in self.measured_names:
                    self.sources[(swept, measured)] = ColumnDataSource({
                        'x': [], 
                        'y': []
                    })
                    self.last_points[(swept, measured)] = ColumnDataSource({
                        'x': [], 
                        'y': []
                    })
        elif self.plot_layout == 'matrix':
            # Matrix layout: all parameters vs all other parameters
            all_params = self.swept_names + self.measured_names
            for param1 in all_params:
                for param2 in all_params:
                    if param1 != param2:
                        self.sources[(param1, param2)] = ColumnDataSource({
                            'x': [], 
                            'y': []
                        })
                        self.last_points[(param1, param2)] = ColumnDataSource({
                            'x': [], 
                            'y': []
                        })
        
        # Progress indicator source
        if self.show_progress:
            self.progress_source = ColumnDataSource({
                'value': [0],
                'percent': ['0%']
            })
    
    def _setup_plots(self):
        """Create the Bokeh plot layout."""
        self.plots = {}
        plot_grid = []
        colors = Category10[10]
        
        # Setup tools and hover tooltips
        tools = "pan,wheel_zoom,box_zoom,reset,save"
        tooltips = [("x", "@x{0.0000}"), ("y", "@y{0.0000}")]
        
        # Create progress indicator if needed
        if self.show_progress:
            self.progress_div = Div(
                text=f"<h3>Parameter Sweep Progress: 0%</h3>",
                width=self.plot_width * len(self.swept_names) if self.swept_names else 400,
            )
            
            self.progress_plot = figure(
                height=50,
                width=self.plot_width * len(self.swept_names) if self.swept_names else 400,
                tools="",
                x_range=(0, 100),
                y_range=(0, 1),
                toolbar_location=None
            )
            self.progress_plot.xaxis.visible = True
            self.progress_plot.yaxis.visible = False
            self.progress_plot.grid.visible = False
            self.progress_plot.outline_line_color = None
            self.progress_bar = self.progress_plot.hbar(
                y=0.5, left=0, right='value', height=0.8,
                source=self.progress_source,
                color="#009E73"
            )
        
        # Create plots based on layout type
        if self.plot_layout == 'grid':
            rows = []
            for measured in self.measured_names:
                row_plots = []
                for j, swept in enumerate(self.swept_names):
                    p = figure(
                        width=self.plot_width,
                        height=self.plot_height,
                        tools=tools,
                        x_axis_label=swept,
                        y_axis_label=measured,
                        title=f"{measured} vs {swept}"
                    )
                    
                    # Add hover tool
                    hover = HoverTool(tooltips=tooltips)
                    p.add_tools(hover)
                    
                    # Add line and points
                    line = p.line('x', 'y', source=self.sources[(swept, measured)],
                                 line_width=2, color=colors[j % len(colors)])
                    scatter = p.scatter('x', 'y', source=self.sources[(swept, measured)],
                                      size=6, color=colors[j % len(colors)], alpha=0.5)
                    
                    # Highlight the latest point
                    latest = p.scatter('x', 'y', source=self.last_points[(swept, measured)],
                                      size=10, color=colors[j % len(colors)], alpha=1.0)
                    
                    p.grid.grid_line_alpha = 0.3
                    self.plots[(swept, measured)] = p
                    row_plots.append(p)
                
                rows.append(row_plots)
            
            plot_grid = rows
            
        elif self.plot_layout == 'matrix':
            # Matrix layout for correlation plots
            all_params = self.swept_names + self.measured_names
            n_params = len(all_params)
            matrix = [[None for _ in range(n_params)] for _ in range(n_params)]
            
            for i, param1 in enumerate(all_params):
                for j, param2 in enumerate(all_params):
                    if i != j:  # Skip diagonals
                        p = figure(
                            width=self.plot_width,
                            height=self.plot_height,
                            tools=tools,
                            x_axis_label=param2,
                            y_axis_label=param1,
                            title=f"{param1} vs {param2}"
                        )
                        
                        # Add hover tool
                        hover = HoverTool(tooltips=tooltips)
                        p.add_tools(hover)
                        
                        # Add line and points
                        line = p.line('x', 'y', source=self.sources[(param2, param1)],
                                     line_width=2, color=colors[i % len(colors)])
                        scatter = p.scatter('x', 'y', source=self.sources[(param2, param1)],
                                          size=6, color=colors[i % len(colors)], alpha=0.5)
                        
                        # Highlight the latest point
                        latest = p.scatter('x', 'y', source=self.last_points[(param2, param1)],
                                          size=10, color=colors[i % len(colors)], alpha=1.0)
                        
                        p.grid.grid_line_alpha = 0.3
                        self.plots[(param2, param1)] = p
                        matrix[i][j] = p
            
            # Filter out None values and create compact grid
            plot_grid = []
            for row in matrix:
                filtered_row = [p for p in row if p is not None]
                if filtered_row:
                    plot_grid.append(filtered_row)
        
        # Create the layout
        if plot_grid:
            self.grid = gridplot(plot_grid, toolbar_location="right")
            
            if self.show_progress:
                self.layout = column(self.progress_div, self.progress_plot, self.grid)
            else:
                self.layout = self.grid
                
            # Display the layout
            self.handle = display(BokehModel(self.layout))
            self.plots_initialized = True
    
    def _update_thread_function(self):
        """Background thread function for asynchronous plot updates."""
        while not self.stop_thread:
            # Check if we need to update
            current_time = time.time()
            time_since_last_update = current_time - self.last_update_time
            
            # Update if enough time has passed and there's new data
            if time_since_last_update >= self.update_interval and not self.update_queue.empty():
                self._process_update_queue()
                self.last_update_time = current_time
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.1)

    def _process_update_queue(self):
        """Process all pending updates in the queue."""

        # Process a limited number of updates at once to avoid blocking
        max_updates = 10
        updates_processed = 0
        
        while not self.update_queue.empty() and updates_processed < max_updates:
            try:
                # Just get the item - we don't need the value
                self.update_queue.get(block=False)
                updates_processed += 1
            except queue.Empty:
                break
        
        # Skip if no data
        if len(self.swept_data) == 0:
            return
        
        # Get the current data
        swept_array = np.array(self.swept_data)
        measured_array = np.array(self.measured_data)
        
        # Update plot data sources
        if self.plot_layout == 'grid':
            for i, measured_name in enumerate(self.measured_names):
                for j, swept_name in enumerate(self.swept_names):
                    if (swept_name, measured_name) in self.sources:
                        # Update full dataset
                        x_data = swept_array[:, j].tolist()
                        y_data = measured_array[:, i].tolist()
                        
                        self.sources[(swept_name, measured_name)].data = {
                            'x': x_data,
                            'y': y_data
                        }
                        
                        # Update latest point source
                        if len(x_data) > 0:
                            self.last_points[(swept_name, measured_name)].data = {
                                'x': [x_data[-1]],
                                'y': [y_data[-1]]
                            }
        
        elif self.plot_layout == 'matrix':
            all_names = self.swept_names + self.measured_names
            all_data = np.hstack((swept_array, measured_array))
            
            for i, name1 in enumerate(all_names):
                for j, name2 in enumerate(all_names):
                    if i != j and (name2, name1) in self.sources:
                        # Update full dataset
                        x_data = all_data[:, j].tolist()
                        y_data = all_data[:, i].tolist()
                        
                        self.sources[(name2, name1)].data = {
                            'x': x_data,
                            'y': y_data
                        }
                        
                        # Update latest point source
                        if len(x_data) > 0:
                            self.last_points[(name2, name1)].data = {
                                'x': [x_data[-1]],
                                'y': [y_data[-1]]
                            }
        
        # Update progress if enabled
        if self.show_progress and self.total_points is not None:
            current_index = len(self.swept_data) - 1
            progress_percent = min(100, (current_index + 1) / self.total_points * 100)
            
            self.progress_source.data = {
                'value': [progress_percent],
                'percent': [f"{progress_percent:.1f}%"]
            }
            
            self.progress_div.text = f"<h3>Parameter Sweep Progress: {progress_percent:.1f}%</h3>"
    
    def update_callback(self, index: int, swept_values: np.ndarray, measured_values: np.ndarray):
        """
        Callback function to update plots during a sweep.
        
        This function is designed to be used as the post_callback in 
        sweepMeasureCut and similar methods.
        """

        # Apply masks if needed
        swept_filtered = swept_values[self.swept_mask]
        measured_filtered = measured_values[self.measured_mask]
        
        # Store the data
        self.swept_data.append(swept_filtered)
        self.measured_data.append(measured_filtered)
        
        # Queue an update request (thread-safe)
        self.update_queue.put(index)
        
        # If this is the first point, ensure we update immediately
        if index == 0:
            self.last_update_time = 0
    
    def reset(self):
        """Reset the plotter to start a new sweep."""
        
        # Clear data
        self.swept_data.clear()
        self.measured_data.clear()
        
        # Clear the update queue
        while not self.update_queue.empty():
            try:
                self.update_queue.get(block=False)
            except queue.Empty:
                break
        
        # Reset progress
        if self.show_progress:
            self.progress_source.data = {'value': [0], 'percent': ['0%']}
            self.progress_div.text = "<h3>Parameter Sweep Progress: 0%</h3>"
        
        # Reset all data sources
        for source in self.sources.values():
            source.data = {'x': [], 'y': []}
        
        for source in self.last_points.values():
            source.data = {'x': [], 'y': []}
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the collected data as numpy arrays."""
        return np.array(self.swept_data), np.array(self.measured_data)
    
    def stop(self):
        """Stop the update thread."""
        self.stop_thread = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
    
    def force_update(self):
        """Force an immediate update of all plots."""
        self._process_update_queue()
