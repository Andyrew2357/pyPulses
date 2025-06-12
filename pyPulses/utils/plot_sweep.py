"""
Lazy functions for calling a parameter sweep with SweepPlotter as a callback.
"""

import numpy as np
from typing import Any, List, Tuple
from .param_sweep_measure import ParamSweepMeasure

def create_plotter_config(swept_names: List[str], measured_names: List[str],
                                        total_points: int, **kwargs) -> dict:
    """
    Create a standard configuration dictionary for the SweepPlotter.
    """

    config = {
        'swept_names'       : swept_names,
        'measured_names'    : measured_names,
        'total_points'      : total_points,
        'plot_layout'       : kwargs.get('plot_layout', 'grid'),
        'update_interval'   : kwargs.get('update_interval', 0.1),
        'plot_width'        : kwargs.get('plot_width', 350),
        'plot_height'       : kwargs.get('plot_height', 300),
        'max_history'       : kwargs.get('max_history', None)
    }
    return config

def plotSweep(sweep: ParamSweepMeasure, plotter_class = None, 
              plotter_kwargs: dict = {}) -> Tuple[np.ndarray, Any]:
    """
    Run a parameter sweep with asynchronous plotting.
    """

    # Import the plotter class here to avoid circular imports
    if plotter_class is None:
        # Import default plotter here - keeping this import local prevents 
        # circular imports and allows for alternative plotter classes to be 
        # passed
        from ..plotting.sweep_plotter import SweepPlotter
        plotter_class = SweepPlotter
    
    # Extract names and determine total points
    if (plotter_kwargs is None) or (not 'swept_names' in plotter_kwargs):
        if isinstance(sweep.swept_name, (list, tuple)):
            plotter_kwargs['swept_names'] = sweep.swept_name
        else:
            plotter_kwargs['swept_names'] = [sweep.swept_name]
    
    if (plotter_kwargs is None) or (not 'measured_names' in plotter_kwargs):
        if isinstance(sweep.measured_name, (list, tuple)):
            plotter_kwargs['measured_names'] = sweep.measured_name
        else:
            plotter_kwargs['measured_names'] = [sweep.measured_name]
    
    # Determine number of points
    total_points = getattr(sweep, 'npoints', None)
    
    # Set up the plotter
    plotter_config = create_plotter_config(
        total_points=total_points,
        **plotter_kwargs
    )
    
    plotter = plotter_class(**plotter_config)
    
    # Save original post_callback if it exists
    original_post_callback = getattr(sweep, 'post_callback', None)
    
    # Create a combined callback that updates the plot and calls the og callback
    def combined_callback(index, swept_values, measured_values):
        # Update the plot
        plotter.update_callback(index, swept_values, measured_values)
        
        # Call the original callback if it exists
        if original_post_callback:
            original_post_callback(index, swept_values, measured_values)
    
    # Set the combined callback
    sweep.post_callback = combined_callback
    
    # Run the sweep
    results = sweep.run()
    
    # Restore the original callback
    sweep.post_callback = original_post_callback

    # if plotter supports it, force update for final points
    if hasattr(plotter, 'force_update'):
        plotter.force_update()

    # if the plotter is asynchronous, we need to kill the thread    
    if hasattr(plotter, 'stop'):
        plotter.stop()

    return results, plotter
