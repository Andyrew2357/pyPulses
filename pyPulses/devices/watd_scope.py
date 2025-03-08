
"""
This class is a wrapper for the MSO44 for use as a time domain waveform averager. 
It supports measurements of the initial slope for balancing pulse heights and 
measurements of the integrated 'settled' response for balancing discharge pulses.
"""

from ._registry import DeviceRegistry
from .abstract_device import abstractDevice
from .mso44 import mso44
from typing import Optional, Tuple
import numpy as np
import time
import json
import os

class watdScope(abstractDevice):
    def __init__(self, loggers = None, config = None):
        if not config:
            fname = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                r'watd_scope.json'
            )
            with open(fname, 'r') as f:
                config = json.load(f)

        if loggers:
            try:
                logger, scope_logger = loggers
            except:
                logger = scope_logger = loggers

        super().__init__(logger)

        self.trace_wait = config["trace_wait"]
        
        self.tsl0, self.tsl1 = config["slope_window"]
        self.target_intercept = config["target_intercept"]

        self.tint0, self.tint1 = config["integral_window"]

        scope_address = config["scope_address"]
        self.scope = DeviceRegistry.get_device(scope_address)
        if self.scope is None:
            self.scope = mso44(scope_logger, instrument_id = scope_address)

        self.scope.get_waveform_parameters()

    def run(self, on: bool):
        """Start or stop acquisitions."""
        self.scope.run(on)
        self.info(f"{'Started' if on else 'Stopped'} acquisitions.")

    def is_running(self) -> bool:
        """
        Query whether the scope is taking acquisitions.
        """
        return self.scope.is_running()

    def set_channel(self, ch: int):
        """
        Set the active channel on which measurements are taken.
        """
        self.scope.set_channel(ch)
        self.info(f"Set active channel to {ch}")

    def get_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return t and v."""
        return self.t, self.v

    def take_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query a waveform off the scope and update t and v accordingly.
        """
        # self.scope.clear_trace()
        time.sleep(self.trace_wait)
        self.t, self.v = self.scope.get_waveform()
        self.info(f"Took trace with {self.t.size} points.")
        return self.t, self.v

    def get_slope_int(self) -> float:
        """
        Calculate and return the implied t intercept of the initial slope
        relative to the target intercept using t and v.
        """
        mask = (self.tsl0 <= self.t) & (self.t <= self.tsl1)
        b, m = np.polyfit(self.t[mask], self.v[mask], 1)
        return -(b / m) - self.target_intercept

    def take_slope_int(self) -> float:
        """
        Update t and v; then return the result of get_slope_int.
        """
        self.take_waveform()
        return self.get_slope_int()
    
    def get_integral(self) -> float:
        """
        Calculate and return the integral of the waveform over a specified time
        using a trapezoidal sum.
        """
        mask = (self.tint0 <= self.t) & (self.t <= self.tint1)
        return np.sum(np.diff(self.t[mask]) * 
                      (self.v[mask][1:] + self.v[mask][:-1])) / 2
    
    def take_integral(self) -> float:
        """
        Update t and v; then return the result of get_integral.
        """
        self.take_waveform()
        return self.get_integral()
    
    def get_mean(self) -> float:
        """
        Calculate and return the mean of the waveform over a specified time.
        """
        return self.get_integral() / (self.tint1 - self.tint0)
    
    def take_mean(self) -> float:
        """
        Update t and v; then return the result of get_mean.
        """
        self.take_waveform()
        return self.get_mean()
    
    def get_abs_integral(self) -> float:
        """
        Calculate and return the integral of the absolute value of the waveform
        over a specified time using a trapezoidal sum. 
        """       
        mask = (self.tint0 <= self.t) & (self.t <= self.tint1)
        return np.sum(np.diff(self.t[mask]) * 
                (np.abs(self.v[mask][1:]) + np.abs(self.v[mask][:-1]))) / 2
    
    def take_abs_integral(self) -> float:
        """
        Update t and v; then return the result of get_abs_integral.
        """
        self.take_waveform()
        return self.get_abs_integral()
    
    def get_abs_mean(self) -> float:
        """
        Calculate and return the mean of the absolute value of the integral over
        a specified time.
        """
        return self.get_abs_integral() / (self.tint1 - self.tint0)
    
    def take_abs_mean(self) -> float:
        """
        Update t and v; then return the result of get_abs_mean.
        """
        self.take_waveform()
        return self.get_abs_mean()