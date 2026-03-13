from __future__ import annotations

from .channel_adapter import ScalarChannel

import time
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Protocol, Tuple


"""
Calibration Models
"""

class CalibrationModel(ABC):
    """
    Abstract base class for calibration models. Maps control value to output value.
    """

    @abstractmethod
    def forward(self, c: float) -> float:
        """
        Compute output value from control value.
        
        Parameters
        ----------
        c : float
            control value

        Returns
        -------
        o : float
        """
        ...

    @abstractmethod
    def inverse(self, o: float, c_min: float, c_max: float) -> float:
        """
        Compute control value from desired output value.

        Parameters
        ----------
        o : float
            output value
        c_min : float
            minimum allowed control value
        c_max : float
            maximum allowed control value

        Returns
        -------
        float
        """
        ...

    @abstractmethod
    def output_bounds() -> Tuple[float, float]:
        """
        Compute minimum and maximum output values achievable within the bounds.
        """
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a configuration dictionary"""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'CalibrationModel':
        """Initialize from a configuration dictionary"""
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CalibrationModel':
        model = config.get('model')
        if model == 'trivial':
            return TrivialCalibration.from_dict(config)
        if model == 'polynomial':
            return PolynomialCalibration.from_dict(config)
        raise ValueError(f"Unknown calibration model: {model}")

class CalibratedChannelProtocol(Protocol):
    """Protocol for channels that accept calibration."""
    
    def set_calibration(self, cal: 'CalibrationModel') -> None: ...
    def add_crosstalk(self, other: 'ScalarChannel', coeff: float) -> None: ...

class TrivialCalibration(CalibrationModel):
    """
    Models output as: O(c) = c
    """

    def forward(self, c: float) -> float:
        return c
    
    def inverse(self, o: float, c_min: float, c_max: float):
        if o > c_max or o < c_min:
            raise ValueError(f"Output {o} not in [{c_min}, {c_max}]")
        return o
    
    def output_bounds(self, c_min: float, c_max: float) -> Tuple[float, float]:
        return c_min, c_max
    
    def to_dict(self):
        return {'model': 'trivial',}
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrivialCalibration':
        return cls()

class PolynomialCalibration(CalibrationModel):
    """
    Polynomial calibration: O(c) = sum_i(a[i] * c^i)
    
    Coefficients stored in ascending order [a0, a1, a2, ...].
    
    Optionally supports bounded extrapolation: outside [c_fit_min, c_fit_max],
    uses linear extrapolation instead of the polynomial (prevents blowup).
    """

    def __init__(self, 
        coeffs: List[float] | np.ndarray,
        c_fit_min: float | None = None,
        c_fit_max: float | None = None,
    ):
        self._coeffs = np.array(coeffs, dtype=np.float64)
        self._coeffs_r = self._coeffs[::-1]  # Reversed for np.polyval
        self._degree = len(self._coeffs) - 1
        
        # Compute derivative coefficients for inverse
        if self._degree >= 1:
            self._dcoeffs_r = np.polyder(self._coeffs_r)
        else:
            self._dcoeffs_r = np.array([0.0])
        
        # Bounds for safe extrapolation
        self._c_fit_min = c_fit_min
        self._c_fit_max = c_fit_max
        
        # Precompute boundary values and slopes for extrapolation
        if c_fit_min is not None and c_fit_max is not None:
            self._o_fit_min = float(np.polyval(self._coeffs_r, c_fit_min))
            self._o_fit_max = float(np.polyval(self._coeffs_r, c_fit_max))
            self._slope_min = float(np.polyval(self._dcoeffs_r, c_fit_min))
            self._slope_max = float(np.polyval(self._dcoeffs_r, c_fit_max))
        else:
            self._o_fit_min = None
            self._o_fit_max = None
            self._slope_min = None
            self._slope_max = None

    @property
    def degree(self) -> int:
        return self._degree
    
    @property
    def coefficients(self) -> np.ndarray:
        return self._coeffs.copy()

    def forward(self, c: float) -> float:
        """Evaluate calibration, with linear extrapolation outside fit range."""
        if self._c_fit_min is not None and c < self._c_fit_min:
            # Linear extrapolation below range
            return self._o_fit_min + self._slope_min * (c - self._c_fit_min)
        if self._c_fit_max is not None and c > self._c_fit_max:
            # Linear extrapolation above range
            return self._o_fit_max + self._slope_max * (c - self._c_fit_max)
        return float(np.polyval(self._coeffs_r, c))

    def forward_vectorized(self, cs: np.ndarray) -> np.ndarray:
        """Vectorized forward evaluation."""
        result = np.polyval(self._coeffs_r, cs)
        
        if self._c_fit_min is not None:
            mask_lo = cs < self._c_fit_min
            result[mask_lo] = self._o_fit_min + self._slope_min * (cs[mask_lo] - self._c_fit_min)
        
        if self._c_fit_max is not None:
            mask_hi = cs > self._c_fit_max
            result[mask_hi] = self._o_fit_max + self._slope_max * (cs[mask_hi] - self._c_fit_max)
        
        return result

    def inverse(self, o: float, c_min: float, c_max: float) -> float:
        """Compute control value for desired output using Newton-Raphson."""
        if self._degree == 0:
            if np.isclose(self._coeffs[0], o):
                return (c_min + c_max) / 2
            raise ValueError(f"Constant calibration {self._coeffs[0]} cannot produce {o}")
        
        if self._degree == 1:
            a0, a1 = self._coeffs
            if np.isclose(a1, 0):
                raise ValueError("Degenerate linear calibration with zero slope")
            c = (o - a0) / a1
            return np.clip(c, c_min, c_max)
        
        return self._inverse_newton(o, c_min, c_max)

    def _inverse_newton(self, o: float, c_min: float, c_max: float,
                        max_iter: int = 50, tol: float = 1e-12) -> float:
        """Newton-Raphson with bisection fallback."""
        f_min, f_max = self.forward(c_min), self.forward(c_max)
        
        # Initial guess via linear interpolation
        if np.isclose(f_min, f_max):
            return (c_min + c_max) / 2
        
        t = (o - f_min) / (f_max - f_min)
        c = c_min + t * (c_max - c_min)
        c = np.clip(c, c_min, c_max)

        for _ in range(max_iter):
            f_c = self.forward(c) - o
            if abs(f_c) < tol:
                return float(c)
            
            df_c = float(np.polyval(self._dcoeffs_r, c))
            if np.isclose(df_c, 0):
                # Bisection fallback
                if f_c > 0:
                    c_max = c
                else:
                    c_min = c
                c = (c_min + c_max) / 2
                continue

            c_new = c - f_c / df_c
            c_new = np.clip(c_new, c_min, c_max)
            
            if abs(c_new - c) < tol:
                return float(c_new)
            c = c_new

        return float(c)

    def output_bounds(self, c_min: float, c_max: float) -> Tuple[float, float]:
        """Compute output range, checking critical points for non-monotonic polynomials."""
        vals = [self.forward(c_min), self.forward(c_max)]
        
        if self._degree >= 2:
            # Find critical points (where derivative = 0)
            critical = np.roots(self._dcoeffs_r)
            for cp in critical:
                if np.isreal(cp):
                    cp_real = float(np.real(cp))
                    if c_min < cp_real < c_max:
                        vals.append(self.forward(cp_real))
        
        return min(vals), max(vals)

    def to_dict(self) -> Dict[str, Any]:
        # Use 'model' key for consistency with TrivialCalibration
        return {
            'model': 'polynomial',
            'coefficients': self._coeffs.tolist(),
            'c_fit_min': self._c_fit_min,
            'c_fit_max': self._c_fit_max,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolynomialCalibration':
        return cls(
            coeffs=data['coefficients'],
            c_fit_min=data.get('c_fit_min'),
            c_fit_max=data.get('c_fit_max'),
        )
    
"""
Calibration Measurements
"""

@dataclass
class CalibrationPoint:
    """A single calibration measurement."""
    control: float
    measured: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CalibrationDataset:
    """Collection of calibration measurements."""
    
    points: List[CalibrationPoint] = field(default_factory=list)
    channel_id: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def add(self, control: float, measured: float, **metadata):
        self.points.append(CalibrationPoint(
            control=control,
            measured=measured,
            metadata=metadata,
        ))
    
    def controls(self) -> np.ndarray:
        return np.array([p.control for p in self.points])
    
    def measured(self) -> np.ndarray:
        return np.array([p.measured for p in self.points])
    
    def fit_polynomial(self, degree: int = 2) -> 'PolynomialCalibration':
        """Fit polynomial with bounded extrapolation."""
        c = self.controls()
        m = self.measured()
        coeffs = np.polyfit(c, m, degree)[::-1]  # Ascending order
        return PolynomialCalibration(
            coeffs=coeffs.tolist(),
            c_fit_min=float(c.min()),
            c_fit_max=float(c.max()),
        )
    
    def residuals(self, cal: 'PolynomialCalibration') -> np.ndarray:
        predicted = cal.forward_vectorized(self.controls())
        return self.measured() - predicted
    
    def rms_error(self, cal: 'PolynomialCalibration') -> float:
        return float(np.sqrt(np.mean(self.residuals(cal)**2)))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'channel_id': self.channel_id,
            'conditions': self.conditions,
            'points': [
                {'control': p.control, 'measured': p.measured,
                 'timestamp': p.timestamp, 'metadata': p.metadata}
                for p in self.points
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationDataset':
        ds = cls(
            channel_id=data.get('channel_id', ''),
            conditions=data.get('conditions', {}),
        )
        for p in data.get('points', []):
            ds.points.append(CalibrationPoint(
                control=p['control'],
                measured=p['measured'],
                timestamp=p.get('timestamp', 0),
                metadata=p.get('metadata', {}),
            ))
        return ds

class CalibrationMeasurement(ABC):
    """Abstract base for calibration measurement procedures."""
    
    @abstractmethod
    def measure(self) -> CalibrationDataset:
        """Execute measurement and return raw data."""
        ...
    
    def run(self, degree: int = 2) -> Tuple[CalibrationDataset, PolynomialCalibration]:
        """Measure and fit."""
        data = self.measure()
        fit = data.fit_polynomial(degree)
        return data, fit
    
class ChannelCalibration(CalibrationMeasurement):
    """
    Calibrate a single channel by sweeping control and measuring output.
    """
    
    def __init__(self,
        channel: ScalarChannel,
        meter: ScalarChannel,
        control_points: np.ndarray,
        settle_time: float = 0.1,
        averages: int = 1,
        channel_id: str = "",
        pre_measure: Callable[[], None] | None = None,
        post_measure: Callable[[], None] | None = None,
    ):
        self.channel = channel
        self.meter = meter
        self.control_points = control_points
        self.settle_time = settle_time
        self.averages = averages
        self.channel_id = channel_id
        self.pre_measure = pre_measure
        self.post_measure = post_measure
    
    def measure(self) -> CalibrationDataset:
        dataset = CalibrationDataset(channel_id=self.channel_id)
        
        if self.pre_measure:
            self.pre_measure()
        
        try:
            for c in self.control_points:
                self.channel.set_output(c)
                time.sleep(self.settle_time)
                
                readings = [self.meter.get_output() for _ in range(self.averages)]
                measured = float(np.mean(readings))
                
                dataset.add(control=c, measured=measured)
        finally:
            if self.post_measure:
                self.post_measure()
        
        return dataset

class CrosstalkMeasurement:
    """
    Measure linear crosstalk coefficient: how much does source affect victim?
    
    Sweeps source channel while victim is at zero, measures victim output.
    """
    
    def __init__(self,
        source_channel: ScalarChannel,
        victim_channel: ScalarChannel,
        victim_meter: ScalarChannel,
        control_points: np.ndarray,
        settle_time: float = 0.1,
        averages: int = 1,
        source_id: str = "",
        victim_id: str = "",
    ):
        self.source_channel = source_channel
        self.victim_channel = victim_channel
        self.victim_meter = victim_meter
        self.control_points = control_points
        self.settle_time = settle_time
        self.averages = averages
        self.source_id = source_id
        self.victim_id = victim_id
    
    def measure(self) -> Tuple[CalibrationDataset, float]:
        """
        Measure crosstalk and fit linear coefficient.
        
        Returns
        -------
        dataset : CalibrationDataset
            Raw measurement data.
        coefficient : float
            Linear crosstalk coefficient (victim_output / source_control).
        """
        dataset = CalibrationDataset(
            channel_id=f"{self.source_id}_to_{self.victim_id}",
            conditions={'measurement_type': 'crosstalk'},
        )
        
        # Zero victim channel
        self.victim_channel.set_output(0.0)
        time.sleep(self.settle_time)
        
        for c in self.control_points:
            self.source_channel.set_output(c)
            time.sleep(self.settle_time)
            
            readings = [self.victim_meter.get_output() for _ in range(self.averages)]
            measured = float(np.mean(readings))
            
            dataset.add(control=c, measured=measured)
        
        # Fit linear coefficient
        controls = dataset.controls()
        measured = dataset.measured()
        
        if len(controls) > 1:
            # Linear fit: measured = coeff * control + offset
            coeffs = np.polyfit(controls, measured, 1)
            coefficient = float(coeffs[0])
        else:
            coefficient = 0.0
        
        return dataset, coefficient
