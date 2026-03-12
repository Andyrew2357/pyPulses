from __future__ import annotations

from .channel_adapter import CompPair, ScalarChannel

import time
import json
import numpy as np
from logging import Logger
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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
    
@dataclass
class PulseShaperRawData:
    """
    Raw measurement data from pulse shaper calibration.
    
    Four passes total:
    - Passes 1,2: Sweep X with Y=0, toggle polarity
    - Passes 3,4: Sweep Y with X=0, toggle polarity
    
    Each pass records both meter readings.
    """
    x_controls: np.ndarray = None
    y_controls: np.ndarray = None
    
    # Passes 1,2: X sweeps, Y=0
    # Pass 1: xpolarity=True
    pass1_x_meter: np.ndarray = None  # X_high(X_ctrl)
    pass1_y_meter: np.ndarray = None  # Y_crosstalk_low(X_ctrl)
    # Pass 2: xpolarity=False
    pass2_x_meter: np.ndarray = None  # X_low(X_ctrl)
    pass2_y_meter: np.ndarray = None  # Y_crosstalk_high(X_ctrl)
    
    # Passes 3,4: Y sweeps, X=0
    # Pass 3: xpolarity=True (Y is low here)
    pass3_x_meter: np.ndarray = None  # X_crosstalk_high(Y_ctrl)
    pass3_y_meter: np.ndarray = None  # Y_low(Y_ctrl)
    # Pass 4: xpolarity=False (Y is high here)
    pass4_x_meter: np.ndarray = None  # X_crosstalk_low(Y_ctrl)
    pass4_y_meter: np.ndarray = None  # Y_high(Y_ctrl)
    
    def to_dict(self) -> Dict[str, Any]:
        def _arr(a):
            return a.tolist() if a is not None else None
        return {
            'x_controls': _arr(self.x_controls),
            'y_controls': _arr(self.y_controls),
            'pass1_x_meter': _arr(self.pass1_x_meter),
            'pass1_y_meter': _arr(self.pass1_y_meter),
            'pass2_x_meter': _arr(self.pass2_x_meter),
            'pass2_y_meter': _arr(self.pass2_y_meter),
            'pass3_x_meter': _arr(self.pass3_x_meter),
            'pass3_y_meter': _arr(self.pass3_y_meter),
            'pass4_x_meter': _arr(self.pass4_x_meter),
            'pass4_y_meter': _arr(self.pass4_y_meter),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PulseShaperRawData':
        def _arr(key):
            return np.array(data[key]) if data.get(key) else None
        return cls(
            x_controls=_arr('x_controls'),
            y_controls=_arr('y_controls'),
            pass1_x_meter=_arr('pass1_x_meter'),
            pass1_y_meter=_arr('pass1_y_meter'),
            pass2_x_meter=_arr('pass2_x_meter'),
            pass2_y_meter=_arr('pass2_y_meter'),
            pass3_x_meter=_arr('pass3_x_meter'),
            pass3_y_meter=_arr('pass3_y_meter'),
            pass4_x_meter=_arr('pass4_x_meter'),
            pass4_y_meter=_arr('pass4_y_meter'),
        )

@dataclass
class PulseShaperCalibrationResult:
    """Complete calibration results for pulse shaper."""
    
    x_cal: 'PolynomialCalibration' = None
    y_cal: 'PolynomialCalibration' = None
    x_data: CalibrationDataset = None
    y_data: CalibrationDataset = None
    raw_data: PulseShaperRawData = None
    
    cal_attenuation: float = 1.0
    
    crosstalk_x_to_y: float = 0.0  # Y_effect = coeff * X_output
    crosstalk_y_to_x: float = 0.0  # X_effect = coeff * Y_output
    crosstalk_x_to_y_data: CalibrationDataset = None
    crosstalk_y_to_x_data: CalibrationDataset = None
    
    timestamp: float = field(default_factory=time.time)
    
    def summary(self) -> str:
        """Return human-readable summary of calibration results."""
        lines = [
            f"Pulse Shaper Calibration ({time.ctime(self.timestamp)})",
            f"  Calibration attenuation: {self.cal_attenuation}",
            f"  X calibration: degree {self.x_cal.degree}, coeffs {self.x_cal.coefficients}" if self.x_cal else "  X calibration: None",
        ]
        if self.x_data and self.x_cal:
            lines.append(f"     RMS error: {self.x_data.rms_error(self.x_cal):.6g}")
        lines.append(
            f"  Y calibration: degree {self.y_cal.degree}, coeffs {self.y_cal.coefficients}" if self.y_cal else "  Y calibration: None"
        )
        if self.y_data and self.y_cal:
            lines.append(f"     RMS error: {self.y_data.rms_error(self.y_cal):.6g}")
        lines.extend([
            f"  Crosstalk X->Y: {self.crosstalk_x_to_y:.6g}",
            f"  Crosstalk Y->X: {self.crosstalk_y_to_x:.6g}",
        ])
        return '\n'.join(lines)
    
    def apply_to_channels(self,
        x_calibrated: CalibratedChannelProtocol,
        y_calibrated: CalibratedChannelProtocol,
    ):
        """
        Apply calibration results to CalibratedChannel instances.
        
        Sets calibrations and crosstalk coefficients.
        """
        if self.x_cal:
            x_calibrated.set_calibration(self.x_cal)
        
        if self.y_cal:
            y_calibrated.set_calibration(self.y_cal)
        
        # Crosstalk: X_effect = coeff * Y_output, so X has crosstalk from Y
        if self.crosstalk_y_to_x != 0.0:
            x_calibrated.add_crosstalk(y_calibrated, self.crosstalk_y_to_x)
        
        # Y_effect = coeff * X_output, so Y has crosstalk from X
        if self.crosstalk_x_to_y != 0.0:
            y_calibrated.add_crosstalk(x_calibrated, self.crosstalk_x_to_y)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cal_attenuation': self.cal_attenuation,
            'x_cal': self.x_cal.to_dict() if self.x_cal else None,
            'y_cal': self.y_cal.to_dict() if self.y_cal else None,
            'x_data': self.x_data.to_dict() if self.x_data else None,
            'y_data': self.y_data.to_dict() if self.y_data else None,
            'raw_data': self.raw_data.to_dict() if self.raw_data else None,
            'crosstalk_x_to_y': self.crosstalk_x_to_y,
            'crosstalk_y_to_x': self.crosstalk_y_to_x,
            'crosstalk_x_to_y_data': self.crosstalk_x_to_y_data.to_dict() if self.crosstalk_x_to_y_data else None,
            'crosstalk_y_to_x_data': self.crosstalk_y_to_x_data.to_dict() if self.crosstalk_y_to_x_data else None,
        }
    
    def save(self, path: str | Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PulseShaperCalibrationResult':
        from calibration import PolynomialCalibration
        
        result = cls(
            timestamp=data.get('timestamp', time.time()),
            cal_attenuation=data.get('cal_attenuation', 1.0),
            crosstalk_x_to_y=data.get('crosstalk_x_to_y', 0.0),
            crosstalk_y_to_x=data.get('crosstalk_y_to_x', 0.0),
        )
        
        if data.get('x_cal'):
            result.x_cal = PolynomialCalibration.from_dict(data['x_cal'])
        if data.get('y_cal'):
            result.y_cal = PolynomialCalibration.from_dict(data['y_cal'])
        if data.get('x_data'):
            result.x_data = CalibrationDataset.from_dict(data['x_data'])
        if data.get('y_data'):
            result.y_data = CalibrationDataset.from_dict(data['y_data'])
        if data.get('raw_data'):
            result.raw_data = PulseShaperRawData.from_dict(data['raw_data'])
        if data.get('crosstalk_x_to_y_data'):
            result.crosstalk_x_to_y_data = CalibrationDataset.from_dict(data['crosstalk_x_to_y_data'])
        if data.get('crosstalk_y_to_x_data'):
            result.crosstalk_y_to_x_data = CalibrationDataset.from_dict(data['crosstalk_y_to_x_data'])
        
        return result
    
    @classmethod
    def load(cls, path: str | Path) -> 'PulseShaperCalibrationResult':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class PulseShaperCalibration:
    """
    Calibration procedure for pulse shaper X and Y channels.
    
    Four-pass measurement:
    - Passes 1,2: Sweep X control with Y=0, polarity high then low
    - Passes 3,4: Sweep Y control with X=0, polarity high then low
    
    Each pass measures BOTH meters, giving us calibration and crosstalk
    simultaneously.
    
    Results are compensated for calibration attenuator to give true output.
    """
    
    def __init__(self,
        relay: CompPair,
        x_channel: ScalarChannel,
        y_channel: ScalarChannel,
        x_meter: ScalarChannel,
        y_meter: ScalarChannel,
        control_points: np.ndarray,
        cal_attenuation: float = 1.0,
        settle_time: float = 0.5,
        averages: int = 3,
        x_high_level: float | None = None,
        x_low_level: float | None = None,
        y_high_level: float | None = None,
        y_low_level: float | None = None,
        parallel_meters: bool = True,
        logger: Logger | None = None,
    ):
        self.relay = relay
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.x_meter = x_meter
        self.y_meter = y_meter
        self.control_points = np.asarray(control_points)
        self.cal_attenuation = cal_attenuation
        self.settle_time = settle_time
        self.averages = averages
        
        self.x_high_level = x_high_level
        self.x_low_level = x_low_level
        self.y_high_level = y_high_level
        self.y_low_level = y_low_level

        self.parallel_meters = parallel_meters
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel_meters else None

        self.logger = logger

    def close(self):
        """Shutdown executor if present."""
        if self._executor:
            self._executor.shutdown(wait=False)

    def __del__(self):
        self.close()

    def log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)
    
    def _configure_levels(self):
        """Set DTG high/low levels if specified."""
        if self.x_high_level is not None:
            self.relay.Xhigh(self.x_high_level)
        if self.x_low_level is not None:
            self.relay.Xlow(self.x_low_level)
        if self.y_high_level is not None:
            self.relay.Yhigh(self.y_high_level)
        if self.y_low_level is not None:
            self.relay.Ylow(self.y_low_level)
    
    def _read_meters(self) -> Tuple[float, float]:
        """Read both meters, optionally in parallel."""
        def read_avg(meter):
            readings = [meter.get_output() for _ in range(self.averages)]
            return float(np.mean(readings))
        
        if self._executor:
            x_future = self._executor.submit(read_avg, self.x_meter)
            y_future = self._executor.submit(read_avg, self.y_meter)
            return x_future.result(), y_future.result()
        else:
            return read_avg(self.x_meter), read_avg(self.y_meter)
    
    def _sweep_single_channel(self, 
        sweep_channel: ScalarChannel,
        zero_channel: ScalarChannel,
        x_polarity: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep one channel (other at zero), fixed polarity, read both meters.
        
        Returns (x_meter_readings, y_meter_readings)
        """
        zero_channel.set_output(0.0)
        self.relay.xpolarity(x_polarity)
        self.relay.ypolarity(not x_polarity)
        time.sleep(self.settle_time)
        
        n = len(self.control_points)
        x_readings = np.zeros(n)
        y_readings = np.zeros(n)
        
        for i, c in enumerate(self.control_points):
            sweep_channel.set_output(c)
            time.sleep(self.settle_time)
            x_readings[i], y_readings[i] = self._read_meters()
            self.log(f"Readings at control point {i} ({c:.5e}): X={x_readings[i]:.5e}, Y={y_readings[i]:.5e}")
        
        return x_readings, y_readings
    
    def measure(self) -> PulseShaperRawData:
        """
        Execute four-pass measurement.
        
        Assumes DTG run is already disabled by caller.
        """
        self._configure_levels()
        
        raw = PulseShaperRawData(
            x_controls=self.control_points.copy(),
            y_controls=self.control_points.copy(),
        )
        
        # Passes 1,2: Sweep X, Y=0
        self.log("Starting pass 1 (X sweep, Y=0)")
        raw.pass1_x_meter, raw.pass1_y_meter = self._sweep_single_channel(
            self.x_channel, self.y_channel, x_polarity=True
        )
        self.log("Starting pass 2 (X sweep, Y=0)")
        raw.pass2_x_meter, raw.pass2_y_meter = self._sweep_single_channel(
            self.x_channel, self.y_channel, x_polarity=False
        )
        
        # Passes 3,4: Sweep Y, X=0
        self.log("Starting pass 3 (Y sweep, X=0)")
        raw.pass3_x_meter, raw.pass3_y_meter = self._sweep_single_channel(
            self.y_channel, self.x_channel, x_polarity=True
        )
        self.log("Starting pass 4 (Y sweep, X=0)")
        raw.pass4_x_meter, raw.pass4_y_meter = self._sweep_single_channel(
            self.y_channel, self.x_channel, x_polarity=False
        )
        
        # Reset both channels
        self.x_channel.set_output(0.0)
        self.y_channel.set_output(0.0)
        
        return raw
        
    def process(self, raw: PulseShaperRawData) -> PulseShaperCalibrationResult:
        """
        Process raw data into calibration datasets.
        
        X amplitude from passes 1,2 (X sweep, Y=0):
            X_out = (pass1_x_meter - pass2_x_meter) / attenuation
        
        Y amplitude from passes 3,4 (Y sweep, X=0):
            Y_out = (pass4_y_meter - pass3_y_meter) / attenuation
        
        Crosstalk is measured as a function of OUTPUT, not control:
        
        X->Y crosstalk from passes 1,2:
            Y_effect = (pass2_y_meter - pass1_y_meter) / attenuation
            X_output = (pass1_x_meter - pass2_x_meter) / attenuation
            crosstalk_x_to_y = slope of Y_effect vs X_output
        
        Y->X crosstalk from passes 3,4:
            X_effect = (pass3_x_meter - pass4_x_meter) / attenuation
            Y_output = (pass4_y_meter - pass3_y_meter) / attenuation
            crosstalk_y_to_x = slope of X_effect vs Y_output
        """
        atten = self.cal_attenuation
        
        # X calibration: control -> output
        # pass1: xpol=True -> X high, pass2: xpol=False -> X low
        x_output = (raw.pass1_x_meter - raw.pass2_x_meter) / atten
        
        # Y calibration: control -> output
        # pass3: xpol=True -> Y low, pass4: xpol=False -> Y high
        y_output = (raw.pass4_y_meter - raw.pass3_y_meter) / atten
        
        # Crosstalk X->Y: effect on Y as function of X output
        # pass1: Y is low, pass2: Y is high
        # Y_effect is the crosstalk-induced change in Y
        y_effect_from_x = (raw.pass2_y_meter - raw.pass1_y_meter) / atten
        
        # Crosstalk Y->X: effect on X as function of Y output
        # pass3: X is high, pass4: X is low
        x_effect_from_y = (raw.pass3_x_meter - raw.pass4_x_meter) / atten
        
        # Build calibration datasets (control -> output)
        x_data = CalibrationDataset(
            channel_id='X',
            conditions={'cal_attenuation': atten},
        )
        y_data = CalibrationDataset(
            channel_id='Y',
            conditions={'cal_attenuation': atten},
        )
        
        for i, c in enumerate(raw.x_controls):
            x_data.add(control=c, measured=x_output[i])
        
        for i, c in enumerate(raw.y_controls):
            y_data.add(control=c, measured=y_output[i])
        
        # Build crosstalk datasets (source OUTPUT -> victim effect)
        # These are indexed by source output, not control
        x_to_y_data = CalibrationDataset(
            channel_id='X_to_Y',
            conditions={
                'cal_attenuation': atten, 
                'crosstalk': True,
                'indexed_by': 'source_output',
            },
        )
        y_to_x_data = CalibrationDataset(
            channel_id='Y_to_X',
            conditions={
                'cal_attenuation': atten, 
                'crosstalk': True,
                'indexed_by': 'source_output',
            },
        )
        
        for i in range(len(raw.x_controls)):
            # X output is the independent variable, Y effect is dependent
            x_to_y_data.add(control=x_output[i], measured=y_effect_from_x[i])
        
        for i in range(len(raw.y_controls)):
            # Y output is the independent variable, X effect is dependent
            y_to_x_data.add(control=y_output[i], measured=x_effect_from_y[i])
        
        # Build result
        result = PulseShaperCalibrationResult()
        result.cal_attenuation = atten
        result.raw_data = raw
        result.x_data = x_data
        result.y_data = y_data
        result.crosstalk_x_to_y_data = x_to_y_data
        result.crosstalk_y_to_x_data = y_to_x_data
        
        return result


    def run(self, degree: int = 2) -> PulseShaperCalibrationResult:
        """
        Run complete calibration.
        
        Assumes DTG run is already disabled by caller.
        """
        self.log("Starting calibration...")
        raw = self.measure()
        self.log("Measurement complete, processing data...")
        result = self.process(raw)
        self.log("Data processing complete, fitting calibrations...")
        
        # Fit polynomial calibrations (control -> output)
        result.x_cal = result.x_data.fit_polynomial(degree)
        result.y_cal = result.y_data.fit_polynomial(degree)
        
        # Fit linear crosstalk coefficients (output -> effect)
        # crosstalk_x_to_y: Y_effect = coeff * X_output
        x_out = result.crosstalk_x_to_y_data.controls()  # This is X output, not control
        y_effect = result.crosstalk_x_to_y_data.measured()
        if len(x_out) > 1:
            result.crosstalk_x_to_y = float(np.polyfit(x_out, y_effect, 1)[0])
        
        # crosstalk_y_to_x: X_effect = coeff * Y_output
        y_out = result.crosstalk_y_to_x_data.controls()  # This is Y output, not control
        x_effect = result.crosstalk_y_to_x_data.measured()
        if len(y_out) > 1:
            result.crosstalk_y_to_x = float(np.polyfit(y_out, x_effect, 1)[0])
        
        self.log("Calibration fitting complete.")
        self.log(result.summary())

        return result