"""
Calibration measurement for pulse shaper + DTG + DC box.
(SEE NOTES FOR THE NOTATION AND EXPLANATION OF HOW THIS PROCEDURE WORKS)

Measures F_{p1,p2}(c1, c2) for all polarity combinations, then derives
calibrations for PolarityCalibratedChannel.

Key relationships:
    Δ1_{sig1,sig2} = -Δ1_{-sig1,sig2}  (depends only on sig2)
    Δ2_{sig1,sig2} = -Δ2_{sig1,-sig2}  (depends only on sig1)

So we only need two calibrations per channel, indexed by the OTHER pair's polarity.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import time
import json

from .channel_adapter import ScalarChannel, CompPair
from .calibration import PolynomialCalibration


"""
Calibration Result
"""

@dataclass
class PulseShaperCalibrationResult:
    """
    Processed calibration result.
    
    Contains calibrations for each channel indexed by the OTHER pair's polarity.
    Ready to be loaded into PolarityCalibratedChannel.
    
    For channel 1 (charging): indexed by sig2 (discharging pair's X polarity)
    For channel 2 (discharging): indexed by sig1 (charging pair's X polarity)
    """
    # X calibrations keyed by other pair's polarity
    x1_cal_pos: PolynomialCalibration = None  # When discharging pair X polarity is True
    x1_cal_neg: PolynomialCalibration = None  # When discharging pair X polarity is False
    x2_cal_pos: PolynomialCalibration = None  # When charging pair X polarity is True
    x2_cal_neg: PolynomialCalibration = None  # When charging pair X polarity is False
    
    # Y calibrations (optional, only if y_meter provided)
    y1_cal_pos: PolynomialCalibration = None
    y1_cal_neg: PolynomialCalibration = None
    y2_cal_pos: PolynomialCalibration = None
    y2_cal_neg: PolynomialCalibration = None
    
    # Fit metadata
    fit_degree: int = 8
    rms_errors: Dict[str, float] = field(default_factory=dict)
    cross_term_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    cal_attenuation: float = 1.0
    timestamp: float = field(default_factory=time.time)
    raw_data_path: str | None = None
    
    def summary(self) -> str:
        lines = [
            "Pulse Shaper Calibration Result",
            f"  Timestamp: {time.ctime(self.timestamp)}",
            f"  Attenuation: {self.cal_attenuation}",
            f"  Polynomial degree: {self.fit_degree}",
            "",
        ]
        
        def cal_summary(name: str, cal: PolynomialCalibration | None) -> str:
            if cal is None:
                return f"  {name}: not calibrated"
            linear_coeff = cal.coefficients[1] if len(cal.coefficients) > 1 else 0
            return f"  {name}: linear={linear_coeff:.6g}, RMS={self.rms_errors.get(name, 0):.2e}"
        
        lines.append(cal_summary("X1(sig2=+)", self.x1_cal_pos))
        lines.append(cal_summary("X1(sig2=-)", self.x1_cal_neg))
        lines.append(cal_summary("X2(sig1=+)", self.x2_cal_pos))
        lines.append(cal_summary("X2(sig1=-)", self.x2_cal_neg))
        
        if self.y1_cal_pos is not None:
            lines.append("")
            lines.append(cal_summary("Y1(sig2=+)", self.y1_cal_pos))
            lines.append(cal_summary("Y1(sig2=-)", self.y1_cal_neg))
            lines.append(cal_summary("Y2(sig1=+)", self.y2_cal_pos))
            lines.append(cal_summary("Y2(sig1=-)", self.y2_cal_neg))
        
        if self.cross_term_diagnostics:
            lines.append("")
            lines.append("  Cross-term diagnostics:")
            for k, v in self.cross_term_diagnostics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4g}")
        
        if self.raw_data_path:
            lines.append("")
            lines.append(f"  Raw data: {self.raw_data_path}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        def cal_to_dict(cal):
            return cal.to_dict() if cal is not None else None
        
        return {
            'x1_cal_pos': cal_to_dict(self.x1_cal_pos),
            'x1_cal_neg': cal_to_dict(self.x1_cal_neg),
            'x2_cal_pos': cal_to_dict(self.x2_cal_pos),
            'x2_cal_neg': cal_to_dict(self.x2_cal_neg),
            'y1_cal_pos': cal_to_dict(self.y1_cal_pos),
            'y1_cal_neg': cal_to_dict(self.y1_cal_neg),
            'y2_cal_pos': cal_to_dict(self.y2_cal_pos),
            'y2_cal_neg': cal_to_dict(self.y2_cal_neg),
            'fit_degree': self.fit_degree,
            'rms_errors': self.rms_errors,
            'cross_term_diagnostics': self.cross_term_diagnostics,
            'cal_attenuation': self.cal_attenuation,
            'timestamp': self.timestamp,
            'raw_data_path': self.raw_data_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PulseShaperCalibrationResult':
        def cal_from_dict(d):
            return PolynomialCalibration.from_dict(d) if d is not None else None
        
        return cls(
            x1_cal_pos=cal_from_dict(data.get('x1_cal_pos')),
            x1_cal_neg=cal_from_dict(data.get('x1_cal_neg')),
            x2_cal_pos=cal_from_dict(data.get('x2_cal_pos')),
            x2_cal_neg=cal_from_dict(data.get('x2_cal_neg')),
            y1_cal_pos=cal_from_dict(data.get('y1_cal_pos')),
            y1_cal_neg=cal_from_dict(data.get('y1_cal_neg')),
            y2_cal_pos=cal_from_dict(data.get('y2_cal_pos')),
            y2_cal_neg=cal_from_dict(data.get('y2_cal_neg')),
            fit_degree=data.get('fit_degree', 8),
            rms_errors=data.get('rms_errors', {}),
            cross_term_diagnostics=data.get('cross_term_diagnostics', {}),
            cal_attenuation=data.get('cal_attenuation', 1.0),
            timestamp=data.get('timestamp', time.time()),
            raw_data_path=data.get('raw_data_path'),
        )
    
    def save(self, path: str | Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved calibration result to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'PulseShaperCalibrationResult':
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        result = cls.from_dict(data)
        print(f"Loaded calibration result from {path}")
        return result
    
    def load_raw_data(self) -> Dict[str, np.ndarray]:
        """Load raw data from the npz file."""
        if self.raw_data_path is None:
            raise ValueError("No raw data path stored")
        return dict(np.load(self.raw_data_path))


"""
Calibration Measurement
"""

class PulseShaperCalibration:
    """
    Calibration measurement procedure for the pulse shaper box.
    
    Measures F_{p1,p2}(c1, c2) for all polarity combinations, then computes
    calibrations for PolarityCalibratedChannel.
    
    Physical setup:
    - relay1 controls charging pair (X1/Y1 polarities are complementary)
    - relay2 controls discharging pair (X2/Y2 polarities are complementary)
    - x_meter measures X output (required)
    - y_meter measures Y output (optional)
    
    Key insight: calibration for channel 1 only depends on sig2 (other pair's polarity),
    and calibration for channel 2 only depends on sig1.
    """
    
    POLARITY_KEYS = ['pp', 'pm', 'mp', 'mm']
    POLARITY_VALUES = {
        'pp': (True, True),
        'pm': (True, False),
        'mp': (False, True),
        'mm': (False, False),
    }
    
    def __init__(self,
        relay1: CompPair,
        relay2: CompPair,
        ch1_x: ScalarChannel,
        ch2_x: ScalarChannel,
        x_meter: ScalarChannel,
        ch1_y: ScalarChannel | None = None,
        ch2_y: ScalarChannel | None = None,
        y_meter: ScalarChannel | None = None,
        axis_controls: np.ndarray | None = None,
        cal_attenuation: float = 1.0,
        settle_time: float = 0.5,
        averages: int = 3,
        sparse_downsample: int = 4,
        output_dir: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        relay1, relay2 : CompPair
            Relays controlling charging and discharging pair polarities.
        ch1_x, ch2_x : ScalarChannel
            DC controls for charging and discharging pairs (X output).
        x_meter : ScalarChannel
            DMM measuring X output.
        ch1_y, ch2_y : ScalarChannel, optional
            DC controls for Y output. Required if y_meter is provided.
        y_meter : ScalarChannel, optional
            DMM measuring Y output. If None, only X is calibrated.
        axis_controls : np.ndarray, optional
            Dense control points for axis sweeps. Default: linspace(0, 10, 101).
        cal_attenuation : float
            Output attenuation to compensate in results.
        settle_time : float
            Time to wait after changing settings.
        averages : int
            Number of DMM readings to average.
        sparse_downsample : int
            Downsampling factor for joint sweep grid.
        output_dir : str | Path, optional
            Directory to save raw data incrementally. If None, uses current directory.
        """
        self.relay1 = relay1
        self.relay2 = relay2
        self.ch1_x = ch1_x
        self.ch2_x = ch2_x
        self.x_meter = x_meter
        
        self.ch1_y = ch1_y
        self.ch2_y = ch2_y
        self.y_meter = y_meter
        self._has_y = (ch1_y is not None and ch2_y is not None and y_meter is not None)
        
        if axis_controls is None:
            axis_controls = np.linspace(0, 10, 101)
        self.axis_controls = np.asarray(axis_controls)
        self.sparse_controls = self.axis_controls[::sparse_downsample]
        self.cal_attenuation = cal_attenuation
        self.settle_time = settle_time
        self.averages = averages
        
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path('.')
        
        # Parallel meter reading if we have both meters
        self._executor = ThreadPoolExecutor(max_workers=2) if self._has_y else None
        
        # Raw data storage
        self._raw_data: Dict[str, np.ndarray] = {}
        self._raw_data_path: Path | None = None
        
        print(
            f"PulseShaperCalibration initialized: "
            f"{len(self.axis_controls)} axis points, "
            f"{len(self.sparse_controls)} sparse points, "
            f"attenuation={cal_attenuation}, "
            f"has_y={self._has_y}"
        )
    
    def _set_polarities(self, p1: bool, p2: bool):
        """Set relay polarities (X gets p, Y gets not p for each relay)."""
        self.relay1.xpolarity(p1)
        self.relay1.ypolarity(not p1)
        self.relay2.xpolarity(p2)
        self.relay2.ypolarity(not p2)
    
    def _read_meters(self) -> Tuple[float, float | None]:
        """Read meters, optionally in parallel if both exist."""
        def read_avg(meter):
            readings = [meter.get_output() for _ in range(self.averages)]
            return float(np.mean(readings))
        
        if self._has_y and self._executor:
            x_future = self._executor.submit(read_avg, self.x_meter)
            y_future = self._executor.submit(read_avg, self.y_meter)
            return x_future.result(), y_future.result()
        else:
            return read_avg(self.x_meter), None
    
    def _zero_all(self):
        """Set all DC controls to zero."""
        self.ch1_x.set_output(0.0)
        self.ch2_x.set_output(0.0)
        if self._has_y:
            self.ch1_y.set_output(0.0)
            self.ch2_y.set_output(0.0)
    
    def _save_raw_data(self):
        """Save raw data incrementally to npz file."""
        if self._raw_data_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._raw_data_path = self.output_dir / f"pulse_shaper_raw_{timestamp}.npz"
        
        np.savez(self._raw_data_path, **self._raw_data)
    
    def _sweep_axis1(self, p1: bool, p2: bool) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Sweep axis 1: vary c1 with c2 = 0.
        
        Returns (x_readings, y_readings or None).
        """
        self._set_polarities(p1, p2)
        self.ch2_x.set_output(0.0)
        if self._has_y:
            self.ch2_y.set_output(0.0)
        time.sleep(self.settle_time)
        
        n = len(self.axis_controls)
        x_readings = np.zeros(n)
        y_readings = np.zeros(n) if self._has_y else None
        
        for i, c in enumerate(self.axis_controls):
            self.ch1_x.set_output(c)
            if self._has_y:
                self.ch1_y.set_output(c)
            time.sleep(self.settle_time)
            x_val, y_val = self._read_meters()
            x_readings[i] = x_val
            if y_readings is not None:
                y_readings[i] = y_val
            
            try:
                pct = 100.0 * (i + 1) / n
                if self._has_y:
                    msg = (
                        f"[{i+1:4d}/{n:4d}] "
                        f"c={c:10.4f} "
                        f"x={x_val:12.6f} "
                        f"y={y_val:12.6f} "
                        f"{pct:5.1f}%"
                    )
                else:
                    msg = (
                        f"[{i+1:4d}/{n:4d}] "
                        f"c={c:10.4f} "
                        f"x={x_val:12.6f} "
                        f"{pct:5.1f}%"
                    )
                print('\r' + msg, end='', flush=True)
            except Exception:
                pass
        
        print()
        return x_readings, y_readings
    
    def _sweep_axis2(self, p1: bool, p2: bool) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Sweep axis 2: vary c2 with c1 = 0.
        
        Returns (x_readings, y_readings or None).
        """
        self._set_polarities(p1, p2)
        self.ch1_x.set_output(0.0)
        if self._has_y:
            self.ch1_y.set_output(0.0)
        time.sleep(self.settle_time)
        
        n = len(self.axis_controls)
        x_readings = np.zeros(n)
        y_readings = np.zeros(n) if self._has_y else None
        
        for i, c in enumerate(self.axis_controls):
            self.ch2_x.set_output(c)
            if self._has_y:
                self.ch2_y.set_output(c)
            time.sleep(self.settle_time)
            x_val, y_val = self._read_meters()
            x_readings[i] = x_val
            if y_readings is not None:
                y_readings[i] = y_val
            
            try:
                pct = 100.0 * (i + 1) / n
                if self._has_y:
                    msg = (
                        f"[{i+1:4d}/{n:4d}] "
                        f"c={c:10.4f} "
                        f"x={x_val:12.6f} "
                        f"y={y_val:12.6f} "
                        f"{pct:5.1f}%"
                    )
                else:
                    msg = (
                        f"[{i+1:4d}/{n:4d}] "
                        f"c={c:10.4f} "
                        f"x={x_val:12.6f} "
                        f"{pct:5.1f}%"
                    )
                print('\r' + msg, end='', flush=True)
            except Exception:
                pass
        
        print()
        return x_readings, y_readings
    
    def _sweep_joint(self, p1: bool, p2: bool) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Sweep joint grid on sparse controls.
        
        Returns (x_grid, y_grid or None) each of shape (n_sparse, n_sparse).
        """
        self._set_polarities(p1, p2)
        time.sleep(self.settle_time)
        
        n = len(self.sparse_controls)
        x_grid = np.zeros((n, n))
        y_grid = np.zeros((n, n)) if self._has_y else None
        total = n * n
        
        for i, c1 in enumerate(self.sparse_controls):
            self.ch1_x.set_output(c1)
            if self._has_y:
                self.ch1_y.set_output(c1)
            for j, c2 in enumerate(self.sparse_controls):
                self.ch2_x.set_output(c2)
                if self._has_y:
                    self.ch2_y.set_output(c2)
                time.sleep(self.settle_time)
                x_val, y_val = self._read_meters()
                x_grid[i, j] = x_val
                if y_grid is not None:
                    y_grid[i, j] = y_val
                
                try:
                    idx = i * n + j + 1
                    pct = 100.0 * idx / total
                    if self._has_y:
                        msg = (
                            f"[{idx:4d}/{total:4d}] "
                            f"c1={c1:8.3f} c2={c2:8.3f} "
                            f"x={x_val:12.6f} "
                            f"y={y_val:12.6f} "
                            f"{pct:5.1f}%"
                        )
                    else:
                        msg = (
                            f"[{idx:4d}/{total:4d}] "
                            f"c1={c1:8.3f} c2={c2:8.3f} "
                            f"x={x_val:12.6f} "
                            f"{pct:5.1f}%"
                        )
                    print('\r' + msg, end='', flush=True)
                except Exception:
                    pass
        
        print()
        return x_grid, y_grid
    
    def measure(self) -> Dict[str, np.ndarray]:
        """
        Execute full measurement.
        
        Assumes DTG run is already disabled.
        Raw data is saved incrementally to npz file after each sweep.
        
        Returns raw data dict (also accessible via load_raw_data on result).
        """
        print("=" * 60)
        print("Starting pulse shaper calibration measurement")
        print("=" * 60)
        start = time.time()
        
        self._raw_data = {
            'axis_controls': self.axis_controls.copy(),
            'sparse_controls': self.sparse_controls.copy(),
        }
        self._save_raw_data()
        
        # Axis 1 sweeps (4 polarity combinations)
        print("\nMeasuring axis 1 (charging pair)...")
        for key in self.POLARITY_KEYS:
            p1, p2 = self.POLARITY_VALUES[key]
            print(f"  Sweep F_{key}(c1, 0)  [p1={p1}, p2={p2}]")
            x_rd, y_rd = self._sweep_axis1(p1, p2)
            self._raw_data[f'axis1_F_{key}_x'] = x_rd
            if y_rd is not None:
                self._raw_data[f'axis1_F_{key}_y'] = y_rd
            self._save_raw_data()
        
        # Axis 2 sweeps (4 polarity combinations)
        print("\nMeasuring axis 2 (discharging pair)...")
        for key in self.POLARITY_KEYS:
            p1, p2 = self.POLARITY_VALUES[key]
            print(f"  Sweep F_{key}(0, c2)  [p1={p1}, p2={p2}]")
            x_rd, y_rd = self._sweep_axis2(p1, p2)
            self._raw_data[f'axis2_F_{key}_x'] = x_rd
            if y_rd is not None:
                self._raw_data[f'axis2_F_{key}_y'] = y_rd
            self._save_raw_data()
        
        # Joint sweeps (4 polarity combinations)
        n_sparse = len(self.sparse_controls)
        print(f"\nMeasuring joint grid ({n_sparse}x{n_sparse})...")
        for key in self.POLARITY_KEYS:
            p1, p2 = self.POLARITY_VALUES[key]
            print(f"  Sweep F_{key}(c1, c2)  [p1={p1}, p2={p2}]")
            x_grid, y_grid = self._sweep_joint(p1, p2)
            self._raw_data[f'joint_F_{key}_x'] = x_grid
            if y_grid is not None:
                self._raw_data[f'joint_F_{key}_y'] = y_grid
            self._save_raw_data()
        
        self._zero_all()
        elapsed = time.time() - start
        print(f"\nMeasurement complete in {elapsed:.1f}s")
        print(f"Raw data saved to: {self._raw_data_path}")
        
        return self._raw_data
    
    def process(self, 
        raw: Dict[str, np.ndarray] | None = None, 
        degree: int = 8,
    ) -> PulseShaperCalibrationResult:
        """
        Process raw data into calibrations.
        
        Key insight: calibration for channel 1 only depends on sig2,
        and calibration for channel 2 only depends on sig1.
        
        We compute:
            Δ1(c1; sig2=+) = F_{p,m}(c1,0) - F_{m,m}(c1,0)
            Δ1(c1; sig2=-) = F_{p,p}(c1,0) - F_{m,p}(c1,0)
            Δ2(c2; sig1=+) = F_{m,p}(0,c2) - F_{m,m}(0,c2)
            Δ2(c2; sig1=-) = F_{p,p}(0,c2) - F_{p,m}(0,c2)
        
        Sign correction is applied at the end to ensure positive linear coefficient.
        """
        if raw is None:
            raw = self._raw_data
        
        print("\nProcessing calibration data...")
        atten = self.cal_attenuation
        controls = raw['axis_controls']
        
        result = PulseShaperCalibrationResult(
            fit_degree=degree,
            cal_attenuation=atten,
            raw_data_path=str(self._raw_data_path) if self._raw_data_path else None,
        )
        
        def fit_polynomial(ctrl: np.ndarray, delta: np.ndarray) -> Tuple[PolynomialCalibration, float]:
            """Fit polynomial and return (calibration, rms_error)."""
            coeffs = np.polyfit(ctrl, delta, degree)[::-1]
            cal = PolynomialCalibration(
                coeffs=list(coeffs),
                c_fit_min=float(ctrl.min()),
                c_fit_max=float(ctrl.max()),
            )
            predicted = cal.forward_vectorized(ctrl)
            rms = float(np.sqrt(np.mean((delta - predicted)**2)))
            return cal, rms
        
        def ensure_positive_linear(cal: PolynomialCalibration) -> PolynomialCalibration:
            """Flip coefficients if linear term is negative."""
            coeffs = cal.coefficients
            if len(coeffs) > 1 and coeffs[1] < 0:
                return PolynomialCalibration(
                    coeffs=[-c for c in coeffs],
                    c_fit_min=cal._c_fit_min,
                    c_fit_max=cal._c_fit_max,
                )
            return cal
        
        # X1: depends on sig2 (discharging pair's X polarity)
        # When sig2=True (p2=True for X): Δ = F_{pm} - F_{mm}
        # When sig2=False (p2=False for X): Δ = F_{pp} - F_{mp}
        
        delta_x1_pos = (raw['axis1_F_pm_x'] - raw['axis1_F_mm_x']) / atten
        cal, rms = fit_polynomial(controls, delta_x1_pos)
        result.x1_cal_pos = ensure_positive_linear(cal)
        result.rms_errors['X1(sig2=+)'] = rms
        
        delta_x1_neg = (raw['axis1_F_pp_x'] - raw['axis1_F_mp_x']) / atten
        cal, rms = fit_polynomial(controls, delta_x1_neg)
        result.x1_cal_neg = ensure_positive_linear(cal)
        result.rms_errors['X1(sig2=-)'] = rms
        
        # X2: depends on sig1 (charging pair's X polarity)
        # When sig1=True (p1=True for X): Δ = F_{mp} - F_{mm}
        # When sig1=False (p1=False for X): Δ = F_{pp} - F_{pm}
        
        delta_x2_pos = (raw['axis2_F_mp_x'] - raw['axis2_F_mm_x']) / atten
        cal, rms = fit_polynomial(controls, delta_x2_pos)
        result.x2_cal_pos = ensure_positive_linear(cal)
        result.rms_errors['X2(sig1=+)'] = rms
        
        delta_x2_neg = (raw['axis2_F_pp_x'] - raw['axis2_F_pm_x']) / atten
        cal, rms = fit_polynomial(controls, delta_x2_neg)
        result.x2_cal_neg = ensure_positive_linear(cal)
        result.rms_errors['X2(sig1=-)'] = rms
        
        # Y calibrations (if available)
        # Y uses same indexing by X polarity for consistency with PolarityCalibratedChannel
        # Y1: Δ = F_{mp} - F_{pp} (when sig2=+) or F_{mm} - F_{pm} (when sig2=-)
        # Y2: Δ = F_{pm} - F_{pp} (when sig1=+) or F_{mm} - F_{mp} (when sig1=-)
        
        if self._has_y:
            delta_y1_pos = (raw['axis1_F_mp_y'] - raw['axis1_F_pp_y']) / atten
            cal, rms = fit_polynomial(controls, delta_y1_pos)
            result.y1_cal_pos = ensure_positive_linear(cal)
            result.rms_errors['Y1(sig2=+)'] = rms
            
            delta_y1_neg = (raw['axis1_F_mm_y'] - raw['axis1_F_pm_y']) / atten
            cal, rms = fit_polynomial(controls, delta_y1_neg)
            result.y1_cal_neg = ensure_positive_linear(cal)
            result.rms_errors['Y1(sig2=-)'] = rms
            
            delta_y2_pos = (raw['axis2_F_pm_y'] - raw['axis2_F_pp_y']) / atten
            cal, rms = fit_polynomial(controls, delta_y2_pos)
            result.y2_cal_pos = ensure_positive_linear(cal)
            result.rms_errors['Y2(sig1=+)'] = rms
            
            delta_y2_neg = (raw['axis2_F_mm_y'] - raw['axis2_F_mp_y']) / atten
            cal, rms = fit_polynomial(controls, delta_y2_neg)
            result.y2_cal_neg = ensure_positive_linear(cal)
            result.rms_errors['Y2(sig1=-)'] = rms
        
        # Cross-term analysis
        result.cross_term_diagnostics = self._analyze_cross_terms(raw, result)
        
        print("Processing complete.")
        return result
    
    def _analyze_cross_terms(self,
        raw: Dict[str, np.ndarray],
        result: PulseShaperCalibrationResult,
    ) -> Dict[str, Any]:
        """
        Validate that Δ is approximately independent of the other control.
        """
        atten = self.cal_attenuation
        sparse = raw['sparse_controls']
        
        diagnostics = {}
        
        def analyze_channel(name: str,
                            joint_a_key: str,
                            joint_b_key: str,
                            cal: PolynomialCalibration,
                            axis: int,
                            output: str = 'x'):
            """
            Analyze cross-term for one channel.
            
            axis=0 means delta should be independent of c2
            axis=1 means delta should be independent of c1
            """
            key_a = f'joint_F_{joint_a_key}_{output}'
            key_b = f'joint_F_{joint_b_key}_{output}'
            
            if key_a not in raw or key_b not in raw:
                return
            
            joint_delta = (raw[key_a] - raw[key_b]) / atten
            predicted = cal.forward_vectorized(sparse)
            
            # Match sign to calibration (which has been corrected)
            diag = np.diag(joint_delta)
            if np.sum(diag * predicted < 0) > len(diag) // 2:
                joint_delta = -joint_delta
            
            if axis == 0:
                predicted_grid = predicted[:, np.newaxis]
            else:
                predicted_grid = predicted[np.newaxis, :]
            
            residual = joint_delta - predicted_grid
            
            rms_signal = np.sqrt(np.mean(joint_delta**2))
            rms_residual = np.sqrt(np.mean(residual**2))
            
            diagnostics[f'{name}_rms'] = float(rms_residual)
            if rms_signal > 0:
                diagnostics[f'{name}_ratio'] = float(rms_residual / rms_signal)
        
        # X1 (sig2=+): Δ = F_{pm} - F_{mm}
        analyze_channel('X1_cross', 'pm', 'mm', result.x1_cal_pos, axis=0, output='x')
        
        # X2 (sig1=+): Δ = F_{mp} - F_{mm}
        analyze_channel('X2_cross', 'mp', 'mm', result.x2_cal_pos, axis=1, output='x')
        
        if self._has_y:
            # Y1 (sig2=+): Δ = F_{mp} - F_{pp}
            analyze_channel('Y1_cross', 'mp', 'pp', result.y1_cal_pos, axis=0, output='y')
            
            # Y2 (sig1=+): Δ = F_{pm} - F_{pp}
            analyze_channel('Y2_cross', 'pm', 'pp', result.y2_cal_pos, axis=1, output='y')
        
        if diagnostics:
            cross_ratios = [v for k, v in diagnostics.items() if 'ratio' in k]
            if cross_ratios:
                print(f"  Cross-term ratios: {[f'{r*100:.1f}%' for r in cross_ratios]}")
        
        return diagnostics
    
    def plot_calibration(self,
        raw: Dict[str, np.ndarray] | None = None,
        result: PulseShaperCalibrationResult | None = None,
        save_path: str | Path | None = None,
    ):
        """
        Plot calibration results.
        
        Creates a figure showing calibration curves and cross-term analysis.
        """
        import matplotlib.pyplot as plt
        
        if raw is None:
            raw = self._raw_data
        
        atten = self.cal_attenuation
        controls = raw['axis_controls']
        sparse = raw['sparse_controls']
        
        # Determine layout based on whether Y is available
        if self._has_y:
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            channels = [
                ('X1(sig2=+)', 'axis1_F_pm_x', 'axis1_F_mm_x', result.x1_cal_pos, 'joint_F_pm_x', 'joint_F_mm_x', 0),
                ('X2(sig1=+)', 'axis2_F_mp_x', 'axis2_F_mm_x', result.x2_cal_pos, 'joint_F_mp_x', 'joint_F_mm_x', 1),
                ('Y1(sig2=+)', 'axis1_F_mp_y', 'axis1_F_pp_y', result.y1_cal_pos, 'joint_F_mp_y', 'joint_F_pp_y', 0),
                ('Y2(sig1=+)', 'axis2_F_pm_y', 'axis2_F_pp_y', result.y2_cal_pos, 'joint_F_pm_y', 'joint_F_pp_y', 1),
            ]
        else:
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            channels = [
                ('X1(sig2=+)', 'axis1_F_pm_x', 'axis1_F_mm_x', result.x1_cal_pos, 'joint_F_pm_x', 'joint_F_mm_x', 0),
                ('X2(sig1=+)', 'axis2_F_mp_x', 'axis2_F_mm_x', result.x2_cal_pos, 'joint_F_mp_x', 'joint_F_mm_x', 1),
            ]
        
        fig.suptitle('Pulse Shaper Calibration', fontsize=14)
        
        for col, (name, key_a, key_b, cal, joint_a, joint_b, axis) in enumerate(channels):
            # Compute delta
            delta = (raw[key_a] - raw[key_b]) / atten
            pred = cal.forward_vectorized(controls)
            
            # Match sign
            if np.sum(delta * pred < 0) > len(delta) // 2:
                delta = -delta
            
            # Row 0: Calibration curve
            ax = axes[0, col]
            ax.plot(controls, delta, 'o', markersize=2, alpha=0.5, label='Measured')
            ax.plot(controls, pred, '-', linewidth=2, label='Fit')
            ax.set_xlabel('Control (V)')
            ax.set_ylabel('Δ Output (V)')
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Row 1: Residuals
            ax = axes[1, col]
            residual = delta - pred
            ax.plot(controls, residual, 'o', markersize=2)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Control (V)')
            ax.set_ylabel('Residual (V)')
            rms_val = np.sqrt(np.mean(residual**2))
            ax.set_title(f'Residuals (RMS={rms_val:.2e})')
            ax.grid(True, alpha=0.3)
            
            # Row 2: Cross-term heatmap
            ax = axes[2, col]
            
            joint_delta = (raw[joint_a] - raw[joint_b]) / atten
            pred_sparse = cal.forward_vectorized(sparse)
            
            # Match sign
            diag = np.diag(joint_delta)
            if np.sum(diag * pred_sparse < 0) > len(diag) // 2:
                joint_delta = -joint_delta
            
            if axis == 0:
                expected = pred_sparse[:, np.newaxis] * np.ones((1, len(sparse)))
            else:
                expected = np.ones((len(sparse), 1)) * pred_sparse[np.newaxis, :]
            
            cross_term = joint_delta - expected
            
            vmax = np.max(np.abs(cross_term))
            if vmax == 0:
                vmax = 1.0
            im = ax.imshow(cross_term, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, aspect='auto',
                          extent=[sparse[0], sparse[-1], sparse[0], sparse[-1]])
            ax.set_xlabel('c₂ (V)')
            ax.set_ylabel('c₁ (V)')
            ax.set_title('Cross-term')
            plt.colorbar(im, ax=ax, label='Deviation (V)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved calibration plot to {save_path}")
        
        return fig, axes
    
    def run(self, degree: int = 8) -> PulseShaperCalibrationResult:
        """
        Run complete calibration.
        
        Assumes DTG run is already disabled.
        """
        raw = self.measure()
        result = self.process(raw, degree=degree)
        
        print("=" * 60)
        print("Calibration complete")
        print("=" * 60)
        print("\n" + result.summary())
        
        return result
    
    def close(self):
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False