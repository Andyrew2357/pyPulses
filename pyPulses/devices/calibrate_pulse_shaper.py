"""
Calibration measurement for pulse shaper + DTG + DC box.

Measures F^{(x)}_{p1,p2}(X1, X2) for all polarity combinations, then derives
Δ calibrations for each (sig1, sig2) operating point.

See my notes for the logic behind this procedure.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List
import numpy as np
import time
import json
import logging
from pathlib import Path

from .channel_adapter import ScalarChannel, CompPair
from .calibration import PolynomialCalibration


logger = logging.getLogger(__name__)


"""
Raw Data Storage
"""

@dataclass
class PulseShaperRawData:
    """
    Raw F_{p1,p2}(c1, c2) measurements for all polarity combinations.
    
    Axis measurements: dense sweeps along (c1, 0) and (0, c2)
    Joint measurements: sparse sweeps on full (c1, c2) grid
    
    Naming convention:
        F_pp = F_{+,+}, F_pm = F_{+,-}, F_mp = F_{-,+}, F_mm = F_{-,-}
        where first index is p1 (relay 1 polarity), second is p2 (relay 2 polarity)
    """
    # Control points
    axis_controls: np.ndarray = None      # Dense 1D array for axis sweeps
    sparse_controls: np.ndarray = None    # Sparse 1D array for joint sweeps
    
    # Axis 1 sweeps: F(c1, 0) - shape (n_axis,)
    axis1_F_pp_x: np.ndarray = None
    axis1_F_pm_x: np.ndarray = None
    axis1_F_mp_x: np.ndarray = None
    axis1_F_mm_x: np.ndarray = None
    axis1_F_pp_y: np.ndarray = None
    axis1_F_pm_y: np.ndarray = None
    axis1_F_mp_y: np.ndarray = None
    axis1_F_mm_y: np.ndarray = None
    
    # Axis 2 sweeps: F(0, c2) - shape (n_axis,)
    axis2_F_pp_x: np.ndarray = None
    axis2_F_pm_x: np.ndarray = None
    axis2_F_mp_x: np.ndarray = None
    axis2_F_mm_x: np.ndarray = None
    axis2_F_pp_y: np.ndarray = None
    axis2_F_pm_y: np.ndarray = None
    axis2_F_mp_y: np.ndarray = None
    axis2_F_mm_y: np.ndarray = None
    
    # Joint sweeps: F(c1, c2) - shape (n_sparse, n_sparse)
    joint_F_pp_x: np.ndarray = None
    joint_F_pm_x: np.ndarray = None
    joint_F_mp_x: np.ndarray = None
    joint_F_mm_x: np.ndarray = None
    joint_F_pp_y: np.ndarray = None
    joint_F_pm_y: np.ndarray = None
    joint_F_mp_y: np.ndarray = None
    joint_F_mm_y: np.ndarray = None
    
    def to_dict(self) -> Dict[str, Any]:
        def _arr(a):
            return a.tolist() if a is not None else None
        return {k: _arr(v) for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PulseShaperRawData':
        def _arr(v):
            return np.array(v) if v is not None else None
        return cls(**{k: _arr(v) for k, v in data.items()})


"""
Calibration Result
"""

@dataclass 
class PulseShaperCalibrationResult:
    """
    Processed calibration result.
    
    Contains Δ calibrations for each channel and each (sig1, sig2) combination.
    Ready to be loaded into PulseAmplitudeManager.
    """
    # Calibrations keyed by (sigma1, sigma2)
    x1_calibrations: Dict[Tuple[bool, bool], PolynomialCalibration] = field(default_factory=dict)
    x2_calibrations: Dict[Tuple[bool, bool], PolynomialCalibration] = field(default_factory=dict)
    y1_calibrations: Dict[Tuple[bool, bool], PolynomialCalibration] = field(default_factory=dict)
    y2_calibrations: Dict[Tuple[bool, bool], PolynomialCalibration] = field(default_factory=dict)
    
    # Raw data for diagnostics
    raw_data: PulseShaperRawData = None
    
    # Fit quality metrics
    fit_degree: int = 2
    rms_errors: Dict[str, float] = field(default_factory=dict)
    cross_term_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    cal_attenuation: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def summary(self) -> str:
        lines = [
            "Pulse Shaper Calibration Result",
            f"  Timestamp: {time.ctime(self.timestamp)}",
            f"  Attenuation: {self.cal_attenuation}",
            f"  Polynomial degree: {self.fit_degree}",
            "",
            "  RMS fit errors:",
        ]
        for name, err in self.rms_errors.items():
            lines.append(f"    {name}: {err:.6g}")
        
        if self.cross_term_diagnostics:
            lines.append("")
            lines.append("  Cross-term diagnostics:")
            for k, v in self.cross_term_diagnostics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4g}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        def serialize_cal_dict(d):
            return {f"{s1},{s2}": cal.to_dict() for (s1, s2), cal in d.items()}
        
        return {
            'x1_calibrations': serialize_cal_dict(self.x1_calibrations),
            'x2_calibrations': serialize_cal_dict(self.x2_calibrations),
            'y1_calibrations': serialize_cal_dict(self.y1_calibrations),
            'y2_calibrations': serialize_cal_dict(self.y2_calibrations),
            'raw_data': self.raw_data.to_dict() if self.raw_data else None,
            'fit_degree': self.fit_degree,
            'rms_errors': self.rms_errors,
            'cross_term_diagnostics': self.cross_term_diagnostics,
            'cal_attenuation': self.cal_attenuation,
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PulseShaperCalibrationResult':
        def deserialize_cal_dict(d):
            result = {}
            for key, cal_data in d.items():
                s1_str, s2_str = key.split(',')
                s1 = s1_str.strip() == 'True'
                s2 = s2_str.strip() == 'True'
                result[(s1, s2)] = PolynomialCalibration.from_dict(cal_data)
            return result
        
        return cls(
            x1_calibrations=deserialize_cal_dict(data.get('x1_calibrations', {})),
            x2_calibrations=deserialize_cal_dict(data.get('x2_calibrations', {})),
            y1_calibrations=deserialize_cal_dict(data.get('y1_calibrations', {})),
            y2_calibrations=deserialize_cal_dict(data.get('y2_calibrations', {})),
            raw_data=PulseShaperRawData.from_dict(data['raw_data']) if data.get('raw_data') else None,
            fit_degree=data.get('fit_degree', 2),
            rms_errors=data.get('rms_errors', {}),
            cross_term_diagnostics=data.get('cross_term_diagnostics', {}),
            cal_attenuation=data.get('cal_attenuation', 1.0),
            timestamp=data.get('timestamp', time.time()),
        )
    
    def save(self, path: str | Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved calibration to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'PulseShaperCalibrationResult':
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        result = cls.from_dict(data)
        logger.info(f"Loaded calibration from {path}")
        return result


"""
Calibration Measurement
"""

class PulseShaperCalibration:
    """
    Calibration measurement procedure for dual pulse pair system.
    
    Measures F_{p1,p2}(c1, c2) for all polarity combinations, then computes
    Δ calibrations for each (sig1, sig2) operating point.
    
    Physical setup:
    - relay1 controls charging pair (X1/Y1 polarities are complementary)
    - relay2 controls discharging pair (X2/Y2 polarities are complementary)
    - X output measured by x_meter, Y output measured by y_meter
    - ch1_x, ch1_y are DC controls for charging pair
    - ch2_x, ch2_y are DC controls for discharging pair
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
        ch1_y: ScalarChannel,
        ch2_x: ScalarChannel,
        ch2_y: ScalarChannel,
        x_meter: ScalarChannel,
        y_meter: ScalarChannel,
        axis_controls: np.ndarray,
        cal_attenuation: float = 1.0,
        settle_time: float = 0.5,
        averages: int = 3,
        parallel_meters: bool = True,
        sparse_downsample: int = 4,
    ):
        """
        Parameters
        ----------
        relay1, relay2 : CompPair
            Relays controlling charging and discharging pair polarities.
        ch1_x, ch1_y : ScalarChannel
            DC controls for charging pair (X and Y outputs).
        ch2_x, ch2_y : ScalarChannel
            DC controls for discharging pair (X and Y outputs).
        x_meter, y_meter : ScalarChannel
            DMMs measuring X and Y outputs.
        axis_controls : np.ndarray
            Dense control points for axis sweeps.
        cal_attenuation : float
            Output attenuation to compensate in results.
        settle_time : float
            Time to wait after changing settings.
        averages : int
            Number of DMM readings to average.
        parallel_meters : bool
            If True, read X and Y meters in parallel.
        sparse_downsample : int
            Downsampling factor for joint sweep grid.
        """
        self.relay1 = relay1
        self.relay2 = relay2
        self.ch1_x = ch1_x
        self.ch1_y = ch1_y
        self.ch2_x = ch2_x
        self.ch2_y = ch2_y
        self.x_meter = x_meter
        self.y_meter = y_meter
        
        self.axis_controls = np.asarray(axis_controls)
        self.sparse_controls = self.axis_controls[::sparse_downsample]
        self.cal_attenuation = cal_attenuation
        self.settle_time = settle_time
        self.averages = averages
        self.parallel_meters = parallel_meters
        
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel_meters else None
        
        logger.info(
            f"PulseShaperCalibration initialized: "
            f"{len(self.axis_controls)} axis points, "
            f"{len(self.sparse_controls)} sparse points, "
            f"attenuation={cal_attenuation}"
        )
    
    def _set_polarities(self, p1: bool, p2: bool):
        """Set relay polarities (X gets p, Y gets not p for each relay)."""
        self.relay1.xpolarity(p1)
        self.relay1.ypolarity(not p1)
        self.relay2.xpolarity(p2)
        self.relay2.ypolarity(not p2)
        logger.debug(f"Polarities: p1={p1}, p2={p2}")
    
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
    
    def _zero_all(self):
        """Set all DC controls to zero."""
        self.ch1_x.set_output(0.0)
        self.ch1_y.set_output(0.0)
        self.ch2_x.set_output(0.0)
        self.ch2_y.set_output(0.0)
    
    def _sweep_axis1(self, p1: bool, p2: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep axis 1: vary (c1, c1) with (c2, c2) = 0.
        
        Returns (x_readings, y_readings).
        """
        self._set_polarities(p1, p2)
        self.ch2_x.set_output(0.0)
        self.ch2_y.set_output(0.0)
        time.sleep(self.settle_time)
        
        n = len(self.axis_controls)
        x_readings = np.zeros(n)
        y_readings = np.zeros(n)
        
        for i, c in enumerate(self.axis_controls):
            self.ch1_x.set_output(c)
            self.ch1_y.set_output(c)
            time.sleep(self.settle_time)
            x_readings[i], y_readings[i] = self._read_meters()
        
        return x_readings, y_readings
    
    def _sweep_axis2(self, p1: bool, p2: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep axis 2: vary (c2, c2) with (c1, c1) = 0.
        
        Returns (x_readings, y_readings).
        """
        self._set_polarities(p1, p2)
        self.ch1_x.set_output(0.0)
        self.ch1_y.set_output(0.0)
        time.sleep(self.settle_time)
        
        n = len(self.axis_controls)
        x_readings = np.zeros(n)
        y_readings = np.zeros(n)
        
        for i, c in enumerate(self.axis_controls):
            self.ch2_x.set_output(c)
            self.ch2_y.set_output(c)
            time.sleep(self.settle_time)
            x_readings[i], y_readings[i] = self._read_meters()
        
        return x_readings, y_readings
    
    def _sweep_joint(self, p1: bool, p2: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep joint grid on sparse controls.
        
        Returns (x_grid, y_grid) each of shape (n_sparse, n_sparse).
        First index is c1, second index is c2.
        """
        self._set_polarities(p1, p2)
        time.sleep(self.settle_time)
        
        n = len(self.sparse_controls)
        x_grid = np.zeros((n, n))
        y_grid = np.zeros((n, n))
        
        for i, c1 in enumerate(self.sparse_controls):
            self.ch1_x.set_output(c1)
            self.ch1_y.set_output(c1)
            for j, c2 in enumerate(self.sparse_controls):
                self.ch2_x.set_output(c2)
                self.ch2_y.set_output(c2)
                time.sleep(self.settle_time)
                x_grid[i, j], y_grid[i, j] = self._read_meters()
        
        return x_grid, y_grid
    
    def measure(self) -> PulseShaperRawData:
        """
        Execute full measurement.
        
        Assumes DTG run is already disabled.
        """
        logger.info("=" * 60)
        logger.info("Starting pulse shaper calibration measurement")
        logger.info("=" * 60)
        start = time.time()
        
        raw = PulseShaperRawData(
            axis_controls=self.axis_controls.copy(),
            sparse_controls=self.sparse_controls.copy(),
        )
        
        # Axis 1 sweeps (4 polarity combinations)
        logger.info("Measuring axis 1 (charging pair)...")
        for key in self.POLARITY_KEYS:
            p1, p2 = self.POLARITY_VALUES[key]
            logger.debug(f"  Sweep F_{key}(c1, 0)")
            x_rd, y_rd = self._sweep_axis1(p1, p2)
            setattr(raw, f'axis1_F_{key}_x', x_rd)
            setattr(raw, f'axis1_F_{key}_y', y_rd)
        
        # Axis 2 sweeps (4 polarity combinations)
        logger.info("Measuring axis 2 (discharging pair)...")
        for key in self.POLARITY_KEYS:
            p1, p2 = self.POLARITY_VALUES[key]
            logger.debug(f"  Sweep F_{key}(0, c2)")
            x_rd, y_rd = self._sweep_axis2(p1, p2)
            setattr(raw, f'axis2_F_{key}_x', x_rd)
            setattr(raw, f'axis2_F_{key}_y', y_rd)
        
        # Joint sweeps (4 polarity combinations)
        n_sparse = len(self.sparse_controls)
        logger.info(f"Measuring joint grid ({n_sparse}x{n_sparse})...")
        for key in self.POLARITY_KEYS:
            p1, p2 = self.POLARITY_VALUES[key]
            logger.debug(f"  Sweep F_{key}(c1, c2)")
            x_grid, y_grid = self._sweep_joint(p1, p2)
            setattr(raw, f'joint_F_{key}_x', x_grid)
            setattr(raw, f'joint_F_{key}_y', y_grid)
        
        self._zero_all()
        elapsed = time.time() - start
        logger.info(f"Measurement complete in {elapsed:.1f}s")
        
        return raw

    def process(self, raw: PulseShaperRawData, degree: int = 2) -> PulseShaperCalibrationResult:
        """
        Process raw data into calibrations.
        
        For each (sig1, sig2), compute:
            Δ^{X1}_{sig1,sig2}(c1) = F_{sig1,-sig2}(c1,0) - F_{-sig1,-sig2}(c1,0)
            Δ^{X2}_{sig1,sig2}(c2) = F_{-sig1,sig2}(0,c2) - F_{-sig1,-sig2}(0,c2)
        
        and similarly for Y1, Y2.
        """
        logger.info("Processing calibration data...")
        atten = self.cal_attenuation
        controls = raw.axis_controls
        
        result = PulseShaperCalibrationResult(
            raw_data=raw,
            fit_degree=degree,
            cal_attenuation=atten,
        )
        
        def pol_key(p: bool) -> str:
            return 'p' if p else 'm'
        
        def get_axis1(key: str, output: str) -> np.ndarray:
            return getattr(raw, f'axis1_F_{key}_{output}')
        
        def get_axis2(key: str, output: str) -> np.ndarray:
            return getattr(raw, f'axis2_F_{key}_{output}')
        
        def fit_cal(controls: np.ndarray, delta: np.ndarray) -> PolynomialCalibration:
            """Fit polynomial without sign correction."""
            coeffs = np.polyfit(controls, delta, degree)[::-1]
            return PolynomialCalibration(
                coeffs=coeffs.tolist(),
                c_fit_min=float(controls.min()),
                c_fit_max=float(controls.max()),
            )
        
        def rms(delta: np.ndarray, cal: PolynomialCalibration) -> float:
            predicted = cal.forward_vectorized(controls)
            return float(np.sqrt(np.mean((delta - predicted)**2)))
        
        # Compute calibrations for each (sig1, sig2)
        for sigma1 in [True, False]:
            for sigma2 in [True, False]:
                s1, s2 = pol_key(sigma1), pol_key(sigma2)
                ns1, ns2 = pol_key(not sigma1), pol_key(not sigma2)
                
                # X1: Δ = F_{sig1,-sig2}(c1,0) - F_{-sig1,-sig2}(c1,0)
                key_a = s1 + ns2
                key_b = ns1 + ns2
                delta_x1 = (get_axis1(key_a, 'x') - get_axis1(key_b, 'x')) / atten
                cal_x1 = fit_cal(controls, delta_x1)
                result.x1_calibrations[(sigma1, sigma2)] = cal_x1
                result.rms_errors[f'X1_({sigma1},{sigma2})'] = rms(delta_x1, cal_x1)
                
                # X2: Δ = F_{-sig1,sig2}(0,c2) - F_{-sig1,-sig2}(0,c2)
                key_a = ns1 + s2
                key_b = ns1 + ns2
                delta_x2 = (get_axis2(key_a, 'x') - get_axis2(key_b, 'x')) / atten
                cal_x2 = fit_cal(controls, delta_x2)
                result.x2_calibrations[(sigma1, sigma2)] = cal_x2
                result.rms_errors[f'X2_({sigma1},{sigma2})'] = rms(delta_x2, cal_x2)
                
                # Y1: Δ = F_{-sig1,sig2}(c1,0) - F_{sig1,sig2}(c1,0)
                key_a = ns1 + s2
                key_b = s1 + s2
                delta_y1 = (get_axis1(key_a, 'y') - get_axis1(key_b, 'y')) / atten
                cal_y1 = fit_cal(controls, delta_y1)
                result.y1_calibrations[(sigma1, sigma2)] = cal_y1
                result.rms_errors[f'Y1_({sigma1},{sigma2})'] = rms(delta_y1, cal_y1)

                # Y2: Δ = F_{sig1,-sig2}(0,c2) - F_{sig1,sig2}(0,c2)
                key_a = s1 + ns2
                key_b = s1 + s2
                delta_y2 = (get_axis2(key_a, 'y') - get_axis2(key_b, 'y')) / atten
                cal_y2 = fit_cal(controls, delta_y2)
                result.y2_calibrations[(sigma1, sigma2)] = cal_y2
                result.rms_errors[f'Y2_({sigma1},{sigma2})'] = rms(delta_y2, cal_y2)
        
        # Cross-term validation using joint measurements
        result.cross_term_diagnostics = self._analyze_cross_terms(raw, result)
        
        # Apply sign correction at the end
        self._apply_sign_corrections(result)
        
        logger.info("Processing complete.")
        return result

    def _analyze_cross_terms(self,
        raw: PulseShaperRawData,
        result: PulseShaperCalibrationResult,
    ) -> Dict[str, Any]:
        """
        Validate that Δ is approximately independent of the other control.
        
        Compare measured Δ on joint grid to predicted Δ from axis calibrations.
        No sign correction here - we use raw deltas.
        """
        atten = self.cal_attenuation
        sparse = raw.sparse_controls
        
        diagnostics = {}
        
        def analyze_channel(name: str, 
                            joint_a: np.ndarray, 
                            joint_b: np.ndarray,
                            cal: PolynomialCalibration,
                            axis: int):
            """
            Analyze cross-term for one channel.
            
            axis=0 means delta should be independent of c2 (broadcast along axis 1)
            axis=1 means delta should be independent of c1 (broadcast along axis 0)
            """
            joint_delta = (joint_a - joint_b) / atten
            predicted = cal.forward_vectorized(sparse)
            
            if axis == 0:
                predicted_grid = predicted[:, np.newaxis]
            else:
                predicted_grid = predicted[np.newaxis, :]
            
            residual = joint_delta - predicted_grid
            
            rms_signal = np.sqrt(np.mean(joint_delta**2))
            rms_residual = np.sqrt(np.mean(residual**2))
            
            diagnostics[f'{name}_cross_term_rms'] = float(rms_residual)
            diagnostics[f'{name}_cross_term_ratio'] = float(rms_residual / rms_signal) if rms_signal > 0 else 0.0
        
        # X1: varies with c1 (axis 0), should be independent of c2
        analyze_channel('x1', raw.joint_F_pm_x, raw.joint_F_mm_x,
                        result.x1_calibrations[(True, True)], axis=0)
        
        # X2: varies with c2 (axis 1), should be independent of c1
        analyze_channel('x2', raw.joint_F_mp_x, raw.joint_F_mm_x,
                        result.x2_calibrations[(True, True)], axis=1)
        
        # Y1: varies with c1 (axis 0), should be independent of c2
        analyze_channel('y1', raw.joint_F_mp_y, raw.joint_F_pp_y,
                        result.y1_calibrations[(True, True)], axis=0)
        
        # Y2: varies with c2 (axis 1), should be independent of c1
        analyze_channel('y2', raw.joint_F_pm_y, raw.joint_F_pp_y,
                        result.y2_calibrations[(True, True)], axis=1)
        
        logger.info(
            f"Cross-term analysis: "
            f"X1={diagnostics['x1_cross_term_ratio']*100:.2f}%, "
            f"X2={diagnostics['x2_cross_term_ratio']*100:.2f}%, "
            f"Y1={diagnostics['y1_cross_term_ratio']*100:.2f}%, "
            f"Y2={diagnostics['y2_cross_term_ratio']*100:.2f}%"
        )
        
        return diagnostics

    def _apply_sign_corrections(self, result: PulseShaperCalibrationResult):
        """
        Apply sign corrections to calibrations so outputs are majority positive.
        
        For each channel, check if majority of outputs are negative and flip if so.
        All (sigma1, sigma2) combinations for a channel are flipped together.
        """
        controls = result.raw_data.axis_controls
        
        def needs_flip(calibrations: Dict[Tuple[bool, bool], PolynomialCalibration]) -> bool:
            """Check if calibrations produce majority negative outputs."""
            total_negative = 0
            total_positive = 0
            for cal in calibrations.values():
                outputs = cal.forward_vectorized(controls)
                total_negative += np.sum(outputs < 0)
                total_positive += np.sum(outputs > 0)
            return total_negative > total_positive
        
        def flip_calibrations(calibrations: Dict[Tuple[bool, bool], PolynomialCalibration]):
            """Flip sign of all calibration coefficients."""
            for key, cal in calibrations.items():
                flipped_coeffs = [-c for c in cal.coefficients]
                calibrations[key] = PolynomialCalibration(
                    coeffs=flipped_coeffs,
                    c_fit_min=cal._c_fit_min,
                    c_fit_max=cal._c_fit_max,
                )
        
        # Check and flip each channel
        for name, cals in [
            ('X1', result.x1_calibrations),
            ('X2', result.x2_calibrations),
            ('Y1', result.y1_calibrations),
            ('Y2', result.y2_calibrations),
        ]:
            if needs_flip(cals):
                flip_calibrations(cals)
                logger.info(f"Applied sign correction to {name} calibrations")

    def plot_calibration(self, 
        raw: PulseShaperRawData, 
        result: PulseShaperCalibrationResult,
        sigma1: bool = True,
        sigma2: bool = True,
        save_path: str | Path | None = None,
    ):
        """
        Plot calibration results for a given (sigma1, sigma2) operating point.
        
        Creates a figure with:
        - Row 1: X1, X2, Y1, Y2 calibration curves with fits
        - Row 2: Fit residuals
        - Row 3: Cross-term heatmaps (joint grid)
        """
        import matplotlib.pyplot as plt
        
        atten = self.cal_attenuation
        controls = raw.axis_controls
        sparse = raw.sparse_controls
        
        def pol_key(p: bool) -> str:
            return 'p' if p else 'm'
        
        s1, s2 = pol_key(sigma1), pol_key(sigma2)
        ns1, ns2 = pol_key(not sigma1), pol_key(not sigma2)
        
        def get_axis1(key: str, output: str) -> np.ndarray:
            return getattr(raw, f'axis1_F_{key}_{output}')
        
        def get_axis2(key: str, output: str) -> np.ndarray:
            return getattr(raw, f'axis2_F_{key}_{output}')
        
        def get_joint(key: str, output: str) -> np.ndarray:
            return getattr(raw, f'joint_F_{key}_{output}')
        
        # Compute raw deltas (no sign correction)
        delta_x1 = (get_axis1(s1 + ns2, 'x') - get_axis1(ns1 + ns2, 'x')) / atten
        delta_x2 = (get_axis2(ns1 + s2, 'x') - get_axis2(ns1 + ns2, 'x')) / atten
        delta_y1 = (get_axis1(ns1 + s2, 'y') - get_axis1(s1 + s2, 'y')) / atten
        delta_y2 = (get_axis2(s1 + ns2, 'y') - get_axis2(s1 + s2, 'y')) / atten
        
        # Get calibrations (already sign-corrected)
        cal_x1 = result.x1_calibrations[(sigma1, sigma2)]
        cal_x2 = result.x2_calibrations[(sigma1, sigma2)]
        cal_y1 = result.y1_calibrations[(sigma1, sigma2)]
        cal_y2 = result.y2_calibrations[(sigma1, sigma2)]
        
        # Predictions from calibrations
        pred_x1 = cal_x1.forward_vectorized(controls)
        pred_x2 = cal_x2.forward_vectorized(controls)
        pred_y1 = cal_y1.forward_vectorized(controls)
        pred_y2 = cal_y2.forward_vectorized(controls)
        
        # Match delta signs to calibration signs for plotting
        # (calibration may have been flipped)
        def match_sign(delta, pred):
            if np.sum(delta * pred < 0) > len(delta) // 2:
                return -delta
            return delta
        
        delta_x1 = match_sign(delta_x1, pred_x1)
        delta_x2 = match_sign(delta_x2, pred_x2)
        delta_y1 = match_sign(delta_y1, pred_y1)
        delta_y2 = match_sign(delta_y2, pred_y2)
        
        # Joint deltas
        joint_delta_x1 = (get_joint(s1 + ns2, 'x') - get_joint(ns1 + ns2, 'x')) / atten
        joint_delta_x2 = (get_joint(ns1 + s2, 'x') - get_joint(ns1 + ns2, 'x')) / atten
        joint_delta_y1 = (get_joint(ns1 + s2, 'y') - get_joint(s1 + s2, 'y')) / atten
        joint_delta_y2 = (get_joint(s1 + ns2, 'y') - get_joint(s1 + s2, 'y')) / atten
        
        # Match joint delta signs
        def match_sign_joint(joint_delta, cal):
            pred = cal.forward_vectorized(sparse)
            diag = np.diag(joint_delta)
            if np.sum(diag * pred < 0) > len(diag) // 2:
                return -joint_delta
            return joint_delta
        
        joint_delta_x1 = match_sign_joint(joint_delta_x1, cal_x1)
        joint_delta_x2 = match_sign_joint(joint_delta_x2, cal_x2)
        joint_delta_y1 = match_sign_joint(joint_delta_y1, cal_y1)
        joint_delta_y2 = match_sign_joint(joint_delta_y2, cal_y2)
        
        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Pulse Shaper Calibration (σ₁={sigma1}, σ₂={sigma2})', fontsize=14)
        
        channels = [
            ('X1', delta_x1, pred_x1, cal_x1, joint_delta_x1, 0),
            ('X2', delta_x2, pred_x2, cal_x2, joint_delta_x2, 1),
            ('Y1', delta_y1, pred_y1, cal_y1, joint_delta_y1, 0),
            ('Y2', delta_y2, pred_y2, cal_y2, joint_delta_y2, 1),
        ]
        
        for col, (name, delta, pred, cal, joint_delta, axis) in enumerate(channels):
            # Row 0: Calibration curve + fit
            ax = axes[0, col]
            ax.plot(controls, delta, 'o', markersize=3, alpha=0.5, label='Measured')
            ax.plot(controls, pred, '-', linewidth=2, label='Fit')
            ax.set_xlabel('Control (V)')
            ax.set_ylabel('Δ Output (V)')
            ax.set_title(f'{name}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Row 1: Residuals
            ax = axes[1, col]
            residual = delta - pred
            ax.plot(controls, residual, 'o', markersize=3)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Control (V)')
            ax.set_ylabel('Residual (V)')
            rms_val = np.sqrt(np.mean(residual**2))
            ax.set_title(f'Residuals (RMS={rms_val:.2e})')
            ax.grid(True, alpha=0.3)
            
            # Row 2: Cross-term heatmap
            ax = axes[2, col]
            
            pred_sparse = cal.forward_vectorized(sparse)
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
            ax.set_title(f'Cross-term')
            plt.colorbar(im, ax=ax, label='Deviation (V)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved calibration plot to {save_path}")
        
        return fig, axes

    def run(self, degree: int = 2) -> PulseShaperCalibrationResult:
        """
        Run complete calibration.
        
        Assumes DTG run is already disabled.
        """
        raw = self.measure()
        result = self.process(raw, degree=degree)
        
        logger.info("=" * 60)
        logger.info("Calibration complete")
        logger.info("=" * 60)
        logger.info("\n" + result.summary())
        
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