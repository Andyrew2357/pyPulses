from .abstract_device import abstractDevice
from .attenuator import Attenuator
from .channel_adapter import ScalarChannel
from .registry import (
    register_device_class, 
    format_reference, 
    resolve_reference, 
    DeferredReference,
    DeviceRegistry,
)

import numpy as np
from logging import Logger
from abc import ABC, abstractmethod
from typing import List, Tuple

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

class PolynomialCalibration(CalibrationModel):
    """
    Models output as: O(c) = sum_i(a[i] * c^i for i)

    Coefficients are stored in ascending order [a0, a1, a2, ...].
    """

    def __init__(self, coeffs: List[float] | np.ndarray):
        self._coeffs = np.array(coeffs, dtype=np.float64)
        self._coeffs_r = self._coeffs[::-1]
        self._dcoeffs = np.polyder(self._coeffs)
        self._critical = np.roots(self._dcoeffs)
        self._degree = len(self._coeffs) - 1

    @property
    def degree(self) -> int:
        return self._degree
    
    def forward(self, c: float) -> float:
        return float(np.polyval(self._coeffs_r, c))
    
    def forward_vectorized(self, cs: np.ndarray) -> np.ndarray:
        return np.polyval(self._coeffs_r, cs)
    
    def inverse(self, o: float, c_min: float, c_max: float) -> float:
        if self.degree == 0:
            if np.isclose(self._coeffs[0], o):
                return (c_min + c_max) / 2
            raise ValueError(f"Constant calibration {self._coeffs[0]} cannot produce {o}")
        
        if self._degree == 1:
            a0, a1 = self._coeffs
            if np.isclose(a1, 0):
                raise ValueError("Degenerate linear calibration with zero slope")
            c = (o - a0) / a1
            return self._clamp_and_validate(c, c_min, c_max, o)
        
        if self._degree == 2:
            return self._inverse_quadratic(o, c_min, c_max)
        
        return self._inverse_newton(o, c_min, c_max)
    
    def _inverse_quadratic(self, o: float, c_min: float, c_max: float) -> float:
        a0, a1, a2 = self._coeffs
        a, b, c = a2, a1, a0 - o
        disc = b*b - 4*a*c
        if disc < 0:
            raise ValueError(f"Output {o} not achievable (no real roots)")
        
        sqrt_disc = np.sqrt(disc)
        roots = [(-b + sqrt_disc)/(2*a), (-b - sqrt_disc)/(2*a)]
        valid_roots = [r for r in roots if c_min <= r <= c_max]
        if not valid_roots:
            raise ValueError(f"Output {o} not achievable within [{c_min}, {c_max}]")
        
        if len(valid_roots) == 2:
            mid = (c_min + c_max) / 2
            return min(valid_roots, key=lambda r: abs(r - mid))
        
        return valid_roots[0]
    
    def _inverse_newton(self, o: float, c_min: float, c_max: float, 
                        max_itr: int = 50, tol: float = 1e-12) -> float:
        """Newton-Raphson inverse"""

        f_min, f_max = self.forward(c_min), self.forward(c_max)
        if np.isclose(f_min, f_max):
            raise ValueError("Degenerate calibration: constant over voltage range")
        
        t = (o - f_min) / (f_max - f_min)
        v = c_min + t * (c_max - c_min)
        v = np.clip(v, c_min, c_max)

        for _ in range(max_itr):
            f_v = np.polyval(self._coeffs_r, v) - o
            if abs(f_v) < tol:
                return float(v)
            
            df_v = np.polyval(self._dcoeffs, v)
            if np.isclose(df_v, 0):
                if f_v > 0:
                    c_max = v
                else:
                    c_min = v
                v = (c_min + c_max) / 2
                continue

            v_new = v - f_v / df_v
            v_new = np.clip(v_new, c_min, c_max)
            if abs(v_new - v) < tol:
                return float(v_new)
            v = v_new

        return float(v)
    
    def _clamp_and_validate(self, c: float, c_min: float, c_max: float, o: float) -> float:
        if c < c_min or c > c_max:
            raise ValueError(
                f"Output {o} requires control value {c}, outside bounds [{c_min}, {c_max}]"
            )
        return c
    
    def output_bounds(self, c_min: float, c_max: float) -> Tuple[float, float]:
        """
        Compute output amplitude bounds.

        For monotonic polynomials, just evaluates at endpoints.
        For non-monotonic, also checks critical points.
        """

        vals = [self.forward(c_min), self.forward(c_max)]
        if self._degree >= 2:
            for cp in self._critical:
                if np.isreal(cp):
                    cp_real = float(np.real(cp))
                    if c_min < cp_real < c_max:
                        vals.append(self.forward(cp_real))

        return min(vals), max(vals)
    
"""
Calibrated Channel
"""

class AttenuatorMode:
    """How aggressively to adjust the attenuator."""
    PASSIVE = 'passive'
    AGRESSIVE = 'aggressive'

class AttenuatorPreference:
    """Which attenuation to prefer when multiple settings work."""
    CLOSEST = 'closest'
    SMALLEST = 'smallest'
    LARGEST = 'largest'

@register_device_class("CalibratedChannel")
class CalibratedChannel(abstractDevice):
    """
    A channel with calibration, optional attenuation, and crosstalk compensation.
    """

    def __init__(self,
        hardware: ScalarChannel | DeferredReference,
        calibration: CalibrationModel | None,
        control_min: float,
        control_max: float,
        output_min: float,
        output_max: float,
        attenuator: Attenuator | DeferredReference | None = None,
        attenuator_active: bool = True,
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):

        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

        self._hw = hardware
        self._calibration = calibration
        self._attenuator = attenuator
        self._attenuator_active = attenuator_active

        # control and output rails
        self._c_min = control_min
        self._c_max = control_max
        self._o_min = output_min
        self._o_max = output_max

        # attenuator settings
        self._attenuator_mode = AttenuatorMode.PASSIVE
        self._attenuator_preference = AttenuatorPreference.CLOSEST
        self._attenuator_locked = False

        # crosstalk compensation
        self._crosstalk: List[Tuple[str, float]] = []

        # cached values
        self._cached_control: float | None = None
        self._cached_attenuation: float | None = None

    @property
    def output_min(self) -> float:
        """Minimum achievable output (at current attenuation)"""
        atten = self._get_attenuation(lazy=True)
        return self._o_min * atten
    
    @property
    def output_max(self) -> float:
        """Maximum achievable output (at current attenuation)"""
        atten = self._get_attenuation(lazy=True)
        return self._o_max * atten

    def add_crosstalk(self, other: 'CalibratedChannel | str', coeff: float):
        """Register crosstalk from another channel."""
        if isinstance(other, str):
            ref = other
        else:
            ref = format_reference(other)
        self._crosstalk.append((ref, coeff))

    def clear_crosstalk(self) -> None:
        """Remove all crosstalk registrations."""

        self._crosstalk.clear()

    """
    Core Output Interface
    """

    def get_output(self, lazy: bool = True) -> float:
        """
        Get the current output, including crosstalk correction.

        Parameters
        ----------
        lazy : bool
            If true, we use cached control values when available
        """

        c = self._get_control(lazy=lazy)
        o = self._calibration.forward(c) * self._get_attenuation()
        for other, coeff in self._crosstalk:
            o += coeff * resolve_reference(other).get_output()

        return o
    
    def set_output(self, o: float):
        """
        Set the desired output. 
        Adjusts the attenuator (if available and not locked) and control to 
        achieve the requested amplitude.

        Note: Crosstalk is NOT compensated when setting (by design, to avoid
        complexity and additional hardware queries).
        """

        if np.isnan(o):
            raise ValueError(f"Cannot set amplitude to NaN")
        
        if o == 0:
            self._hw.set_output(0.0)
            self._cached_control = (0.0)
            if self._attenuator and self._attenuator_active \
                and not self._attenuator_locked:
                self._cached_attenuation = self._attenuator.set_higher(0.0)
            return
        
        # Check polarity
        if (self._o_min > 0 and o < 0) or (self._o_max < 0 and o > 0):
            raise ValueError(
                f"Cannot achieve output value {o} with unipolar output "
                f"[{self._o_min}, {self._o_max}]"
            )
        
        # Determine attenuation
        if self._attenuator is not None:
            atten = self._pick_attenuation(o)
            target_raw = o / atten
        else:
            atten = 1.0
            target_raw = o

        # Clamp to achievable range
        if target_raw < self._o_min:
            self.warn(f"Requested output {o} is below allowed range {self._o_min}; Clamping...")
            target_raw = self._o_min
        elif target_raw > self._o_max:
            self.warn(f"Requested output {o} is above allowed range {self._o_max}; Clamping...")
            target_raw = self._o_max

        # Compute required control setting
        control = self._calibration.inverse(target_raw, self._c_min, self._c_max)

        # Set hardware
        self._hw.set_output(control)
        self._cached_control = control

    def __call__(self, value: float | None = None) -> float | None:
        if value is None:
            return self.get_output()
        return self.set_output(value)

    """
    Attenuator Handling
    """

    def _get_attenuation(self, lazy: bool = True) -> float:
        """Get current attenuation factor."""
        
        if self._attenuator is None:
            return 1.0
        
        if lazy and self._cached_attenuation is not None:
            return self._cached_attenuation
        
        self._cached_attenuation = self._attenuator.get()
        return self._cached_attenuation
    
    def _pick_attenuation(self, target_output: float) -> float:
        """Choose appropriate attenuation for target output."""

        if not self._attenuator_active or self._attenuator_locked:
            return self._get_attenuation(lazy=True)
        
        current_atten = self._get_attenuation(lazy=True)

        # In passive mode, only change if we can't reach target
        if self._attenuator_mode == AttenuatorMode.PASSIVE:
            achievable_min = self._o_min * current_atten
            achievable_max = self._o_max * current_atten
            if achievable_min <= target_output <= achievable_max:
                return current_atten
            
        # Need to pick a new attenuation
        abs_target = abs(target_output)

        if self._attenuator_preference == AttenuatorPreference.SMALLEST:
            # Least attenuation that can still reach target
            max_raw = max(abs(self._o_min), abs(self._o_max))
            ideal = abs_target / max_raw
            actual = self._attenuator.set_lower(ideal)
        
        elif self._attenuator_preference == AttenuatorPreference.LARGEST:
            # Most attenuation that can still reach target
            min_raw = min(abs(self._o_min), abs(self._o_max))
            if min_raw > 0:
                ideal = abs_target / min_raw
            else:
                ideal = abs_target / max(abs(self._o_min), abs(self._o_max))
            actual = self._attenuator.set_higher(ideal)
        
        else:  # CLOSEST
            # Geometric mean of output range
            geo_mean = np.sqrt(abs(self._o_min * self._o_max))
            if geo_mean > 0:
                ideal = abs_target / geo_mean
            else:
                ideal = abs_target / max(abs(self._o_min), abs(self._o_max))
            actual = self._attenuator.set_closest(ideal)
        
        self._cached_attenuation = actual
        return actual
    
    """
    Control Access
    """

    def _get_control(self, lazy: bool = True) -> float:
        """Get current control."""
        if lazy and self._cached_control is not None:
            return self._cached_control
        
        self._cached_control = self._hw.get_output()
        return self._cached_control
    
    def invalidate_cache(self) -> None:
        """Force next query to read from hardware."""
        self._cached_control = None
        self._cached_attenuation = None

    """
    Serialization support
    """

    def _serialize_state(self) -> dict:
        config = {
            'hardware': format_reference(self._hw),
            'attenuator': format_reference(self._attenuator),
            'crosstalk': self._crosstalk,
            'c_min': self._c_min,
            'c_max': self._c_max,
            'o_min': self._o_min,
            'o_max': self._o_max,
            'attenuator_active': self._attenuator_active,
            'attenuator_mode': self._attenuator_mode,
            'attenuator_preference': self._attenuator_preference,
            'attenuator_locked': self._attenuator_locked,
        }
        
        # Add calibration coefficients if polynomial
        if isinstance(self._calibration, PolynomialCalibration):
            config['calibration_type'] = 'polynomial'
            config['calibration_coefficients'] = self._calibration._coeffs.tolist()
        elif isinstance(self._calibration, TrivialCalibration):
            config['calibration_type'] = 'trivial'
        
        return config
    
    def _deserialize_state(self, config: dict):
        
        cal_type = config.get('calibration_type', 'polynomial')
        if cal_type == 'polynomial':
            self._calibration = PolynomialCalibration(config['calibration_coefficients'])
        elif cal_type == 'trivial':
            self._calibration = TrivialCalibration()
        else:
            raise ValueError(f"Unknown calibration type: {cal_type}")
        
        if 'hardware' in config:
            self._hw = DeferredReference(config['hardware'])
        if 'attenuator' in config:
            self._attenuator = config['attenuator']
            if self._attenuator is not None:
                self._attenuator = DeferredReference(self._attenuator)
        if 'crosstalk' in config:
            self._crosstalk = config['crosstalk']

        if 'c_min' in config:
            self._c_min = config['c_min']
        if 'c_max' in config:
            self._c_max = config['c_max']
        if 'o_min' in config:
            self._o_min = config['o_min']
        if 'o_max' in config:
            self._o_max = config['o_max']
        if 'attenuator_active' in config:
            self._attenuator_active = config['attenuator_active']
        if 'attenuator_mode' in config:
            self._attenuator_mode = config['attenuator_mode']
        if 'attenuator_preference' in config:
            self._attenuator_preference = config['attenuator_preference']
        if 'attenuator_locked' in config:
            self._attenuator_locked = config['attenuator_locked']

        self._cached_control = None
        self._cached_attenuation = None

        self._resolve_references()

    def _resolve_references(self):
        if isinstance(self._hw, DeferredReference):
            self._hw = self._hw.unwrap()
        if isinstance(self._attenuator, DeferredReference):
            self._attenuator = self._attenuator.unwrap()

    @classmethod
    def from_config(cls, config) -> 'CalibratedChannel':

        hardware = config.pop("hardware")
        if hardware is not None:
            hardware = DeferredReference(hardware)
        calibration = None
        control_min = config.pop('c_min')
        control_max = config.pop('c_max')
        output_min = config.pop('o_min')
        output_max = config.pop('o_max')
        attenuator = config.pop('attenuator')
        if attenuator is not None:
            attenuator = DeferredReference(attenuator)
        attenuator_active = config.pop('attenuator_active')
        registry_id = config.pop('registry_id')

        instance = cls(
            hardware=hardware,
            calibration=calibration,
            control_min=control_min,
            control_max=control_max,
            output_min=output_min,
            output_max=output_max,
            attenuator=attenuator,
            attenuator_active=attenuator_active,
            registry_id=registry_id,
        )

        return instance