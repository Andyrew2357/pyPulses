from .cap_filter import CapacitanceFilter
from .dis_filter import DischargeFilter, DischargeExtrapolator

from ...devices.registry import resolve_reference, format_reference
from ...devices.wfatd import wfAverager, wfBalance, wfCurveData
from ...devices.pulse_pair import pulsePair

from dataclasses import dataclass
from typing import Any, Callable, Dict
import numpy as np
import logging
import json

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class TDPTContext():

    # Time-domain waveform averager accessed by the balance parameters
    averager: wfAverager

    # Charging pair and error parameter
    charge_pair: pulsePair
    capacitance_error: wfBalance

    # Discharging pair and error parameter
    discharge_pair: pulsePair | None = None
    dis_eta: int = -1
    discharge_error: wfBalance | None = None
    integrated_error: wfBalance | None = None

    # Discharge amplitude takes cues from charging amplitude
    discharge_X_ratio: float | None = None

    # Limits on the control parameters
    max_pulse_height_x: float = 50.0e-3
    max_pulse_height_y: float = 50.0e-3
    min_pulse_width: float = 1.0e-6
    max_pulse_width: float = 5.0e-6
    pulse_width_resolution: float = 2.7e-10

    # Configuration for adjusting the discharge pulse amplitudes
    dis_amp_adjustment_threshold: float = 4.0e-9
    dbal_dev: float = 0.1

    # Termination criteria — error must be < threshold * sqrt(variance)
    cap_balance_threshold:  float = 1.0
    dis_balance_threshold:  float = 1.0
    int_balance_threshold:  float = 1.0
    max_iterations: int = 10
    good_sweep_threshold: int = 1

    # Balance loop clamping limits — dimensionless ratios |Δ(Y/X)|
    max_Y1_step:      float = 0.5
    max_Y2_step:      float = 0.5
    max_Y2_int_step:  float = 0.1

    # Miscellaneous
    settle_time: float = 0.2
    high_resolution_sweep_multiplier: int = 1
    dis_extrap_support: int = 7
    dis_extrap_order: int = 2

    # Callbacks
    pre_sweep_callback:  Callable | None = None
    post_sweep_callback: Callable | None = None

    logger: logging.Logger | None = None

    def __post_init__(self):
        # Filters — populated by initialize_filters
        self.cap_filter: CapacitanceFilter | None = None
        self.dis_filter: DischargeFilter | None = None

        # Extrapolator — constructed here since it only needs context params
        self.extrapolator: DischargeExtrapolator | None = None
        if self.is_discharge():
            self.extrapolator = DischargeExtrapolator(
                extrap_support = self.dis_extrap_support,
                extrap_order   = self.dis_extrap_order,
                W_min          = self.min_pulse_width,
                W_max          = self.max_pulse_width,
            )

        # Override for Y2/X2 ratio, set by map-level health monitor.
        self._next_Y2_ratio_override: float | None = None

        # Map-level health monitoring counters
        self._rail_hi_streak: int = 0
        self._rail_lo_streak: int = 0
        self._failure_streak: int = 0
        self._consecutive_good: int = 0

        # Most recent balance result, used by map-level health monitor
        self._previous_result = None

    """Convenience predicates"""

    def is_discharge(self) -> bool:
        return self.discharge_pair is not None

    def is_integrated(self) -> bool:
        return self.integrated_error is not None

    """Logging"""

    def log(self, *args, **kwargs):
        if self.logger:
            self.logger.info(*args, **kwargs)

    """Filter management"""

    def add_capacitance_filter(self, f: CapacitanceFilter):
        self.cap_filter = f
        f.ctx = self

    def add_discharge_filter(self, f: DischargeFilter):
        self.dis_filter = f
        f.ctx = self

    def reinitialize_positive_filter(self):
        self.cap_filter.positive_initialize()
        if self.dis_filter is not None:
            self.dis_filter.positive_initialize()

    def reinitialize_negative_filter(self):
        self.cap_filter.negative_initialize()
        if self.dis_filter is not None:
            self.dis_filter.negative_initialize()

    def log_capacitance_filter_status(self):
        self.log(f"Capacitance Filter Status:\n{self.cap_filter}")

    def log_discharge_filter_status(self):
        self.log(f"Discharge Filter Status:\n{self.dis_filter}")

    """Kalman predict convenience methods"""

    def predict_excitation_change(self):
        self.cap_filter.predict(balance_change=False)
        if self.is_discharge():
            self.dis_filter.predict(balance_change=False)

    def predict_balance_change(self):
        self.cap_filter.predict(balance_change=True)
        if self.is_discharge():
            self.dis_filter.predict(balance_change=True)

    """
    Excitation parameter accessors

    Each accessor follows the same pattern:
      - Called with no argument: returns current hardware value
      - Called with a value: sets hardware, updates any tracked filter
        state, and returns the value actually set

    Bounds checking is performed here. Step limiting is the
    responsibility of the balance loop, not the accessor.
    """

    def X1(self, v: float | None = None) -> float:
        if v is not None:
            if not (-self.max_pulse_height_x <= v <= self.max_pulse_height_x):
                v = float(np.clip(v, -self.max_pulse_height_x, self.max_pulse_height_x))
                self.log(f"X1 clamped to {v:.5e}")
            self.charge_pair.X(v)

        set_v = self.charge_pair.X()
        if self.cap_filter is not None:
            self.cap_filter.Xamp = set_v
        return set_v

    def Y1(self, v: float | None = None) -> float:
        if v is not None:
            if not (-self.max_pulse_height_y <= v <= self.max_pulse_height_y):
                v = float(np.clip(v, -self.max_pulse_height_y, self.max_pulse_height_y))
                self.log(f"Y1 clamped to {v:.5e}")
            self.charge_pair.Y(v)

        set_v = self.charge_pair.Y()
        if self.cap_filter is not None:
            self.cap_filter.Yamp = set_v
        return set_v

    def X2(self, v: float | None = None) -> float | None:
        if not self.is_discharge():
            return None
        if v is not None:
            if not (-self.max_pulse_height_x <= v <= self.max_pulse_height_x):
                v = float(np.clip(v, -self.max_pulse_height_x, self.max_pulse_height_x))
                self.log(f"X2 clamped to {v:.5e}")
            self.discharge_pair.X(v)
        return self.discharge_pair.X()

    def Y2(self, v: float | None = None) -> float | None:
        if not self.is_discharge():
            return None
        if v is not None:
            if not (-self.max_pulse_height_y <= v <= self.max_pulse_height_y):
                v = float(np.clip(v, -self.max_pulse_height_y, self.max_pulse_height_y))
                self.log(f"Y2 clamped to {v:.5e}")
            self.discharge_pair.Y(v)
        return self.discharge_pair.Y()

    def W(self, w: float | None = None) -> float | None:
        if not self.is_discharge():
            return None
        if w is not None:
            if not (self.min_pulse_width <= w <= self.max_pulse_width):
                w = float(np.clip(w, self.min_pulse_width, self.max_pulse_width))
                self.log(f"W clamped to {w:.5e}")
            self.discharge_pair.W(w)
        return self.discharge_pair.W()

    """Sweep control"""

    def take_sweep(self, high_resolution: bool = False):
        multiplier = self.high_resolution_sweep_multiplier if high_resolution else 1
        self.averager.take_curve(multiplier)

    def get_curve_data(self) -> wfCurveData:
        return self.averager.get_curve_data()

    def get_G(self) -> float:
        return self.cap_filter.get_G()

    def get_Cac(self) -> float:
        return self.cap_filter.get_Cac()

    def get_dMdW(self) -> float:
        return self.dis_filter.get_dMdW()

    """Handling for Polarity Changes"""

    def reset_for_polarity_switch(self, W0: float | None = None):
        positive = self.X1() > 0

        if positive:
            self.reinitialize_positive_filter()
        else:
            self.reinitialize_negative_filter()

        if W0 is None:
            W0 = self.dis_filter._home_W
        W0 = self.W(W0)

        if self.extrapolator is not None:
            self.extrapolator.clear()
            self.extrapolator.push_W0(W0)

        self.log(
            f"Polarity switch to {'positive' if positive else 'negative'}. "
            f"W set to {W0:.5e}. Filters reinitialized. Extrapolator histories cleared."
        )

    """Serialization"""

    @classmethod
    def from_json(cls, path: str) -> 'TDPTContext':
        with open(path, 'r') as f:
            conf = json.load(f)
        return cls.from_config(conf)

    @classmethod
    def from_config(cls, conf: Dict[str, Any]) -> 'TDPTContext':
        device_refs = conf.get('device_refs', {})

        averager          = resolve_reference(device_refs['averager'])
        charge_pair       = resolve_reference(device_refs['charge_pair'])
        discharge_pair    = resolve_reference(device_refs['discharge_pair']) \
            if device_refs.get('discharge_pair') else None
        capacitance_error = resolve_reference(device_refs['capacitance_error'])
        discharge_error   = resolve_reference(device_refs['discharge_error']) \
            if device_refs.get('discharge_error') else None
        integrated_error  = resolve_reference(device_refs['integrated_error']) \
            if device_refs.get('integrated_error') else None

        kwargs = conf.get('TDPTContext_kwargs', {})
        obj = cls(
            averager          = averager,
            charge_pair       = charge_pair,
            discharge_pair    = discharge_pair,
            capacitance_error = capacitance_error,
            discharge_error   = discharge_error,
            integrated_error  = integrated_error,
            **kwargs,
        )
        obj._deserialize_state(conf)
        return obj

    def load_state_json(self, path: str):
        with open(path, 'r') as f:
            state = json.load(f)
        self._deserialize_state(state)

    def save_state_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._serialize_state(), f, indent=2, cls=_NumpyEncoder)

    def _serialize_state(self) -> dict:
        result = {
            'TDPTContext_kwargs': dict(
                cap_balance_threshold        = float(self.cap_balance_threshold),
                dis_balance_threshold        = float(self.dis_balance_threshold),
                int_balance_threshold        = float(self.int_balance_threshold),
                max_pulse_height_x           = float(self.max_pulse_height_x),
                max_pulse_height_y           = float(self.max_pulse_height_y),
                discharge_X_ratio            = float(self.discharge_X_ratio) \
                    if self.discharge_X_ratio is not None else None,
                min_pulse_width              = float(self.min_pulse_width),
                max_pulse_width              = float(self.max_pulse_width),
                pulse_width_resolution       = float(self.pulse_width_resolution),
                dis_amp_adjustment_threshold = float(self.dis_amp_adjustment_threshold),
                dbal_dev                     = float(self.dbal_dev),
                max_Y1_step                  = float(self.max_Y1_step),
                max_Y2_step                  = float(self.max_Y2_step),
                max_Y2_int_step              = float(self.max_Y2_int_step),
                settle_time                  = float(self.settle_time),
                max_iterations               = int(self.max_iterations),
                good_sweep_threshold         = int(self.good_sweep_threshold),
                high_resolution_sweep_multiplier = int(self.high_resolution_sweep_multiplier),
                dis_extrap_support           = int(self.dis_extrap_support),
                dis_extrap_order             = int(self.dis_extrap_order),
                dis_eta                      = int(self.dis_eta),
            ),
            'device_refs': dict(
                averager          = format_reference(self.averager),
                charge_pair       = format_reference(self.charge_pair),
                discharge_pair    = format_reference(self.discharge_pair) \
                    if self.discharge_pair else None,
                capacitance_error = format_reference(self.capacitance_error),
                discharge_error   = format_reference(self.discharge_error) \
                    if self.discharge_error else None,
                integrated_error  = format_reference(self.integrated_error) \
                    if self.integrated_error else None,
            ),
            'health': dict(
                _rail_hi_streak        = self._rail_hi_streak,
                _rail_lo_streak        = self._rail_lo_streak,
                _failure_streak        = self._failure_streak,
                _consecutive_good      = self._consecutive_good,
                _next_Y2_ratio_override = self._next_Y2_ratio_override,
            ),
        }

        if self.cap_filter is not None:
            result.update(self.cap_filter._serialize_state())
        if self.dis_filter is not None:
            result.update(self.dis_filter._serialize_state())
        if self.extrapolator is not None:
            result.update(self.extrapolator._serialize_state())

        return result

    def _deserialize_state(self, state: dict):
        # Restore context kwargs
        for key, value in state.get('TDPTContext_kwargs', {}).items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Restore health counters
        health = state.get('health', {})
        self._rail_hi_streak   = health.get('_rail_hi_streak',   0)
        self._rail_lo_streak   = health.get('_rail_lo_streak',   0)
        self._failure_streak   = health.get('_failure_streak',   0)
        self._consecutive_good = health.get('_consecutive_good', 0)
        self._next_Y2_ratio_override = health.get('_next_Y2_ratio_override', None)

        # Restore filters
        if state.get('CapacitanceFilter_kwargs') is not None:
            if self.cap_filter is None:
                self.cap_filter = CapacitanceFilter(self)
            self.cap_filter._deserialize_state(
                state['CapacitanceFilter_kwargs']
            )

        if state.get('DischargeFilter_kwargs') is not None:
            if self.dis_filter is None:
                self.dis_filter = DischargeFilter(self)
            self.dis_filter._deserialize_state(
                state['DischargeFilter_kwargs']
            )

        # Restore extrapolator, reconstructing with current context params
        # if it doesn't exist yet
        if state.get('DischargeExtrapolator_kwargs') is not None:
            if self.extrapolator is None and self.is_discharge():
                self.extrapolator = DischargeExtrapolator(
                    extrap_support = self.dis_extrap_support,
                    extrap_order   = self.dis_extrap_order,
                    W_min          = self.min_pulse_width,
                    W_max          = self.max_pulse_width,
                )
            if self.extrapolator is not None:
                self.extrapolator._deserialize_state(
                    state['DischargeExtrapolator_kwargs']
                )