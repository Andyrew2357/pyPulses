from ..pyPulses.routines.balance_parameter import balanceKnob
from ..pyPulses.routines.linear_balance import lstsqBalance, bracket1d

from ..pyPulses.devices.registry import resolve_reference, format_reference
from ..pyPulses.devices.wfatd import wfAverager, wfBalance, wfCurveData
from ..pyPulses.devices.pulse_pair import pulsePair
from ..pyPulses.utils import kalman
from ..pyPulses.thread_job import _checkpoint

from dataclasses import dataclass
import numpy as np
import logging
import time
import json

from typing import Any, Callable, Dict, Tuple

class _TDPT_CONF():
    """There are some magic numbers that we wrap in a little config"""
    HEURISTIC_NOISE_MULT    : float = 1.0e-3
    DIS_INIT_BAL_UNC        : float = 0.0
    DIS_INIT_EXC_UNC        : float = 0.0
    DIS_INIT_P_MULT         : float = 1.0e-4
    CAP_INIT_BAL_QG_MULT    : float = 1.0e-3
    CAP_INIT_BAL_QC         : float = 1.0e-3
    CAP_INIT_BAL_QdC        : float = 1.0e-4
    CAP_INIT_EXC_QG_MULT    : float = 1.0e-2
    CAP_INIT_EXC_QC         : float = 1.0e-3
    CAP_INIT_EXC_QdC        : float = 1.0e-3
    CAP_INIT_PG_MULT        : float = 0.1
    CAP_INIT_PC             : float = 1.0e-3
    CAP_INIT_PdC            : float = 1.0
    dMdW_R_MULT             : float = 4.0
    INT_ADJ_LAMBDA          : float = 1.0

@dataclass
class TDPTContext: ...

@dataclass
class TDPT_control_state: ...

@dataclass
class TDPT_error_state: ...

@dataclass
class TDPT_optimal_state: ...

@dataclass
class TDPT_initial_filter_config(): ...

class DischargeFilter(): ...

class CapacitanceFilter(): ...

TDPT_POST_CALLBACK = Callable[[TDPTContext, TDPT_control_state, TDPT_error_state, TDPT_optimal_state, wfCurveData], Any]
TDPT_PRE_CALLBACK = Callable[[TDPTContext, TDPT_control_state, TDPT_error_state, TDPT_optimal_state], Any]

@dataclass
class TDPTContext():

    averager: wfAverager
    charge_pair: pulsePair
    
    capacitance_error: wfBalance
    capacitance_error_threshold: float = 1e-3
    max_pulse_height: float = 50.0e-3

    # If no discharge pulses are needed, these can be None
    discharge_pair: pulsePair | None = None    
    discharge_error: wfBalance | None = None
    discharge_error_threshold: float = 40.0
    discharge_X_ratio: float | None = None
    min_pulse_width: float = 1.0e-6
    max_pulse_width: float = 5.0e-6
    pulse_width_resolution: float = 2.7e-10

    # If no discharge pulses are needed, or integrated errors are not a concern,
    # these can be None
    integrated_error: wfBalance | None = None
    integrated_error_threshold: float = 1.0e-6

    # Maximum discharge pulse width setting beyond which we consider changing
    # the Y2 amplitude
    dis_amp_adjustment_threshold: float = 4.0e-9

    # Maximum steps to take in the controls for refinements
    max_Y1_refinement: float = 2.0e-3
    max_Y2_refinement: float = 2.0e-3
    max_Y2_int_refinement: float = 1.0e-3
    max_W_refinement: float = 5.0e-6

    settle_time: float = 0.2
    max_refinements: float = 3
    good_sweep_threshold: int = 1 # Number of consecutive good sweeps needed

    # Various checks to avoid dawdling over discharge conditions
    bracket_rut_condition: int = 5 # Extra condition to decide that we have amalformed width bracket
    bracket_min_width: float = 1.0e-8 # If the width bracket gets this small, we should clear one bound
    max_dis_width_collisions: int = 3 # If the width is railed this many times without satisfaction, discharge balance is probably infeasible

    high_resolution_sweep_multiplier: int = 1
    dis_extrap_support: int = 7
    dis_extrap_order: int = 2
    dis_eta: int = -1 # Relative sign difference between charge and discharge pulse height conventions

    pre_initial_trace_callback: Callable[[TDPTContext, TDPT_control_state], Any] | None = None
    post_initial_trace_callback: TDPT_POST_CALLBACK |  None = None
    pre_refinement_callback: TDPT_PRE_CALLBACK | None = None
    post_refinement_callback: TDPT_POST_CALLBACK |  None = None

    logger: logging.Logger | None = None

    def __post_init__(self):
        self.dis_filter: DischargeFilter | None = None
        self.cap_filter: CapacitanceFilter | None = None
        self._dis_width_bracket: bracket1d | None = None
        self._previous_result: TDPT_filter_balance_result | None = None

    @classmethod
    def from_json(cls, path: str) -> 'TDPTContext':
        with open(path, 'r') as f:
            conf = json.load(f)
        return cls.from_config(conf)

    @classmethod
    def from_config(cls, conf: Dict[str, Any]) -> 'TDPTContext':
                
        device_refs = conf.get('device_refs', {})
        
        # Resolve device references from registries
        averager = resolve_reference(device_refs['averager'])
        charge_pair = resolve_reference(device_refs['charge_pair'])
        
        discharge_pair = None
        if device_refs.get('discharge_pair'):
            discharge_pair = resolve_reference(device_refs['discharge_pair'])
        
        capacitance_error = resolve_reference(device_refs['capacitance_error'])
        
        discharge_error = None
        if device_refs.get('discharge_error'):
            discharge_error = resolve_reference(device_refs['discharge_error'])
        
        integrated_error = None
        if device_refs.get('integrated_error'):
            integrated_error = resolve_reference(device_refs['integrated_error'])

        # Get kwargs, using defaults from dataclass if not present
        kwargs = conf.get('TDPTContext_kwargs', {})

        obj = cls(
            averager=averager,
            charge_pair=charge_pair,
            discharge_pair=discharge_pair,
            capacitance_error=capacitance_error,
            discharge_error=discharge_error,
            integrated_error=integrated_error,
            **kwargs,
        )
        
        # Restore filter states
        obj._deserialize_state(conf)
        
        return obj
    
    def load_state_json(self, path: str):
        with open(path, 'r') as f:
            state = json.load(f)
        self._deserialize_state(state)
    
    def save_state_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._serialize_state(), f, indent=2)

    def _serialize_state(self) -> dict:
        result = {
            'TDPTContext_kwargs': dict(
                capacitance_error_threshold = float(self.capacitance_error_threshold),
                max_pulse_height = float(self.max_pulse_height),
                discharge_error_threshold = float(self.discharge_error_threshold),
                discharge_X_ratio = float(self.discharge_X_ratio) \
                    if self.discharge_X_ratio is not None else None,
                min_pulse_width = float(self.min_pulse_width),
                max_pulse_width = float(self.max_pulse_width),
                pulse_width_resolution = float(self.pulse_width_resolution),
                integrated_error_threshold = float(self.integrated_error_threshold),
                dis_amp_adjustment_threshold = float(self.dis_amp_adjustment_threshold),
                max_Y1_refinement = float(self.max_Y1_refinement),
                max_Y2_refinement = float(self.max_Y2_refinement),
                max_Y2_int_refinement = float(self.max_Y2_int_refinement),
                max_W_refinement = float(self.max_W_refinement),
                settle_time = float(self.settle_time),
                max_refinements = int(self.max_refinements),
                good_sweep_threshold = int(self.good_sweep_threshold),
                bracket_rut_condition = int(self.bracket_rut_condition),
                bracket_min_width = float(self.bracket_min_width),
                max_dis_width_collisions = int(self.max_dis_width_collisions),
                high_resolution_sweep_multiplier = int(self.high_resolution_sweep_multiplier),
                dis_extrap_support = int(self.dis_extrap_support),
                dis_extrap_order = int(self.dis_extrap_order),
                dis_eta = int(self.dis_eta),
            ),
            'device_refs': dict(
                averager = format_reference(self.averager),
                charge_pair = format_reference(self.charge_pair),
                discharge_pair = format_reference(self.discharge_pair) \
                    if self.discharge_pair else None,
                capacitance_error = format_reference(self.capacitance_error),
                discharge_error = format_reference(self.discharge_error) \
                    if self.discharge_error else None,
                integrated_error = format_reference(self.integrated_error) \
                    if self.integrated_error else None,
            ),
        }
        if self.cap_filter is not None:
            result.update(self.cap_filter._serialize_state())
        if self.dis_filter is not None:
            result.update(self.dis_filter._serialize_state())

        return result
    
    def _deserialize_state(self, state: dict):
        TDPTContext_kwargs = state.get('TDPTContext_kwargs')
        if TDPTContext_kwargs is not None:
            for key, value in TDPTContext_kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        CapacitanceFilter_kwargs = state.get('CapacitanceFilter_kwargs')
        if CapacitanceFilter_kwargs is not None:
            if self.cap_filter is None:
                self.cap_filter = CapacitanceFilter(self)
            self.cap_filter._deserialize_state(CapacitanceFilter_kwargs)
        
        DischargeFilter_kwargs = state.get('DischargeFilter_kwargs')
        if DischargeFilter_kwargs is not None:
            if self.dis_filter is None:
                self.dis_filter = DischargeFilter(self)
            self.dis_filter._deserialize_state(DischargeFilter_kwargs)

    def log(self, *args, **kwargs):
        if self.logger:
            self.logger.info(*args, **kwargs)

    """Filter Actions"""

    def add_discharge_filter(self, filter: DischargeFilter):
        self.dis_filter = filter
        filter.ctx = self

    def add_capacitance_filter(self, filter: CapacitanceFilter):
        self.cap_filter = filter
        filter.ctx = self

    def initialize_filters(self, 
        pos_config: TDPT_initial_filter_config, 
        neg_config: TDPT_initial_filter_config,
    ):
        self.initialize_positive_filter(pos_config)
        self.initialize_negative_filter(neg_config)

    def initialize_positive_filter(self, config: TDPT_initial_filter_config):
        if self.cap_filter is None:
            self.cap_filter = CapacitanceFilter(self)
        self.cap_filter.set_positive_init_params(**config.cap_filter_init_parms)

        if config.dis_filter_init_parms is not None:
            if self.dis_filter is None:
                self.dis_filter = DischargeFilter(self)
            self.dis_filter.set_positive_init_params(**config.dis_filter_init_parms)

    def initialize_negative_filter(self, config: TDPT_initial_filter_config):
        if self.cap_filter is None:
            self.cap_filter = CapacitanceFilter(self)
        self.cap_filter.set_negative_init_params(**config.cap_filter_init_parms)

        if config.dis_filter_init_parms is not None:
            if self.dis_filter is None:
                self.dis_filter = DischargeFilter(self)
            self.dis_filter.set_negative_init_params(**config.dis_filter_init_parms)

    def reinitialize_positive_filter(self):
        self.cap_filter.positive_initialize()
        if self.dis_filter is not None:
            self.dis_filter.positive_initialize()

    def reinitialize_negative_filter(self):
        self.cap_filter.negative_initialize()
        if self.dis_filter is not None:
            self.dis_filter.negative_initialize()

    def log_capacitance_filter_status(self):
        self.log(
            "AC Capacitance Filter Status:\n"
            f"{self.cap_filter}"
        )

    def log_discharge_filter_status(self):
        self.log(
            "Discharge Filter Status:\n"
            f"{self.dis_filter}"
        )

    def cap_process(self, balance_change: bool = True):
        self.cap_filter.predict(balance_change)

    def dis_process(self, balance_change: bool = True):
        self.dis_filter.predict(balance_change)

    def cap_update(self, dQ: float, dQ_R: float):
        self.cap_filter.update(dQ, dQ_R)

    def dis_update(self, o_dMdW: float, o_dMdW_R: float):
        self.dis_filter.update(o_dMdW, o_dMdW_R)

    def extrapolate_W0(self) -> float:
        return self.dis_filter.extrapolate_W0()

    """Excitation Parameters"""

    def X1(self, v: float | None = None) -> float:
        """
        Amplitude of the X side of the charging pulse pair
        """

        if v is not None:
            if not (-self.max_pulse_height <= v <= self.max_pulse_height):
                tv = min(self.max_pulse_height, max(-self.max_pulse_height, v))
                self.log(
                    f"Requested pulse X1 ({v:.5e}) falls outside the allowed range.\n"
                    f"Truncating to {tv:.5e}."
                )
                v = tv
            self.charge_pair.X(v)

        set_v = self.charge_pair.X()
        if self.cap_filter:
            self.cap_filter.Xamp = set_v
        
        return set_v
    
    def Y1(self, v: float | None = None, v0: float | None = None) -> float:
        """
        Amplitude of the Y side of the charging pulse pair
        """
        
        if v is not None:
            if (v0 is not None) and abs(v - v0) > self.max_Y1_refinement:
                tv = min(v0 + self.max_Y1_refinement, max(v0 - self.max_Y1_refinement, v))
                self.log(
                    f"Requested change in Y1 from {v0:.5e} to {v:.5e} exceeds the"
                    f" maximum allowed refinement ({self.max_Y1_refinement:.5e})\n"
                    f"Truncating to {tv:.5e}."
                )
                v = tv
            if not (-self.max_pulse_height <= v <= self.max_pulse_height):
                tv = min(self.max_pulse_height, max(-self.max_pulse_height, v))
                self.log(
                    f"Requested pulse Y1 ({v:.5e}) falls outside the allowed range.\n"
                    f"Truncating to {tv:.5e}."
                )
                v = tv
            self.charge_pair.Y(v)

        set_v = self.charge_pair.Y()
        if self.cap_filter:
            self.cap_filter.Yamp = set_v
        
        return set_v
    
    def X2(self, v: float | None = None) -> float:
        """
        Amplitude of the X side of the discharging pulse pair
        """
        if self.discharge_pair is None:
            return

        if v is not None:
            if not (-self.max_pulse_height <= v <= self.max_pulse_height):
                tv = min(self.max_pulse_height, max(-self.max_pulse_height, v))
                self.log(
                    f"Requested pulse X2 ({v:.5e}) falls outside the allowed range.\n"
                    f"Truncating to {tv:.5e}."
                )
                v = tv
            self.discharge_pair.X(v)
        return self.discharge_pair.X()
    
    def Y2(self, v: float | None = None, v0: float | None = None) -> float:
        """
        Amplitude of the Y side of the discharging pulse pair
        """
        if self.discharge_pair is None:
            return

        if v is not None:
            if (v0 is not None) and abs(v - v0) > self.max_Y2_refinement:
                tv = min(v0 + self.max_Y2_refinement, max(v0 - self.max_Y2_refinement, v))
                self.log(
                    f"Requested change in Y2 from {v0:.5e} to {v:.5e} exceeds the"
                    f" maximum allowed refinement ({self.max_Y2_refinement:.5e})\n"
                    f"Truncating to {tv:.5e}."
                )
                v = tv
            if not (-self.max_pulse_height <= v <= self.max_pulse_height):
                tv = min(self.max_pulse_height, max(-self.max_pulse_height, v))
                self.log(
                    f"Requested pulse Y2 ({v:.5e}) falls outside the allowed range.\n"
                    f"Truncating to {tv:.5e}."
                )
                v = tv
            self.discharge_pair.Y(v)
        return self.discharge_pair.Y()
    
    def set_Y2_integrated_balance(self, v: float, v0: float):
        if abs(v - v0) > self.max_Y2_refinement:
            tv = min(v0 + self.max_Y2_int_refinement, max(v0 - self.max_Y2_int_refinement, v))
            self.log(
                f"Requested change in Y2 from {v0:.5e} to {v:.5e} exceeds the"
                f" maximum allowed refinement for an integrated balance ({self.max_Y2_int_refinement:.5e})\n"
                f"Truncating to {tv:.5e}."
            )
            v = tv
        self.discharge_pair.Y(v)
        return self.discharge_pair.Y()

    def W(self, w: float | None = None, w0: float | None = None) -> float:
        """
        Width of the discharging pulse pair
        """

        if w is not None:
            if (w0 is not None) and abs(w - w0) > self.max_W_refinement:
                tw = min(w0 + self.max_W_refinement, max(w0 - self.max_W_refinement, w))
                self.log(
                    f"Requested change in W from {w0:.5e} to {w:.5e} exceeds the"
                    f" maximum allowed refinement ({self.max_W_refinement:.5e})\n"
                    f"Truncating to {tw:.5e}."
                )
                w = tw
            if not (self.min_pulse_width <= w <= self.max_pulse_width):
                tw = min(self.max_pulse_width, max(self.min_pulse_width, w))
                self.log(
                    f"Requested pulse W ({w:.5e}) falls outside the allowed range.\n"
                    f"Truncating to {tw:.5e}."
                )
                w = tw
            self.discharge_pair.W(w)
        
        return self.discharge_pair.W()
    
    """Utilities"""

    def take_sweep(self, high_resolution: bool):
        self.averager.take_curve( \
            self.high_resolution_sweep_multiplier if high_resolution else 1)
        
    def get_curve_data(self) -> wfCurveData:
        return self.averager.get_curve_data()

    def predict_filter_change(self, balance_change: bool = True):
        self.cap_filter.predict(balance_change)
        if self.is_discharge():
            self.dis_filter.predict(balance_change)

    def get_G(self) -> float:
        return self.cap_filter.get_G()

    def get_Cac(self) -> float:
        return self.cap_filter.get_Cac()

    def get_dMdW(self) -> float:
        return self.dis_filter.get_dMdW()

    def is_discharge(self) -> bool:
        return self.discharge_pair is not None
    
    def is_integrated(self) -> bool:
        return self.integrated_error is not None
    
    def set_extrap_dis_width(self) -> float:
        if len(self.dis_filter.W0_history) == 0:
            return self.W()
        W0extrap = self.dis_filter.extrapolate_W0()
        return self.W(W0extrap)
    
    def mean_dis_width(self) -> float:
        if len(self.dis_filter.W0_history) == 0:
            return self.W()
        return float(np.mean(self.dis_filter.W0_history))

    def clear_dis_width_bracket(self):
        del self._dis_width_bracket
        self._dis_width_bracket = bracket1d(
            self.min_pulse_width, 
            self.max_pulse_width,
            padding = self.pulse_width_resolution,
        )

    def update_dis_width_bracket(self, W: float, M: float):
        return self._dis_width_bracket.update(W, M, self.get_dMdW())

    def iterate_dis_width_bracket(self) -> Tuple[float | None, bool]:
        return self._dis_width_bracket.iterate(self.get_dMdW())
    
    def detect_bracket_rut(self, itr: int, good_count: int):
        if itr - good_count > self.bracket_rut_condition:
            return True
        span = self._dis_width_bracket.span()
        return (span is not None) and (span < self.bracket_min_width)
    
    def set_extrap_discharge_Y(self) -> float:
        if len(self.dis_filter.Y2_history) == 0:
            return self.Y2(-self.get_Cac() * self.X2())
        Yextrap = self.dis_filter.extrapolate_Y2()
        return self.set_Y2_integrated_balance(Yextrap, self.dis_filter.Y2_history[-1])

def _extrap(x: np.ndarray, order: int) -> float:
    coeff = np.polyfit(np.arange(len(x)), x, min(order, len(x) - 1))
    poly = np.poly1d(coeff)
    return float(poly(len(x)))

class DischargeFilter():
    """
    Manages a Kalman Filter for dMdW and keeps track of discharge pulse width
    history to extrapolate new balance points.
    """

    def __init__(self,ctx: TDPTContext | None = None):
        
        self.ctx = ctx # To which context are we tied
        
        # Generally the filters for a positive and negative excitation are distinct
        self.is_positive_initialized = False
        self.is_negative_initialized = False
        self._positive_init_params: dict | None = None
        self._negative_init_params: dict | None = None

        # How to treat process noise
        self.use_heuristic_process_noise = True
        self._previous_dMdW: float | None = None

        # These are unused if we use a heuristic process noise
        self.balance_change_uncertainty: float | None = None # process noise on a balance change
        self.excitation_change_uncertainty: float| None = None # process noise on an excitation change

        # Since this filter is trivial, I forgo using my ordinary Kalman
        # filter class and implement it directly.
        self.dMdW: float | None = None # state
        self.P: float | None = None # state uncertainty

        self.W0_history = []
        self.Y2_history = []

    def add_context(self, ctx: TDPTContext):
        self.ctx = ctx
        ctx.dis_filter = self

    def _initialize(self, init_params: dict):
        self.balance_change_uncertainty = init_params.get('balance_change_uncertainty')
        self.excitation_change_uncertainty = init_params.get('excitation_change_uncertainty')
        self.dMdW = init_params.get('dMdW')
        self._previous_dMdW = self.dMdW
        self.P = init_params.get('P')
        self.W0_history = []
        if 'W0' in init_params:
            self.W0_history.append(init_params.get('W0'))
        self.Y2_history = []

    def positive_initialize(self):
        """
        Set up the filter parameters to the initial conditions for a positive excitation.
        """

        if self.is_positive_initialized:
            return
        if self._positive_init_params is None:
            raise RuntimeError(
                "CapacitanceFilter has no positive initialization parameters to invoke."
            )
        self._initialize(self._positive_init_params)
        self.is_negative_initialized = False
        self.is_positive_initialized = True

    def negative_initialize(self):
        """
        Set up the filter parameters to the initial conditions for a negative excitation.
        """

        if self.is_negative_initialized:
            return
        if self._negative_init_params is None:
            raise RuntimeError(
                "CapacitanceFilter has no negative initialization parameters to invoke."
            )
        self._initialize(self._negative_init_params)
        self.is_negative_initialized = True
        self.is_positive_initialized = False

    def set_positive_init_params(self,
        bal_change_Q: float, 
        exc_change_Q: float,
        dMdW: float,
        P: float,
        W0: float,
    ):
        self._positive_init_params = {
            'balance_change_uncertainty': bal_change_Q,
            'excitation_change_uncertainty': exc_change_Q,
            'dMdW': dMdW,
            'P': P,
            'W0': W0,
        }

    def set_negative_init_params(self,
        bal_change_Q: float, 
        exc_change_Q: float,
        dMdW: float,
        P: float,
        W0: float,
    ):
        self._negative_init_params = {
            'balance_change_uncertainty': bal_change_Q,
            'excitation_change_uncertainty': exc_change_Q,
            'dMdW': dMdW,
            'P': P,
            'W0': W0,
        }

    def load_state_json(self, path: str):
        self._deserialize_state(json.load(path).get('DischargeFilter_kwargs'))
        
    def save_state_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._serialize_state(), f, indent=2)
    
    def _serialize_state(self) -> dict:
        return {
            'DischargeFilter_kwargs': dict(
                is_positive_initialized = self.is_positive_initialized,
                is_negative_initialized = self.is_negative_initialized,
                _positive_init_params = self._positive_init_params,
                _negative_init_params = self._negative_init_params,
                use_heuristic_process_noise = self.use_heuristic_process_noise,
                _previous_dMdW = self._previous_dMdW,
                balance_change_uncertainty = self.balance_change_uncertainty,
                excitation_change_uncertainty = self.excitation_change_uncertainty,
                dMdW = self.dMdW,
                P = self.P,
                W0_history = [float(v) for v in self.W0_history],
                Y2_history = [float(v) for v in self.Y2_history],
            )
        }

    def _deserialize_state(self, state: dict):
        self._positive_init_params = state.get('_positive_init_params')
        self._negative_init_params = state.get('_negative_init_params')
        self.is_positive_initialized = state.get('is_positive_initialized')
        self.is_negative_initialized = state.get('is_negative_initialized')
        if self.is_positive_initialized:
            self.positive_initialize()
        elif self.is_negative_initialized:
            self.negative_initialize()
        self.use_heuristic_process_noise = state.get('use_heuristic_process_noise')
        self._previous_dMdW = state.get('_previous_dMdW')
        self.balance_change_uncertainty = state.get('balance_change_uncertainty')
        self.excitation_change_uncertainty = state.get('excitation_change_uncertainty')
        self.dMdW = state.get('dMdW')
        self.P = state.get('P')
        self.W0_history = list(state.get('W0_history', []))
        self.Y2_history = list(state.get('Y2_history', []))

    """Kalman Filter Functionality"""

    def predict(self, 
        heuristic_noise: bool | None = None, 
        balance_change: bool = True
    ):
        """
        Performs the Kalman prediction step. This filter is trivial, so this
        just adds the process noise to P.

        Parameters
        ----------
        heuristic_noise: bool
            Whether we use the process noise parameter we're initialized with
            or instead use the heuristic noise inferred from the previous state
        balance_change : bool
            Whether this prediction is induced by a balance change
        """
        
        if balance_change:
            if heuristic_noise is True or \
                (heuristic_noise is None and self.use_heuristic_process_noise):
                self.P += _TDPT_CONF.HEURISTIC_NOISE_MULT*((self.dMdW - self._previous_dMdW)**2 + self.dMdW**2)
            else:
                self.P += self.balance_change_uncertainty
        else:
            self.P += self.excitation_change_uncertainty

    def update(self, o_dMdW: float, R: float):
        """
        Performs the Kalman update step. This filter is trivial, so the mixing
        is very simple.

        Parameters
        ----------
        o_dMdW : float
            Observed change in discharge slope with respect to discharge pulse width
        R : float
            Uncertainty in `o_dMdW`
        """

        self._previous_dMdW = self.dMdW
        K = self.P / (self.P + R)
        self.dMdW += K * (o_dMdW - self.dMdW)
        self.P *= (1 - K)


    """Balance Point Extrapolation"""

    def push_W0(self, W0: float):
        """
        Add a new discharge width balance point to the history
        
        Parameters
        ----------
        W0 : float
            New good discharge pulse width
        """
        self.W0_history.append(float(W0))

    def push_Y2(self, Y2: float):
        """
        Add a new discharge Y amplitude point to the history
        
        Parameters
        ----------
        Y2 : float
            New discharge Y amplitude
        """
        self.Y2_history.append(float(Y2))

    def extrapolate_W0(self) -> float:
        """
        Extrapolate the new discharge balance point from previous balance points

        Returns
        -------
        float
        """

        # Truncate to support
        self.W0_history = self.W0_history[-self.ctx.dis_extrap_support:]
        return _extrap(self.W0_history, self.ctx.dis_extrap_order)
    
    def extrapolate_Y2(self) -> float:
        """
        Extrapolate the new discharge Y amplitude from previous balance points

        Returns
        -------
        float
        """

        # Truncate to support
        self.Y2_history = self.Y2_history[-self.ctx.dis_extrap_support:]
        return _extrap(self.Y2_history, self.ctx.dis_extrap_order)

    """Utilities"""

    def get_dMdW(self) -> float:
        return self.dMdW

    def __str__(self) -> str:
        return (
            f"dM/dW = {self.dMdW:.5e} +- {np.sqrt(self.P):.5e}\n"
            f"W0 history: {self.W0_history}\n"
            f"Y2 history: {self.Y2_history}\n"
        )
    

class CapacitanceFilter():
    """
    Manages a Kalman Filter for G, Cac, and dCac. Semantically, these are the
    effective gain from the balance point to the scope, the AC Capacitance ratio,
    and the change in AC capacitance ratio upon a change in excitation settings.
    """

    def __init__(self, ctx: TDPTContext | None = None):

        self.ctx = ctx

        # Generally, the filters for positive and negative excitations are distinct
        self.is_negative_initialized = False
        self.is_positive_initialized = False
        self._positive_init_params: dict | None = None
        self._negative_init_params: dict | None = None
        self.kfilter: kalman | None = None
        self.balance_change_Q: np.ndarray | None = None
        self.excitation_change_Q: np.ndarray | None = None

        # Current state of the relevant excitation parameters
        self.Xamp: float | None = None
        self.Yamp: float | None = None


    def add_context(self, ctx: TDPTContext):
        self.ctx = ctx
        ctx.cap_filter = self

    def _initialize(self, init_params: dict):
        self.balance_change_Q = init_params.get('balance_change_Q')
        self.excitation_change_Q = init_params.get('excitation_change_Q')
        self.kfilter = kalman(
            f_F = self._f_F,
            h_H = self._h_H,
            x = init_params.get('x_init'),
            P = init_params.get('P_init'),
        )

    def positive_initialize(self):
        """
        Set up the filter parameters to the initial conditions for a positive excitation.
        """

        if self.is_positive_initialized:
            return
        if self._positive_init_params is None:
            raise RuntimeError(
                "CapacitanceFilter has no positive initialization parameters to invoke."
            )
        self._initialize(self._positive_init_params)
        self.is_negative_initialized = False
        self.is_positive_initialized = True

    def negative_initialize(self):
        """
        Set up the filter parameters to the initial conditions for a negative excitation.
        """

        if self.is_negative_initialized:
            return
        if self._negative_init_params is None:
            raise RuntimeError(
                "CapacitanceFilter has no negative initialization parameters to invoke."
            )
        self._initialize(self._negative_init_params)
        self.is_negative_initialized = True
        self.is_positive_initialized = False

    def set_positive_init_params(self, 
        bal_change_Q: np.ndarray,
        exc_change_Q: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
    ):
        self._positive_init_params = {
            'balance_change_Q': bal_change_Q.reshape((3, 3)),
            'excitation_change_Q': exc_change_Q.reshape((3, 3)),
            'x_init': x.reshape((3, 1)),
            'P_init': np.diag(P.flatten()).reshape((3, 3)),
        }

    def set_negative_init_params(self, 
        bal_change_Q: np.ndarray,
        exc_change_Q: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
    ):
        self._negative_init_params = {
            'balance_change_Q': bal_change_Q.reshape((3, 3)),
            'excitation_change_Q': exc_change_Q.reshape((3, 3)),
            'x_init': x.reshape((3, 1)),
            'P_init': np.diag(P.flatten()).reshape((3, 3)),
        }   
    
    def load_state_json(self, path: str):
        self._deserialize_state(json.load(path).get('CapacitanceFilter_kwargs'))
        
    def save_state_json(self, path: str):
        json.dump(self._serialize_state(), path)

    def _serialize_state(self) -> dict:
        def _to_list(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr
        
        return {
            'CapacitanceFilter_kwargs': dict(
                is_negative_initialized = self.is_negative_initialized,
                is_positive_initialized = self.is_positive_initialized,
                _positive_init_params = {k: _to_list(v) for k, v in self._positive_init_params.items()} \
                    if self._positive_init_params else None,
                _negative_init_params = {k: _to_list(v) for k, v in self._negative_init_params.items()} \
                    if self._negative_init_params else None,
                balance_change_Q = _to_list(self.balance_change_Q),
                excitation_change_Q = _to_list(self.excitation_change_Q),
                Xamp = self.Xamp,
                Yamp = self.Yamp,
                kfilter_x = self.kfilter.x.tolist() if self.kfilter else None,
                kfilter_P = self.kfilter.P.tolist() if self.kfilter else None,
            )
        }

    def _deserialize_state(self, state: dict):
        def _to_array(val):
            if val is None:
                return None
            return np.array(val)
        
        def _restore_init_params(params):
            if params is None:
                return None
            return {k: _to_array(v) if k in ('balance_change_Q', 'excitation_change_Q', 'x_init', 'P_init') else v 
                    for k, v in params.items()}
        
        self.is_negative_initialized = state.get('is_negative_initialized')
        self.is_positive_initialized = state.get('is_positive_initialized')
        self._positive_init_params = _restore_init_params(state.get('_positive_init_params'))
        self._negative_init_params = _restore_init_params(state.get('_negative_init_params'))
        
        if self.is_negative_initialized:
            self.negative_initialize()
        elif self.is_positive_initialized:
            self.positive_initialize()
        
        self.balance_change_Q = _to_array(state.get('balance_change_Q'))
        self.excitation_change_Q = _to_array(state.get('excitation_change_Q'))
        self.Xamp = state.get('Xamp')
        self.Yamp = state.get('Yamp')
        
        if self.kfilter and state.get('kfilter_x') is not None:
            self.kfilter.x = np.array(state.get('kfilter_x'))
            self.kfilter.P = np.array(state.get('kfilter_P'))

    """Kalman Filter Functionality"""

    def _f_F(self, X:np.ndarray, balance_change: bool = True):
        X[1, 0] += X[2, 0] # Cac -> Cac + dCac
        F = np.eye(3)
        if balance_change:
            Q = self.balance_change_Q
        else:
            Q = self.excitation_change_Q
            F[1, 2] += 1
        return X, F, Q

    def _h_H(self, X: np.ndarray, Xamp: float, Yamp: float):
        G, Cac, dCac = X.flatten()
        z = np.array([[G * (Yamp + Cac * Xamp)]])
        H = np.array([[Yamp + Cac * Xamp, G * Xamp, 0]])
        return z, H

    def predict(self, balance_change: bool = True):
        """
        Performs the Kalman prediction step.

        Parameters
        ----------
        balance_change : bool
            Whether this prediction is induced by a balance change
        """
        self.kfilter.predict(balance_change)

    def update(self, 
        dQ: float, R: float, 
        Xamp: float | None = None, 
        Yamp: float | None = None,
    ):
        """
        Performs the Kalman update step. 

        Parameters
        ----------
        dQ : float
            Observed capacitive jump at the start of the excitation pulse
        R : float
            Uncertainty in `dQ`
        Xamp : float
            amplitude of the X side of the excitation pulse
        Yamp: float
            amplitude of the X side of the excitation pulse
        """
        
        if Xamp is None:
            Xamp = self.Xamp
        if Yamp is None:
            Yamp = self.Yamp

        # We prevent any unphyiscal changes in the sign of G or Cac
        pG, pCac, _ = self.kfilter.x.flatten()
        self.kfilter.update(np.array([[dQ]]), np.array([[R]]), Xamp, Yamp)
        G, Cac, _ = self.kfilter.x.flatten()
        
        if pCac * Cac < 0:
            self.ctx.log(
                f"WARNING: Requested sign change in Cac estimate is ignored\n"
                f"         Was {pCac:.5e}, requested {Cac:.5e}."
            )
            self.kfilter.x[1, 0] = pCac

        if pG * G < 0:
            self.ctx.log(
                f"WARNING: Requested sign change in G estimate is ignored\n"
                f"         Was {pG:.5e}, requested {G:.5e}."
            )
            self.cfilter.x[0, 0] = pG

    """Utilities"""

    def get_state(self) -> Tuple[float, float, float]:
        return tuple(self.kfilter.x.flatten())

    def get_G(self) -> float:
        return self.kfilter.x[0, 0]

    def get_Cac(self) -> float:
        return self.kfilter.x[1, 0]

    def __str__(self) -> str:
        G, Cac, dCac = self.kfilter.x.flatten()
        PG, PCac, PdCac = self.kfilter.P.diagonal()
        P = self.kfilter.P
        return (
            f"G = {G:.5e} +- {np.sqrt(PG):.5e}\n"
            f"Cac = {Cac:.5e} +- {np.sqrt(PCac):.5e}\n"
            f"dCac = {dCac:.5e} +- {np.sqrt(PdCac):.5e}\n"
            f"     Covariance: ┏                                      ┓\n"
            f"                 ┃ {f"{P[0,0]:.5e}":<12}{f"{P[0,1]:.5e}":>12}{f"{P[0,2]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{P[1,0]:.5e}":<12}{f"{P[1,1]:.5e}":>12}{f"{P[1,2]:.5e}":>12} ┃\n"
            f"                 ┃ {f"{P[2,0]:.5e}":<12}{f"{P[2,1]:.5e}":>12}{f"{P[2,2]:.5e}":>12} ┃\n"
            f"                 ┗                                      ┛"
        )

@dataclass
class TDPT_initial_filter_metadata():
    X1: float
    X2: float

@dataclass
class TDPT_initial_filter_config():
    status: bool
    cap_status: bool
    dis_status: bool
    positive: bool

    cap_err: float
    cap_var: float
    cap_filter_init_parms: dict

    dis_err: float | None
    dis_var: float | None
    dis_filter_init_parms: dict | None

    meta: TDPT_initial_filter_metadata

    def __str__(self) -> str:
        r = f"Status: {'GOOD' if self.status else 'BAD'}\nCapacitance Status:" \
            f" {'GOOD' if self.cap_status else 'BAD'} ({self.cap_err:.5e} +- {np.sqrt(self.cap_var):.5e})" \
            f"\n    x = {self.cap_filter_init_parms.get('x')} +- {np.sqrt(self.cap_filter_init_parms.get('P'))}" \
            f"\n    Qbal =\n{self.cap_filter_init_parms.get('bal_change_Q')}" \
            f"\n    Qexc =\n{self.cap_filter_init_parms.get('exc_change_Q')}"
        if self.dis_err is not None:
            r+= f"\nDischarge Status: {'GOOD' if self.dis_status else 'BAD'} " \
                f"({self.dis_err:.5e} +- {np.sqrt(self.dis_var):.5e})" \
                f"\n    dMdW = {self.dis_filter_init_parms.get('dMdW'):.5e} +- " \
                f"{np.sqrt(self.dis_filter_init_parms.get('P')):.5e}" \
                f"\n    W0 = {self.dis_filter_init_parms.get('W0'):.5e}"
        return r

def TDPT_initialize_filters(
    ctx: TDPTContext, 
    
    cap_guess: float | None = None,
    cap_low_high: Tuple[float, float] = (0.05, 2.0),
    cap_absolute: bool = False,
    cap_error_threshold: float | None = None,

    dis_low_high: Tuple[float, float] = (0.5, 2.0),
    dis_error_threshold: float | None = None,

    X1: float | None = None, # 10s of mV is reasonable
    X2: float | None = None, # 10s of mV is reasonable
    W: float | None = None, # usually a few us
    reguess: bool = True,

    refinements: int | None = None,
    deviation_samples: int = 3,
    settle_time: float | None = None,

    post_measurement_callback: Callable | None = None,

) -> Tuple[TDPT_initial_filter_config, TDPT_initial_filter_config]:
    
    if cap_guess is None:
        if ctx.cap_filter is not None:
            try:
                cap_guess = ctx.get_Cac()
            except:
                cap_guess = -ctx.Y1()/ctx.X1()
        else:
            cap_guess = -ctx.Y1()/ctx.X1()

    if settle_time is None:
        settle_time = ctx.settle_time

    if refinements is None:
        refinements = ctx.max_refinements

    if cap_error_threshold is None:
        cap_error_threshold = ctx.capacitance_error_threshold

    if dis_error_threshold is None:
        dis_error_threshold = ctx.discharge_error_threshold

    X1 = ctx.X1(X1)
    X2 = ctx.X2(X2)
    Y1 = ctx.Y1(-cap_guess * X1)
    Y2 = ctx.Y2(-cap_guess * X2)
    W = ctx.W(W)

    msg = "Starting test to initialize a the Kalman filters for a " \
         f"{'posi' if X1 > 0 else 'nega'}tive charging pulse..." \
         f"\nInitial Parameters: X1={X1:.5e}, Y2={Y1:.5e}"
    if X2 is not None:
        msg += f", X2={X2:.5e}, Y2={Y2:.5e}"
    ctx.log(msg)

    controls = []
    errors = []

    # Y2 follows along with Y1 to respect Cac
    def Y1_Y2(v: float | None = None):
        if v is None:
            return ctx.Y1()
        nonlocal X1, X2
        ctx.Y1(v)
        ctx.Y2(v * (X2 / X1))

    controls.append(
        balanceKnob(
            bounds = (-ctx.max_pulse_height, ctx.max_pulse_height),
            guess = -cap_guess * X1,
            l = cap_low_high[0],
            h = cap_low_high[1],
            f = Y1_Y2,
            absolute = cap_absolute,
            logger = ctx.logger,
            name = "Y",
        )
    )

    errors.append(ctx.capacitance_error)

    # If we are also initializing a discharge
    if ctx.discharge_pair is not None:
        controls.append(
            balanceKnob(
                bounds = (ctx.min_pulse_width, ctx.max_pulse_width),
                guess = W,
                l = dis_low_high[0],
                h = dis_low_high[1],
                f = ctx.W,
                absolute = False,
                logger = ctx.logger,
                name = "W",
            )
        )

        errors.append(ctx.discharge_error)

    linear_balance = lstsqBalance(
        controls = controls,
        error_parms = errors,
        pre_measurement_callback = lambda *args, **kwargs: \
            ctx.averager.take_curve(ctx.high_resolution_sweep_multiplier),
        post_measurement_callback = post_measurement_callback,
        settle_time = settle_time,
        logger = ctx.logger,
    )

    # Positive Balance
    positive_result = _TDPT_initialize_filter_pass(
        linear_balance, 
        ctx.averager, 
        refinements, 
        cap_error_threshold,
        dis_error_threshold,
        deviation_samples, 
        X1, X2,
        reguess,
    )

    ctx.log(
        f"Obtained a {'posi' if X1 > 0 else 'nega'}tive result:\n{positive_result}"
    )

    # Negative Balance
    X1 = ctx.X1(-X1)
    X2 = ctx.X2(-X2)
    Y1 = ctx.Y1(-cap_guess * X1)
    Y2 = ctx.Y2(-cap_guess * X2)
    linear_balance.controls[0].guess = -cap_guess * X1

    msg = "Starting test to initialize a the Kalman filters for a " \
         f"{'posi' if X1 > 0 else 'nega'}tive charging pulse..." \
         f"\nInitial Parameters: X1={X1:.5e}, Y2={Y1:.5e}"
    if X2 is not None:
        msg += f", X2={X2:.5e}, Y2={Y2:.5e}"
    ctx.log(msg)

    negative_result = _TDPT_initialize_filter_pass(
        linear_balance, 
        ctx.averager, 
        refinements, 
        cap_error_threshold,
        dis_error_threshold,
        deviation_samples, 
        X1, X2,
        reguess,
    )

    ctx.log(
        f"Obtained a {'posi' if X1 > 0 else 'nega'}tive result:\n{negative_result}"
    )

    # If we got the signs mixed up, just swap them
    if positive_result.meta.X1 < 0:
        positive_result, negative_result = negative_result, positive_result

    return positive_result, negative_result

def _TDPT_initialize_filter_pass(
    linear_balance: lstsqBalance,
    averager: wfAverager,
    refinements: int,
    cap_error_threshold: float,
    dis_error_threshold: float | None,
    deviation_samples: int,
    X1: float, X2: float,
    reguess: bool,
):
    Cac, (dQ, dQvar), (M, Mvar) = _TDPT_initialize_filter_inner_balance(
        linear_balance, 
        averager, 
        refinements,
        deviation_samples, 
        X1,
        reguess,
    )

    cap_status = abs(dQ) < cap_error_threshold
    dis_status = abs(M) < dis_error_threshold
    G = linear_balance._A.item(0)
    if linear_balance.M == 2:
        dMdW = linear_balance._A[1, 1]
        dis_filter_init_parms = {
            'bal_change_Q': _TDPT_CONF.DIS_INIT_BAL_UNC**2,
            'exc_change_Q': _TDPT_CONF.DIS_INIT_EXC_UNC**2,
            'dMdW': dMdW,
            'P': _TDPT_CONF.DIS_INIT_P_MULT*(dMdW)**2,
            'W0': linear_balance.controls[1].get_val()
        }
    else:
        dis_filter_init_parms = None

    return TDPT_initial_filter_config(
        status = cap_status and dis_status,
        cap_status = cap_status,
        dis_status = dis_status,
        positive = X1 > 0.0,
        cap_err = dQ,
        cap_var = dQvar,
        cap_filter_init_parms = {
            'bal_change_Q': np.diag([_TDPT_CONF.CAP_INIT_BAL_QG_MULT*G, _TDPT_CONF.CAP_INIT_BAL_QC, _TDPT_CONF.CAP_INIT_BAL_QdC])**2,
            'exc_change_Q': np.diag([_TDPT_CONF.CAP_INIT_EXC_QG_MULT*G, _TDPT_CONF.CAP_INIT_EXC_QC, _TDPT_CONF.CAP_INIT_EXC_QdC])**2,
            'x': np.array([G, Cac, 0.0]),
            'P': np.array([_TDPT_CONF.CAP_INIT_PG_MULT*G, _TDPT_CONF.CAP_INIT_PC, _TDPT_CONF.CAP_INIT_PdC])**2,
        },
        dis_err = M,
        dis_var = Mvar,
        dis_filter_init_parms = dis_filter_init_parms,
        meta = TDPT_initial_filter_metadata(X1, X2)
    )

def _TDPT_initialize_filter_inner_balance(
    linear_balance: lstsqBalance,
    averager: wfAverager,
    refinements: int,
    deviation_samples: int,
    X1: float,
    reguess: bool,
) -> Tuple[float, Tuple[float, float], Tuple[float | None, float | None]]:

    """rebalances linear_balance and returns Cac, error parameters, and deviations."""

    # Reset
    linear_balance.reset()
    linear_balance.balance()
    if reguess:
        linear_balance.set_good(True)
        linear_balance.refine()
        linear_balance.reset()
        linear_balance.balance()

    # Perform refinements to the balance
    linear_balance.set_good(True)
    for i in range(refinements):
        _checkpoint()
        
        # If the error thresholds are met, mark the balance as good so it stays
        # close to and extrapolates from the balance point
        if linear_balance.M == 2:
            dQ = linear_balance.error_parms[0].get_error()
            M = linear_balance.error_parms[1].get_error()
            # good_balance = abs(dQ) < cap_error_threshold and \
            #                 abs(M) < dis_error_threshold
        else:
            dQ = linear_balance.error_parms[0].get_error()
            # good_balance = abs(dQ) < cap_error_threshold
        # linear_balance.set_good(good_balance)

        linear_balance.refine()

    if linear_balance.M == 2:
        # Check the deviation in the error parameters at the balance point
        dQ = linear_balance.error_parms[0].get_error()
        dQsq = dQ*dQ
        M = linear_balance.error_parms[1].get_error()
        Msq = M * M

        for _ in range(deviation_samples):
            averager.take_curve()

            dQ_, _ = linear_balance.error_parms[0]()
            dQ += dQ_
            dQsq += dQ_*dQ_

            M_, _ = linear_balance.error_parms[1]()
            M += M_
            Msq += M_*M_

        dQ /= (deviation_samples + 1)
        dQsq /= (deviation_samples + 1)
        M /= (deviation_samples + 1)
        Msq /= (deviation_samples + 1)
        if deviation_samples == 0:
            dQvar = dQ
            Mvar = M
        else:
            dQvar = dQsq - dQ*dQ
            Mvar = Msq - M * M

        return -linear_balance._x0[0] / X1, (dQ, dQvar), (M, Mvar)
    
    else:
        # Check the deviation in the error parameters at the balance point
        dQ = linear_balance.error_parms[0].get_error()
        dQsq = dQ*dQ

        for _ in range(deviation_samples):
            averager.take_curve()

            dQ_, _ = linear_balance.error_parms[0]()
            dQ += dQ_
            dQsq += dQ_*dQ_

        dQ /= (deviation_samples + 1)
        dQsq /= (deviation_samples + 1)
        if deviation_samples == 0:
            dQvar = dQ
        else:
            dQvar = dQsq - dQ*dQ

        return -linear_balance._x0[0] / X1, (dQ, dQvar), (None, None)
    
@dataclass
class TDPT_control_state():
    X1: float
    Y1: float
    X2: float | None
    Y2: float | None
    W: float | None

    def __str__(self) -> str:
        r = f"X1 = {self.X1:.5e}, Y1 = {self.Y1:.5e}"
        if self.X2 is not None:
            r += f", X2 = {self.X2:.5e}, Y2 = {self.Y2:.5e}, W = {self.W:.5e}"
        return r
    
@dataclass
class TDPT_optimal_state():
    Cac: float
    W0: float | None = None
    good_width_prediction: bool | None = None
    rail_intervention: bool | None = None
    low_collision: bool | None = None
    high_collision: bool | None = None
    discharge_ratio: float | None = None

    def __str__(self) -> str:
        r = f"Cac = {self.Cac:.5e}"
        if self.W0 is not None:
            r += f", W0 = {self.W0:.5e} ({'GOOD' if self.good_width_prediction else 'BAD'})" \
                 f"\n    (Collisions: {'LOW' if self.low_collision else 'HIGH' if self.high_collision else 'NONE'})"
        if self.discharge_ratio is not None:
            r += f", Y2/X2 = {self.discharge_ratio:.5e}"
        return r

@dataclass
class TDPT_error_state():
    # Termination Status
    good: bool
    good_sweep_count: int
    # dQ: Capacitive jump in the charge signal at the start of the charging pulse
    dQ_satisfied: bool
    dQ: float
    dQ_var: float
    dQ_R: float
    # M: Slope of the charge signal after discharge pulses
    M_satisfied: bool | None = None
    M: float | None = None
    M_var: float | None = None
    # I: Integrated charge signal over the measurement window
    Y2_adjusted: bool | None = None
    I_satisfied: bool | None = None
    I: float | None = None
    I_var: float | None = None

    def __str__(self) -> str:
        r = f"Termination Status: {'GOOD' if self.good else 'BAD'} (streak = {self.good_sweep_count})"
        r += f"\n  dQ ({'GOOD' if self.dQ_satisfied else 'BAD'}) = {self.dQ:.5e} +- {np.sqrt(self.dQ_var):.5e}"
        if self.M is not None:
            r += f"\n   M ({'GOOD' if self.M_satisfied else 'BAD'}) = {self.M:.5e} +- {np.sqrt(self.M_var):.5e}"
        if self.I is not None:
            r += f"\n   I ({'GOOD' if self.I_satisfied else 'BAD'}) = {self.I:.5e} +- {np.sqrt(self.I_var):.5e}"
            r += "\n(Y2 was adjusted.)" if self.Y2_adjusted else "\n(Y2 was not adjusted.)"
        return r

@dataclass
class TDPT_filter_balance_result():
    status: bool
    iterations: int
    controls: TDPT_control_state
    errors: TDPT_error_state
    optimal: TDPT_optimal_state
    curve: wfCurveData

    def __str__(self) -> str:
        line = "="*80
        return (
            line + 
            f"\nBALANCE on iteration {self.iterations} was {'' if self.status else 'UN'}SUCCESSFUL" \
            f"\nCONTROLS:\n{self.controls}\nERRORS:\n{self.errors}\nOPTIMAL_SETTINGS:\n{self.optimal}\n" + \
            line
        )

def TDPT_filter_balance(ctx: TDPTContext) -> TDPT_filter_balance_result:
    ctx.log("Initiating Balance...")

    # Set these up ahead of time so that they're defined
    X1 = Y1 = X2 = Y2 = W = None
    dQ = dQ_var = dQ_R = None
    M = M_var = None
    I = I_var = None

    # Check that the context filters have been initialized
    if ctx.cap_filter is None:
        raise RuntimeError(
            "Cannot perform `TDPT_balance` if context filters are uninitialized"
        )

    # Check if the Kalman initialization matches the sign of the charging pulse
    X1 = ctx.X1()
    if (X1 >= 0 and not ctx.cap_filter.is_positive_initialized) or \
       (X1 < 0 and not ctx.cap_filter.is_negative_initialized):
        ctx.log("Filter initialization polarity did not match X1; Reinitializing...")
        if X1 >= 0:
            ctx.reinitialize_positive_filter()
        else:
            ctx.reinitialize_negative_filter()
    
    # Induce a kalman prediction step on the filters to reflect an excitation change
    ctx.predict_filter_change(balance_change = False)
    ctx.log("INITIAL FILTER STATE:")
    ctx.log_capacitance_filter_status()
    if ctx.is_discharge():
        ctx.log_discharge_filter_status()
    Y1 = ctx.Y1(-ctx.get_Cac() * X1) # Balance away dQ

    # If we are also using discharge pulses, we have to clear the stale bracket. 
    # We also extrapolate the discharge pulse width and retrieve X2.
    if ctx.is_discharge():
        ctx.clear_dis_width_bracket()
        W = ctx.set_extrap_dis_width()
        prev_W = W
        prev_M = None
        if ctx.discharge_X_ratio is None:
            X2 = ctx.X2()
        else:
            X2 = ctx.X2(-ctx.dis_eta * ctx.discharge_X_ratio * X1)

    # If we are performing an integrated balance, extrapolate the discharge Y
    # amplitude from previous measurements.
    if ctx.is_integrated():
        Y2 = ctx.set_extrap_discharge_Y()
    else:
        Y2 = ctx.Y2(-ctx.get_Cac() * X1)

    # Perform refinements until we converge on a balance or terminate unhappily
    dQ_satisfied = False
    M_satisfied = not ctx.is_discharge()
    I_satisfied = not ctx.is_integrated()
    Y2_adjusted = False
    success = False
    good_sweep_count = 0
    W_low_collisions = 0
    W_high_collisions = 0
    for itr in range(ctx.max_refinements + 1):
        _checkpoint()

        # Set ourselves to the new predicted optimal control point
        KF_Cac = ctx.get_Cac()
        if itr != 0:
            Y1 = ctx.Y1(-KF_Cac * X1, Y1)        
            if not ctx.is_integrated():
                # If not integrated, the ratios Yi/Xi should be the same
                Y2 = ctx.Y2(-KF_Cac * X2, Y2)

        if ctx.is_discharge():
            if itr == 0:
                # On the first iteration, we haven't doen this yet
                W0, good_width_prediction = ctx.iterate_dis_width_bracket()
            if good_width_prediction:
                W = ctx.W(W0, W)
        
        # THERE IS SOME TERMINATION CHECKING STUFF HERE IF WE'RE HITTING THE EDGE OF THE BRACKET IN THE ORIGINAL
        # THINK ABOUT IF WE WANT TO REPLICATE THIS

        # We induce a kalman prediction to reflect the balance change
        ctx.predict_filter_change(balance_change = True)

        # Now we actually take the sweep
        current_controls = TDPT_control_state(X1, Y1, X2, Y2, W)
        if itr == 0:
            if ctx.pre_initial_trace_callback is not None:
                ctx.pre_initial_trace_callback(ctx, current_controls)
        else:
            if ctx.pre_refinement_callback is not None:
                ctx.pre_refinement_callback(
                    ctx,
                    current_controls, 
                    error_state, 
                    optimal_state,
                )

        ctx.log(f"ITERATION {itr + 1}; Taking a sweep at:\n{current_controls}")
        time.sleep(ctx.settle_time)
        # Conditions for a higher resolution sweep measurement
        took_high_resolution = (good_sweep_count > 0) or (ctx.good_sweep_threshold <= 1)
        ctx.take_sweep(took_high_resolution)
        curve_data = ctx.get_curve_data()

        # Retrieve the relevant error parameters and update the models accordingly
        dQ, dQ_var = ctx.capacitance_error()
        dQ_R = dQ_var
        ctx.cap_update(dQ, dQ_R)
        
        if ctx.is_discharge():
            M, M_var = ctx.discharge_error()
            
            if itr != 0:
                dW = W - prev_W
                if abs(dW) > ctx.pulse_width_resolution:
                    o_dMdW = (M - prev_M) / dW
                    dMdW_R = _TDPT_CONF.dMdW_R_MULT * ctx.discharge_error_threshold**2 / dW * dW
                    ctx.dis_update(o_dMdW, dMdW_R)

            ctx.update_dis_width_bracket(W, M)
            prev_W = W
            prev_M = M

            # Use the bracket and some slope information to predict the optimal 
            # discharge pulse width. On the first iteration, this is never a 
            # "good_width_prediction"
            W0, good_width_prediction = ctx.iterate_dis_width_bracket()

            # We perform some sanity checks on to avoid getting stuck at the rails
            rail_intervention = False
            if ctx._dis_width_bracket.hitting_min_rail \
                and (W <= ctx.min_pulse_width + ctx.pulse_width_resolution):
            
                W0 = ctx.mean_dis_width()
                rail_intervention = True
                ctx.log(
                    "Warning: Discharge pulse width is hitting the lower rail."
                    "\nThis may indicate that the optimal width is below the allowed range."
                    f"\nReverting to a middling value {W0:.5e} to avoid getting stuck at the rail."
                )
                
            if ctx._dis_width_bracket.hitting_max_rail \
                and (W >= ctx.max_pulse_width - ctx.pulse_width_resolution):
                
                W0 = ctx.mean_dis_width()
                rail_intervention = True
                ctx.log(
                    "Warning: Discharge pulse width is hitting the upper rail."
                    "\nThis may indicate that the optimal width is above the allowed range."
                    f"\nReverting to a middling value {W0:.5e} to avoid getting stuck at the rail."
                )
                
            if ctx.is_integrated():
                I, I_var = ctx.integrated_error()
                Y2_adjusted = W > ctx.dis_amp_adjustment_threshold
                if Y2_adjusted:
                    # Step to an adjusted Y2. dis_eta accounts for a sign difference
                    # between the charge and discharge pulse height conventions
                    dY2 = -ctx.dis_eta * I / (_TDPT_CONF.INT_ADJ_LAMBDA * ctx.get_G() * W)
                    ctx.log(
                        f"Adjusting the integrated balance Y2 by {dY2:.5e} to "
                        f"account for integrated error of {I:.5e}."
                    )
                    Y2 = ctx.set_Y2_integrated_balance(Y2 + dY2, Y2)
                    ctx.predict_filter_change(balance_change = True)

        # Check termination conditions
        dQ_satisfied = abs(dQ) < ctx.capacitance_error_threshold
        if ctx.is_discharge():
            M_satisfied = abs(M) < ctx.discharge_error_threshold
            if ctx.is_integrated():
                I_satisfied = abs(I) < ctx.integrated_error_threshold

        good = dQ_satisfied and M_satisfied and (I_satisfied or not Y2_adjusted)

        # We require a certain streak of good sweeps to terminate successfully
        if good:
            good_sweep_count+=1
        else:
            good_sweep_count = 0

        error_state = TDPT_error_state(
            good, good_sweep_count,
            dQ_satisfied, dQ, dQ_var, dQ_R, 
            M_satisfied, M, M_var, 
            Y2_adjusted, I_satisfied, I, I_var
        )

        if ctx.is_discharge():
            if ctx.is_integrated():
                optimal_state = TDPT_optimal_state(
                    ctx.get_Cac(), 
                    W0, good_width_prediction, rail_intervention,
                    ctx._dis_width_bracket.hitting_min_rail,
                    ctx._dis_width_bracket.hitting_max_rail,
                    Y2/X2,
                )
            else:
                optimal_state = TDPT_optimal_state(
                    ctx.get_Cac(), 
                    W0, good_width_prediction, rail_intervention,
                    ctx._dis_width_bracket.hitting_min_rail,
                    ctx._dis_width_bracket.hitting_max_rail,
                )
        else:
            optimal_state = TDPT_optimal_state(ctx.get_Cac())

        ctx.log(
            "SWEEP COMPLETE; Updating internal models based on last measurement..."
            f"\nERRORS:\n{error_state}\nOPTIMAL SETTINGS:\n{optimal_state}"
            f"\nGOOD COUNT: {good_sweep_count} ( / {ctx.good_sweep_threshold})."    
        )

        if took_high_resolution and (good_sweep_count >= ctx.good_sweep_threshold):
            success = True
            break

        if itr == 0:
            if ctx.post_initial_trace_callback is not None:
                ctx.post_initial_trace_callback(
                    ctx,
                    current_controls, 
                    error_state, 
                    optimal_state,
                    curve_data,
                )
        else:
            if ctx.post_refinement_callback is not None:
                ctx.post_refinement_callback(
                    ctx,
                    current_controls,
                    error_state,
                    optimal_state,
                    curve_data,
                )

        if not ctx.is_discharge():
            continue

        # We want to avoid dawdling when the discharg error bounds are infeasible
        # given the allowed discharge pulse widths range. For this, we keep track
        # of consecutive runs where the width is railed while the discharge is still
        # bad and terminate if we accumulate too many
        if ctx._dis_width_bracket.hitting_min_rail:
            ctx.log("Discharge pulse width is hitting the lower rail.")
            W_low_collisions += 1
        else:
            W_low_collisions = 0

        if ctx._dis_width_bracket.hitting_max_rail:
            ctx.log("Discharge pulse width is hitting the upper rail.")
            W_high_collisions += 1
        else:
            W_high_collisions = 0

        if not (M_satisfied and (I_satisfied or not Y2_adjusted)) and \
            (W_low_collisions > ctx.max_dis_width_collisions or \
             W_high_collisions > ctx.max_dis_width_collisions):
            ctx.log(
                "Discharge errors are not satisfied despite repeatedly hitting rails."
                "\nLikely infeasible to discharge; Terminating unsuccessfully."    
            )
            success = False
            break

        # Check if we have hit a rut (likely due to a malformed discharge width bracket)
        if (not M_satisfied) and ctx.detect_bracket_rut(itr, good_sweep_count):
            # We clear whichever is likely to be the problematic bound
            if M > 0:
                ctx._dis_width_bracket.low.clear()
            elif M < 0:
                ctx._dis_width_bracket.high.clear()
    
    if success and ctx.is_discharge():
        if optimal_state.good_width_prediction and \
            not (optimal_state.low_collision or optimal_state.high_collision):
            ctx.dis_filter.push_W0(optimal_state.W0)

    if ctx.is_integrated():
        ctx.dis_filter.push_Y2(Y2)

    # RETURNS
    result = TDPT_filter_balance_result(
        status = success,
        iterations = itr + 1,
        controls = current_controls,
        errors = error_state,
        optimal = optimal_state,
        curve = curve_data,
    )
    ctx._previous_result = result
    ctx.log(result)
    return result