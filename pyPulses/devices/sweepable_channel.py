from __future__ import annotations

from .channel_adapter import ScalarChannel
from .registry import (
    register_device_class,
    format_reference,
    DeferredReference,
    DeviceRegistry,
    HardwareRegistry,
)

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class SweepConfig:
    """
    Sweep constraints for a single parameter.

    Attributes
    ----------
    max_step : float or None
        Maximum step size per ramp iteration. None is treated as unbounded.
    min_step : float or None
        Minimum step size; smaller changes are skipped to avoid hammering
        costly setters. None is treated as 0. If tolerance > min_step,
        tolerance takes precedence.
    tolerance : float
        Allowed residual error between the requested target and the final
        set point. Defaults to 0 (exact).
    max_rate : float or None
        Maximum rate of change in output units per second.
    settle_time : float
        Seconds to wait after each step for hardware to settle.
    async_set : bool
        If True, _tandemSweep dispatches this channel's set_output call
        asynchronously at the start of the sweep and awaits it at the end,
        so that other (sync) channels can move concurrently. The channel
        must implement set_output_async(). Use AsyncChannel as the base
        class to get this behaviour automatically.
    """
    max_step    : float | None = None
    min_step    : float | None = None
    tolerance   : float        = 0.0
    max_rate    : float | None = None
    settle_time : float        = 0.0
    async_set   : bool         = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_step'   : self.max_step,
            'min_step'   : self.min_step,
            'tolerance'  : self.tolerance,
            'max_rate'   : self.max_rate,
            'settle_time': self.settle_time,
            'async_set'  : self.async_set,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SweepConfig':
        return cls(
            max_step    = d.get('max_step'),
            min_step    = d.get('min_step'),
            tolerance   = d.get('tolerance',   0.0),
            max_rate    = d.get('max_rate'),
            settle_time = d.get('settle_time', 0.0),
            async_set   = d.get('async_set',   False),
        )


@register_device_class("SweepableChannel")
class SweepableChannel:
    """
    A scalar channel paired with sweep constraints and display metadata.

    This is the primary unit consumed by tandemSweep and param_sweep_measure.
    It can wrap any ScalarChannel — including CalibratedChannel — and adds
    the sweep behavior that the underlying channel knows nothing about.

    Registration is opt-in via registry_id. Simple one-off sweepable channels
    do not need to be registered; only those that are reused across experiments
    and whose configs should survive serialization warrant registration.

    Attributes
    ----------
    channel : ScalarChannel
        The underlying channel to get/set.
    config : SweepConfig
        Sweep constraints (step sizes, tolerance).
    name : str or None
        Short identifier used in file headers and as a dict key when
        specifying sweep targets. Should be something easy to type.
    long_name : str or None
        Display name, may contain LaTeX syntax for plot axis labels.
    unit : str or None
        Physical unit string (e.g. 'V', 'A', r'\\mu T').
    """

    def __init__(self,
        channel     : ScalarChannel | DeferredReference,
        config      : SweepConfig | None = None,
        name        : str | None = None,
        long_name   : str | None = None,
        unit        : str | None = None,
        registry_id : str | None = None,
    ):
        self._channel   = channel
        self._config    = config or SweepConfig()
        self.name       = name
        self.long_name  = long_name
        self.unit       = unit

        if registry_id is not None:
            DeviceRegistry.register(self, registry_id=registry_id)

    # ------------------------------------------------------------------
    # ScalarChannel passthrough
    # ------------------------------------------------------------------

    def get_output(self) -> float:
        return self._channel.get_output()

    def set_output(self, value: float) -> None:
        self._channel.set_output(value)

    def __call__(self, value: float | None = None) -> float | None:
        if value is None:
            return self.get_output()
        self.set_output(value)
        return None

    # ------------------------------------------------------------------
    # Config access
    # ------------------------------------------------------------------

    @property
    def config(self) -> SweepConfig:
        return self._config

    @config.setter
    def config(self, value: SweepConfig) -> None:
        self._config = value

    @property
    def channel(self) -> ScalarChannel:
        return self._channel

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize_state(self) -> Dict[str, Any]:
        return {
            'channel'  : format_reference(self._channel),
            'config'   : self._config.to_dict(),
            'name'     : self.name,
            'long_name': self.long_name,
            'unit'     : self.unit,
        }

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        if 'channel' in state:
            self._channel = DeferredReference(state['channel'])
        if 'config' in state:
            self._config = SweepConfig.from_dict(state['config'])
        if 'name' in state:
            self.name = state['name']
        if 'long_name' in state:
            self.long_name = state['long_name']
        if 'unit' in state:
            self.unit = state['unit']

        self._resolve_references()

    def _resolve_references(self) -> None:
        if isinstance(self._channel, DeferredReference):
            self._channel = self._channel.unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SweepableChannel':
        channel = config.pop('channel')
        if channel is not None:
            channel = DeferredReference(channel)

        sweep_config = SweepConfig.from_dict(config.pop('config', {}))
        registry_id  = config.pop('registry_id', None)

        instance = cls(
            channel     = channel,
            config      = sweep_config,
            name        = config.pop('name',      None),
            long_name   = config.pop('long_name', None),
            unit        = config.pop('unit',      None),
            registry_id = registry_id,
        )
        return instance

class AsyncChannel(SweepableChannel):
    """
    Abstract base for instruments that own their own internal ramp and must
    never receive intermediate stepped values from _tandemSweep.

    When _tandemSweep encounters a channel whose SweepConfig has async_set=True,
    it dispatches set_output_async() once at the very start of the sweep and
    awaits the result only after all sync channels have finished moving. This
    allows slow instruments (primarily superconducting magnet power supplies)
    to ramp concurrently with fast channels sweeping back to the start of the
    next scan line.

    The SweepConfig is hard-coded here and must not be overridden, since the
    async dispatch contract depends on async_set=True and the absence of any
    step constraints.

    The single-worker executor serializes concurrent set_output_async() calls:
    submitting a new target before the previous completes will queue rather
    than overlap.

    Subclasses
    ----------
    Subclasses must implement _get() and _set(), and should be decorated with
    @register_device_class so they participate in serialization. They must also
    implement _serialize_state, _deserialize_state, and from_config following
    the DeviceRegistry pattern: _serialize_state stores format_reference of the
    underlying instrument; from_config stores a DeferredReference which is
    resolved in _deserialize_state during the second deserialization pass.
    """

    _ASYNC_CONFIG = SweepConfig(
        max_step    = None,
        min_step    = None,
        tolerance   = 0.0,
        async_set   = True,
    )

    def __init__(self,
        name        : str | None = None,
        long_name   : str | None = None,
        unit        : str | None = None,
        registry_id : str | None = None,
    ):
        # Pass channel=None; get/set_output are overridden below and
        # self._channel is never accessed on AsyncChannel instances.
        super().__init__(
            channel     = None,
            config      = self._ASYNC_CONFIG,
            name        = name,
            long_name   = long_name,
            unit        = unit,
            registry_id = registry_id,
        )
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get(self) -> float:
        raise NotImplementedError

    def _set(self, value: float) -> None:
        raise NotImplementedError

    def get_output(self) -> float:
        return self._get()

    def set_output(self, value: float) -> None:
        self._set(value)

    def set_output_async(self, value: float) -> Future:
        """Submit _set to the executor and return a Future."""
        return self._executor.submit(self._set, value)

    def __del__(self):
        self._executor.shutdown(wait=False)