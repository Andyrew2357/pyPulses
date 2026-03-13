
@register_device_class("PolarityCalibratedChannel")
class PolarityCalibratedChannel(abstractDevice):
    """
    A channel whose calibration depends on an external polarity signal.
    
    Delegates to two CalibratedChannel instances (one per polarity) and
    selects between them based on the polarity source.
    
    This is useful for hardware like pulse shapers where the DC control
    always outputs positive values, but the effective output polarity
    is determined by relay timing.
    """
    
    def __init__(self,
        polarity_source: BoolChannel | DeferredReference,
        pos_channel: CalibratedChannel | DeferredReference,
        neg_channel: CalibratedChannel | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)
        
        self._polarity_source = polarity_source
        self._pos = pos_channel
        self._neg = neg_channel
    
    @property
    def polarity(self) -> bool:
        """Current polarity state."""
        return self._polarity_source()
    
    @property
    def _active(self) -> CalibratedChannel:
        """The currently active calibrated channel based on polarity."""
        return self._pos if self.polarity else self._neg
    
    @property
    def _inactive(self) -> CalibratedChannel:
        """The currently inactive calibrated channel."""
        return self._neg if self.polarity else self._pos
    
    def _sync_caches(self):
        """
        Synchronize cached values from active channel to inactive channel.
        
        Since both channels share the same hardware, their cached control
        values must stay in sync.
        """
        self._inactive._cached_control = self._active._cached_control
    
    # Forward ScalarChannel interface to active channel
    
    def get_output(self, lazy: bool = True) -> float:
        """
        Get output from active channel, with sign based on polarity.
        
        Positive polarity returns positive values, negative polarity
        returns negative values.
        """
        magnitude = self._active.get_output(lazy=lazy)
        self._sync_caches()
        return magnitude if self.polarity else -magnitude
    
    def set_output(self, o: float):
        """
        Set output on active channel.
        
        The polarity of the requested value should match the current
        hardware polarity. If not, the magnitude is used and a warning
        is issued (polarity switching should be done at a higher level).
        """
        if (o >= 0) != self.polarity:
            self.warn(
                f"Requested output {o} has opposite sign to current polarity "
                f"({'positive' if self.polarity else 'negative'}). "
                f"Using magnitude {abs(o)}."
            )
        self._active.set_output(abs(o))
        self._sync_caches()
    
    def __call__(self, v: float | None = None) -> float | None:
        if v is None:
            return self.get_output()
        self.set_output(v)
    
    # Expose properties from active channel
    
    @property
    def output_min(self) -> float:
        """Minimum achievable output (signed based on polarity)."""
        if self.polarity:
            return self._active.output_min
        else:
            return -self._active.output_max
    
    @property
    def output_max(self) -> float:
        """Maximum achievable output (signed based on polarity)."""
        if self.polarity:
            return self._active.output_max
        else:
            return -self._active.output_min
    
    def invalidate_cache(self) -> None:
        """Invalidate cache on both channels."""
        self._pos.invalidate_cache()
        self._neg.invalidate_cache()
    
    # Attenuator access
    
    @property
    def attenuator_locked(self) -> bool:
        return self._active._attenuator_locked
    
    @attenuator_locked.setter
    def attenuator_locked(self, value: bool):
        # Lock both to keep them in sync
        self._pos._attenuator_locked = value
        self._neg._attenuator_locked = value
    
    @property
    def attenuator_mode(self) -> AttenuatorMode:
        return self._active._attenuator_mode
    
    @attenuator_mode.setter
    def attenuator_mode(self, value: AttenuatorMode):
        self._pos._attenuator_mode = value
        self._neg._attenuator_mode = value
    
    @property
    def attenuator_preference(self) -> AttenuatorPreference:
        return self._active._attenuator_preference
    
    @attenuator_preference.setter
    def attenuator_preference(self, value: AttenuatorPreference):
        self._pos._attenuator_preference = value
        self._neg._attenuator_preference = value
    
    # Crosstalk management
    
    def add_crosstalk(self, 
        other: ScalarChannel | str, 
        coeff: float, 
        polarity: bool | None = None
    ):
        """
        Add crosstalk from another channel.
        
        Parameters
        ----------
        other : ScalarChannel | str
            The channel causing crosstalk, or a reference string.
        coeff : float
            Crosstalk coefficient.
        polarity : bool | None
            If None, add to both polarities. Otherwise add only to the
            specified polarity's channel.
        """
        if polarity is None:
            self._pos.add_crosstalk(other, coeff)
            self._neg.add_crosstalk(other, coeff)
        elif polarity:
            self._pos.add_crosstalk(other, coeff)
        else:
            self._neg.add_crosstalk(other, coeff)
    
    def clear_crosstalk(self, polarity: bool | None = None):
        """Clear crosstalk registrations."""
        if polarity is None:
            self._pos.clear_crosstalk()
            self._neg.clear_crosstalk()
        elif polarity:
            self._pos.clear_crosstalk()
        else:
            self._neg.clear_crosstalk()
    
    # Access to underlying channels for direct manipulation
    
    @property
    def pos_channel(self) -> CalibratedChannel:
        return self._pos
    
    @property
    def neg_channel(self) -> CalibratedChannel:
        return self._neg
    
    def get_channel(self, polarity: bool) -> CalibratedChannel:
        """Get the calibrated channel for a specific polarity."""
        return self._pos if polarity else self._neg
    
    # Calibration access
    
    def get_calibration(self, polarity: bool | None = None) -> CalibrationModel:
        """
        Get calibration for specified polarity, or active polarity if None.
        """
        if polarity is None:
            polarity = self.polarity
        return self._pos._calibration if polarity else self._neg._calibration
    
    def set_calibration(self, calibration: CalibrationModel, polarity: bool):
        """Set calibration for specified polarity."""
        channel = self._pos if polarity else self._neg
        channel._calibration = calibration
        # Recompute output bounds
        channel._o_min, channel._o_max = calibration.output_bounds(
            channel._c_min, channel._c_max
        )
    
    # Serialization
    
    def _serialize_state(self) -> Dict[str, Any]:
        return {
            'polarity_source': format_reference(self._polarity_source),
            'pos_channel': format_reference(self._pos),
            'neg_channel': format_reference(self._neg),
        }
    
    def _deserialize_state(self, state: Dict[str, Any]):
        if 'polarity_source' in state:
            self._polarity_source = DeferredReference(state['polarity_source'])
        if 'pos_channel' in state:
            self._pos = DeferredReference(state['pos_channel'])
        if 'neg_channel' in state:
            self._neg = DeferredReference(state['neg_channel'])
        
        self._resolve_references()
    
    def _resolve_references(self):
        if isinstance(self._polarity_source, DeferredReference):
            self._polarity_source = self._polarity_source.unwrap()
        if isinstance(self._pos, DeferredReference):
            self._pos = self._pos.unwrap()
        if isinstance(self._neg, DeferredReference):
            self._neg = self._neg.unwrap()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PolarityCalibratedChannel':
        registry_id = config.pop('registry_id')
        
        polarity_source = config.pop('polarity_source', None)
        if polarity_source is not None:
            polarity_source = DeferredReference(polarity_source)
        
        pos_channel = config.pop('pos_channel', None)
        if pos_channel is not None:
            pos_channel = DeferredReference(pos_channel)
        
        neg_channel = config.pop('neg_channel', None)
        if neg_channel is not None:
            neg_channel = DeferredReference(neg_channel)
        
        return cls(
            polarity_source=polarity_source,
            pos_channel=pos_channel,
            neg_channel=neg_channel,
            registry_id=registry_id,
        )