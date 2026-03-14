from .abstract_device import abstractDevice
from .registry import (
    DeviceRegistry, 
    DeferredReference,
    register_device_class,
    format_reference,
)

from .channel_adapter import ScalarChannelAdapter
from .calibration import CalibrationModel
from .calibrated_channel import CalibratedChannel

from logging import Logger
from typing import Any, Dict, List, Tuple

@register_device_class("PulseAmplitudeManager")
class PulseAmplitudeManager(abstractDevice):
    def __init__(self,
        ch1: CalibratedChannel | DeferredReference,
        ch2: CalibratedChannel | DeferredReference,
        ch1_cal: Dict[Tuple[bool, bool], CalibrationModel],
        ch2_cal: Dict[Tuple[bool, bool], CalibrationModel],
        sigmas: List[bool],
        registry_id: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

        self.channel_handlers = [
            PulseAmplitudeChannel(self, 'ch1', 0),
            PulseAmplitudeChannel(self, 'ch2', 1)
        ]
        self.channels = [ch1, ch2]
        self.calibrations = [ch1_cal, ch2_cal]
        self.sigmas = list(sigmas)
        assert len(self.sigmas) == 2
        self._polarity_changed = True

    def update_polarity(self, ind: int, sigma: bool):
        if self.sigmas[ind] != sigma:
            self._polarity_changed = True
            self.sigmas[ind] = sigma

    def recalibrate(self):
        s1, s2 = self.sigmas
        if not (s1 is None or s2 is None):
            for ch, cal in zip(self.channels, self.calibrations):
                ch.set_calibration(cal[(s1, s2)])
        for handler, ch in zip(self.channel_handlers, self.channels):
            if handler._target is not None:
                ch.set_output(handler._target)
        self._polarity_changed = False

    def set_output(self, ind: int, val: float):
        if self._polarity_changed:
            self.recalibrate()
        else:
            self.channels[ind].set_output(val)

    def get_output(self, ind: int) -> float:
        if self._polarity_changed:
            self.recalibrate()
        return self.channels[ind].get_output()
    
    def resolve(self, accessor: str) -> 'PulseAmplitudeChannel':
        if accessor == 'ch1':
            return self.channel_handlers[0]
        if accessor == 'ch2':
            return self.channel_handlers[1]
        raise ValueError(f"Unknown accessor: {accessor}")
    
    def _serialize_state(self) -> Dict[str, Any]:
        # Serialize calibration dicts: convert tuple keys to string keys
        def serialize_cal_dict(cal_dict: Dict[Tuple[bool, bool], CalibrationModel]) -> Dict[str, Any]:
            return {
                f"{s1},{s2}": cal.to_dict()
                for (s1, s2), cal in cal_dict.items()
            }
        
        return {
            'ch1': format_reference(self.channels[0]),
            'ch2': format_reference(self.channels[1]),
            'ch1_cal': serialize_cal_dict(self.calibrations[0]),
            'ch2_cal': serialize_cal_dict(self.calibrations[1]),
            'sigmas': self.sigmas,
            'targets': [h._target for h in self.channel_handlers],
        }
    
    def _deserialize_state(self, state: Dict[str, Any]):
        # Deserialize calibration dicts: convert string keys back to tuple keys
        def deserialize_cal_dict(cal_dict: Dict[str, Any]) -> Dict[Tuple[bool, bool], CalibrationModel]:
            result = {}
            for key, cal_data in cal_dict.items():
                s1_str, s2_str = key.split(',')
                s1 = s1_str.strip() == 'True'
                s2 = s2_str.strip() == 'True'
                result[(s1, s2)] = CalibrationModel.from_config(cal_data)
            return result
        
        if 'ch1' in state:
            self.channels[0] = DeferredReference(state['ch1'])
        if 'ch2' in state:
            self.channels[1] = DeferredReference(state['ch2'])
        
        if 'ch1_cal' in state:
            self.calibrations[0] = deserialize_cal_dict(state['ch1_cal'])
        if 'ch2_cal' in state:
            self.calibrations[1] = deserialize_cal_dict(state['ch2_cal'])
        
        if 'sigmas' in state:
            self.sigmas = list(state['sigmas'])
        
        if 'targets' in state:
            for handler, target in zip(self.channel_handlers, state['targets']):
                handler._target = target
        
        self._polarity_changed = True
        self._resolve_references()

    def _resolve_references(self):
        if isinstance(self.channels[0], DeferredReference):
            self.channels[0] = self.channels[0].unwrap()
        if isinstance(self.channels[1], DeferredReference):
            self.channels[1] = self.channels[1].unwrap()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PulseAmplitudeManager':
        # Deserialize calibration dicts
        def deserialize_cal_dict(cal_dict: Dict[str, Any]) -> Dict[Tuple[bool, bool], CalibrationModel]:
            result = {}
            for key, cal_data in cal_dict.items():
                s1_str, s2_str = key.split(',')
                s1 = s1_str.strip() == 'True'
                s2 = s2_str.strip() == 'True'
                result[(s1, s2)] = CalibrationModel.from_config(cal_data)
            return result
        
        ch1 = config.pop('ch1')
        if ch1 is not None:
            ch1 = DeferredReference(ch1)
        
        ch2 = config.pop('ch2')
        if ch2 is not None:
            ch2 = DeferredReference(ch2)
        
        ch1_cal = deserialize_cal_dict(config.pop('ch1_cal'))
        ch2_cal = deserialize_cal_dict(config.pop('ch2_cal'))
        sigmas = config.pop('sigmas')
        registry_id = config.pop('registry_id')
        
        instance = cls(
            ch1=ch1,
            ch2=ch2,
            ch1_cal=ch1_cal,
            ch2_cal=ch2_cal,
            sigmas=sigmas,
            registry_id=registry_id,
        )
        
        # Restore targets if present
        if 'targets' in config:
            for handler, target in zip(instance.channel_handlers, config['targets']):
                handler._target = target
        
        return instance
        

class PulseAmplitudeChannel(ScalarChannelAdapter):
    def __init__(self, 
        parent: PulseAmplitudeManager, 
        accessor: str,
        ind: int,
    ):
        super().__init__(parent, accessor)
        self._ind = ind
        self._target: float | None = None

    def update_polarity(self, sigma: bool):
        self._parent.update_polarity(self._ind, sigma)

    def recalibrate(self):
        self.parent.recalibrate()

    def set_output(self, val: float):
        self._target = val
        self._parent.set_output(self._ind, val)

    def get_output(self) -> float:
        return self._parent.get_output(self._ind)