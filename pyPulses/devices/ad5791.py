from .pyvisa_device import pyvisaDevice
from .channel_adapter import ScalarChannelAdapter
from .registry import register_hardware_class

import numpy as np
import time
import json
import os
from math import ceil
from logging import Logger
from typing import Any, Dict, List

@register_hardware_class("ad5791")
class ad5791(pyvisaDevice):
    """
    Class representation of the Arduino-AD5791 DC box.
    
    The instrument in question has an Arduino Uno connected to the Analog Devices 
    DAC that takes serial bus input to set 20-bit bipolar DC outputs on 8 channels.
    """

    DEFAULT_PYVISA_CONFIG = {
        'baud_rate': 115200,
        'write_termination': '\n',
        'read_termination': '\n',
        'max_retries': 1,
        'min_interval': 0.05,
        'timeout': 3000
    }

    # maximum bounds on channel values
    max_V = 10.
    min_V = -10.

    def __init__(self,
        resource_name: str,
        registry_id: str | None = None,
        logger: Logger | None = None, 
        skip_connect: bool = False,
        calibration: Dict[int, List[float]] | None = None,
        **kwargs
    ):
        
        """
        Parameters
        ----------
        resource_name : str
            VISA resource name.
        registry_id : str, optional
            Name to register this instance under in the HardwareRegistry
        logger : Logger, optional
            logger used by abstractDevice.
        calibration : dict, optional
            calibration data for the DC channels of the form [a, b] (Vout = a * Vraw + b)
        **kwargs
        """

        super().__init__(resource_name, registry_id, logger, skip_connect, **kwargs)

        # sweep parameters
        self.max_step = 0.05
        self.wait = 0.1

        if calibration is None:
            self.calibration = {ch: [1.0, 0.0] for ch in range(self.NUM_CHANNELS)}
        else:
            self.calibration = {int(k): v for k, v in calibration.items()}

        # arduino reset delay after connect
        if not skip_connect:
            time.sleep(2.5)

    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize all state needed to reconstruct and restore this device.
        """

        config = super()._serialize_state()
        
        config.update({
            'max_step': self.max_step,
            'step_wait': self.wait,
            'calibration': self.calibration,
        })
        
        return config

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """
        Restore device state from serialized config.
        
        Called when device already exists and we want to apply saved settings.
        """

        super()._deserialize_state(state)
        
        if 'max_step' in state:
            self.max_step = state['max_step']
        if 'step_wait' in state:
            self.wait = state['step_wait']
        if 'calibration' in state:
            self.calibration = {int(k): v for k, v in state['calibration'].items()}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ad5791':
        """
        Construct from serialized config.
        
        Parameters
        ----------
        config : dict
            Output from _serialize_state(), plus 'registry_id'.
        """

        # Extract required fields
        registry_id = config.pop('registry_id')
        resource_name = config.pop('resource_name')
    
        # Construct instance
        instance = cls(
            resource_name=resource_name,
            registry_id=registry_id,
            skip_connect=False,
            **config  # Remaining kwargs go to pyvisaDevice
        )
        instance._deserialize_state(config)

        return instance

    def resolve(self, accessor: str) -> 'ad5791_channel':
        try:
            assert accessor.startswith('ch')
            ch = int(accessor[2:])
            assert 0 <= ch <= 7
        except:
            return None
        return ad5791_channel(self, accessor, ch)

    # Output enable

    def output(self, ch: int, on: bool = None) -> bool | None:
        """
        Set or query the output state on the desired channel.

        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        on : bool, Optional

        Returns
        -------
        bool or None
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if on is None:
            return int(self.query(f"OUTP{ch}?"))
        
        self.write(f"OUTP{ch} {int(on)}")
        self.info(f"{'En' if on else 'Dis'}abled channel {ch} output; "
                   "setting to a maximally clean zero...")
        self.set_V(ch, 0.0)

    def output_all(self, on: bool):
        """
        Enable or disable all outputs.
        """

        for ch in range(8):
            self.output(ch, on)

    # Raw uncalibrated voltages

    def sweep_raw_V(self, ch: int, V: float, 
                    max_step: float = None, wait: float = None):
        """
        Sweep DC value of a given channel smoothly to the target. This is the 
        uncalibrated value inferred from +-10V rails.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        max_step : float, default=None
            maximum step between voltages while sweeping.
        wait : float, default=None
            wait time between steps while sweeping.
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5791 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        if not max_step:
            max_step = self.max_step
        if not wait:
            wait = self.wait

        start = self.get_raw_V(ch)
        dist = abs(V - start)
        num_step = ceil(dist / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            time.sleep(wait)
            self.set_raw_V(ch, v, chatty = False)
        
        self.info(f"Swept channel {ch} to {V} V (raw).")

    def get_raw_V(self, ch: int):
        """
        Get the raw DC value on a given channel. This is the uncalibrated value
        inferred from +-10V rails.

        Parameters
        ----------
        ch : int
            target channel.

        Returns
        -------
        V : float
            voltage on the target channel Note: This is not a true query. It 
            simply returns what is saved on the computer. There is currently 
            no way to ask the arduino directly.
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return None

        return float(self.query(f"VOLT{ch}?"))       

    def set_raw_V(self, ch: int, V: float, chatty: bool = True):
        """
        Set the raw DC value of a given channel. This is the uncalibrated value
        inferred from +-10V rails.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        chatty : bool, default=True
            whether to log the change in channel settings.
        """

        if ch not in range(0, 8):
            self.error(f"AD5791 does not have a channel {ch}.")
            return
        
        if V > self.max_V or V < self.min_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"{V} on AD5791 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt

        self.write(f"VOLT{ch} {V}")
        self.info(f"Set channel {ch} to {V} V (raw).")

    # Calibrated voltages

    def _raw_to_cal(self, ch: int, V: float) -> float:
        """
        Return the calibrated voltage corresponding to a raw, uncalibrated 
        voltage.
        """

        a, b = self.calibration[ch]
        return a*V + b
    
    def _cal_to_raw(self, ch: int, V: float) -> float:
        """
        Return the raw, uncalibrated voltage corresponding to a calibrated
        voltage.
        """

        a, b = self.calibration[ch]
        return (V - b) / a

    def sweep_V(self, ch: int, V: float, 
                max_step: float = None, wait: float = None):
        """
        Sweep DC value of a given channel smoothly to the target.
        
        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        max_step : float, default=None
            maximum step between voltages while sweeping.
        wait : float, default=None
            wait time between steps while sweeping.
        """

        self.sweep_raw_V(ch, self._cal_to_raw(ch, V), max_step, wait)
        self.info(f"Swept channel {ch} to calibrated value {V} V.")

    def get_V(self, ch: int):
        """
        Get the DC value on a given channel.

        Parameters
        ----------
        ch : int
            target channel.

        Returns
        -------
        V : float
            voltage on the target channel Note: This is not a true query. It 
            simply returns what is saved on the computer. There is currently 
            no way to ask the arduino directly.
        """

        return self._raw_to_cal(ch, self.get_raw_V(ch))
    
    def set_V(self, ch: int, V: float, chatty: bool = True):
        """
        Set the DC value of a given channel.

        Parameters
        ----------
        ch : int
            target channel (0 through 7).
        V : float
            target voltage.
        chatty : bool, default=True
            whether to log the change in channel settings.
        """

        self.set_raw_V(ch, self._cal_to_raw(ch, V), chatty = False)
        if chatty:
            self.info(f"Set channel {ch} to calibrated value {V} V.")

    def load_calibration(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                r'ad5791_cal.json'
            )
        with open(path, 'r') as f:
            self.calibration = json.load(f)
            self.calibration = {int(k): v for k, v in self.calibration.items()}

class ad5791_channel(ScalarChannelAdapter):
    def __init__(self, parent: ad5791, accessor: str, ch: int):
        super().__init__(parent, accessor)
        self.ch = ch

    def get_output(self) -> float:
        return self._parent.get_V(self.ch)

    def set_output(self, value: float):
        self._parent.set_V(self.ch, value, chatty=False)

    def get_enable(self) -> bool:
        return self._parent.output()

    def set_enable(self, enabled: bool):
        self._parent.output(enabled)