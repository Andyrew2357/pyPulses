from .pyvisa_device import pyvisaDevice
import numpy as np

from typing import Tuple

class tektronix11801b(pyvisaDevice):
    def __init__(self, logger = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """
        self.pyvisa_config = {
            "resource_name": "GPIB0::17::INSTR",
            "timeout": 10,
            "input_buffer_size": 131072,
            "gpib_eos_char": ord('\n'),
            "gpib_eoi_mode": True,
            "gpib_eos_mode": False,

            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

    def get_trace(self) -> Tuple[float, np.ndarray, np.ndarray]:
        
        data = map(float, self.query("CURVe?").split(','))
        spec = data[0]
        data = np.array(data[1:])
        dt = int(self.query("WFMP? XIN")[14:])
        time = dt * np.arange(data.size)
        return spec, time, data
