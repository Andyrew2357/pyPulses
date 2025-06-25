"""This class is an interface to the HP34401A digital multimeter"""

from .pyvisa_device import pyvisaDevice

class hp34401a(pyvisaDevice):
    def __init__(self, logger = None, instrument_id: str = None):
        
        self.pyvisa_config = {
            "resource_name" : "GPIB1::18::INSTR",
            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

    def get_V(self) -> float:
        """Query the voltage."""
        return float(self.device.query(":MEAS:VOLT:DC?").strip())
