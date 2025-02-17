# This class is a bare-bones framework for low-level devices that use the pyvisa 
# package for communication. This includes the majority of standalone instruments.

from .abstract_device import abstractDevice
import pyvisa

class pyvisaDevice(abstractDevice):
    def __init__(self, pyvisa_config, logger = None):
        """Standard initialization, calling ResourceManager.open_resource."""
        super().__init__(logger)
        rm = pyvisa.ResourceManager('@py')
        self.device = rm.open_resource(**pyvisa_config)
