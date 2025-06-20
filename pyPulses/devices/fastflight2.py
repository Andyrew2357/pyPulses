from fastflight2_utils.fastflight64 import FastFlight64
from .abstract_device import abstractDevice
from ._registry import DeviceRegistry

class FastFlight2(abstractDevice):
    def __init__(self, logger = None):
        super().__init__(logger)
        self.FF2 = FastFlight64()
        DeviceRegistry.register_device('FASTFLIGHT2', self)

        # IMPLEMENT ME!!
