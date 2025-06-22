from .fastflight2_utils import FastFlight64
from .fastflight_scopeview import FFScopeView
from .abstract_device import abstractDevice
from ._registry import DeviceRegistry

class FastFlight2(abstractDevice):
    def __init__(self, logger = None):
        super().__init__(logger)
        self.ff2 = FastFlight64()
        DeviceRegistry.register_device('FASTFLIGHT2', self)

        # IMPLEMENT ME!!

    def launch_scope_view(self):
        """
        Launch a GUI to use the fastflight as an Oscilloscope
        """
        FFScopeView(self.ff2)
