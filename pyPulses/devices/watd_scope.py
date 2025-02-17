
from .abstract_device import abstractDevice
from .mso44 import mso44

class watdScope(abstractDevice):
    def __init__(self, config = None, logger = None):
        super.__init__(logger)
    
        # need to get window for slope fitting
        # need to get window for integration
