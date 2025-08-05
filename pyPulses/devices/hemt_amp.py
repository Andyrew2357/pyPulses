
from .abstract_device import abstractDevice
from ..utils import curves, ezTandemSweep

from typing import Callable, Tuple

class HEMTCommonSource(abstractDevice):
    def __init__(self, VG: Callable, VDD: Callable, RD: float, RS: float,
                 IDS: Callable[[], float],
                 VSS: Callable | str = 'GND', 
                 VM: Callable | None = None,
                 VG_range   : Tuple[float, float] = (-1.2, 0.0),
                 VDD_range  : Tuple[float, float] = (0.0, 4.0),
                 VSS_range  : Tuple[float, float] = (-4.0, 0.0),
                 VM_range   : Tuple[float, float] = (-1.2, 0.0),
                 logger = None):        
        super().__init__(logger)

        self.VG  = VG
        self.VDD = VDD
        self.VSS = VSS
        self.VM  = VM # or (lambda *args: 0.0)

        self.IDS = IDS

        self.RD = RD
        self.RS = RS

        self.VGr  = VG_range
        self.VDDr = VDD_range
        self.VSSr = VSS_range
        self.VMr  = VM_range

    def calibrate(self):
        pass

    def auto_bias(self):
        pass

    def set_power(self):
        pass

    def set_VDD(self):
        pass

    def zero(self):
        pass
