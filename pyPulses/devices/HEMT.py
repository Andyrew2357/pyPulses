from .abstract_device import abstractDevice
from .channel_adapter import ScalarChannel
from .sweepable_channel import SweepableChannel

class HEMTCommonSource(abstractDevice):

    def __init__(self,
        VG: SweepableChannel,
        VDD: SweepableChannel,
        RDD: float,
        RSS: float = 0.0,
        VD: ScalarChannel | None = None,
        VS: SweepableChannel | ScalarChannel | None = None,
    ):
        pass

    def ID(self) -> float:
        pass

    def VDS(self) -> float:
        pass

    def VGS(self) -> float:
        pass

    def dissipated_power(self) -> float:
        pass