from .nivisa_utils import find_and_load_gpib
if not find_and_load_gpib():
    print("WARNING: The GPIB library was not found or loaded correctly.")
    print("If you wish to use GPIB instruments, make sure NI488.2 is installed.")

from .ad5764 import ad5764
# from .ad5791 import ad5791
from .dtg5274 import dtg5274
from ._registry import DeviceRegistry
from .keithley2400 import keithley2400
from .keithley2450 import keithley2450
from .mso44 import mso44
from .pulse_generator import pulseGenerator
from .watd_scope import watdScope

__all__ = [
    ad5764,
    # ad5791,
    dtg5274,
    DeviceRegistry,
    keithley2400,
    keithley2450,
    mso44,
    pulseGenerator,
    watdScope
]

visa_dll = 'C:/Windows/System32/visa64.dll'
