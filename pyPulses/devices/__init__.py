from .nivisa_utils import find_and_load_gpib
if not find_and_load_gpib():
    print("WARNING: The GPIB library was not found or loaded correctly.")
    print("If you wish to use GPIB instruments, make sure NI488.2 is installed.")

from .ad5764 import ad5764
# from .ad5791 import ad5791
from .dtg5274 import dtg5274
from .mso44 import mso44
from .pulse_generator import pulseGenerator
from .watd_scope import watdScope

__all__ = [
    ad5764,
    # ad5791,
    dtg5274,
    mso44,
    pulseGenerator,
    watdScope
]

visa_dll = 'C:/Windows/System32/visa64.dll'