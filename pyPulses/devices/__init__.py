from .nivisa_utils import find_and_load_gpib
if not find_and_load_gpib():
    pass # Commented out pedantic warning
    # print("WARNING: The GPIB library was not found or loaded correctly.")
    # print("If you wish to use GPIB instruments, make sure NI488.2 is installed.")

from ._registry import DeviceRegistry
from .ad5764 import ad5764
# from .ad5791 import ad5791
from .ad9854 import ad9854
from .cryomagnetics_4G import cryomagnetics4G
from .dtg import dtg5274
from .dtg_diff_pair import dtgDifferentialPair
from .fastflight2 import FastFlight2
from .hf2li import (hf2li, hf2liACout, hf2liDemodChannel, 
                    hf2liOscillator, hf2liOutputChannel)
from .hp34401a import hp34401a
from .ips120 import ips120
from .keithley2000 import keithley2000
from .keithley2400 import keithley2400
from .keithley2450 import keithley2450
from .mso44 import mso44
from .pid_dcbox import PIDbox
from .srs_lockin import (sr830, sr844, sr850, sr860, sr865a)
from .watd_scope import watdScope

__all__ = [
    "ad5764",
    # "ad5791",
    "ad9854",
    "cryomagnetics4G",
    "DeviceRegistry",
    "dtgDifferentialPair",
    "dtg5274",
    "FastFlight2",
    "hf2li",
    "hf2liACout",
    "hf2liDemodChannel",
    "hf2liOscillator",
    "hf2liOutputChannel",
    "hp34401a",
    "ips120",
    "keithley2000",
    "keithley2400",
    "keithley2450",
    "mso44",
    "PIDbox",
    "sr830",
    "sr844",
    "sr850",
    "sr860",
    "sr865a",
    "watdScope"
]

visa_dll = 'C:/Windows/System32/visa64.dll'
