from .nivisa_utils import find_and_load_gpib
if not find_and_load_gpib():
    pass # Commented out pedantic warning
    # print("WARNING: The GPIB library was not found or loaded correctly.")
    # print("If you wish to use GPIB instruments, make sure NI488.2 is installed.")

from .ad5764 import ad5764
from .ad5791 import ad5791
from .ad9854 import ad9854
from .attenuator import FixedAttenuator
from .calibrated_channel import CalibratedChannel, PolarityCalibratedChannel
from .calibration import PolynomialCalibration, TrivialCalibration
from .calibrate_pulse_shaper import PulseShaperCalibration, PulseShaperCalibrationResult
from .cryomagnetics_4G import cryomagnetics4G
from .dtg import dtg5274
from .dtg_comp_pair import dtgCompPair
from .fastflight2 import FastFlight2
# from .hemt_amp import HEMTCommonSource
from .hf2li import (hf2li, hf2liACout, hf2liDemodChannel, 
                    hf2liOscillator, hf2liOutputChannel)
from .hp34401a import hp34401a
from .ips120 import ips120
from .keithley2000 import keithley2000
from .keithley2400 import keithley2400
from .keithley2450 import keithley2450
from .keithley2700 import keithley2700
from .mso44 import mso44
from .pcm1704 import pcm1704
from .pid_dcbox import PIDbox
from .pulse_pair import pulsePair
from .srs_lockin import (sr830, sr844, sr850, sr860, sr865a)
from .wfatd import wfAverager, wfBalance, wfJump, wfSlope

__all__ = [
    "ad5764",
    "ad5791",
    "ad9854",
    "cryomagnetics4G",
    "CalibratedChannel",
    "dtgCompPair",
    "dtg5274",
    "FastFlight2",
    "FixedAttenuator",
    # "HEMTCommonSource",
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
    "keithley2700",
    "mso44",
    "pcm1704",
    "PIDbox",
    "PolarityCalibratedChannel",
    "PolynomialCalibration",
    "pulsePair",
    "PulseShaperCalibration",
    "PulseShaperCalibrationResult",
    "sr830",
    "sr844",
    "sr850",
    "sr860",
    "sr865a",
    "TrivialCalibration",
    "wfAverager", 
    "wfBalance", 
    "wfJump", 
    "wfSlope"
]

visa_dll = 'C:/Windows/System32/visa64.dll'
