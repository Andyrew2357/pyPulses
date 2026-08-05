"""
Vendor-neutral lock-in amplifier infrastructure shared by lock-in drivers
(SRS sr830/sr844/sr850/sr860/sr865a, AMETEK/Signal Recovery 7280, ...).
"""

from .base import LockinLike
from .channels import lockin_channel, sensitivity_channel, series_correlated_covariance
from .gui import LockinGUI

__all__ = [
    "LockinGUI",
    "LockinLike",
    "lockin_channel",
    "sensitivity_channel",
    "series_correlated_covariance",
]
