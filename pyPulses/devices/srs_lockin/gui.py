"""
Backward-compatibility shim.

The vendor-neutral lock-in web GUI now lives in
`pyPulses.devices.lockin.gui`. This module re-exports it under its old name
so existing imports of `pyPulses.devices.srs_lockin.gui.SRSLockinGUI` keep
working.
"""

from ..lockin.gui import LockinGUI

# Backward-compat alias; prefer LockinGUI (pyPulses.devices.lockin.gui).
SRSLockinGUI = LockinGUI
