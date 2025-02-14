"""
This class is an interface for communicating with the DTG5274 data timing
generator. It implements only a small fraction of the functionality offered
by the instrument. It is primarily used by the pulseGenerator class, which
coordinates multiple instruments to generate clean excitation and discharge
pulses.
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
import pyvisa.constants
from typing import Optional

class dtg5274(pyvisaDevice):
    def __init__(self, logger: Optional[str] = None, 
                 instrument_id: Optional[str] = None):
        self.config = {
            "resource_name" : "GPIB0::27::INSTR",
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        # Set Output Buffer Size to 512 bytes
        self.device.set_buffer(pyvisa.constants.VI_WRITE_BUF, 512)
        
        # Set EOS Character Code to LF (Line Feed, ASCII 10)
        self.device.set_visa_attribute(
            pyvisa.constants.VI_ATTR_TERMCHAR, ord('\n')
        )

        # Set EOI Mode to 'on'
        self.device.send_end = True

        # Set EOS Mode to 'none'
        self.device.set_visa_attribute(
            pyvisa.constants.VI_ATTR_TERMCHAR_EN, False
        )

        # Do a level calibration
        self.device.query("*CAL?")

        # Right now we only support pulse generator mode
        self.device.write("TBAS:OMODe PULS")

    def run(self, on: bool):
        """Turn all outputs on or off."""
        self.device.write(f"TBAS:RUN {'ON' if on else 'OFF'}")
        self.info(f"DTG5274: Turned {'on' if on else 'off'} all outputs.")

    def set_frequency(self, f: int):
        """Set the internal clock frequency (input is in Hz)."""
        self.device.write(f"TBAS:FREQ {f}")
        self.info(f"DTG5274: Set frequency to {f}.")

    def get_frequency(self) -> int:
        """Query the device frequency."""
        return int(self.device.query("TBAS:FREQ?"))
    
    def set_relative_rate(self, prate, slot, channel, mainframe = 1):
        """
        Set the relative pulse rate on a given output.
        Describes the rate of pulses with respect to the clock rate:
        'NORM', 'HALF', 'QUAR', 'EIGH', 'SIXT', or 'OFF'
        """
        self.device.write(f"PGEN{slot}{mainframe}:CH{channel}:PRATe {prate}")
        self.info(
            f"DTG5274: Set prate on {slot}{mainframe}:{channel} to {prate}"
        )

    def get_relative_rate(self, slot, channel, mainframe = 1):
        """Query the relative pulse rate on a given output."""
        return self.device.query(f"PGEN{slot}{mainframe}:CH{channel}:PRATe?")[:-1]

    def set_polarity(self, pos: bool, slot, channel, mainframe = 1):
        """Set the polarity of the pulses on a given output."""
        self.device.write(
            f"PGEN{slot}{mainframe}:CH{channel}:POLarity {'NORM' if pos else 'INV'}"
        )
        self.info(
            f"DTG5274: Set polarity on {slot}{mainframe}:{channel} 
            {'positive' if pos else 'negative'}")
        
    def get_polarity(self, slot, channel, mainframe = 1) -> bool:
        """Query the polarity of the pulses on a given output."""
        pol = self.device.query(f"PGEN{slot}{mainframe}:CH{channel}:POLarity?")
        return pol == 'NORM\n'
    
    def set_low(self, V, slot, channel, mainframe = 1):
        """Set the low level for pulses on a given output (intput is in V)."""
        if not (-1.2 <= V <= 2.6):
            Vt = min(2.6, max(-1.2, V))
            self.warn(
                f"DTG5274: Logical low {V}V is out of range; truncating to {Vt}V"
            )
            V = Vt

        self.device.write(f"PGEN{slot}{mainframe}:CH{channel}:LOW {V}")
        self.info(
            f"DTG5274: Set Logical low on {slot}{mainframe}:{channel} to {V}V"
        )

    def get_low(self, slot, channel, mainframe = 1) -> float:
        """Get the low level for pulses on a given output."""
        return float(
            self.device.query(f"PGEN{slot}{mainframe}:CH{channel}:LOW?")
        )
    
    def set_high(self, V, slot, channel, mainframe = 1):
        """Set the high level for pulses on a given output (input is in V)."""
        if not (-1.1 <= V <= 2.7):
            Vt = min(2.7, max(-1.1, V))
            self.warn(
                f"DTG5274: Logical high {V}V is out of range; truncating to {Vt}V"
            )
            V = Vt

        self.device.write(f"PGEN{slot}{mainframe}:CH{channel}:HIGH {V}")
        self.info(
            f"DTG5274: Set Logical high on {slot}{mainframe}:{channel} to {V}V"
        )

    def get_low(self, slot, channel, mainframe = 1) -> float:
        """Get the high level for pulses on a given output."""
        return float(
            self.device.query(f"PGEN{slot}{mainframe}:CH{channel}:HIGH?")
        )

    def set_width(self, W, slot, channel, mainframe = 1):
        """Set the pulse width on a given output (input is in s)"""
        self.device.write(f"PGEN{slot}{mainframe}:CH{channel}:WIDTh {W}")
        self.info(
            f"DTG5274: Set pulse width on {slot}{mainframe}:{channel} to {W}"
        )

    def get_width(self, slot, channel, mainframe = 1) -> float:
        """Query the pulse width for pulses on a given output."""
        return float(
            self.device.query(f"PGEN{slot}{mainframe}:CH{channel}:WIDTh?")
        )
