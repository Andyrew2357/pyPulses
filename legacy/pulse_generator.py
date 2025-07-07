"""
This class represents a multi-instrument object that controls a dc box and dtg
to produce excitation and discharge pulses with the desired heights. The DC box
is wired to Xsg1, Ysg1, Xsg2, and Ysg2 on the pulse shaper box, and the DTG is
wired to all of the clock inputs (AC1, AC1bar, AC2, AC2bar) and the repetitive 
signal averager's trigger input.
"""

from ..pyPulses.devices._registry import DeviceRegistry
from ..pyPulses.devices.abstract_device import abstractDevice
from ..pyPulses.devices.ad5764 import ad5764
from .dtg5274 import dtg5274
from typing import Optional
import json
import os

class pulseGenerator(abstractDevice):
    def __init__(self, loggers = None, config = None):
        if not config:
            fname = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                r'pulse_generator.json'
            )
            with open(fname, 'r') as f:
                config = json.load(f)

        if loggers:
            try:
                logger, dc_logger, dtg_logger = loggers
            except:
                logger = dc_logger = dtg_logger = loggers
        
        super().__init__(logger)        

        self.max_V = config["max_V"]
        self.min_V = config["min_V"]

        self.dcbox_map = config["dcbox_map"]

        self.ac1    = tuple(config["ac1"])
        self.ac1bar = tuple(config["ac1bar"])
        self.ac2    = tuple(config["ac2"])
        self.ac2bar = tuple(config["ac2bar"])
        self.trig   = tuple(config["trig"])

        dcbox_address   = config["dcbox_address"]
        dtg_address     = config["dtg_address"]

        self.dcbox = DeviceRegistry.get_device(dcbox_address)
        if self.dcbox is None:
            self.dcbox = ad5764(dc_logger, instrument_id = dcbox_address)
        
        self.dtg = DeviceRegistry.get_device(dtg_address)
        if self.dtg is None:
            self.dtg = dtg5274(dtg_logger, instrument_id = dtg_address)
        
        self.get_status()
        self.dtg.run(True)

        self.wait = 0.05
        self.max_step = 0.1
    
    def get_V(self, ch):
        """Get the pulse height of a given output."""
        return self.dcbox.get_V(self.dcbox_map[ch])
    
    def set_V(self, ch, V, **kwargs) -> float:
        """Set the pulse height of a given output."""
        if "wait" not in kwargs: kwargs["wait"] = self.wait
        if "max_step" not in kwargs: kwargs["max_step"] = self.max_step

        if not self.min_V <= V < self.max_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"Setting {V} V on {ch} is out of range; Truncating to {Vt} V."
            )
            V = Vt

        self.dcbox.sweep_V(self.dcbox_map[ch], V, **kwargs)
        self.info(f"Set {ch} to {V} V.")
        return V

    def set_clock_rate(self, f):
        """Set the clock rate of the dtg."""
        self.dtg.set_frequency(f)
        self.info(f"Set frequency to {f} Hz.")

    def set_prate(self, prate):
        """Set the relative pulse rate consistently accross all outputs."""
        if self.exc_on():
            self.dtg.set_relative_rate(prate, *self.ac1)
            self.dtg.set_relative_rate(prate, *self.ac1bar)
        if self.dis_on():
            self.dtg.set_relative_rate(prate, *self.ac2)
            self.dtg.set_relative_rate(prate, *self.ac2bar)
        self.dtg.set_relative_rate(prate, *self.trig)
        self.prate = prate
        self.info(f"Set the relative pulse rate to {prate}.")

    def exc_on(self, on: Optional[bool] = None) -> Optional[bool]:
        """Query or set whether excitation pulses are on or off."""
        if on is None:
            print(self.dtg.get_relative_rate(*self.ac1))
            return self.dtg.get_relative_rate(*self.ac1) != 'OFF'

        if on:
            self.dtg.set_relative_rate(self.prate, *self.ac1)
            self.dtg.set_relative_rate(self.prate, *self.ac1bar)
        else:
            self.dtg.set_relative_rate('OFF', *self.ac1)
            self.dtg.set_relative_rate('OFF', *self.ac1bar)
        self.info(f"Turned {'on' if on else 'off'} excitation pulses.")

    def dis_on(self, on: Optional[bool] = None) -> Optional[bool]:
        """Query or set whether discharge pulses are on or off."""
        if on is None:
            return self.dtg.get_relative_rate(*self.ac2) != 'OFF'

        if on:
            self.dtg.set_relative_rate(self.prate, *self.ac2)
            self.dtg.set_relative_rate(self.prate, *self.ac2bar)
        else:
            self.dtg.set_relative_rate('OFF', *self.ac2)
            self.dtg.set_relative_rate('OFF', *self.ac2bar)
        self.info(f"Turned {'on' if on else 'off'} discharge pulses.")

    def set_polarity(self, exc: bool):
        """
        Set the polarity of the excitation and discharge pulses.
        Make sure both ends are consistent with one another.
        """
        self.dtg.set_polarity(exc, *self.ac1)
        self.dtg.set_polarity(not exc, *self.ac1bar)

        self.dtg.set_polarity(not exc, *self.ac2)
        self.dtg.set_polarity(exc, *self.ac2bar)

        self.info(
            f"Set pulse polarities for {'positive' if exc else 'negative'} excitation."
        )

    def set_exc_width(self, W):
        """Set the width for both ends of the excitation pulse."""
        self.dtg.set_width(W, *self.ac1)
        self.dtg.set_width(W, *self.ac1bar)
        self.t_exc = W
        self.info(f"Set excitation pulse width to {W}s.")

    def set_dis_width(self, W):
        """Set the width for both ends of the discharge pulse."""
        self.dtg.set_width(W, *self.ac2)
        self.dtg.set_width(W, *self.ac2bar)
        self.t_dis = W
        self.info(f"Set discharge pulse width to {W}s.")

    def get_widths(self):
        """Make sure the pulse widths we keep internally are accurate."""
        self.t_exc = self.dtg.get_width(*self.ac1)
        self.t_dis = self.dtg.get_width(*self.ac2)

        t_exc_bar = self.dtg.get_width(*self.ac1bar)
        if t_exc_bar != self.t_exc:
            self.warn(
                f"Mismatch in excitation pulse widths detected {self.t_exc} != {t_exc_bar}."
            )
        t_dis_bar = self.dtg.get_width(*self.ac2bar)
        if t_dis_bar != self.t_dis:
            self.warn(
                f"Mismatch in discharge pulse widths detected {self.t_exc} != {t_exc_bar}."
            )

    def get_status(self):
        self.prate = self.dtg.get_relative_rate(*self.trig)
        if self.prate == 'OFF':
            self.warn("Relative rate is set to OFF. This is likely a mistake.")
            self.warn("Use set_prate to set a meaningful relative pulse rate.")

        for ac in [self.ac1, self.ac1bar, self.ac2, self.ac2bar, self.trig]:
            if self.dtg.get_relative_rate(*ac) not in [self.prate, 'OFF']:
                self.warn(
                    f"Relative rates are not set consistently between clocks."
                )
                break
        self.get_widths()
