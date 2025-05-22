"""
This class represents a wrapper over the dtg that functions as a pulse generator
without the pulse shaper box.
"""

from ._registry import DeviceRegistry
from .abstract_device import abstractDevice
from .dtg5274 import dtg5274
from typing import Optional

class pulseGeneratorDTG(abstractDevice):
    def __init__(self, loggers = None, config = None):
        if not config:
            config = {
                "max_V" : 2.5,
                "min_V" : 0.0,
                "Vx1"   : ["A", 1],
                "Vy1"   : ["A", 2],
                "Vx2"   : ["B", 1],
                "Vy2"   : ["B", 2],
                "trig"  : ["C", 1],
            }
        self.config = config

        if loggers:
            try:
                logger, dtg_logger = loggers
            except:
                logger = dtg_logger = loggers
        
        super().__init__(logger)        

        self.max_V = config["max_V"]
        self.min_V = config["min_V"]


        self.Vx1    = tuple(config["Vx1"])
        self.Vy1    = tuple(config["Vy1"])
        self.Vx2    = tuple(config["Vx2"])
        self.Vy2    = tuple(config["Vy2"])
        self.trig   = tuple(config["trig"])

        dtg_address = config["dtg_address"]
        self.dtg = DeviceRegistry.get_device(dtg_address)
        if self.dtg is None:
            self.dtg = dtg5274(dtg_logger, instrument_id = dtg_address)
        
        self.get_status()
        self.dtg.run(True)
    
    def get_V(self, ch):
        """Get the pulse height of a given output."""
        return self.dtg.get_high(*self.config[ch])
    
    def set_V(self, ch, V) -> float:
        """Set the pulse height of a given output."""

        if not self.min_V <= V < self.max_V:
            Vt = min(self.max_V, max(self.min_V, V))
            self.warn(
                f"Setting {V} V on {ch} is out of range; Truncating to {Vt} V."
            )
            V = Vt

        self.dtg.set_high(*self.config[ch], V)
        self.info(f"Set {ch} to {V} V.")
        return V

    def set_clock_rate(self, f):
        """Set the clock rate of the dtg."""
        self.dtg.set_frequency(f)
        self.info(f"Set frequency to {f} Hz.")

    def set_prate(self, prate):
        """Set the relative pulse rate consistently accross all outputs."""
        if self.exc_on():
            self.dtg.set_relative_rate(prate, *self.Vx1)
            self.dtg.set_relative_rate(prate, *self.Vy1)
        if self.dis_on():
            self.dtg.set_relative_rate(prate, *self.Vx2)
            self.dtg.set_relative_rate(prate, *self.Vy2)
        self.dtg.set_relative_rate(prate, *self.trig)
        self.prate = prate
        self.info(f"Set the relative pulse rate to {prate}.")

    def exc_on(self, on: Optional[bool] = None) -> Optional[bool]:
        """Query or set whether excitation pulses are on or off."""
        if on is None:
            print(self.dtg.get_relative_rate(*self.Vx1))
            return self.dtg.get_relative_rate(*self.Vx1) != 'OFF'

        if on:
            self.dtg.set_relative_rate(self.prate, *self.Vx1)
            self.dtg.set_relative_rate(self.prate, *self.Vy1)
        else:
            self.dtg.set_relative_rate('OFF', *self.Vx1)
            self.dtg.set_relative_rate('OFF', *self.Vy1)
        self.info(f"Turned {'on' if on else 'off'} excitation pulses.")

    def dis_on(self, on: Optional[bool] = None) -> Optional[bool]:
        """Query or set whether discharge pulses are on or off."""
        if on is None:
            return self.dtg.get_relative_rate(*self.Vx2) != 'OFF'

        if on:
            self.dtg.set_relative_rate(self.prate, *self.Vx2)
            self.dtg.set_relative_rate(self.prate, *self.Vy2)
        else:
            self.dtg.set_relative_rate('OFF', *self.Vx2)
            self.dtg.set_relative_rate('OFF', *self.Vy2)
        self.info(f"Turned {'on' if on else 'off'} discharge pulses.")

    def set_polarity(self, exc: bool):
        """
        Set the polarity of the excitation and discharge pulses.
        Make sure both ends are consistent with one another.
        """
        self.dtg.set_polarity(exc, *self.Vx1)
        self.dtg.set_polarity(not exc, *self.Vy1)

        self.dtg.set_polarity(not exc, *self.Vx2)
        self.dtg.set_polarity(exc, *self.Vy2)

        self.info(
            f"Set pulse polarities for {'positive' if exc else 'negative'} excitation."
        )

    def set_exc_width(self, W):
        """Set the width for both ends of the excitation pulse."""
        self.dtg.set_width(W, *self.Vx1)
        self.dtg.set_width(W, *self.Vy1)
        self.t_exc = W
        self.info(f"Set excitation pulse width to {W}s.")

    def set_dis_width(self, W):
        """Set the width for both ends of the discharge pulse."""
        self.dtg.set_width(W, *self.Vx2)
        self.dtg.set_width(W, *self.Vy2)
        self.t_dis = W
        self.info(f"Set discharge pulse width to {W}s.")

    def get_widths(self):
        """Make sure the pulse widths we keep internally are accurate."""
        self.t_exc = self.dtg.get_width(*self.Vx1)
        self.t_dis = self.dtg.get_width(*self.Vx2)

        t_exc_bar = self.dtg.get_width(*self.Vy1)
        if t_exc_bar != self.t_exc:
            self.warn(
                f"Mismatch in excitation pulse widths detected {self.t_exc} != {t_exc_bar}."
            )
        t_dis_bar = self.dtg.get_width(*self.Vy2)
        if t_dis_bar != self.t_dis:
            self.warn(
                f"Mismatch in discharge pulse widths detected {self.t_exc} != {t_exc_bar}."
            )

    def get_status(self):
        self.prate = self.dtg.get_relative_rate(*self.trig)
        if self.prate == 'OFF':
            self.warn("Relative rate is set to OFF. This is likely a mistake.")
            self.warn("Use set_prate to set a meaningful relative pulse rate.")

        for ac in [self.Vx1, self.Vy1, self.Vx2, self.Vy2, self.trig]:
            if self.dtg.get_relative_rate(*ac) not in [self.prate, 'OFF']:
                self.warn(
                    f"Relative rates are not set consistently between clocks."
                )
                break
        self.get_widths()
