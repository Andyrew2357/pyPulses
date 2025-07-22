"""
Differential pair for use with Oliver's pulse shaper box. It manipulates two
DTG channels in software to make them behave in the appropriate manner
"""

from .dtg import DTG
from .dtg_utils import Channel

class dtgDifferentialPair():
    """Differential pair for use with Oliver's pulse shaper box"""
    def __init__(self, dtg: DTG, chx: str | Channel, chy: str | Channel):
        """
        Parameters
        ----------
        dtg : DTG
            parent DTG control class that owns the channels
        chx : str | Channel
            channel on the dtg
        chy : str | Channel
            channel on the dtg
        """

        self.dtg = dtg
        self.chx = dtg.get_channel(chx)
        self.chy = dtg.get_channel(chy)

    def enable(self, on: bool):
        """
        Enable or disable the pair output

        Parameters
        ----------
        on : bool
            true means enable
        """
        self.dtg.chan_output(self.chx, on)
        self.dtg.chan_output(self.chy, on)

    @property
    def ldelay(self) -> float:
        """Lead delay of `chx`"""
        if self.chx.ldelay is None:
            return self.dtg.lead_delay(self.chx)
        else:
            return self.chx.ldelay
        
    @ldelay.setter
    def ldelay(self, l: float):
        toff = self.toff
        self.dtg.lead_delay(self.chx, l)
        self.dtg.lead_delay(self.chy, l + toff)

    @property
    def toff(self) -> float:
        """Lead delay of `chy` relative to `chx`"""
        if self.chy.ldelay is None:
            self.dtg.lead_delay(self.chy)

        return self.chy.ldelay - self.chy.ldelay
    
    @toff.setter
    def toff(self, dt: float):
        self.dtg.lead_delay(self.chy, self.ldelay + dt)

    @property
    def width(self) -> float:
        """width of the `chx` pulse"""
        if self.chx.width is None:
            return self.dtg.pulse_width(self.chx)
        else:
            return self.chx.width
        
    @width.setter
    def width(self, w: float):
        woff = self.woff
        self.dtg.pulse_width(self.chx, w)
        self.dtg.pulse_width(self.chy, w + woff)

    @property
    def woff(self) -> float:
        """width of the `chy` pulse relative to `chx`"""
        if self.chy.width is None:
            wy = self.dtg.pulse_width(self.chy)
        else:
            wy = self.chy.width

        return wy - self.width

    @woff.setter
    def woff(self, dw: float):
        self.dtg.pulse_width(self.chx, self.width)
        self.dtg.pulse_width(self.chy, self.width + dw)

    @property
    def polarity(self) -> bool:
        """polarity of `chx`; `chy` is opposite"""
        return self.chx.polarity
    
    @polarity.setter
    def polarity(self, pos: bool):
        self.dtg.polarity(self.chx, pos)
        self.dtg.polarity(self.chy, not pos)

    @property
    def Xlow(self) -> float:
        """logical low level of `chx`"""
        if self.chx.low is None:
            return self.dtg.low_level(self.chx)
        else:
            return self.chx.low
        
    @Xlow.setter
    def Xlow(self, V: float):
        self.dtg.low_level(self.chx, V)

    @property
    def Xhigh(self) -> float:
        """logical high level of `chx`"""
        if self.chx.high is None:
            return self.dtg.high_level(self.chx)
        else:
            return self.chx.high
        
    @Xhigh.setter
    def Xhigh(self, V: float):
        self.dtg.high_level(self.chx, V)

    @property
    def Ylow(self) -> float:
        """logical low level of `chy`"""
        if self.chy.low is None:
            return self.dtg.low_level(self.chy)
        else:
            return self.chy.low
        
    @Ylow.setter
    def Ylow(self, V: float):
        self.dtg.low_level(self.chy, V)

    @property
    def Yhigh(self) -> float:
        """logical high level of `chy`"""
        if self.chy.high is None:
            return self.dtg.high_level(self.chy)
        else:
            return self.chy.high
        
    @Yhigh.setter
    def Yhigh(self, V: float):
        self.dtg.high_level(self.chy, V)
