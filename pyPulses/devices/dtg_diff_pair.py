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
        self._post_init()
    
    def _post_init(self):
        self.chx.lhold('LDEL')
        self.chy.lhold('LDEL')
        self.chx.thold('TDEL')
        self.chy.thold('TDEL')

    # Enabled
    def enable(self, on: bool | None = None) -> bool | None:
        """
        Enable or disable the pair output

        Parameters
        ----------
        on : bool
            true means enable
        """
        if on is None:
            return self.chx.enabled()
        self.chx.enabled(on)
        self.chy.enabled(on)

    # Lead Delay
    def ldelay(self, l: float | None = None) -> float | None:
        """Lead delay of `chx`"""
        if l is None:
            return self.chx.ldelay()
        toff = self.ldoff()
        self.chx.ldelay(l)
        self.chy.ldelay(l + toff)
        
    def ldoff(self, dt: float | None = None) -> float | None:
        """Lead delay of `chy` relative to `chx`"""
        if dt is None:
            return self.chy.ldelay() - self.chx.ldelay()
        ldel = self.ldelay()
        self.chy(ldel + dt)
    
    # Trail Delays
    def tdelay(self, t: float | None = None) -> float | None:
        """Trail delay of `chx`"""
        if t is None:
            return self.chx.tdelay()
        toff = self.tdoff()
        self.chx.tdelay(t)
        self.chy.tdelay(t + toff)
        
    def tdoff(self, dt: float | None = None) -> float | None:
        """Trail delay of `chy` relative to `chx`"""
        if dt is None:
            return self.chy.tdelay() - self.chx.tdelay()
        tdel = self.chx.tdelay()
        self.chy.tdelay(tdel + dt)
    
    # Width
    def width(self, w: float | None = None) -> float | None:
        """width of the `chx` pulse"""
        if w is None:
            return self.chx.width()
        woff = self.woff()
        self.chx.width(w)
        self.chy.width(w + woff)

    def woff(self, dw: float | None = None) -> float | None:
        """width of the `chy` pulse relative to `chx`"""
        if dw is None:
            return self.chy.width() - self.chx.width()
        width = self.width()
        self.chy.width(width + dw)

    # Polarity
    def polarity(self, pos: bool | None = None) -> bool | None:
        """polarity of `chx`; `chy` is opposite"""
        if pos is None:
            return self.chx.polarity()
        self.chx.polarity(pos)
        self.chy.polarity(not pos)
    
    # X Levels
    def Xlow(self, V: float | None = None) -> float | None:
        """logical low level of `chx`"""
        return self.chx.low(V)

    def Xhigh(self, V: float | None = None) -> float | None:
        """logical high level of `chx`"""
        return self.chx.high(V)
        
    # Y Levels
    def Ylow(self, V: float | None = None) -> float | None:
        """logical low level of `chy`"""
        return self.chy.low(V)
    
    def Yhigh(self, V: float | None = None) -> float | None:
        """logical high level of `chy`"""
        return self.chy.high(V)