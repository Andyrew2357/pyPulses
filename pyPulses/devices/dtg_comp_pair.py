"""
Differential pair for use with Oliver's pulse shaper box. It manipulates two
DTG channels in software to make them behave in the appropriate manner
"""

from .dtg import DTG
from .dtg_utils import dtgChannel
from .abstract_device import abstractDevice
from .registry import (
    register_device_class, 
    format_reference, 
    DeferredReference,
    DeviceRegistry,
)

from logging import Logger
from typing import Any, Dict

@register_device_class("dtgCompPair")
class dtgCompPair(abstractDevice):
    """Complementary pair for use with Oliver's pulse shaper box"""
    def __init__(self, 
        dtg: DTG | DeferredReference, 
        chx: str | dtgChannel | DeferredReference, 
        chy: str | dtgChannel | DeferredReference,
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_post_init: bool = False,
    ):
        """
        Parameters
        ----------
        dtg : DTG
            parent DTG control class that owns the channels
        chx : str | dtgChannel
            channel on the dtg
        chy : str | dtgChannel
            channel on the dtg
        """

        super().__init__(logger)
        DeviceRegistry.register(self, registry_id=registry_id)

        self.dtg = dtg
        self.chx = chx
        self.chy = chy
        if not skip_post_init:
            self._post_init()
    
    def _post_init(self):
        self.chx = self.dtg.get_channel(self.chx)
        self.chy = self.dtg.get_channel(self.chy)
        self.chx.lhold('LDEL')
        self.chy.lhold('LDEL')
        self.chx.thold('TDEL')
        self.chy.thold('TDEL')

    def _serialize_state(self) -> Dict[str, Any]:
        try:
            dtg = format_reference(self.dtg)
            chx = format_reference(self.chx)
            chy = format_reference(self.chy)
        except:
            dtg = None
            chx = None
            chy = None

        return {
            'DTG': dtg,
            'CHX': chx,
            'CHY': chy,
        }

    def _deserialize_state(self, state: Dict[str, Any]):
        if 'DTG' in state:
            self.dtg = DeferredReference(state['DTG'])
        if 'CHX' in state:
            self.chx = DeferredReference(state['CHX'])
        if 'CHY' in state:
            self.chy = DeferredReference(state['CHY'])
        
        self._resolve_references()
        self._post_init()
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'dtgCompPair':
        dtg = config.pop('DTG')
        if dtg is not None:
            dtg = DeferredReference(dtg)
        chx = config.pop('CHX')
        if chx is not None:
            chx = DeferredReference(chx)
        chy = config.pop('CHY')
        if chy is not None:
            chy = DeferredReference(chy)
        registry_id = config.pop('registry_id')

        instance = cls(
            dtg=dtg,
            chx=chx,
            chy=chy,
            registry_id=registry_id,
            skip_post_init=True, # We only call post_init after _deserialize_state
        )
        return instance

    def _resolve_references(self):
        if isinstance(self.dtg, DeferredReference):
            self.dtg = self.dtg.unwrap()
        if isinstance(self.chx, DeferredReference):
            self.chx = self.chx.unwrap()
        if isinstance(self.chy, DeferredReference):
            self.chy = self.chy.unwrap()

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
            return self.chx.enable()
        self.chx.enable(on)
        self.chy.enable(on)

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
        self.chy.ldelay(ldel + dt)
    
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
    def xpolarity(self, pos: bool | None = None) -> bool | None:
        """polarity of `chx`"""
        if pos is None:
            return self.chx.polarity()
        self.chx.polarity(pos)

    def ypolarity(self, pos: bool | None = None) -> bool | None:
        """polarity of `chy`"""
        if pos is None:
            return self.chy.polarity()
        self.chy.polarity(pos)

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