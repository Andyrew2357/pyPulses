"""Utility classes for tracking DTG state"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from bitarray import bitarray

@dataclass
class Channel:
    name : str
    ch   : int
    slot : str
    mf   : int
    min_V: float
    max_V: float
    min_V_diff: float
    min_width : float
    rise_time : float

    def __post_init__(self):
        self.enabled    : bool    = None
        self.polarity   : bool    = None
        self._high      : float   = None
        self._low       : float   = None
        self._width     : float   = None
        self._ldelay    : float   = None
        self._tdelay    : float   = None
        self.prate      : float   = None
        self.termination_Z: float = None
        self.termination_V: float = None

    def __str__(self) -> str:
        return f"PGEN{self.slot}{self.mf}:CH{self.ch}"
    
    def _id(self) -> str:
        return f"{self.mf}{self.slot}{self.ch}"
    
    @property
    def high(self) -> Optional[float]:
        return self._high
    
    @high.setter
    def high(self, V: float):
        if self._low is not None:
            self._low = min(V - self.min_V_diff, self._low)
        self._high = V

    @property
    def low(self) -> Optional[float]:
        return self._low
    
    @low.setter
    def low(self, V: float):
        if self._high is not None:
            self._high = max(V + self.min_V_diff, self._high)
        self._low = V

    @property
    def width(self) -> Optional[float]:
        return self._width
    
    @width.setter
    def width(self, W: float):
        if self._ldelay is not None:
            self._tdelay = self._ldelay + W
        self._width = W
    
    @property
    def ldelay(self) -> Optional[float]:
        return self._ldelay
    
    @ldelay.setter
    def ldelay(self, l: float):
        if self._width is not None:
            self._tdelay = l + self._width
        self._ldelay = l

    @property
    def tdelay(self) -> Optional[float]:
        return self._tdelay
    
    @tdelay.setter
    def tdelay(self, t: float):
        if self._ldelay is not None:
            self._width = t - self._ldelay
        self._tdelay = t

class Group():
    """Logical bus corresponding to physical channels of the device"""
    def __init__(self, name: str, channels: int | List[Channel | None]):
        self.name = name
        self.width: int = None
        self.channels: List[Channel] = None
        if type(channels) == int:
            self.width = channels
            self.channels = [None] * channels
        else:
            self.width = len(channels)
            self.channels = channels

class Block():
    """
    Pattern data represented in bits. 
    Views of these bits are associated with groups
    """
    def __init__(self, name: str, length: int):
        self.name   = name
        self.length = length
        self.waveforms: List[BlockData] = []

    def add_data(self, group: Group, ch_idx: int, data: bitarray, 
                 offset: int = 0, length: int = None):
        """Add data to the block associated to some logical channel"""
        
        if length is None:
            length = self.length - offset
        waveform = BlockData(group, ch_idx, data, offset, length)
        self.waveforms.append(waveform)
        return waveform

class BlockData():
    def __init__(self, group: Group, ch_idx: int, data: bitarray, 
                 offset: int, length: int):
        self.group  = group
        self.ch_idx = ch_idx
        self.off    = offset
        self.bits   = data[:length]

    def to_array(self) -> np.ndarray:
        return np.array(self.bits.tolist(), dtype = np.uint8).astype(float)
