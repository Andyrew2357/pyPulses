"""
Utility classes for tracking DTG state
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from bitarray import bitarray

from .pyvisa_device import pyvisaDevice

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
    dtg: pyvisaDevice

    def __str__(self) -> str:
        return f"PGEN{self.slot}{self.mf}:CH{self.ch}"
    
    def _id(self) -> str:
        return f"{self.mf}{self.slot}{self.ch}"
    
    def query(self, *args, **kwargs):
        return self.dtg.query(*args, **kwargs)
    
    def write(self, *args, **kwargs):
        self.dtg.write(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.dtg.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.dtg.warn(*args, **kwargs)
    
    def high(self, V: float | None = None) -> float | None:
        if V is None:
            return float(self.query(f"{self}:HIGH?"))
        V = min(self.max_V, max(self.min_V + self.min_V_diff, V))
        self.write(f"{self}:HIGH {V}")
        self.info(f"Set logical high level of channel {self._id()} to {V} V.")

    def low(self, V: float | None = None) -> float | None:
        if V is None:
            return float(self.query(f"{self}:LOW?"))
        V = min(self.max_V - self.min_V_diff, max(self.min_V, V))
        self.write(f"{self}:LOW {V}")
        self.info(f"Set logical low level of channel {self._id()} to {V} V.")

    def lhold(self, mode: str | None = None) -> str | None:
        if mode is None:
            return self.query(f"{self}:LHOLD?").strip()
        if mode not in ['LDEL', 'PHAS']:
            self.warn('Invalid request for LHOLD. Try `LDEL` or `PHAS`.')
            return
        self.write(f"{self}:LHOLD {mode}")
        self.info(f"Set lead hold mode of channel {self._id()} to {mode}.")

    def thold(self, mode: str | None = None) -> str | None:
        if mode is None:
            return self.query(f"{self}:THOLD?").strip()
        if mode not in ['TDEL', 'DCYC', 'WIDT']:
            self.warn('Invalid request for THOLD. Try `TDEL`, `DCYC`, or `WIDT`.')
            return
        self.write(f"{self}:THOLD {mode}")
        self.info(f'Set trail hold mode of channel {self._id()} to {mode}.')

    def width(self, W: float | None = None) -> float | None:
        if W is None:
            return float(self.query(f"{self}:WIDTh?"))
        self.write(f"{self}:WIDTh {W}")
        self.info(f"Set pulse width of channel {self._id()} to {W} s.")

    def ldelay(self, l: float | None = None) -> float | None:
        if l is None:
            return float(self.query(f"{self}:LDELay?"))
        self.write(f"{self}:LDELay {l}")
        self.info(f"Set lead delay on channel {self._id()} to {l} s.")

    def tdelay(self, t: float | None = None) -> float | None:
        if t is None:
            return float(self.query(f"{self}:TDELay?"))
        self.write(f"{self}:TDELay {t}")
        self.info(f"Set trail delay on channel {self._id()} to {t} s.")

    def prate(self, prate: float | None = None) -> float | None:        
        
        rel = {
            'NORM': 1.0, 
            'HALF': 0.5, 
            'QUAR': 0.25, 
            'EIGH': 0.125, 
            'SIXT': 0.0625, 
            'OFF': 0
        }
        
        if prate is None:
            return rel[self.query(f"{self}:PRATE?").strip()]

        if prate >= 1.0:
            val = 'NORM'
        elif prate >= 0.5:
            val = 'HALF'
        elif prate >= 0.25:
            val = 'QUAR'
        elif prate >= 0.125:
            val = 'EIGH'
        elif prate >= 0.0625:
            val = 'SIXT'
        else:
            val = 'OFF'

        self.write(f"{self}:PRATE {val}")
        self.info(f"Set relative rate of channel {self._id()} to {rel[val]}")

    def polarity(self, pos: bool | None = None) -> bool | None:
        if pos is None:
            return self.query(f"{self}:POLarity?").strip() == 'NORM'
        self.write(f"{self}:POLarity {'NORM' if pos else 'INV'}")
        self.info(f"Set polarity of channel {self._id()} to {'posi' if pos else 'nega'}tive.")

    def enabled(self, on: bool | None = None) -> bool | None:
        if on is None:
            return int(self.query(f"{self}:OUTPut?")) == 1
        self.write(f"{self}:OUTPut {'ON' if on else 'OFF'}")
        self.info(f"{'En' if on else 'Dis'}abled channel {self._id()}.")

    def termination_Z(self, Z: float | None = None) -> float | None:
        if Z is None:
            return float(self.query(f"{self}:TIMPedance?"))
        if Z >= 1e3:
            Z = 1e3
        else:
            Z = 50
        self.write(f"{self}:TIMPedance {Z}")
        self.info(f"Set termination impedance on channel {self._id()} to {Z} Ohm.")

    def termination_V(self, V: float | None = None) -> float | None:
        if V is None:
            return float(self.query(f"{self}:TVOLtage?"))
        V = max(-2.0, min(5.0, V))
        self.write(f"{self}:TVOLtage {V}")
        self.info(f"Set termination voltage on channel {self._id()} to {V} V.")

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
