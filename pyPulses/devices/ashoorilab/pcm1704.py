"""
This class is an interface to the PCM1704-based 24-bit DC box.
"""

from ...utils.curves import MonotonicPiecewiseLinear

from ..channel_adapter import ScalarChannelAdapter
from ..registry import register_hardware_class

from .usb6501 import USB6501

import os
import time
import json
import numpy as np
from math import ceil
from typing import Any, Dict


@register_hardware_class("pcm1704")
class pcm1704(USB6501):
    """
    Class representation of the PCM1704-based 24-bit DC box.
    
    Wiring (USB-6501 Port A)
    ------------------------
        Bit 0  SCLK   serial clock
        Bit 1  SDATA  serial data (MSB first)
        Bit 2  AD2    channel-address bit 2
        Bit 3  AD1    channel-address bit 1
        Bit 4  AD0    channel-address bit 0
        Bit 5  WCE1   write/chip-enable 1  (latch pulse)
        Bit 6  WCE0   write/chip-enable 0  (data-load window)
    
    Port B is wired to the optoisolator outputs and is used only by the hardware
    self-test; it plays no role in normal DAC writes.
    """

    # Bit assignments on Port A
    SCLK  = 0x01
    SDATA = 0x02
    AD2   = 0x04
    AD1   = 0x08
    AD0   = 0x10
    WCE1  = 0x20
    WCE0  = 0x40
 
    _CTRL_PORT = USB6501.PORT_A   # output: SPI control lines → PCM1704
    _MON_PORT  = USB6501.PORT_B   # input:  optoisolator feedback (self-test only)

    v_fullscale   = 12.0
    bits_max      = 0xFFFFFF
    bits_half_max = 0x7FFFFF
    n_ch          = 8

    max_step = 0.05
    wait     = 0.1

    def __init__(
        self,
        serial: str | None = None,
        calibration: dict | str = 'Darjeeling',
        registry_id: str | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        serial : str or None, default=None
            USB serial number of the NI USB-6501 (e.g. ``'013597CB'``).
            If None, the first USB-6501 found on any bus is used.
        calibration : dict or str, default='Darjeeling'
            DAC calibration data.  The presets `'Lipton'` and `'Darjeeling'`
            correspond to the two boxes in the lab.
        logger : Logger, optional
        """
        super().__init__(serial=serial, registry_id=registry_id, logger=logger, **kwargs)

        if isinstance(calibration, str):
            self.load_calibration(calibration)
        else:
            self.calibration = calibration

        self.ch_bits = [0.0] * self.n_ch

        # Configure port directions
        self.set_input_mode(self._CTRL_PORT, is_input=False)
        self.set_input_mode(self._MON_PORT,  is_input=True)
 
        # Drive the idle state: WCE1 asserted, everything else low
        self.set_output(self._CTRL_PORT, self.WCE1)

    # ---------------------------------------------------------------------- #
    # Serialization                                                          #
    # ---------------------------------------------------------------------- #

    def _serialize_state(self) -> Dict[str, Any]:
        return {
            'serial': self.serial,
        }

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        super()._deserialize_state()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "pcm1704":
        registry_id = config.pop('registry_id')
        serial      = config.pop('serial')
        calibration = config.pop('calibration') or 'Darjeeling'

        return cls(
            serial      = serial,
            calibration = calibration,
            registry_id = registry_id,
            **config,
        )

    def resolve(self, accessor: str) -> 'pcm1704_channel | pcm1704_raw_channel':
        try:
            raw = accessor.endswith('_raw')
            stem = accessor[:-4] if raw else accessor
            assert stem.startswith('ch')
            ch = int(stem[2:])
            assert 0 <= ch <= 7
        except Exception:
            return None
        return pcm1704_raw_channel(self, accessor, ch) if raw \
               else pcm1704_channel(self, accessor, ch)

    # ---------------------------------------------------------------------- #
    # Raw bit / voltage conversion                                           #
    # ---------------------------------------------------------------------- #

    def _raw_voltage_to_bits(self, rawV: float) -> int:
        if abs(rawV) > self.v_fullscale:
            rawV = self.v_fullscale * (1 if rawV > 0 else -1)
        bits = int(
            ((rawV * self.bits_half_max) / self.v_fullscale)
            + (0.5 if rawV >= 0 else -0.5)
        )
        return bits if bits >= 0 else (self.bits_max + bits + 1)

    def _bits_to_raw_voltage(self, bits: int) -> float:
        if bits < self.bits_half_max + 1:
            return bits * self.v_fullscale / self.bits_half_max
        return (bits - self.bits_max - 1) * self.v_fullscale / self.bits_half_max

    def set_bits(self, ch: int, bits: int) -> None:
        v = max(0, min(bits, self.bits_max))
        self.ch_bits[ch] = v
        self._set_bits(ch, bits)

    def _get_bits(self, ch: int) -> int:
        return self.ch_bits[ch]

    # ---------------------------------------------------------------------- #
    # Raw-voltage API                                                        #
    # ---------------------------------------------------------------------- #

    def set_raw_V(self, ch: int, V: float, chatty: bool = True):
        minv = self.calibration['raw_min'][ch]
        maxv = self.calibration['raw_max'][ch]
        if V > maxv or V < minv:
            Vt = min(maxv, max(minv, V))
            self.warn(
                f"{V} on PCM1704 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt
        bits = self._raw_voltage_to_bits(V)
        self.set_bits(ch, bits)
        if chatty:
            self.info(f"Set channel {ch} to raw value {V} V (0x{bits:X})")

    def get_raw_V(self, ch: int) -> float:
        return self._bits_to_raw_voltage(self.ch_bits[ch])

    def sweep_raw_V(
        self, ch: int, V: float,
        max_step: float = None, wait: float = None,
    ):
        minv = self.calibration['raw_min'][ch]
        maxv = self.calibration['raw_max'][ch]
        if V > maxv or V < minv:
            Vt = min(maxv, max(minv, V))
            self.warn(
                f"{V} on PCM1704 channel {ch} is out of range; truncating to {Vt}."
            )
            V = Vt
        if not max_step:
            max_step = self.max_step
        if not wait:
            wait = self.wait
        start    = self.get_raw_V(ch)
        num_step = ceil(abs(V - start) / max_step)
        for v in np.linspace(start, V, num_step + 1)[1:]:
            time.sleep(wait)
            self.set_raw_V(ch, v, chatty=False)
        self.info(f"Swept channel {ch} to {V} V (raw).")

    # ---------------------------------------------------------------------- #
    # Calibrated-voltage API                                                 #
    # ---------------------------------------------------------------------- #

    def _raw_to_cal(self, ch: int, V: float) -> float:
        return self.calibration['pwl_fit'][ch](V)

    def _cal_to_raw(self, ch: int, V: float) -> float:
        return self.calibration['pwl_fit'][ch].inverse(V)

    def set_V(self, ch: int, V: float, chatty: bool = True):
        self.set_raw_V(ch, self._cal_to_raw(ch, V), chatty=False)
        if chatty:
            self.info(f"Set channel {ch} to calibrated value {V} V.")

    def get_V(self, ch: int) -> float:
        return self._raw_to_cal(ch, self.get_raw_V(ch))

    def sweep_V(self, ch: int, V: float,
                max_step: float = None, wait: float = None):
        self.sweep_raw_V(ch, self._cal_to_raw(ch, V), max_step, wait)
        self.info(f"Swept channel {ch} to calibrated value {V} V.")

    # ---------------------------------------------------------------------- #
    # Calibration loading                                                    #
    # ---------------------------------------------------------------------- #

    def load_calibration(self, path: str = None):
        if path in ('Lipton', 'Darjeeling'):
            mypath = os.path.dirname(os.path.abspath(__file__))
            try:
                with open(os.path.join(mypath, 'pcm1704_cal.json'), 'r') as f:
                    self.calibration = json.load(f)[path]
            except Exception:
                self.warn("No calibration file found; using identity calibration.")
                self.calibration = {
                    'raw_min': {ch: -self.v_fullscale for ch in range(self.n_ch)},
                    'raw_max': {ch:  self.v_fullscale for ch in range(self.n_ch)},
                    'pwl_fit': {
                        ch: {
                            'x_breaks': [-self.v_fullscale, self.v_fullscale],
                            'y_breaks': [-self.v_fullscale, self.v_fullscale],
                        }
                        for ch in range(self.n_ch)
                    },
                }
        else:
            with open(path, 'r') as f:
                self.calibration = json.load(f)

        self.calibration['raw_min'] = {
            int(k): v for k, v in self.calibration['raw_min'].items()
        }
        self.calibration['raw_max'] = {
            int(k): v for k, v in self.calibration['raw_max'].items()
        }
        self.calibration['pwl_fit'] = {
            int(k): MonotonicPiecewiseLinear(p['x_breaks'], p['y_breaks'])
            for k, p in self.calibration['pwl_fit'].items()
        }

    # ---------------------------------------------------------------------- #
    # Backend                                                                #
    # ---------------------------------------------------------------------- #

    def _set_bits(self, channel: int, bits: int) -> None:
        """
        Write a 24-bit value to one PCM1704 DAC channel.
 
        Constructs the full SPI waveform (77 port states) and dispatches it as
        a single pipelined batch via write_sequence().
 
        Parameters
        ----------
        channel : int
            DAC output channel, 0-7.
        bits : int
            24-bit raw DAC code, 0x000000-0xFFFFFF.
            Uses two's-complement offset-binary encoding: 
            0x7FFFFF ≈ +Vfullscale, 0x800001 ≈ -Vfullscale.
        """
        if not 0 <= channel <= 7:
            raise ValueError(f"channel must be 0..7, got {channel!r}")
        bits &= 0xFFFFFF
 
        # Encode the 3-bit channel address on AD[2:0]
        addr: int = (
            (self.AD0 if channel & 0x1 else 0) |
            (self.AD1 if channel & 0x2 else 0) |
            (self.AD2 if channel & 0x4 else 0)
        )
 
        # ------------------------------------------------------------------ #
        # Build the complete Port A state sequence                           #
        # ------------------------------------------------------------------ #
        #
        # The PCM1704 SPI framing has three phases:
        #
        #   Phase 1 (i=0)      : WCE1 high, WCE0 low  — channel address set up,
        #                         chip selected but not yet loading data.
        #   Phase 2 (i=1..23)  : WCE1 low,  WCE0 high — data-load window;
        #                         24 bits clocked in MSB first.
        #   Phase 3 (after loop): WCE0 low,  WCE1 high — latch; two extra SCLK
        #                         pulses shift the serial register into the
        #                         parallel output latch.
        #
        # Bit 23 (the LSB of the 24-bit word) is forced to 0.
        # This matches the original drivers, where a comment notes the LSB 
        # appears miscalibrated in hardware.
        #
        # Total states: 24 iterations × 3 + 1 latch + 2×2 SCLK pulses = 77.
 
        states: list[int] = []
        state: int = addr | self.WCE1  # Phase 1 initial value
 
        for i in range(24):
            if i == 1:
                # Transition to Phase 2: drop WCE1, assert WCE0
                state = (state & ~(self.WCE0 | self.WCE1)) | self.WCE0
 
            # Set or clear SDATA for bit (23 - i), MSB first; force LSB to 0
            if i == 23:
                state &= ~self.SDATA
            elif (bits >> (23 - i)) & 1:
                state |= self.SDATA
            else:
                state &= ~self.SDATA
 
            states.append(state)                  # SCLK low, data stable (setup)
            states.append(state | self.SCLK)      # SCLK rising edge  (latch in DAC)
            states.append(state & ~self.SCLK)     # SCLK falling edge (ready for next)
            state &= ~self.SCLK
 
        # Phase 3: latch — drop WCE0, re-assert WCE1
        state = (state & ~(self.WCE0 | self.WCE1)) | self.WCE1
        states.append(state)
 
        # Two SCLK pulses transfer the serial shift register to the parallel DAC latch
        for _ in range(2):
            states.append(state | self.SCLK)
            states.append(state & ~self.SCLK)
 
        # Dispatch all 77 states in one pipelined USB batch
        self.write_sequence(self._CTRL_PORT, states)


    def test(self) -> bool:
        """
        Hardware self-test: verify optoisolator continuity on all control lines.
 
        Toggles each bit of Port A low and high in turn and reads back the result
        on Port B.  Returns True if all channels pass, False otherwise.
 
        Note: Port B must be wired to the optoisolator outputs for this to be
        meaningful.
        """
        full = 0x7F  # bits 0–6 (SCLK through WCE0)
        errors_low  = 0
        errors_high = 0
 
        # Set all high, then walk each bit low (counting down to avoid latching)
        self.set_output(self._CTRL_PORT, full)
        accumulated = 0
        for bit in range(6, -1, -1):
            mask = 1 << bit
            accumulated |= mask
            self.set_output(self._CTRL_PORT, 0, mask)
            fb = self.get_input(self._MON_PORT)
            if fb & accumulated:    # any bit set low so far is still showing high
                errors_low |= mask

        # Walk each bit high
        self.set_output(self._CTRL_PORT, 0, full)
        accumulated = 0
        for bit in range(7):
            mask = 1 << bit
            accumulated |= mask
            self.set_output(self._CTRL_PORT, mask, mask)
            fb = self.get_input(self._MON_PORT)
            if not (fb & accumulated):  # any bit set high so far is still showing low
                errors_high |= mask
 
        # Return to idle state
        self.set_output(self._CTRL_PORT, self.WCE1)
 
        if errors_low == full:
            print("Self-test: all channels failed to go low — "
                  "check power and serial cable.")
            return False
        if errors_low:
            print(f"Self-test: channels 0x{errors_low:02X} failed to go low.")
            return False
        if errors_high:
            print(f"Self-test: channels 0x{errors_high:02X} failed to go high.")
            return False
 
        print("Self-test: all channels OK.")
        return True

class pcm1704_channel(ScalarChannelAdapter):
    def __init__(self, parent: pcm1704, accessor: str, ch: int):
        super().__init__(parent, accessor)
        self.ch = ch

    def get_output(self) -> float:
        return self._parent.get_V(self.ch)

    def set_output(self, value: float):
        self._parent.set_V(self.ch, value, chatty=False)

class pcm1704_raw_channel(ScalarChannelAdapter):
    def __init__(self, parent: pcm1704, accessor: str, ch: int):
        super().__init__(parent, accessor)
        self.ch = ch

    def get_output(self) -> float:
        return self._parent.get_raw_V(self.ch)

    def set_output(self, value: float):
        self._parent.set_raw_V(self.ch, value, chatty=False)