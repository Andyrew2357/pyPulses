"""
Instrument control for the Signal Recovery / ORTEC FastFlight2
repetitive signal averager.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gui import FastFlight2GUI

try:
    import libusb_package
    libusb_package.get_libusb1_backend() # registers the bundled DLL
except ImportError:
    pass  # not on Windows, or user has system libusb — either is fine

import sys
import math
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import usb.core
import usb.util
import usb.control

from ..abstract_device import abstractDevice
from ..registry import register_hardware_class, HardwareRegistry


_FIRMWARE_DIR = Path(__file__).parent / "firmware"

# ---------------------------------------------------------------------------
# USB device constants
# ---------------------------------------------------------------------------
_VENDOR_ID  = 0x0a2d
_DEVICE_ID  = 0x0015

_CONTROL_OUT = 0x01
_CONTROL_IN  = 0x81
_SPECTRA_OUT = 0x08
_SPECTRA_IN  = 0x82

_MEMORY_SET_REQUEST = 0xa3

_MISC_CONTROL_PTR    = 0xa10f
_RUN_MASK            = 0x01
_EXT_TRIGGER_MASK    = 0x02
_RAPID_PROTOCOL_MASK = 0x04
_UNKNOWN_START       = 0x08
_TIMER_RESET_MASK    = 0x10

_PROTOCOL_SET_PTR = 0xa1fe
_PROTOCOL_BASE    = 0xa200
_PROTOCOL_STEP    = 0x0020

_TRIGGER_PARAMETER     = 0x06
_TRIGGER_POLARITY_MASK = 0x10
_TRIGGER_RISING_MASK   = 0x02

_MAX_BULK_SIZE   = 0xFFC0
_DEFAULT_TIMEOUT = 500     # ms

_BM_VENDOR_OUT = 0x40   # USB_TYPE_VENDOR | USB_RECIP_DEVICE
_BM_VENDOR_IN  = 0xC0   # USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_ENDPOINT_IN

_FW_CHUNK_DATA = 0x1F   # firmware payload bytes per bulk transfer (batch=0x20)

# ---------------------------------------------------------------------------
# Firmware "strange dance" initialisation sequence
# ---------------------------------------------------------------------------
_DANCE = [
    0x23, 0x21, 0x20,
    -0x20, 0x22, 0x23,
    0x22, 0x20, 0x21, 0x20,
    0x22, 0x23, 0x22, 0x20,
    0x21, 0x20, 0x20, 0x21, 0x20,
    0x20, 0x21, 0x20, 0x20, 0x21,
    0x20, 0x22, 0x23, -0x23, 0x22, -0xa2,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    # 0x22 -0xa2 is C arithmetic → 0x22 - 0xa2 = -0x80
    0x22, 0x23, -0x23, 0x22 - 0xa2,
    0x22, 0x23, 0x22,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    0x22, 0x23, 0x22, 0x22, 0x23, 0x22,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    0x22, 0x23, -0x23,
    0x22, 0x23, 0x21,
    0x20, -0x20,
    0x22, 0x23, 0x22, 0x20, 0x21, 0x20,
    0x22, 0x23, 0x22, 0x20, 0x21, 0x20,
    0x20, 0x21, 0x20, 0x20, 0x21, 0x20, 0x20, 0x21, 0x20,
    0x22, 0x23, 0x22,
    0x22, 0x23, -0x23,
    0x22, -0x22,
    0x22, 0x23, -0x23, 0x22, 0x23, -0x23,
    0x22, 0x23, -0x23, 0x22, 0x23, -0x23,
    0x22, 0x23, -0x23,
    0x22, 0x23, -0xa3, 0x22, 0x23, -0xa3,
    0x22, 0x23, -0x23,
    0x22, 0x20, 0x21,
    0x20, -0x20,
    0x22, 0x23, -0xa3,
    0x22, 0x23, -0x23, 0x22, 0x23, -0x23,
    0x22, 0x23, -0xa3, 0x22, 0x23, -0xa3,
    0x22, 0x23, -0xa3, 0x22, 0x23, -0xa3,
    0x22, 0x20, 0x21, 0x20, -0x20,
    0x22, 0x23, -0xa3,
    0x22, 0x23, -0x23, 0x22, 0x23, -0x23,
    0x22, 0x23, -0xa3,
    0x22, 0x23, -0x23, 0x22, 0x23, -0x23,
    0x22, 0x23, -0x23, 0x22, 0x23, -0x23,
    0x22, 0x22, 0x23, 0x22, 0x20,
    0x21, 0x23, 0x03,
]

# ---------------------------------------------------------------------------
# Spectrum word-stream code types
# ---------------------------------------------------------------------------
_DATA_16BIT     = 0x0
_DATA_24BIT     = 0x1
_DATA_STICK     = 0x2
_SPECTRUM_BEGIN = 0x3
_SPECTRUM_END   = 0x3    # same numeric value as BEGIN
_TIME_LOW       = 0x4
_TIME_HIGH      = 0x5
_PROTOCOL_CODE  = 0x6
_ION_COUNT      = 0x7    # also == SYNC
_NOT_CODE       = 0xFF

_CODE_DATA_MASK = 0x001FFFFF


def _code_type(word: int) -> int:
    if (word & 0xFF000000) != 0xFF000000:
        return _NOT_CODE
    code = (word & 0x00E00000) >> 21
    if _DATA_16BIT <= code <= _ION_COUNT:
        return code
    return _NOT_CODE

# ---------------------------------------------------------------------------
# Internal goto-replacement exceptions
# ---------------------------------------------------------------------------
class _Retry(Exception):
    """Re-synchronize from scratch"""

class _Resync(Exception):
    """Reset byte count and re-read header without re-synchronizing"""

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
class FastFlight2Protocol:
    """
    Acquisition protocol parameters for the FastFlight2.

    After modifying fields, call `stuff` to serialise into the byte arrays 
    (`b1`, `b2`) that are written to device memory, or pass the protocol 
    directly to `FastFlight2.send_protocol` which calls `stuff` internally.

    All time values are in **nanoseconds**; voltages in **volts**.
    """

    # Time-per-point enumeration values (match device register encoding)
    INT_250ps = 0x10
    INT_500ps = 0x20
    INT_1ns   = 0x40
    INT_2ns   = 0x80

    PEAK_ONLY = 0x0
    LOSSLESS  = 0x1
    STICK     = 0x2

    SENS_2x   = 0x0
    SENS_3x   = 0x01
    SENS_4x   = 0x02

    _TPP_NS = {
        INT_250ps: 0.25,
        INT_500ps: 0.5,
        INT_1ns:   1.0,
        INT_2ns:   2.0,
    }
    _NS_TPP = {v: k for k, v in _TPP_NS.items()}

    def __init__(self):
        self.record_length          = 1e4    # ns  (10 µs)
        self.time_offset            = 16.0   # ns
        self.voltage_offset         = -0.25  # V
        self.records_per_spectrum   = 256
        self.precision_enhancer     = True
        self.tpp                    = self.INT_500ps
        self.compression            = self.LOSSLESS
        self.ringing_protection     = 2
        self.sensitivity            = self.SENS_2x
        self.background_interval    = 200    # points
        self.adjacent_background    = 0x10   # points
        self.correlated_subtraction = False

    # ------------------------------------------------------------------
    def time_per_point(self, ns: float = None) -> float:
        """
        Get or set the time per point in nanoseconds.

        Valid values: 0.25, 0.5, 1.0, 2.0.  Setting rounds to nearest.
        """
        if ns is None:
            return self._TPP_NS[self.tpp]
        if   ns <= 0.251: self.tpp = self.INT_250ps
        elif ns <  0.51:  self.tpp = self.INT_500ps
        elif ns <  1.01:  self.tpp = self.INT_1ns
        else:             self.tpp = self.INT_2ns
        return self._TPP_NS[self.tpp]

    @property
    def n_points(self) -> int:
        """Number of time-bins for the current record_length and tpp."""
        return int(round(self.record_length / self._TPP_NS[self.tpp]))

    def stuff(self):
        """
        Serialise protocol fields into ``b1`` (13 bytes) and ``b2`` (18 bytes)
        for direct write into device memory.  Also quantises ``record_length``,
        ``time_offset``, and ``voltage_offset`` in-place to reflect the values
        actually encoded (matching C++ Protocol::stuff() behaviour).

        Returns
        -------
        b1, b2 : bytes
        """
        b1 = bytearray(0x0d)
        b2 = bytearray(0x12)

        tpp_ns = self._TPP_NS[self.tpp]
        points = int(round(self.record_length / tpp_ns))
        points = max(16, min(1_500_000, points))
        self.record_length = points * tpp_ns

        divider = int(0x10 * 0.5 / tpp_ns)
        b1[0] = (points //  divider)          & 0xFF
        b1[1] = (points // (divider * 0x100)) & 0xFF

        toi = int(self.time_offset / 16.0)
        toi = max(1, min(0xFFFF, toi))
        self.time_offset = toi * 16.0
        b1[3] = toi        & 0xFF
        b1[4] = (toi >> 8) & 0xFF

        b1[5] = self.records_per_spectrum        & 0xFF
        b1[6] = (self.records_per_spectrum >> 8) & 0xFF

        i = int(round(((self.voltage_offset + 0.25) / 0.5) * 65535))
        i = max(0, min(0xFFFF, i))
        b1[8] = i & 0xFF
        b1[9] = (i >> 8) & 0xFF
        self.voltage_offset = 0.5 * (i / 65535.0) - 0.25

        b1[0xb] = {
            self.INT_250ps: 0x00,
            self.INT_500ps: 0x01,
            self.INT_1ns:   0x02,
            self.INT_2ns:   0x03,
        }[self.tpp]
        b1[0xc] = 1 if self.precision_enhancer else 0

        b2[0] = self.compression | self.tpp
        b2[1] = (self.ringing_protection * 0x10) | self.sensitivity
        b2[2] = 0x0a
        b2[3] = 0x00
        b2[4] = 0x30
        b2[5] = (self.background_interval // 4) & 0xFF
        b2[6] = (self.adjacent_background & 0x7F) | (0x80 if self.correlated_subtraction else 0)

        p2 = (points // 8) * 8 - 2
        b2[7]    = 0x64
        b2[8]    = 0x04
        b2[9]    =  p2        & 0xFF
        b2[0xa]  = (p2 >>  8) & 0xFF
        b2[0xb]  = (p2 >> 16) & 0x1F
        b2[0x10] = self.records_per_spectrum        & 0xFF
        b2[0x11] = (self.records_per_spectrum >> 8) & 0xFF

        return bytes(b1), bytes(b2)

    def to_dict(self) -> dict:
        return {
            'record_length'         : self.record_length,
            'time_offset'           : self.time_offset,
            'voltage_offset'        : self.voltage_offset,
            'records_per_spectrum'  : self.records_per_spectrum,
            'precision_enhancer'    : self.precision_enhancer,
            'tpp'                   : self.tpp,
            'compression'           : self.compression,
            'ringing_protection'    : self.ringing_protection,
            'sensitivity'           : self.sensitivity,
            'background_interval'   : self.background_interval,
            'adjacent_background'   : self.adjacent_background,
            'correlated_subtraction': self.correlated_subtraction,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FastFlight2Protocol':
        p = cls()
        for k, v in d.items():
            setattr(p, k, v)
        return p


# ---------------------------------------------------------------------------
# FastFlight2
# ---------------------------------------------------------------------------
@register_hardware_class("fastflight2")
class FastFlight2(abstractDevice):
    """
    Signal Recovery / ORTEC FastFlight2 repetitive signal averager.

    Connects via USB using PyUSB / libusb-1.0.  Handles firmware upload
    automatically on cold-start.

    Parameters
    ----------
    registry_id : str, optional
        Name to register this instance under in the HardwareRegistry.
    logger : Logger, optional
        Logger used by abstractDevice.
    skip_connect : bool
        If True, skip USB connection and firmware initialisation.
        Useful for constructing an instance purely from a saved state.
    fpga_directory : str
        Path to directory containing FPGA firmware files
        (AcqControl.rbf, pipes.rbf, pipes4P2.rbf, compressionfpga.rbf,
        00_AnalogFPGA.rbf, TrigProcFPGA.rbf, fanout.bin).
        Only read if firmware upload is required.

    Notes
    -----
    On Windows, install the WinUSB driver for the device using Zadig
    (https://zadig.akeo.ie/) before instantiating this class.
    """

    OVERLOAD  = 0x1
    UNDERLOAD = 0x2

    #: Maximum supported number of time-bins per spectrum
    MAX_POINTS = 1_500_000

    #: Number of protocol slots on the device
    MAX_PROTOCOL_SLOTS = 16

    def __init__(
        self,
        registry_id   : str | None  = None,
        logger        : Logger | None = None,
        skip_connect  : bool = False,
        fpga_directory: str  = str(_FIRMWARE_DIR),
    ):
        super().__init__(logger)
        HardwareRegistry.register(self, registry_id=registry_id)
        
        self._fpga_dir      = fpga_directory
        self._dev           = None          # usb.core.Device
        self._last_chip     = -1
        self._acquiring     = False

        # Spectrum stream state
        self._stream_buf    = bytearray()
        self._stream_pos    = 0
        self._bytecount     = 0
        self._unget         = -1

        # Status from last completed spectrum
        self._overload      = 0
        self._last_protocol = -1

        # Active protocol and per-trigger settings (mirrored locally for
        # serialisation without round-tripping to the device)
        self.protocol               = FastFlight2Protocol()
        self._trigger_threshold     = 0.0   # V
        self._trigger_rising        = True
        self._trigger_enable_high   = False
        self._external_trigger      = True
        self._rapid_protocol        = True

        if not skip_connect:
            self._connect()

    # -----------------------------------------------------------------------
    # Connection and firmware
    # -----------------------------------------------------------------------
    def _connect(self):
        """Open the USB device and ensure firmware is loaded."""
        self.info(f"Searching for FastFlight2 "
                  f"(VID=0x{_VENDOR_ID:04x} PID=0x{_DEVICE_ID:04x}) ...")

        self._dev = usb.core.find(idVendor=_VENDOR_ID, idProduct=_DEVICE_ID)
        if self._dev is None:
            self.error(
                "FastFlight2 not found on USB bus.\n"
                "  Windows: install WinUSB driver with Zadig first.\n"
                "  Linux:   check udev rules / permissions."
            )
            return

        try:
            mfr = self._dev.manufacturer
            prd = self._dev.product
        except Exception:
            mfr, prd = "?", "?"
        self.info(f"Found: {mfr} / {prd}  "
                  f"(bus {self._dev.bus}, addr {self._dev.address})")

        if sys.platform != "win32":
            try:
                if self._dev.is_kernel_driver_active(0):
                    self.info("Detaching kernel driver ...")
                    self._dev.detach_kernel_driver(0)
            except Exception as e:
                self.warn(f"Kernel driver detach skipped: {e}")

        try:
            self._dev.set_configuration()
        except usb.core.USBError as e:
            self.warn(f"set_configuration: {e}  (continuing)")

        usb.util.claim_interface(self._dev, 0)
        self.info("Interface 0 claimed.")

        for ep in (_CONTROL_OUT, _CONTROL_IN):
            try:
                usb.control.clear_halt(self._dev, ep)
            except Exception as e:
                self.warn(f"clear_halt(0x{ep:02x}): {e}  (continuing)")

        self._device_init()

    def _device_init(self):
        """
        Uploads firmware if needed, then brings the device to a clean idle.
        """
        if not self.is_initialized():
            self.warn("Device not initialised — uploading firmware ...")
            self._send_firmware()
            self.info("Firmware upload complete.")
        else:
            self.info("Device already initialised.")

        self.stop_acquisition()
        self.clear_buffer()
        self.reset_timer()

    def close(self):
        """Release USB resources cleanly."""
        try:
            self.stop_acquisition()
        except Exception:
            pass
        if self._dev is not None:
            try:
                usb.util.release_interface(self._dev, 0)
            except Exception:
                pass
            usb.util.dispose_resources(self._dev)
            self._dev = None

    # -----------------------------------------------------------------------
    # Low-level USB primitives
    # -----------------------------------------------------------------------
    def _bulk_write(self, ep, data, timeout=_DEFAULT_TIMEOUT):
        return self._dev.write(ep, data, timeout)

    def _bulk_read(self, ep, size, timeout=_DEFAULT_TIMEOUT):
        return self._dev.read(ep, size, timeout)

    def _ctrl(self, bm, req, value, index, data_or_len, timeout=_DEFAULT_TIMEOUT):
        return self._dev.ctrl_transfer(bm, req, value, index, data_or_len, timeout)

    # -----------------------------------------------------------------------
    # Device memory (vendor control transfers)
    # -----------------------------------------------------------------------
    def _set_memory(self, address: int, value: int) -> bool:
        r = self._ctrl(_BM_VENDOR_OUT, _MEMORY_SET_REQUEST,
                       address, 0, bytes([value & 0xFF]))
        return r == 1

    def _e_set_memory(self, address: int, value: int):
        if not self._set_memory(address, value):
            raise RuntimeError(
                f"_set_memory(0x{address:04x}, 0x{value:02x}) failed")

    def _get_memory(self) -> int:
        r = self._ctrl(_BM_VENDOR_IN, _MEMORY_SET_REQUEST,
                       _MISC_CONTROL_PTR, 0, 1)
        return r[0]

    # -----------------------------------------------------------------------
    # Parameter channel
    # -----------------------------------------------------------------------
    def _set_parameter(self, parameter: int, value: int) -> bool:
        self._bulk_write(_CONTROL_OUT,
                         bytes([0x10, parameter & 0xFF, value & 0xFF]))
        resp = self._bulk_read(_CONTROL_IN, 1)
        if resp[0] not in (0, 1):
            self.warn(f"Unexpected _set_parameter response 0x{resp[0]:02x} "
                      f"(param=0x{parameter:02x} val=0x{value:02x})")
        return resp[0] == 1

    def _e_set_parameter(self, parameter: int, value: int):
        if not self._set_parameter(parameter, value):
            raise RuntimeError(
                f"_set_parameter(0x{parameter:02x}, 0x{value:02x}) failed")

    def _get_parameter(self, parameter: int) -> int:
        self._bulk_write(_CONTROL_OUT, bytes([0x11, parameter & 0xFF]))
        return self._bulk_read(_CONTROL_IN, 1)[0]

    # -----------------------------------------------------------------------
    # Initialisation check
    # -----------------------------------------------------------------------
    def is_initialized(self) -> bool:
        """
        Query whether the device has valid firmware loaded.
        """
        self._bulk_write(_CONTROL_OUT, bytes([0x0F]))
        resp = self._bulk_read(_CONTROL_IN, 1)[0]
        if resp not in (0, 1):
            self.warn(f"Unexpected is_initialized response: 0x{resp:02x}")
        return resp == 1

    # -----------------------------------------------------------------------
    # Firmware upload
    # -----------------------------------------------------------------------
    def _setup_chip(self, chip: int):
        """
        Select the target FPGA chip for subsequent firmware writes. 
        Skips the round-trip if the chip hasn't changed since the last call.
        """
        if chip == self._last_chip:
            return
        self._last_chip = chip
        self._bulk_write(_CONTROL_OUT, bytes([chip & 0xFF]))
        resp = self._bulk_read(_CONTROL_IN, 1)[0]
        if resp != 0:
            raise RuntimeError(
                f"_setup_chip(0x{chip:02x}): expected 0, got 0x{resp:02x}")

    def _send_file(self, fname: str):
        """
        Send one firmware bitfile to the currently-selected FPGA chip.

        Each USB transfer is [chip_id+1, payload...] where payload is up
        to 31 bytes.  The loop ends when a short (< 31 byte) chunk is sent,
        signalling EOF.
        """
        path = Path(self._fpga_dir) / fname
        self.info(f"  {fname}  →  chip 0x{self._last_chip:02x} ...")

        with open(path, "rb") as fh:
            file_data = fh.read()

        if not file_data:
            raise RuntimeError(f"_send_file: {fname!r} is empty")

        chip_byte = bytes([self._last_chip + 1])
        hunk   = 0
        offset = 0
        total  = len(file_data)

        while offset < total:
            chunk  = file_data[offset: offset + _FW_CHUNK_DATA]
            n      = len(chunk)
            self._bulk_write(_CONTROL_OUT, chip_byte + chunk, timeout=500)
            resp = self._bulk_read(_CONTROL_IN, 1)[0]
            if resp != 0:
                raise RuntimeError(
                    f"_send_file({fname!r}) hunk {hunk:#x}: "
                    f"expected 0, got 0x{resp:02x}")
            hunk   += 1
            offset += n
            if n < _FW_CHUNK_DATA:
                break

        self.info(f"    done ({hunk} hunks, {total} bytes)")

    def _strange_dance(self):
        """
        Replay a sequence of parameter reads/writes against register 0x10.

        Positive entries -> _e_set_parameter(0x10, value)
        Negative entries -> _get_parameter(0x10), warn on mismatch
        """
        for val in _DANCE:
            if val > 0:
                self._e_set_parameter(0x10, val)
            else:
                p        = self._get_parameter(0x10)
                expected = (-val) & 0xFF
                if p != expected:
                    self.warn(f"Dance mismatch at 0x10: "
                              f"expected 0x{expected:02x}, got 0x{p:02x}")

        # Final fixed-register writes — purpose unknown in original source.
        for reg, val in [
            (0x18, 0x06), (0x19, 0x00), (0x1A, 0x07),
            (0x18, 0x24), (0x19, 0x7d), (0x1a, 0x00),
            (0x18, 0x2c), (0x19, 0x74), (0x1a, 0x00),
            (0x18, 0x9F), (0x19, 0x9f), (0x1a, 0x01),
        ]:
            self._set_parameter(reg, val)

    def _send_firmware(self):
        """
        Upload all FPGA bitfiles and run the initialisation sequence.
        Mirrors FastFlight2::sendFirmware().

        Upload order:
          chip 0x4: AcqControl.rbf
          chip 0x6: pipes.rbf x 3, pipes4P2.rbf
          chip 0x8: compressionfpga.rbf
          chip 0xc: 00_AnalogFPGA.rbf, TrigProcFPGA.rbf
          chip 0xa: fanout.bin

        Followed by a four-step post-upload handshake and the strange dance.
        """
        EXPECTED = bytes([0x42, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
        result   = self._ctrl(_BM_VENDOR_IN, 0xa2, 0xfff0, 0, 8)
        if bytes(result) != EXPECTED:
            self.warn(f"Unexpected pre-upload response: {bytes(result).hex()!r}"
                      f"  (expected {EXPECTED.hex()!r})")

        self.info("Uploading FPGA firmware:")
        self._setup_chip(0x4);  self._send_file("AcqControl.rbf")
        self._setup_chip(0x6)
        self._send_file("pipes.rbf")
        self._send_file("pipes.rbf")
        self._send_file("pipes.rbf")
        self._send_file("pipes4P2.rbf")
        self._setup_chip(0x8);  self._send_file("compressionfpga.rbf")
        self._setup_chip(0xc)
        self._send_file("00_AnalogFPGA.rbf")
        self._send_file("TrigProcFPGA.rbf")
        self._setup_chip(0xa);  self._send_file("fanout.bin")

        # Post-upload handshake - four bulk writes of [0x0E, value].
        self.info("Post-upload handshake ...")
        for val in [0xDF, 0xD7, 0x95, 0x00]:
            self._bulk_write(_CONTROL_OUT, bytes([0x0E, val]))
            resp = self._bulk_read(_CONTROL_IN, 1)[0]
            if resp != 0:
                raise RuntimeError(
                    f"Post-upload handshake (0x0E 0x{val:02x}): "
                    f"expected 0, got 0x{resp:02x}")

        self.info("Running strange-dance initialisation ...")
        self._strange_dance()

    # -----------------------------------------------------------------------
    # Device state control
    # -----------------------------------------------------------------------
    def reset_timer(self):
        """
        Reset the trace elapsed-time counter.
        """
        m = self._get_memory()
        self._e_set_memory(_MISC_CONTROL_PTR, m |  _TIMER_RESET_MASK)
        self._e_set_memory(_MISC_CONTROL_PTR, m & ~_TIMER_RESET_MASK)

    def clear_buffer(self):
        """
        Flush the device acquisition buffer and reset the software byte-stream 
        state.
        """
        self._e_set_parameter(0x07, 0x20)
        self._bulk_write(_CONTROL_OUT, bytes([0x12]))
        resp = self._bulk_read(_CONTROL_IN, 1)[0]
        if resp != 1:
            raise RuntimeError(f"clear_buffer: expected 1, got {resp}")
        self._e_set_parameter(0x07, 0x00)
        self._stream_buf = bytearray()
        self._stream_pos = 0
        self._bytecount  = 0
        self._unget      = -1

    def trigger_threshold(self, v: float = None) -> float:
        """
        Get or set the trigger threshold in volts.

        Parameters
        ----------
        v : float, optional
            Threshold voltage.

        Returns
        -------
        threshold : float
            Current threshold in volts.
        """
        if v is None:
            return self._trigger_threshold

        self._trigger_threshold = float(v)
        v2 = (2.5 - v) / 5.0 * 1024
        i  = int(math.ceil(v2))
        self._e_set_parameter(0x14, (i & 0x3) * 0x40)
        self._e_set_parameter(0x15,  i // 0x04)
        self.info(f"Set trigger threshold to {self._trigger_threshold} V.")
        return self._trigger_threshold

    def trigger_rising(self, rising: bool = None) -> bool:
        """
        Get or set the trigger edge polarity.

        Parameters
        ----------
        rising : bool, optional
            True = rising edge, False = falling edge.

        Returns
        -------
        rising : bool
        """
        if rising is None:
            return self._trigger_rising

        self._trigger_rising = bool(rising)
        c = self._get_parameter(_TRIGGER_PARAMETER)
        c = (c | _TRIGGER_RISING_MASK) if rising else (c & ~_TRIGGER_RISING_MASK)
        self._set_parameter(_TRIGGER_PARAMETER, c)
        self.info(f"Set trigger edge to {'rising' if rising else 'falling'}.")
        return self._trigger_rising

    def trigger_enable_high(self, high: bool = None) -> bool:
        """
        Get or set the trigger enable polarity.

        Parameters
        ----------
        high : bool, optional
            True = trigger enabled when input is high.

        Returns
        -------
        high : bool
        """
        if high is None:
            return self._trigger_enable_high

        self._trigger_enable_high = bool(high)
        c = self._get_parameter(_TRIGGER_PARAMETER)
        c = (c | _TRIGGER_POLARITY_MASK) if high else (c & ~_TRIGGER_POLARITY_MASK)
        self._set_parameter(_TRIGGER_PARAMETER, c)
        self.info(f"Set trigger enable polarity to {'high' if high else 'low'}.")
        return self._trigger_enable_high

    def external_trigger(self, state: bool = None) -> bool:
        """
        Get or set external vs. internal trigger mode.

        Parameters
        ----------
        state : bool, optional
            True = external trigger, False = internal.

        Returns
        -------
        external : bool
        """
        if state is None:
            return self._external_trigger

        self._external_trigger = bool(state)
        c  = self._get_memory()
        nc = (c | _EXT_TRIGGER_MASK) if state else (c & ~_EXT_TRIGGER_MASK)
        if nc != c:
            self._e_set_memory(_MISC_CONTROL_PTR, nc)
        self.info(f"Set trigger source to {'external' if state else 'internal'}.")
        return self._external_trigger

    def rapid_protocol_selection(self, state: bool = None) -> bool:
        """
        Get or set rapid protocol selection mode.

        Parameters
        ----------
        state : bool, optional

        Returns
        -------
        rapid : bool
        """
        if state is None:
            return self._rapid_protocol

        self._rapid_protocol = bool(state)
        c  = self._get_memory()
        nc = (c | _RAPID_PROTOCOL_MASK) if state else (c & ~_RAPID_PROTOCOL_MASK)
        if nc != c:
            self._e_set_memory(_MISC_CONTROL_PTR, nc)
        return self._rapid_protocol

    # -----------------------------------------------------------------------
    # Protocol management
    # -----------------------------------------------------------------------
    def send_protocol(self, protocol: FastFlight2Protocol, slot: int = 0):
        """
        Serialise and write a protocol to a device memory slot.

        Parameters
        ----------
        protocol : FastFlight2Protocol
        slot : int
            Protocol slot index, 0 … MAX_PROTOCOL_SLOTS-1.
        """
        if not (0 <= slot < self.MAX_PROTOCOL_SLOTS):
            self.error(f"Invalid protocol slot {slot}; "
                       f"must be 0-{self.MAX_PROTOCOL_SLOTS-1}")
            return

        b1, b2 = protocol.stuff()
        base   = _PROTOCOL_BASE + slot * _PROTOCOL_STEP
        self._ctrl(_BM_VENDOR_OUT, _MEMORY_SET_REQUEST, base,        0, b1)
        self._ctrl(_BM_VENDOR_OUT, _MEMORY_SET_REQUEST, base + 0x0e, 0, b2)

    def set_protocol_slot(self, slot: int):
        """
        Select the active protocol slot.
        Mirrors FastFlight2::setProtocol().

        Parameters
        ----------
        slot : int
        """
        self._set_memory(_PROTOCOL_SET_PTR, slot)

    # -----------------------------------------------------------------------
    # Acquisition control
    # -----------------------------------------------------------------------
    def start_acquisition(self):
        """
        Begin acquisition with the currently active protocol slot.

        Internally calls clear_buffer() first.
        """
        self.clear_buffer()
        self._e_set_parameter(0x05, 0xd0)

        # Mystery sequence
        for val in [0x00, 0x00, 0x10, 0x10, 0x00, 0x00, 0x50]:
            self._e_set_memory(0xa1fc, val)
        self._e_set_memory(0xa1fb, 0x08)
        self._e_set_memory(0xa1fb, 0x18)
        self._e_set_memory(0xa1fe, 0x00)

        e = self._get_memory()
        self._e_set_memory(_MISC_CONTROL_PTR, e |  _UNKNOWN_START)
        self._e_set_memory(_MISC_CONTROL_PTR, e & ~_UNKNOWN_START)
        self._e_set_memory(_MISC_CONTROL_PTR, e |  _RUN_MASK)
        self._acquiring = True

    def stop_acquisition(self):
        """
        Halt acquisition.
        """
        e = self._get_memory()
        self._e_set_memory(_MISC_CONTROL_PTR, e & ~_RUN_MASK)
        self._acquiring = False

    # -----------------------------------------------------------------------
    # Spectrum acquisition — data stream layer
    # -----------------------------------------------------------------------
    def _fetch_chunk(self) -> bytes:
        """
        Send the data-request handshake to spectraOut and read one chunk
        from spectraIn. Returns raw bytes, possibly empty on timeout.
        """
        cmd = bytes([0xFF, 0x03])
        off = 0
        for _ in range(3):
            try:
                off += self._dev.write(_SPECTRA_OUT, cmd[off:], 50)
                if off >= len(cmd):
                    break
            except usb.core.USBTimeoutError:
                break
            except usb.core.USBError as e:
                self.warn(f"spectraOut write error: {e}")
                break

        try:
            return bytes(self._dev.read(_SPECTRA_IN, _MAX_BULK_SIZE, 1000))
        except usb.core.USBTimeoutError:
            return b""
        except usb.core.USBError as e:
            self.warn(f"spectraIn read error: {e}")
            return b""

    def _fetch_exactly(self, n_bytes: int) -> bytes:
        """
        Accumulate exactly `n_bytes` from the spectra stream, issuing
        as many _fetch_chunk() calls as necessary.

        Incorporates any bytes left in the current stream buffer before
        fetching more from the device.
        """
        buf = bytearray()

        # Drain whatever is already in the stream buffer first.
        remaining_in_buf = len(self._stream_buf) - self._stream_pos
        if remaining_in_buf > 0:
            take = min(remaining_in_buf, n_bytes)
            buf.extend(self._stream_buf[self._stream_pos:self._stream_pos + take])
            self._stream_pos += take

        while len(buf) < n_bytes:
            chunk = self._fetch_chunk()
            if chunk:
                need  = n_bytes - len(buf)
                buf.extend(chunk[:need])
                # Put any remainder back into the stream buffer for next time.
                if len(chunk) > need:
                    self._stream_buf = bytearray(chunk[need:])
                    self._stream_pos = 0

        return bytes(buf)

    def _get_byte(self) -> int:
        """
        Return the next byte from the stream (byte-by-byte path used only for 
        the sync and two-word header).
        """
        if self._unget != -1:
            val = self._unget
            self._unget = -1
            self._bytecount += 1
            return val

        while True:
            if self._stream_pos < len(self._stream_buf):
                val = self._stream_buf[self._stream_pos]
                self._stream_pos += 1
                self._bytecount  += 1
                return val
            chunk = self._fetch_chunk()
            if chunk:
                self._stream_buf = bytearray(chunk)
                self._stream_pos = 0

    def _unget_byte(self, c: int):
        self._unget = c
        self._bytecount -= 1

    def _get_word_stream(self) -> int:
        """Read 4 bytes LE as uint32 from the byte-by-byte stream."""
        b0 = self._get_byte()
        b1 = self._get_byte()
        b2 = self._get_byte()
        b3 = self._get_byte()
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

    def _synchronize(self):
        """
        Consume bytes until 8 consecutive 0xFF bytes are seen.
        """
        TARGET = 8
        count  = 0
        total  = 0
        while True:
            total += 1
            b = self._get_byte()
            if b == 0xFF:
                count += 1
                if count == TARGET:
                    if total != TARGET:
                        self.info(f"[sync] consumed {total:#x} bytes to find sync")
                    # Peek ahead - the last 0xFF might belong to a stale command.
                    c = self._get_byte()
                    if c != 0xFF:
                        self._unget_byte(c)
                    return
            else:
                count = 0

    # -----------------------------------------------------------------------
    # Spectrum acquisition — decode layer
    # -----------------------------------------------------------------------
    def _decode_body(
        self,
        words : np.ndarray,
        out   : np.ndarray,
        max_points: int,
    ) -> int:
        """
        Decode the word array representing one spectrum body (everything
        after the two SPECTRUM_BEGIN header words).

        This operates entirely on pre-fetched in-memory data, so there
        is no USB I/O overhead per word.

        Returns the number of valid points decoded, or raises
        `_Retry` / `_Resync` on stream errors.
        """
        i          = 0          # word index into `words`
        index      = 0          # output-array index (bin number)
        last_code  = _SPECTRUM_BEGIN
        n          = len(words)

        while i < n:
            w  = int(words[i]);  i += 1
            ct = _code_type(w)

            # --- Address / data-type code ---
            if ct == _DATA_16BIT or ct == _DATA_24BIT:
                jump = w & _CODE_DATA_MASK
                if index != jump:
                    self.warn(f"[decode] index jump {index} → {jump}")
                    index = jump
                    if index > max_points:
                        self.warn("[decode] illegal jump; retrying")
                        raise _Retry()
                last_code = ct
                if i >= n:
                    raise _Retry()
                w  = int(words[i]);  i += 1
                ct = _NOT_CODE    # fall-through to data decode

            # --- Data word(s) ---
            if ct == _NOT_CODE:
                if index + 4 > max_points:
                    self.warn(f"[decode] buffer overflow at {index}; retrying")
                    raise _Retry()

                if last_code == _DATA_16BIT:
                    if i >= n:
                        raise _Retry()
                    u = int(words[i]);  i += 1
                    out[index]   = ((w >> 24) & 0xFF) << 8 | ((u >> 24) & 0xFF)
                    out[index+1] = ((w >> 16) & 0xFF) << 8 | ((u >> 16) & 0xFF)
                    out[index+2] = ((w >>  8) & 0xFF) << 8 | ((u >>  8) & 0xFF)
                    out[index+3] = ( w        & 0xFF) << 8 | ( u        & 0xFF)
                    index += 4

                elif last_code == _DATA_24BIT:
                    if i + 1 >= n:
                        raise _Retry()
                    u = int(words[i]);  i += 1
                    v = int(words[i]);  i += 1
                    out[index]   = (((w>>24)&0xFF)<<16) | (((u>>24)&0xFF)<<8) | ((v>>24)&0xFF)
                    out[index+1] = (((w>>16)&0xFF)<<16) | (((u>>16)&0xFF)<<8) | ((v>>16)&0xFF)
                    out[index+2] = (((w>> 8)&0xFF)<<16) | (((u>> 8)&0xFF)<<8) | ((v>> 8)&0xFF)
                    out[index+3] = (( w     &0xFF)<<16) | (( u     &0xFF)<<8) | ( v     &0xFF)
                    index += 4

                else:
                    self.warn(f"[decode] data word with no preceding type code "
                              f"(last_code=0x{last_code:x})")

            # --- End of spectrum ---
            elif ct == _SPECTRUM_END:
                return index

            # --- Metadata codes ---
            elif ct == _PROTOCOL_CODE:
                self._last_protocol = w & _CODE_DATA_MASK

            elif ct == _TIME_LOW or ct == _TIME_HIGH:
                pass    # timestamps not currently used

            elif ct == _ION_COUNT:   # numerically == _SYNC
                # All 21 data bits set -> this is a rogue sync, not an ion count.
                if (w & _CODE_DATA_MASK) == _CODE_DATA_MASK:
                    if i >= n:
                        raise _Retry()
                    u = int(words[i]);  i += 1
                    if u == 0xFFFFFFFF:
                        self.warn("[decode] unexpected mid-stream sync → resync")
                        raise _Resync()
                    else:
                        self.warn("[decode] unexpected partial sync → retry")
                        raise _Retry()
                else:
                    # Genuine ion count — read overload status word.
                    if i + 2 >= n:
                        raise _Retry()
                    u = int(words[i]);  i += 1
                    i += 2              # two further words unused
                    self._overload = 0
                    if u & 0x00008000:
                        self._overload |= self.OVERLOAD
                    if u & 0x80000000:
                        self._overload |= self.UNDERLOAD

            elif ct == _DATA_STICK:
                self.warn("[decode] STICK data not supported; retrying")
                raise _Retry()

            else:
                self.warn(f"[decode] unhandled code word 0x{w:08x}")

        # Ran off the end of the word array without seeing SPECTRUM_END.
        raise _Retry()

    def get_spectrum(self, out: np.ndarray) -> int:
        """
        Acquire and decode one complete spectrum into a pre-allocated array.

        `out` must be a numpy array of dtype `uint32` with length >= the
        expected number of time-bins.  Allocate with:

            out = np.zeros(ff.protocol.n_points, dtype=np.uint32)

        or more conservatively:

            out = np.zeros(FastFlight2.MAX_POINTS, dtype=np.uint32)

        Parameters
        ----------
        out : numpy.ndarray, dtype=uint32
            Output buffer.  Written in-place.

        Returns
        -------
        n_points : int
            Number of valid time-bins written to `out`.

        Notes
        -----
        Blocks until a valid spectrum is received.  After the two
        SPECTRUM_BEGIN header words are parsed, the remaining bytes are
        fetched in one bulk operation and decoded entirely from memory
        (no per-word USB I/O).
        """
        max_points = len(out)

        while True:         # 'goto retry' in C++ — re-synchronize from scratch
            try:
                self._synchronize()

                while True:     # 'goto resync' in C++ — re-read header only
                    try:
                        self._bytecount = 0   # db.reset()

                        # --- Two header words (byte-by-byte stream) ---
                        t = self._get_word_stream()
                        if _code_type(t) != _SPECTRUM_BEGIN:
                            self.warn(f"[spectrum] bad header word 0x{t:08x}")
                            raise _Retry()
                        spectrum_number = t & _CODE_DATA_MASK

                        t = self._get_word_stream()
                        if _code_type(t) != _SPECTRUM_BEGIN:
                            self.warn(f"[spectrum] bad length word 0x{t:08x}")
                            raise _Retry()
                        spectrum_length = t & _CODE_DATA_MASK

                        self.info(
                            f"[spectrum] #{spectrum_number}  "
                            f"declared_len={spectrum_length} words"
                        )

                        # --- Bulk fetch of remaining body ---
                        # Total bytes = spectrum_length × 4.
                        # We have already consumed 8 bytes (two header words)
                        # tracked by _bytecount.
                        remaining = spectrum_length * 4 - self._bytecount
                        if remaining <= 0:
                            self.warn("[spectrum] non-positive remaining byte "
                                      "count; retrying")
                            raise _Retry()

                        raw   = self._fetch_exactly(remaining)
                        words = np.frombuffer(raw, dtype='<u4')

                        # --- Decode ---
                        n_points = self._decode_body(words, out, max_points)

                        # Byte-count cross-check (mirrors C++ SPECTRUM_END check)
                        actual = self._bytecount + remaining
                        if spectrum_length * 4 != actual:
                            self.warn(
                                f"[spectrum] byte-count mismatch "
                                f"(expected {spectrum_length*4}, "
                                f"got {actual}); retrying"
                            )
                            raise _Retry()

                        return n_points

                    except _Resync:
                        continue    # restart inner loop — no re-sync needed

                    break           # inner loop finished normally

            except _Retry:
                continue            # restart outer loop — re-synchronize

    # -----------------------------------------------------------------------
    # Status accessors
    # -----------------------------------------------------------------------
    def get_overload(self) -> int:
        """
        Return the overload/underload flags from the last completed spectrum.

        Returns
        -------
        flags : int
            Bitfield; test against `FastFlight2.OVERLOAD` and
            `FastFlight2.UNDERLOAD`.
        """
        return self._overload

    def get_last_protocol(self) -> int:
        """
        Return the protocol slot index reported in the last completed spectrum.
        """
        return self._last_protocol

    # -----------------------------------------------------------------------
    # Convenience acquisition helper
    # -----------------------------------------------------------------------
    def acquire(
        self,
        out      : Optional[np.ndarray] = None,
        protocol : Optional[FastFlight2Protocol] = None,
        slot     : int = 0,
    ) -> tuple[np.ndarray, int]:
        """
        High-level single-spectrum acquisition.

        Sends `protocol` (or `self.protocol` if None) to `slot`,
        starts acquisition, waits for one spectrum, stops acquisition, and
        returns the data.

        Parameters
        ----------
        out : numpy.ndarray, dtype=uint32, optional
            Pre-allocated output buffer.  If None, one is allocated with
            `protocol.n_points` elements.
        protocol : FastFlight2Protocol, optional
            Protocol to use.  Defaults to `self.protocol`.
        slot : int
            Protocol slot.

        Returns
        -------
        data : numpy.ndarray
            View of `out` containing the acquired spectrum.
        n_points : int
            Number of valid bins.
        """
        p = protocol if protocol is not None else self.protocol

        if out is None:
            out = np.zeros(p.n_points, dtype=np.uint32)

        self.send_protocol(p, slot)
        self.set_protocol_slot(slot)
        self.start_acquisition()
        try:
            n = self.get_spectrum(out)
        finally:
            self.stop_acquisition()

        return out[:n], n

    # -----------------------------------------------------------------------
    # State serialisation (mirrors DTG pattern)
    # -----------------------------------------------------------------------
    def _serialize_state(self) -> Dict[str, Any]:
        state = super()._serialize_state()
        state.update({
            'fastflight2_config_version': 1,
            'fpga_directory'    : self._fpga_dir,
            'trigger_threshold' : self._trigger_threshold,
            'trigger_rising'    : self._trigger_rising,
            'trigger_enable_high': self._trigger_enable_high,
            'external_trigger'  : self._external_trigger,
            'rapid_protocol'    : self._rapid_protocol,
            'protocol'          : self.protocol.to_dict(),
        })
        return state

    def _deserialize_state(self, state: dict):
        super()._deserialize_state(state)

        if state.get('fastflight2_config_version') != 1:
            self.error("Unrecognised FastFlight2 config version; cannot deserialise")
            return

        self._fpga_dir = state.get('fpga_directory', self._fpga_dir)
        self.protocol  = FastFlight2Protocol.from_dict(state['protocol'])

        self.trigger_threshold(state['trigger_threshold'])
        self.trigger_rising(state['trigger_rising'])
        self.trigger_enable_high(state['trigger_enable_high'])
        self.external_trigger(state['external_trigger'])
        self.rapid_protocol_selection(state['rapid_protocol'])

        self.send_protocol(self.protocol, 0)
        self.set_protocol_slot(0)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FastFlight2':
        """
        Construct a FastFlight2 from a serialised config dict.

        Parameters
        ----------
        config : dict
            Output of `_serialize_state()`, including `registry_id`,
            `resource_name`, and `fpga_directory`.
        """
        registry_id    = config.pop('registry_id')
        resource_name  = config.pop('resource_name')
        fpga_directory = config.pop('fpga_directory', str(_FIRMWARE_DIR))

        instance = cls(
            resource_name  = resource_name,
            registry_id    = registry_id,
            fpga_directory = fpga_directory,
            skip_connect   = False,
        )
        instance._deserialize_state(config)
        return instance

    def resolve(self, accessor: str):
        # FastFlight2 has no sub-channels to resolve.
        return None
    
    # -----------------------------------------------------------------------
    # GUI
    # -----------------------------------------------------------------------

    def launch_gui(self, port: int = 8780) -> 'FastFlight2GUI':
        """
        Launch a localhost web GUI for this instrument.

        Opens a browser-accessible interface at http://localhost:<port>
        with live spectrum display, protocol configuration, and trigger
        settings.  Sets ``self.gui`` to the running GUI instance and
        returns it.

        If a GUI is already running for this instrument, returns the
        existing instance without starting a second one.

        Parameters
        ----------
        port : int, default 8780
            HTTP port.  WebSocket server uses port+1.

        Returns
        -------
        FastFlight2GUI
        """
        from .gui import FastFlight2GUI
        gui = getattr(self, 'gui', None)
        if gui is not None:
            print(f"GUI already running → http://localhost:{gui._port}")
            return gui
        return FastFlight2GUI(self, port=port).start()

    def kill_gui(self):
        """Stop the running GUI, if any."""
        gui = getattr(self, 'gui', None)
        if gui is not None:
            gui.stop()