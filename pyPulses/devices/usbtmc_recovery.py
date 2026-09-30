"""
USBTMC error recovery for pyvisa-py USB INSTR sessions.

pyvisa-py (checked against 0.8.1) never recovers a USBTMC link on its own:

* It never issues CLEAR_FEATURE(ENDPOINT_HALT), so a STALLed bulk endpoint
  stays halted and every later transfer fails with EPIPE.
* USBSession.clear() is not implemented for USB (it returns
  VI_ERROR_NSUP_OPER), so viClear / resource.clear() does nothing useful.
* USBTMC._abort_bulk_in reads a single wMaxPacketSize packet and does not
  re-read when CHECK_ABORT_BULK_IN_STATUS reports bytes still in the FIFO.
* BulkInMessage.from_bytes never checks the response bTag against the
  request, so a stale response is returned as if it were the current one.
* A zero-length bulk-IN packet raises struct.error, which is outside the
  (USBError, ValueError) tuple the abort path catches.

device_clear() performs the USBTMC 1.0 device-clear sequence, which is the
USB equivalent of a GPIB SDC: it discards the device's input and output
buffers and resynchronises both bulk pipes. It does not change instrument
settings.
"""

from __future__ import annotations

import errno
import struct
import time
from typing import Any, Dict, Optional

import usb.core
import usb.control

# USBTMC 1.0 class-specific requests (bRequest)
_INITIATE_CLEAR = 5
_CHECK_CLEAR_STATUS = 6

# USBTMC_status values
_STATUS_SUCCESS = 0x01
_STATUS_PENDING = 0x02

# bmRequestType: device-to-host | class | recipient=interface
_REQTYPE_IN_CLASS_INTF = 0xA1

_CTRL_TIMEOUT_MS = 1000

# Errors that mean "the USBTMC message stream is no longer trustworthy"
# rather than "the device is gone".
#   EPIPE     : endpoint STALL (halt latched until CLEAR_FEATURE)
#   EOVERFLOW : device sent more than requested ("babble"), typically a
#               stale or misframed response
_DESYNC_ERRNOS = {errno.EPIPE, errno.EOVERFLOW}


def get_usbtmc_interface(resource) -> Optional[Any]:
    """
    Return the pyvisa-py USBTMC interface object behind a pyvisa resource,
    or None if the resource is not a pyvisa-py USB INSTR session.
    """
    try:
        from pyvisa_py.protocols.usbtmc import USBTMC
        itf = resource.visalib.sessions[resource.session].interface
    except Exception:
        return None
    return itf if isinstance(itf, USBTMC) else None


def is_desync_error(exc: BaseException) -> bool:
    """True for errors that a USBTMC device clear can recover from."""
    if isinstance(exc, struct.error):
        # Zero-length or truncated bulk-IN transfer
        return True
    if isinstance(exc, usb.core.USBTimeoutError):
        return False
    if isinstance(exc, usb.core.USBError):
        return abs(exc.errno or 0) in _DESYNC_ERRNOS
    return False


def _halted(itf, ep) -> Optional[bool]:
    try:
        return bool(usb.control.get_status(itf.usb_dev, ep) & 1)
    except usb.core.USBError:
        return None


def _drain_bulk_in(itf, timeout_ms: int = 100, max_bytes: int = 1 << 22) -> int:
    """
    Read and discard bulk-IN data until a short packet (including a
    zero-length packet) ends the transfer, or until the device stops
    sending (timeout). Returns the number of bytes discarded.
    """
    ep = itf.usb_recv_ep
    size = ep.wMaxPacketSize * 64
    total = 0
    while total < max_bytes:
        try:
            data = ep.read(size, timeout_ms)
        except usb.core.USBTimeoutError:
            break
        except usb.core.USBError as e:
            if abs(e.errno or 0) == errno.EPIPE:
                itf.usb_dev.clear_halt(ep)
                continue
            raise
        total += len(data)
        if len(data) < size:
            break
    return total


def device_clear(itf, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Run the USBTMC 1.0 device-clear sequence on a pyvisa-py USBTMC interface.

    1. CLEAR_FEATURE(ENDPOINT_HALT) on both bulk endpoints, so the clear
       handshake (which may require reading bulk-IN) can proceed. Per
       USB 2.0 section 9.4.5 this also resets the data toggle to DATA0 on
       both sides, so it is harmless on a non-halted endpoint.
    2. INITIATE_CLEAR, then poll CHECK_CLEAR_STATUS until it is no longer
       PENDING, draining bulk-IN whenever bmClear.D0 reports queued data.
    3. CLEAR_FEATURE(ENDPOINT_HALT) on bulk-OUT, which the spec requires
       after a successful clear.
    4. Drain any bulk-IN residue (e.g. a stranded zero-length packet).

    Returns a report dict for logging. Raises RuntimeError if the device
    rejects or never completes the clear.
    """
    dev = itf.usb_dev
    ep_in, ep_out = itf.usb_recv_ep, itf.usb_send_ep
    intf = itf.usb_intf.bInterfaceNumber

    report: Dict[str, Any] = {
        'in_halted': _halted(itf, ep_in),
        'out_halted': _halted(itf, ep_out),
        'drained_bytes': 0,
    }

    # 1
    for ep in (ep_in, ep_out):
        dev.clear_halt(ep)

    # 2
    r = dev.ctrl_transfer(_REQTYPE_IN_CLASS_INTF, _INITIATE_CLEAR,
                          0, intf, 1, _CTRL_TIMEOUT_MS)
    if r[0] != _STATUS_SUCCESS:
        raise RuntimeError(f"USBTMC INITIATE_CLEAR rejected (status {r[0]:#04x})")

    t0 = time.monotonic()
    while True:
        r = dev.ctrl_transfer(_REQTYPE_IN_CLASS_INTF, _CHECK_CLEAR_STATUS,
                              0, intf, 2, _CTRL_TIMEOUT_MS)
        if r[0] != _STATUS_PENDING:
            break
        if r[1] & 1:
            report['drained_bytes'] += _drain_bulk_in(itf)
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("USBTMC CHECK_CLEAR_STATUS still pending after "
                               f"{timeout} s")
        time.sleep(0.05)
    if r[0] != _STATUS_SUCCESS:
        raise RuntimeError(f"USBTMC clear failed (status {r[0]:#04x})")

    # 3
    dev.clear_halt(ep_out)

    # 4
    report['drained_bytes'] += _drain_bulk_in(itf, timeout_ms=50)
    return report