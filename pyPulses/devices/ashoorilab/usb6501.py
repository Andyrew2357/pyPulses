"""
Pure-Python, OS-agnostic driver for the NI USB-6501 24-channel digital I/O device.
Uses PyUSB (libusb) instead of NI-DAQmx.

Platform setup
--------------
Linux:
        # /etc/udev/rules.d/99-usb6501.rules
        SUBSYSTEM=="usb", ATTR{idVendor}=="3923", ATTR{idProduct}=="718a", MODE="0666"

    Then: sudo udevadm control --reload-rules && sudo udevadm trigger

Windows:
    Use Zadig (https://zadig.akeo.ie) to replace the NI driver for "NI USB-6501"
    with WinUSB or libusb-win32.  Only needs to be done once per machine.
"""

try:
    import libusb_package
    libusb_package.get_libusb1_backend() # registers the bundled DLL
except ImportError:
    pass  # not on Windows, or user has system libusb — either is fine


from ..abstract_device import abstractDevice
from ..registry import HardwareRegistry

import threading
from typing import List
from logging import Logger

import usb.core
import usb.util


class USB6501(abstractDevice):
    """
    Driver for the National Instruments USB-6501 24-channel digital I/O device.

    Communicates directly over USB bulk transfers, bypassing NI-DAQmx entirely.
    Exposes three 8-bit ports (A, B, C) that can be independently configured as
    input or output.

    Thread safety
    -------------
    All public methods are protected by an internal lock and are safe to call from
    multiple threads.  `write_sequence` holds the lock for the full send+drain
    cycle so no other operation can interleave with a sequence in progress.
    """

    VENDOR_ID  = 0x3923
    PRODUCT_ID = 0x718A
    DATA_EP    = 0x01   # bulk-OUT to device
    RESP_EP    = 0x81   # bulk-IN  from device

    # Port constants
    PORT_A  = 0
    PORT_B  = 1
    PORT_C  = 2
    PORT_CL = 3   # lower nibble of PORT_C
    PORT_CH = 4   # upper nibble of PORT_C

    # ---------------------------------------------------------------------- #
    # USB packet templates                                                   #
    # Never modify these in-place; always copy with bytearray() first.       #
    # ---------------------------------------------------------------------- #

    # setOutput: [0x0E] = physical port (0/1/2), [0x11] = output byte value
    _SET_OUTPUT_CMD = bytes([
        0x00, 0x01, 0x00, 0x14, 0x00, 0x10, 0x01, 0x0F,
        0x02, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x01, 0x00, 0x00,
    ])

    # setInputMode: [0x0E/0x0F/0x10] = direction mask for ports A/B/C
    # (0x00 = input, 0xFF = output, values in between for partial nibble mode)
    _SET_MODE_CMD = bytes([
        0x00, 0x01, 0x00, 0x18, 0x00, 0x14, 0x01, 0x12,
        0x02, 0x10, 0x00, 0x00, 0x00, 0x05, 0xFE, 0x00,
        0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
    ])

    # getInput: [0x0E] = physical port; response byte [0x0E] = pin states
    _GET_INPUT_CMD = bytes([
        0x00, 0x01, 0x00, 0x10, 0x00, 0x0C, 0x01, 0x0E,
        0x02, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    ])

    # The device sends this 12-byte response after every successful write command
    _WRITE_OK_RESP = bytes([
        0x00, 0x01, 0x00, 0x0C, 0x00, 0x08, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x02,
    ])

    # ---------------------------------------------------------------------- #
    # Construction / teardown                                                #
    # ---------------------------------------------------------------------- #

    def __init__(self, 
        serial: str | None = None, 
        registry_id: str | None = None,  
        logger: Logger | None = None, 
        **kwargs
    ):
        """
        Open a connection to the USB-6501.

        Parameters
        ----------
        serial : str, optional
            USB serial number of the device.
            If *None* the first USB-6501 found on any bus is used.
        """

        super().__init__(logger=logger)
        HardwareRegistry.register(self, registry_id=registry_id)
        self.serial = serial

        self._lock = threading.Lock()

        # Cached state (the device has no readback for output values)
        self._output      = bytearray(3)                    # ports A, B, C
        self._output_mode = bytearray([0xFF, 0xFF, 0xFF])   # 0xFF = output, 0x00 = input

        find_kwargs: dict = dict(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
        if serial:
            find_kwargs['serial_number'] = serial

        self._dev = usb.core.find(**find_kwargs)
        if self._dev is None:
            raise RuntimeError(
                "NI USB-6501 not found"
                + (f" (serial={serial!r})" if serial else "")
                + ". Is the device plugged in and do you have permission to access it?"
            )

        # Detach kernel driver on Linux if it has claimed the interface
        try:
            if self._dev.is_kernel_driver_active(0):
                self._dev.detach_kernel_driver(0)
        except (NotImplementedError, usb.core.USBError):
            pass  # not applicable on Windows / macOS

        self._dev.set_configuration()
        usb.util.claim_interface(self._dev, 0)
        self._flush()

    def close(self) -> None:
        """Release the USB interface and return it to the OS."""
        try:
            self._flush(timeout_ms=100)
            usb.util.release_interface(self._dev, 0)
            usb.util.dispose_resources(self._dev)
        except Exception:
            pass

    def __del__(self)          : self.close()
    def __enter__(self)        : return self
    def __exit__(self, *args)  : self.close()

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                       #
    # ---------------------------------------------------------------------- #

    def _phys_port(self, port: int) -> int:
        """Map a PORT_* constant to the 0/1/2 index used in USB packets."""
        if port in (self.PORT_A, self.PORT_B):
            return port
        if port in (self.PORT_C, self.PORT_CL, self.PORT_CH):
            return 2
        raise ValueError(f"Unknown port constant: {port!r}")

    def _nibble_mask(self, port: int, mask: int) -> int:
        """Restrict mask to the relevant nibble for PORT_CL / PORT_CH."""
        if port == self.PORT_CL: return mask & 0x0F
        if port == self.PORT_CH: return mask & 0xF0
        return mask & 0xFF

    def _flush(self, timeout_ms: int = 50) -> None:
        """Discard any stale bytes sitting in the device's bulk-IN buffer."""
        try:
            while True:
                self._dev.read(self.RESP_EP, 1024, timeout=timeout_ms)
        except (usb.core.USBTimeoutError, usb.core.USBError):
            pass

    def _verify_write_resp(self, resp: bytes, context: str = "write") -> None:
        if resp != self._WRITE_OK_RESP:
            raise RuntimeError(
                f"USB-6501 unexpected {context} response: "
                f"got {resp.hex()!r}, expected {self._WRITE_OK_RESP.hex()!r}"
            )

    # ---------------------------------------------------------------------- #
    # Public API                                                             #
    # ---------------------------------------------------------------------- #

    def set_input_mode(self, port: int, is_input: bool) -> None:
        """Configure a port's direction.

        Parameters
        ----------
        port     : PORT_A, PORT_B, PORT_C, PORT_CL, or PORT_CH
        is_input : True → configure as input, False → configure as output
        """
        p = self._phys_port(port)

        with self._lock:
            # Update the cached direction mask for the relevant bits
            if port == self.PORT_CL:
                self._output_mode[2] = (
                    (self._output_mode[2] & 0xF0) | (0x00 if is_input else 0x0F)
                )
            elif port == self.PORT_CH:
                self._output_mode[2] = (
                    (self._output_mode[2] & 0x0F) | (0x00 if is_input else 0xF0)
                )
            else:
                self._output_mode[p] = 0x00 if is_input else 0xFF

            cmd = bytearray(self._SET_MODE_CMD)
            cmd[0x0E] = self._output_mode[0]
            cmd[0x0F] = self._output_mode[1]
            cmd[0x10] = self._output_mode[2]

            self._dev.write(self.DATA_EP, bytes(cmd))
            resp = bytes(self._dev.read(self.RESP_EP, 64))

        self._verify_write_resp(resp, context="set_input_mode")

    def set_output(self, port: int, data: int, mask: int = 0xFF) -> None:
        """Set output bits on a port.

        Only the bits that are 1 in *mask* are changed; the rest retain their
        current value.

        Parameters
        ----------
        port : PORT_A, PORT_B, PORT_C, PORT_CL, or PORT_CH
        data : byte value to write
        mask : bit mask (default 0xFF = change all bits)
        """
        mask = self._nibble_mask(port, mask)
        p    = self._phys_port(port)
        new  = ((data & mask) | (self._output[p] & ~mask)) & 0xFF

        cmd = bytearray(self._SET_OUTPUT_CMD)
        cmd[0x0E] = p
        cmd[0x11] = new

        with self._lock:
            self._dev.write(self.DATA_EP, bytes(cmd))
            resp = bytes(self._dev.read(self.RESP_EP, 64))
            self._output[p] = new

        self._verify_write_resp(resp, context="set_output")

    def write_sequence(self, port: int, states: List[int]) -> None:
        """
        Write a sequence of byte values to a port as fast as possible.

        All USB write commands are dispatched *before* any confirmation responses
        are read.  This pipelines USB host scheduling with device processing and
        avoids the 10-100 ms per-response penalty of a naive send-then-wait loop.

        Parameters
        ----------
        port   : PORT_A, PORT_B, or PORT_C
        states : sequence of byte values written to the port in order
        """
        if not states:
            return

        p        = self._phys_port(port)
        template = bytearray(self._SET_OUTPUT_CMD)
        template[0x0E] = p

        # Pre-build all command bytes so the send loop is as tight as possible
        cmds: List[bytes] = []
        for s in states:
            cmd = bytearray(template)
            cmd[0x11] = s & 0xFF
            cmds.append(bytes(cmd))

        with self._lock:
            # Dispatch all commands before reading any responses.
            # The device processes them sequentially while we continue sending.
            for cmd in cmds:
                self._dev.write(self.DATA_EP, cmd)

            # Drain all confirmation responses.  Content is always _WRITE_OK_RESP
            # and carries no data, so we don't verify each one individually.
            for _ in cmds:
                self._dev.read(self.RESP_EP, 64)

            self._output[p] = states[-1] & 0xFF

    def get_input(self, port: int, mask: int = 0xFF) -> int:
        """Read the current logic level of a port's input pins.

        Parameters
        ----------
        port : PORT_A, PORT_B, PORT_C, PORT_CL, or PORT_CH
        mask : bit mask applied to the result

        Returns
        -------
        int
            Masked byte read from the port.
        """
        mask = self._nibble_mask(port, mask)
        p    = self._phys_port(port)

        cmd = bytearray(self._GET_INPUT_CMD)
        cmd[0x0E] = p

        with self._lock:
            self._dev.write(self.DATA_EP, bytes(cmd))
            resp = bytes(self._dev.read(self.RESP_EP, 64))

        return resp[0x0E] & mask

    def get_output(self, port: int, mask: int = 0xFF) -> int:
        """Return the cached output value for a port (not queried from device)."""
        return self._output[self._phys_port(port)] & self._nibble_mask(port, mask)