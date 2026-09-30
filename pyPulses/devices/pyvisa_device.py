"""
Base class for VISA-controlled instruments.
"""

from .registry import HardwareRegistry
from .abstract_device import abstractDevice
from . import usbtmc_recovery
from .errors import RecoverableError, mark_recoverable

import pyvisa
import pyvisa.constants
import pyvisa.errors
import re
import time
from contextlib import contextmanager
from threading import RLock
from typing import Dict, Any, List, Optional
import numpy as np


def parse_IEEE_488_2(data: bytes) -> np.ndarray:
    if not data.startswith(b'#'):
        raise ValueError("Invalid binary block format.")

    n = int(data[1:2]) # number of digits in length
    len_start = 2
    len_end = len_start + n
    length = int(data[len_start:len_end])
    binary_data = data[len_end:len_end + length]

    if len(binary_data) != length:
        raise ValueError("Invalid binary block length.")

    return np.frombuffer(binary_data, dtype = '<f4') # little-endian float32


class DeviceCommunicationError(Exception):
    """
    Raised for protocol-level failures that PyVISA itself doesn't already
    signal via VisaIOError -- e.g. an empty/short response from a device
    that should have replied. transact() raises this so callers get a
    single, predictable exception type for "the exchange didn't make
    sense," on top of whatever raw PyVISA errors also propagate.
    """
    pass


class LinkRecoveredError(DeviceCommunicationError, RecoverableError):
    """
    A transaction kept failing, but the link was cleared and verified in sync
    afterwards: the data is lost, the instrument is in a known state.
    Recoverable in the sense of devices.errors.
    """
    pass


class pyvisaDevice(abstractDevice):
    """
    Base class for instruments controlled via PyVISA.

    Parameters
    ----------
    resource_name : str
        VISA resource string (e.g., "ASRL5::INSTR", "GPIB0::5::INSTR").
    registry_id : str, optional
        Logical ID for HardwareRegistry. If None, auto-generates.
    logger : Logger, optional
        Logger instance for debug/info/warn/error messages.
    skip_connect : bool, default=False
        If True, skip connecting on init (for deserialization).
    **kwargs
    """

    # Subclasses can override/extend this with their own defaults -- see
    # _merged_default_config(), which combines these across the MRO
    # instead of one class's dict silently replacing another's.
    DEFAULT_PYVISA_CONFIG: Dict[str, Any] = {
        'timeout': 5000,
        'write_termination': '\n',
        'read_termination': '\n',
    }

    # Optional query used to confirm a USBTMC link is back in sync after a
    # device clear. Its response is captured at connect() and compared after
    # every clear. Leave None for devices without a stable identity query.
    USBTMC_SYNC_QUERY: Optional[str] = None

    # Non-USB resources only (USBTMC links always clear on timeout): if True,
    # a timeout triggers a VISA device clear (GPIB: SDC via ibclr) and the
    # timeout is re-raised marked recoverable. Off by default because some
    # older instruments treat a device clear as more than an I/O flush; turn
    # it on per class once that has been checked in the instrument manual.
    DEVICE_CLEAR_ON_TIMEOUT: bool = False

    def __init__(
        self,
        resource_name: str,
        registry_id: str | None = None,
        logger=None,
        skip_connect: bool = False,
        **kwargs
    ):
        super().__init__(logger)

        # Build pyvisa config from defaults (merged across the class
        # hierarchy -- see _merged_default_config) + overrides.
        self.pyvisa_config = self._merged_default_config()
        self.pyvisa_config.update(kwargs)
        self.pyvisa_config['resource_name'] = resource_name

        # Minimum interval between I/O operations (seconds)
        self._min_interval: float = self.pyvisa_config.pop('min_interval', 0.0)

        # Default number of attempts for transact() (1 = no retries).
        # Individual transact() calls can still override this with
        # max_attempts=.
        self._max_retries: int = self.pyvisa_config.pop('max_retries', 3)

        # Single, reentrant lock serializing all I/O on this instrument.
        self._device_lock = RLock()
        self._last_called: float = 0.0

        # Connection state
        self.device = None

        # USBTMC recovery state (only used for pyvisa-py USB INSTR sessions)
        self._usbtmc_sync_ref: Optional[str] = None
        self.usbtmc_clear_count: int = 0

        # Register in hardware registry
        HardwareRegistry.register(self, registry_id=registry_id)

        # Connect unless told not to
        if not skip_connect:
            if resource_name == 'DEBUG':
                self.device = dummyResource()
            else:
                self.connect()

    @property
    def resource_name(self) -> str:
        """The VISA resource string."""
        return self.pyvisa_config['resource_name']

    @classmethod
    def _merged_default_config(cls) -> Dict[str, Any]:
        """
        Merge DEFAULT_PYVISA_CONFIG across the class's MRO, base classes
        first, so a subclass only needs to specify the keys it wants to
        add or change.

        Plain `self.DEFAULT_PYVISA_CONFIG.copy()` resolves to whichever
        single class defines the attribute closest in the MRO -- if a
        subclass defines its own DEFAULT_PYVISA_CONFIG dict, that dict
        wholesale REPLACES the base class's, rather than extending it.
        That's exactly what happened with ad5764: its DEFAULT_PYVISA_CONFIG
        never mentioned 'timeout', 'write_termination', or
        'read_termination', so those silently fell back to whatever
        PyVISA/pyserial defaults to, instead of the 5000 ms /
        newline-terminated behavior this base class advertises. Merging
        instead means a subclass is only responsible for the keys it
        actually cares about.
        """
        merged: Dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get('DEFAULT_PYVISA_CONFIG', {}))
        return merged

    """
    -------------------------------------------------------------------------
    Serialization
    -------------------------------------------------------------------------
    """

    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize connection config."""
        config = self.pyvisa_config.copy()
        config['min_interval'] = self._min_interval
        config['max_retries'] = self._max_retries
        return config

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore settings from serialized state."""
        if 'min_interval' in state:
            self._min_interval = state['min_interval']
        if 'max_retries' in state:
            self._max_retries = state['max_retries']

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "pyvisaDevice":
        """
        Construct from serialized config.

        Parameters
        ----------
        config : dict
            Must contain 'registry_id' and 'resource_name'.
        """
        registry_id = config.pop('registry_id')
        resource_name = config.pop('resource_name')

        return cls(
            resource_name=resource_name,
            registry_id=registry_id,
            skip_connect=False,
            **config
        )

    """
    -------------------------------------------------------------------------
    Connection management
    -------------------------------------------------------------------------
    """

    def connect(self):
        """Open the VISA resource with configured settings."""
        resource_name = self.pyvisa_config["resource_name"]

        # Determine interface type
        if re.match(r'^ASRL', resource_name):
            interface_type = 'ASRL'
        elif re.match(r'^GPIB', resource_name):
            interface_type = 'GPIB'
        elif re.match(r'^TCPIP', resource_name):
            interface_type = 'TCPIP'
        else:
            interface_type = 'OTHER'

        # Open resource
        rm = pyvisa.ResourceManager('@py')
        self.device = rm.open_resource(resource_name)

        # Common configuration
        for attr in ['timeout', 'write_termination', 'read_termination']:
            if attr in self.pyvisa_config:
                try:
                    setattr(self.device, attr, self.pyvisa_config[attr])
                except Exception as e:
                    self.warn(f"Could not set attribute {attr}: {e}")

        # Buffer sizes
        if 'output_buffer_size' in self.pyvisa_config:
            try:
                self.device.set_buffer(
                    pyvisa.constants.VI_WRITE_BUF,
                    self.pyvisa_config['output_buffer_size']
                )
            except Exception as e:
                self.warn(f"Could not set output buffer size: {e}")

        if 'input_buffer_size' in self.pyvisa_config:
            try:
                self.device.set_buffer(
                    pyvisa.constants.VI_READ_BUF,
                    self.pyvisa_config['input_buffer_size']
                )
            except Exception as e:
                self.warn(f"Could not set input buffer size: {e}")

        # Interface-specific configuration
        self._configure_interface(interface_type)

        # Reference response for post-clear sync checks on USBTMC links
        self._usbtmc_sync_ref = None
        if self.USBTMC_SYNC_QUERY and self._usbtmc_interface() is not None:
            try:
                self._usbtmc_sync_ref = self.device.query(self.USBTMC_SYNC_QUERY)
            except Exception as e:
                self.warn(f"Could not capture USBTMC sync reference: {e}")

        self.info(f"Connected to instrument {resource_name}.")

    def _configure_interface(self, interface_type: str):
        """Apply interface-specific settings."""
        match interface_type:
            case 'ASRL':
                for attr in ['baud_rate', 'data_bits', 'stop_bits',
                             'parity', 'flow_control']:
                    if attr in self.pyvisa_config:
                        try:
                            setattr(self.device, attr, self.pyvisa_config[attr])
                        except Exception as e:
                            self.warn(f"Could not set attribute {attr}: {e}")

            case 'GPIB':
                if 'gpib_eos_mode' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_TERMCHAR_EN,
                        self.pyvisa_config['gpib_eos_mode']
                    )
                if 'gpib_eoi_mode' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_SEND_END_EN,
                        self.pyvisa_config['gpib_eoi_mode']
                    )
                if 'gpib_eos_char' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_TERMCHAR,
                        self.pyvisa_config['gpib_eos_char']
                    )

            case 'TCPIP':
                if 'tcpip_nodelay' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_TCPIP_NODELAY,
                        self.pyvisa_config['tcpip_nodelay']
                    )
                if 'tcpip_keepalive' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_TCPIP_KEEPALIVE,
                        self.pyvisa_config['tcpip_keepalive']
                    )

    def disconnect(self):
        """Close the VISA resource."""
        if self.device is not None:
            try:
                self.device.close()
            except Exception as e:
                self.warn(f"Error closing device: {e}")
            self.device = None

    def refresh(self):
        """Close and reopen the connection."""
        if self.pyvisa_config['resource_name'] == 'DEBUG':
            return
        self.disconnect()
        self.connect()

    def __del__(self):
        self.disconnect()
        super().__del__()

    """
    -------------------------------------------------------------------------
    Rate limiting and locking
    -------------------------------------------------------------------------
    """

    def _enforce_rate_limit(self):
        """
        Sleep if needed to respect the minimum inter-operation interval.
        Must be called while holding _device_lock.
        """
        if not self._min_interval:
            return
        elapsed = time.time() - self._last_called
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_called = time.time()

    @contextmanager
    def atomic(self):
        """
        Hold the device lock across an entire multi-step transaction (e.g.
        write + settle + read) so that no other call -- from another
        thread, if one is ever introduced -- can interleave I/O on this
        bus mid-sequence. Safe to nest with write_raw()/read_raw()/etc.,
        since _device_lock is reentrant.
        """
        with self._device_lock:
            yield

    """
    -------------------------------------------------------------------------
    USBTMC recovery
    -------------------------------------------------------------------------
    """

    def _usbtmc_interface(self):
        """pyvisa-py USBTMC interface for this resource, or None."""
        if self.device is None:
            return None
        return usbtmc_recovery.get_usbtmc_interface(self.device)

    def usbtmc_clear(self, reason: str = "manual") -> dict:
        """
        Run a USBTMC device clear (USB equivalent of GPIB SDC): un-halt both
        bulk endpoints, flush the device's input/output buffers, and discard
        any stranded bulk-IN data. Instrument settings are not changed.

        Raises DeviceCommunicationError if the resource is not a USBTMC
        session, the clear fails, or the link is still out of sync after it.
        """
        with self._device_lock:
            itf = self._usbtmc_interface()
            if itf is None:
                raise DeviceCommunicationError(
                    "usbtmc_clear() requires a pyvisa-py USB INSTR resource."
                )
            try:
                report = usbtmc_recovery.device_clear(itf)
            except Exception as e:
                raise DeviceCommunicationError(
                    f"USBTMC device clear failed: {e!r}"
                ) from e
            self.usbtmc_clear_count += 1
            self.warn(
                f"USBTMC device clear #{self.usbtmc_clear_count} ({reason}): "
                f"{report}"
            )

            if self.USBTMC_SYNC_QUERY and self._usbtmc_sync_ref is not None:
                try:
                    resp = self.device.query(self.USBTMC_SYNC_QUERY)
                except Exception as e:
                    raise DeviceCommunicationError(
                        f"Sync query failed after device clear: {e!r}"
                    ) from e
                if resp != self._usbtmc_sync_ref:
                    raise DeviceCommunicationError(
                        f"Link still out of sync after device clear: "
                        f"{self.USBTMC_SYNC_QUERY!r} returned {resp!r}, "
                        f"expected {self._usbtmc_sync_ref!r}."
                    )
            return report

    def _usbtmc_recover_quietly(self, reason: str) -> bool:
        """Best-effort clear used on error paths. Never raises; returns True
        if the link was cleared and verified."""
        try:
            self.usbtmc_clear(reason)
            return True
        except Exception as e:
            self.error(f"USBTMC recovery after {reason} failed: {e!r}")
            return False

    def _recover_after_timeout(self, is_usbtmc: bool) -> bool:
        """
        Try to restore a consistent link after a timeout, so the late
        response cannot be read as the answer to the next query. Returns
        True only if that succeeded.
        """
        if is_usbtmc:
            return self._usbtmc_recover_quietly("timeout")
        if self.DEVICE_CLEAR_ON_TIMEOUT:
            try:
                self.device.clear()
                self.warn("Device clear after timeout.")
                return True
            except Exception as e:
                self.error(f"Device clear after timeout failed: {e!r}")
        return False

    def _io(self, fn, *args, retry: bool = True, **kwargs):
        """
        Run one complete I/O transaction under the device lock.

        On pyvisa-py USBTMC links:
          * desync errors (EPIPE / EOVERFLOW / truncated bulk-IN) trigger a
            device clear and, if `retry`, a re-run of the whole transaction,
            up to max_retries attempts. If every attempt fails but the last
            clear succeeded, LinkRecoveredError (recoverable) is raised; if
            the link cannot be restored, DeviceCommunicationError (fatal).
          * a timeout triggers a device clear; the timeout is re-raised,
            marked recoverable if the clear succeeded.
          * KeyboardInterrupt triggers a device clear before re-raising.

        On other resources, a timeout is re-raised unchanged unless
        DEVICE_CLEAR_ON_TIMEOUT is set (see class attribute).

        `retry` must be False for operations that are only half of a
        transaction (a bare read after a separate write): the clear discards
        the response, so re-reading cannot succeed.
        """
        with self._device_lock:
            is_usbtmc = self._usbtmc_interface() is not None
            attempts = max(1, self._max_retries) if (is_usbtmc and retry) else 1
            for attempt in range(1, attempts + 1):
                self._enforce_rate_limit()
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    if is_usbtmc:
                        self._usbtmc_recover_quietly("KeyboardInterrupt mid-transaction")
                    raise
                except pyvisa.errors.VisaIOError as e:
                    if e.error_code == pyvisa.constants.StatusCode.error_timeout \
                            and self._recover_after_timeout(is_usbtmc):
                        raise mark_recoverable(e)
                    raise
                except Exception as e:
                    if not (is_usbtmc and usbtmc_recovery.is_desync_error(e)):
                        raise
                    self.error(
                        f"USBTMC desync on attempt {attempt}/{attempts}: {e!r}"
                    )
                    # Raises DeviceCommunicationError (fatal) if the link
                    # cannot be restored and verified.
                    self.usbtmc_clear(repr(e))
                    if attempt == attempts:
                        raise LinkRecoveredError(
                            f"USBTMC transaction failed after {attempts} "
                            f"attempt(s); link was cleared and verified."
                        ) from e

    """
    -------------------------------------------------------------------------
    Communication methods
    -------------------------------------------------------------------------
    """

    def write(self, *args, **kwargs):
        self.debug(f"Writing: {args[0]}")
        return self._io(self.device.write, *args, **kwargs)

    def write_raw(self, *args, **kwargs):
        self.debug(f"Writing raw: {args[0]}")
        return self._io(self.device.write_raw, *args, **kwargs)

    def read(self, *args, **kwargs):
        response = self._io(self.device.read, *args, retry=False, **kwargs)
        self.debug(f"Read: {response}")
        return response

    def read_raw(self, *args, **kwargs):
        response = self._io(self.device.read_raw, *args, retry=False, **kwargs)
        self.debug(f"Read raw: {response}")
        return response

    def query(self, *args, **kwargs):
        self.debug(f"Querying: {args[0]}")
        start = time.time()
        result = self._io(self.device.query, *args, **kwargs)
        self.debug(f"Response: {result.strip()} (in {time.time() - start:.3f}s)")
        return result

    def query_binary_values(self, message: str, **kwargs):
        """
        Write `message` and read an IEEE 488.2 binary block as one atomic,
        retryable transaction. kwargs are passed to pyvisa's
        query_binary_values (datatype, is_big_endian, container, ...).
        """
        self.debug(f"Querying (binary): {message}")
        return self._io(self.device.query_binary_values, message, **kwargs)

    def flush(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            self.debug(f"Flushing: {args}")
            return self.device.flush(*args, **kwargs)

    def transact(
        self,
        payload: bytes,
        n_response_lines: int = 0,
        settle: float = 0.0,
        max_attempts: Optional[int] = None,
        retry_wait: float = 0.2,
    ) -> Optional[List[str]]:
        """
        Send one fixed-frame command and, optionally, read back a fixed
        number of newline-terminated response lines, as a single logical
        transaction. Intended for simple serial instrument protocols with
        no handshake/checksum (e.g. the Arduino-controlled boxes), where
        the host can't otherwise tell a stale byte from a current one.

        Parameters
        ----------
        payload : bytes
            Exact command bytes to write.
        n_response_lines : int, default=0
            Number of newline-terminated lines to read back and return.
            0 means write-only (fire-and-forget).
        settle : float, default=0.0
            Seconds to wait after writing before reading (or returning).
        max_attempts : int, optional
            Attempts before giving up and raising the last exception.
            Defaults to this device's configured `max_retries` (see
            DEFAULT_PYVISA_CONFIG / __init__).
        retry_wait : float, default=0.2
            Seconds to wait between attempts.

        Returns
        -------
        list of str, or None
            Response lines, if n_response_lines > 0, else None.

        Raises
        ------
        Exception
            Whatever the last attempt raised (typically
            pyvisa.errors.VisaIOError or DeviceCommunicationError), if
            every attempt fails.
        """
        if max_attempts is None:
            max_attempts = self._max_retries

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self.atomic():
                    self.flush(
                        pyvisa.constants.VI_READ_BUF
                        | pyvisa.constants.VI_IO_IN_BUF_DISCARD
                    )

                    # 2. Write the command.
                    self.write_raw(payload)

                    # 3. Let the device actually act on it.
                    if settle:
                        time.sleep(settle)

                    # 4. Optionally read back a response.
                    if n_response_lines:
                        lines = []
                        for _ in range(n_response_lines):
                            line = self.read().strip()
                            if not line:
                                raise DeviceCommunicationError(
                                    "Received empty line from device."
                                )
                            lines.append(line)
                        return lines
                    return None
            except Exception as e:
                last_exc = e
                self.warn(
                    f"Transaction attempt {attempt}/{max_attempts} failed: {e}"
                )
                if attempt < max_attempts:
                    time.sleep(retry_wait)

        self.error(f"Transaction failed after {max_attempts} attempts: {last_exc}")
        raise last_exc


"""
This is a dummy class for debugging instruments without actually sending
commands (important for testing things like magnet power supplies).

There exist far more sophisticated ways of modeling instruments that take SCPI
commands, but not all of ours operate this way (some are arduino controlled),
and this is far easier to implement. The user just has to preprogram certain
responses.

See cryomagnetics_4G.py for a good example of how to use this, including adding
various commands and dealing with simulated attributes.
"""
class dummyResource(abstractDevice):
    def __init__(self, logger = None):
        super().__init__(logger)
        self.history = []
        self.output = ""
        self.commands = {}

        self.attr = {}

    def receive(self, cmd, *args, **kwargs):
        event = {'command': cmd, 'args': args, 'kwargs': kwargs}
        msg = f"{cmd}: args = {args}, kwargs = {kwargs}"
        self.history.append(event)
        self.debug(msg)

    def response(self):
        s = self.output
        self.output = ""
        return s

    def parse(self, command, *args, **kwargs):
        for expression in self.commands:
            hit = re.match(expression, command)
            if hit:
                arguments = hit.groups()
                res = self.commands[expression](self, *arguments,
                                                *args, **kwargs)
                if res is not None:
                    self.output = res

    def add_command(self, regular_expression, function):
        self.commands[regular_expression] = function

    """functions used by the instrument control class."""

    def write(self, *args, **kwargs):
        self.receive('write', *args, **kwargs)
        self.parse(*args, **kwargs)

    def write_raw(self, *args, **kwargs):
        self.receive('write_raw', *args, **kwargs)
        self.parse(*args, **kwargs)

    def flush(self, *args, **kwargs):
        self.receive('flush', *args, **kwargs)
        self.output = ""

    def read_raw(self, *args, **kwargs):
        self.receive('read_raw', *args, **kwargs)
        return self.response()

    def query(self, *args, **kwargs):
        self.receive('query', *args, **kwargs)
        self.parse(*args, **kwargs)
        return self.response()