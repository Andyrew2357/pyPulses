"""
Base class for VISA-controlled instruments.
"""

from .registry import HardwareRegistry
from .abstract_device import abstractDevice

import pyvisa
import pyvisa.constants
import re
import time
from threading import Lock
from typing import Dict, Any
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


class pyvisaDevice(abstractDevice):
    """
    Base class for instruments controlled via PyVISA.
    
    Handles connection management and rate limiting. A single lock (_device_lock)
    is held across every I/O operation so that concurrent threads cannot
    interleave reads and writes on the same instrument bus.

    Subclasses should define their default pyvisa_config and implement
    device-specific methods. Retry logic belongs at the command level in
    subclasses, where the semantics of each command are understood.
    
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
    
    # Subclasses can override this with their default config
    DEFAULT_PYVISA_CONFIG: Dict[str, Any] = {
        'timeout': 5000,
        'write_termination': '\n',
        'read_termination': '\n',
    }
    
    def __init__(
        self,
        resource_name: str,
        registry_id: str | None = None,
        logger=None,
        skip_connect: bool = False,
        **kwargs
    ):
        super().__init__(logger)
        
        # Build pyvisa config from defaults + overrides
        self.pyvisa_config = self.DEFAULT_PYVISA_CONFIG.copy()
        self.pyvisa_config.update(kwargs)
        self.pyvisa_config['resource_name'] = resource_name

        # Minimum interval between I/O operations (seconds)
        self._min_interval: float = self.pyvisa_config.pop('min_interval', 0.0)

        # Single lock serializing all I/O on this instrument
        self._device_lock = Lock()
        self._last_called: float = 0.0
        
        # Connection state
        self.device = None
        
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

    """
    -------------------------------------------------------------------------
    Serialization
    -------------------------------------------------------------------------
    """
    
    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize connection config."""
        config = self.pyvisa_config.copy()
        config['min_interval'] = self._min_interval
        return config

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore settings from serialized state."""
        if 'min_interval' in state:
            self._min_interval = state['min_interval']

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
    Communication methods
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

    def write(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            self.debug(f"Writing: {args[0]}")
            return self.device.write(*args, **kwargs)

    def write_raw(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            self.debug(f"Writing raw: {args[0]}")
            return self.device.write_raw(*args, **kwargs)

    def read(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            response = self.device.read(*args, **kwargs)
            self.debug(f"Read: {response}")
            return response

    def read_raw(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            response = self.device.read_raw(*args, **kwargs)
            self.debug(f"Read raw: {response}")
            return response

    def query(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            self.debug(f"Querying: {args[0]}")
            start = time.time()
            result = self.device.query(*args, **kwargs)
            self.debug(f"Response: {result.strip()} (in {time.time() - start:.3f}s)")
            return result

    def flush(self, *args, **kwargs):
        with self._device_lock:
            self._enforce_rate_limit()
            self.debug(f"Flushing: {args}")
            return self.device.flush(*args, **kwargs)


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