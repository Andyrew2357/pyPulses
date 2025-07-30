"""
This class is a bare-bones framework for low-level devices that use the pyvisa 
package for communication. This includes the majority of standalone instruments.
"""

from ._registry import DeviceRegistry
from .nivisa_utils import visa_dll
from .abstract_device import abstractDevice
import pyvisa.constants
import pyvisa
import re
import functools
import time
from threading import Lock
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
    def __init__(self, pyvisa_config: dict, logger = None, instrument_id = None):
        """Standard initialization, calling ResourceManager.open_resource."""
        
        super().__init__(logger)
        self.pyvisa_config = pyvisa_config
        if instrument_id:
            self.pyvisa_config['resource_name'] = instrument_id
        DeviceRegistry.register_device(self.pyvisa_config['resource_name'], self)
        
        # Retry and rate limit settings (can be updated later)
        self.retry_settings = {
            'max_retries': pyvisa_config.get('max_retries', 3),
            'retry_delay': pyvisa_config.get('retry_delay', 0.1),
            'retry_exceptions': pyvisa_config.get('retry_exceptions', 
                                                  (pyvisa.VisaIOError,)),
            'min_interval': pyvisa_config.get('min_interval', 0.01)
        }
        self._rate_limit_locks = {}
        self._last_called = {}

        # for debugging purposes, we can connect to an object that mimics a
        # pyvisa resource but just logs the messages it recieves. The user may
        # also pre-program responses from the dummy resource for testing. 
        if pyvisa_config['resource_name'] == 'DEBUG':
            self.device = dummyResource()
            return
        
        self.connect()

    def connect(self):
        """
        Open a VISA instrument with the right configuration.
        Works with ASRL, GPIB, and TCPIP instruments.
        """
    
        # Store the resource name and configuration
        resource_name = self.pyvisa_config["resource_name"]
    
        # Determine instrument type
        if re.match(r'^ASRL', resource_name):
            interface_type = 'ASRL'
        elif re.match(r'^GPIB', resource_name):
            interface_type = 'GPIB'
        elif re.match(r'^TCPIP', resource_name):
            interface_type = 'TCPIP'
        else:
            interface_type = 'OTHER'
    
        # Open the instrument with the resource manager
        rm = pyvisa.ResourceManager(visa_dll)
        self.device = rm.open_resource(resource_name)
    
        # Common configuration
        for attr in ['timeout', 'write_termination', 'read_termination']:
            if attr in self.pyvisa_config:
                try:
                    setattr(self.device, attr, self.pyvisa_config[attr])
                except Exception as e:
                    self.warn(f"Could not set attribute {attr}: {e}")

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
        match interface_type:
            case 'ASRL':
                # Serial-specific attributes
                for attr in ['baud_rate', 'data_bits', 'stop_bits', 
                             'parity', 'flow_control', 'write_buffer_size', 
                             'read_buffer_size']:
                    if attr in self.pyvisa_config:
                        try:
                            setattr(self.device, attr, self.pyvisa_config[attr])
                        except Exception as e:
                            self.warn(f"Could not set attribute {attr}: {e}")
                        
            case 'GPIB':
                # GPIB-specific attributes using set_visa_attribute
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
                # TCPIP-specific attributes
                if 'tcpip_nodelay' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_TCPIP_NODELAY, 
                        self.pyvisa_config['tcpip_nodelay']
                    )
                if 'tcpip_keepalive' in self.pyvisa_config:
                    self.device.set_visa_attribute(
                        pyvisa.constants.VI_ATTR_TCPIP_KEEPALIVE, 
                        self.pyvisa_config['tcpip_keepalive'])
            
            case _:
                self.warn(f"Interface type {interface_type} not recognized.")
    
        self.info(f"Connected to instrument {resource_name}.")

    def refresh(self):
        """Close and reopen the device."""
        if self.pyvisa_config['resource_name'] == 'DEBUG':
            return

        self.device.close()
        self.connect()

    def __del__(self):
        try:
            self.device.close()
        except Exception as e:
            self.warn(f"Exception during pyvisaDevice cleanup: {e}")
            
        super().__del__()

    @staticmethod
    def retry_and_rate_limit(method):
        """
        Decorator that retries and rate-limits instance methods, reading 
        settings from self.retry_settings.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Extract instance settings
            settings = getattr(self, "retry_settings", {})
            max_retries = settings.get("max_retries", 3)
            retry_exceptions = settings.get("retry_exceptions", 
                                            (pyvisa.VisaIOError,))
            retry_delay = settings.get("retry_delay", 0.1)
            min_interval = settings.get("min_interval", 0.0)

            # Rate limiting: lock to ensure global timing across threads
            lock = self._rate_limit_locks.setdefault(method.__name__, Lock())
            with lock:
                last_called = self._last_called.setdefault(method.__name__, 0.0)
                elapsed = time.time() - last_called
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                self._last_called[method.__name__] = time.time()

            # Retry logic
            for attempt in range(max_retries):
                try:
                    return method(self, *args, **kwargs)
                except retry_exceptions as e:
                    self.warn(
                        f"{method.__name__} failed (attempt {attempt+1}): {e}"
                    )
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)

        return wrapper
    
    def set_retry_params(self, 
                         max_retries: int = None, 
                         retry_delay: float = None, 
                         min_interval: float = None):
        if max_retries is not None:
            self.retry_settings["max_retries"] = max_retries
        if retry_delay is not None:
            self.retry_settings["retry_delay"] = retry_delay
        if min_interval is not None:
            self.retry_settings["min_interval"] = min_interval

    @retry_and_rate_limit
    def write(self, *args, **kwargs):
        self.debug(f"Writing: {args[0]}")
        return self.device.write(*args, **kwargs)

    @retry_and_rate_limit
    def write_raw(self, *args, **kwargs):
        self.debug(f"Writing raw: {args[0]}")
        return self.device.write_raw(*args, **kwargs)

    @retry_and_rate_limit
    def read(self, *args, **kwargs):
        response = self.device.read(*args, **kwargs)
        self.debug(f"Read: {response}")
        return response

    @retry_and_rate_limit
    def read_raw(self, *args, **kwargs):
        response = self.device.read_raw(*args, **kwargs)
        self.debug(f"Read raw: {response}")
        return response

    @retry_and_rate_limit
    def query(self, *args, **kwargs):
        self.debug(f"Querying: {args[0]}")
        start = time.time()
        result = self.device.query(*args, **kwargs)
        elapsed = time.time() - start
        self.debug(f"Response: {result.strip()} (in {elapsed:.3f}s)")
        return result
    
    @retry_and_rate_limit
    def flush(self, *args, **kwargs):
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
