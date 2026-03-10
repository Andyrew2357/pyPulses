"""
Base class for Zurich Instruments devices.
"""

from .registry import HardwareRegistry
from .abstract_device import abstractDevice

from zhinst.core import ziDAQServer
import time
from typing import Dict, Any


class zhinstDevice(abstractDevice):
    """
    Base class for instruments controlled via Zurich Instruments API.
    
    Parameters
    ----------
    device_id : str
        ZI device identifier (e.g., "dev1234").
    server_host : str, default='localhost'
        Data server hostname.
    server_port : int, default=8005
        Data server port.
    api_level : int, default=6
        ZI API level.
    registry_id : str, optional
        Logical ID for HardwareRegistry. If None, auto-generates.
    logger : Logger, optional
        Logger instance.
    skip_connect : bool, default=False
        If True, skip connecting on init.
    """
    
    def __init__(
        self,
        device_id: str,
        server_host: str = 'localhost',
        server_port: int = 8005,
        api_level: int = 6,
        registry_id: str | None = None,
        logger=None,
        skip_connect: bool = False,
    ):
        super().__init__(logger)
        
        # Store connection parameters
        self.device_id = device_id
        self.server_host = server_host
        self.server_port = server_port
        self.api_level = api_level
        
        self.path_prefix = f"/{device_id}/"
        self.daq = None
        
        # Register in hardware registry
        HardwareRegistry.register(self, registry_id=registry_id)
        
        # Connect unless told not to
        if not skip_connect:
            self.connect()

    """
    -------------------------------------------------------------------------
    Serialization
    -------------------------------------------------------------------------
    """

    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize connection parameters."""
        return {
            'device_id': self.device_id,
            'server_host': self.server_host,
            'server_port': self.server_port,
            'api_level': self.api_level,
        }

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state. Connection params can't change on existing connection,
        so this is mostly a no-op for zhinstDevice.
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "zhinstDevice":
        """Construct from serialized config."""
        registry_id = config.pop('registry_id')
        device_id = config.pop('device_id')
        
        return cls(
            device_id=device_id,
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
        """Connect to the data server."""
        self.daq = ziDAQServer(self.server_host, self.server_port, self.api_level)
        self.info(f"Connected to ZI device {self.device_id} via {self.server_host}:{self.server_port}")

    def disconnect(self):
        """Disconnect from the data server."""
        if self.daq is not None:
            try:
                self.daq.disconnect()
            except Exception as e:
                self.warn(f"Error disconnecting: {e}")
            self.daq = None

    def __del__(self):
        self.disconnect()
        super().__del__()

    """
    -------------------------------------------------------------------------
    ZI-specific methods
    -------------------------------------------------------------------------
    """
    
    def _get_full_path(self, path: str) -> str:
        return self.path_prefix + path if not path.startswith('/') else path
    
    def get_sample_val(self, path: str, key: str):
        full_path = self._get_full_path(path)
        value = self.daq.getSample(full_path)[key]
        self.debug(f"ZHINST: Got {full_path}/{key} = {value}")
        return value
    
    def get_sample(self, path: str) -> dict:
        full_path = self._get_full_path(path)
        data = self.daq.getSample(full_path)
        self.debug(f"ZHINST: Got {full_path} = {data}")
        return data

    def set_double(self, path: str, value):
        full_path = self._get_full_path(path)
        self.daq.setDouble(full_path, value)
        self.debug(f"ZHINST: Set {full_path} = {value}")

    def get_double(self, path: str):
        full_path = self._get_full_path(path)
        value = self.daq.getDouble(full_path)
        self.debug(f"ZHINST: Got {full_path} = {value}")
        return value

    def set_int(self, path: str, value) -> int:
        full_path = self._get_full_path(path)
        self.daq.setInt(full_path, value)
        self.debug(f"ZHINST: Set {full_path} = {value}")
        
    def get_int(self, path: str) -> int:
        full_path = self._get_full_path(path)
        value = self.daq.getInt(full_path)
        self.debug(f"ZHINST: GotInt {full_path} = {value}")
        return value

    def subscribe_and_poll(self, path: str, duration: float = 0.1,
                           poll_timeout: float = 0.2) -> dict:
        full_path = self._get_full_path(path)
        self.daq.subscribe(full_path)
        time.sleep(duration)
        data = self.daq.poll(poll_timeout, 500, 0, True)
        self.daq.unsubscribe(full_path)
        return data