"""
This class is a bare-bones framework for low-level devices that use the 
zhinst-toolkit package for communication.
"""

from ._registry import DeviceRegistry
from .abstract_device import abstractDevice

from zhinst.core import ziDAQServer
import time

class zhinstDevice(abstractDevice):
    def __init__(self, 
                 device_id: str, 
                 server_host: str = 'localhost',
                 server_port: int = 8005, 
                 API_level: int = 6, 
                 logger = None):
        
        super().__init__(logger)

        self.daq = ziDAQServer(server_host, server_port, API_level)
        self.path_prefix = f"/{device_id}/"

        DeviceRegistry.register_device(device_id, self)

    def _get_full_path(self, path: str) -> str:
        return self.path_prefix + path if not path.startswith('/') else path
    
    def get_sample_val(self, path: str, key: str):
        full_path = self._get_full_path(path)
        value = self.daq.getSample(full_path)[key]
        self.info(f"ZHINST: Got {full_path}/{key} = {value}")
        return value
    
    def get_sample(self, path: str) -> dict:
        full_path = self._get_full_path(path)
        data = self.daq.getSample(full_path)
        self.info(f"ZHINST: Got {full_path} = {data}")
        return data

    def set_double(self, path: str, value):
        full_path = self._get_full_path(path)
        self.daq.setDouble(full_path, value)
        self.info(f"ZHINST: Set {full_path} = {value}")

    def get_double(self, path: str):
        full_path = self._get_full_path(path)
        value = self.daq.getDouble(full_path)
        self.info(f"ZHINST: Got {full_path} = {value}")
        return value

    def set_int(self, path: str, value) -> int:
        full_path = self._get_full_path(path)
        value = self.daq.setInt(full_path, value)
        self.info(f"ZHINST: Set {full_path} = {value}")
        
    def get_int(self, path: str) -> int:
        full_path = self._get_full_path(path)
        value = self.daq.getInt(full_path)
        self.info(f"ZHINST: GotInt {full_path} = {value}")
        return value

    def subscribe_and_poll(self, path: str, duration: float = 0.1,
                           poll_timeout: float = 0.2) -> dict:
        full_path = self._get_full_path(path)
        self.daq.subscribe(full_path)
        time.sleep(duration)
        data = self.daq.poll(poll_timeout, 500, 0, True)
        self.daq.unsubscribe(full_path)
        return data

    def __del__(self):
        try:
            self.daq.disconnect()
        except Exception as e:
            self.warn(f"Exception during zhinstDevice cleanup: {e}")

        super().__del__()
