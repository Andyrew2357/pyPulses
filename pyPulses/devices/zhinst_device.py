"""
Base class for Zurich Instruments devices.
"""

from .registry import HardwareRegistry
from .abstract_device import abstractDevice

from zhinst.core import ziDAQServer
import numpy as np
import time
from typing import Dict, Any


class zhinstDevice(abstractDevice):
    """
    Base class for instruments controlled via Zurich Instruments API.
    
    Parameters
    ----------
    device_id : str
        ZI device identifier (e.g., "dev1234"). Stored lowercase, which is how
        the data server spells node paths and poll keys.
    server_host : str, default='localhost'
        Data server hostname.
    server_port : int, default=8005
        Data server port.
    api_level : int, default=6
        ZI API level (the HF2 only supports 1).
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
        device_id = device_id.lower()
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
                self.stop_all_streams()
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

    def sync(self):
        """Block until settings sent so far have reached the device."""
        self.daq.sync()

    """
    -------------------------------------------------------------------------
    Streaming data
    -------------------------------------------------------------------------

    poll() is session-wide: it returns data for every subscribed node. All
    polling goes through _poll, which hands data for persistently subscribed
    ("streaming") nodes to their buffers, so one-off reads and streams can be
    mixed without stealing each other's samples.
    """

    # Cap on buffered chunks per stream that nobody reads
    MAX_STREAM_CHUNKS = 10000

    def _poll(self, duration: float, timeout: float = 0.5) -> dict:
        data = self.daq.poll(duration, int(timeout * 1000), 0, True)
        data = {k.lower(): v for k, v in data.items()}
        for path, chunks in self._streams.items():
            if path in data:
                chunks.append(data[path])
                del chunks[:-self.MAX_STREAM_CHUNKS]
        return data

    @property
    def _streams(self) -> Dict[str, list]:
        # path -> undelivered chunks; created lazily so subclasses that set
        # attributes before super().__init__ are unaffected
        if '_stream_buffers' not in self.__dict__:
            self._stream_buffers = {}
        return self._stream_buffers

    def poll_node(self, path: str, duration: float = 0.1,
                  timeout: float = 0.5) -> dict | None:
        """
        Record a node for `duration` seconds with a fresh subscription.

        Syncs first (so nothing buffered from before the call, or recorded
        before earlier settings took effect, is returned), then subscribes
        and polls for exactly `duration`. Costs a sync and two round trips
        on top of `duration`; see start_stream for repeated reads.

        Parameters
        ----------
        path : str
        duration : float, default=0.1
            Recording time in s.
        timeout : float, default=0.5
            Extra time in s to wait for data in transit.

        Returns
        -------
        dict or None
            The node's data (e.g. 'x', 'y', 'timestamp' arrays for a demod
            sample), or None if nothing arrived.
        """
        full_path = self._get_full_path(path).lower()
        if full_path in self._streams:
            raise RuntimeError(f"{full_path} is streaming; use read_stream.")
        self.daq.sync()
        self.daq.subscribe(full_path)
        try:
            data = self._poll(duration, timeout)
        finally:
            self.daq.unsubscribe(full_path)
        return data.get(full_path)

    def start_stream(self, path: str):
        """
        Subscribe to a node persistently (no-op if already streaming).

        Parameters
        ----------
        path : str
        """
        full_path = self._get_full_path(path).lower()
        if full_path in self._streams:
            return
        self.daq.sync()
        self.daq.subscribe(full_path)
        self._streams[full_path] = []
        self.info(f"ZHINST: streaming {full_path}")

    def stop_stream(self, path: str):
        """
        Parameters
        ----------
        path : str
        """
        full_path = self._get_full_path(path).lower()
        if self._streams.pop(full_path, None) is not None:
            self.daq.unsubscribe(full_path)
            self.info(f"ZHINST: stopped streaming {full_path}")

    def stop_all_streams(self):
        for path in list(self._streams):
            self.stop_stream(path)

    def is_streaming(self, path: str) -> bool:
        return self._get_full_path(path).lower() in self._streams

    def read_stream(self, path: str, n_samples: int, rate: float,
                    timeout: float = 1.0) -> Dict[str, Any]:
        """
        Return the next `n_samples` samples of a streaming node recorded
        after this call starts.

        The start is the timestamp of the node's latest sample (getSample,
        one round trip), so anything buffered from before, e.g. while gates
        moved or a setting settled, is dropped. (Polling with zero recording
        time does not flush the buffer on the HF2, so it can't be used for
        this.) That sample is at most one transfer latency (~10-30 ms on the
        HF2) old, so callers that wait for settling longer than that get only
        settled data. The window is cut by sample count, since one poll
        returns whatever has arrived rather than a fixed time span.

        Parameters
        ----------
        path : str
        n_samples : int
        rate : float
            Sample rate (Sa/s), used to size the polls.
        timeout : float, default=1.0
            Give up if the samples haven't arrived `timeout` s after the
            expected time.

        Returns
        -------
        dict
            Arrays (e.g. 'x', 'y', 'timestamp'), each of length n_samples.
        """
        full_path = self._get_full_path(path).lower()
        chunks = self._streams[full_path]
        start = np.ravel(self.daq.getSample(full_path)['timestamp'])[0]

        got: list = []
        n = 0
        deadline = time.monotonic() + n_samples / rate + timeout
        while True:
            # Keep only samples recorded after `start`, from whatever has
            # been delivered so far (including by other polls)
            for c in chunks:
                keep = np.asarray(c['timestamp']) > start
                piece = {k: np.asarray(v)[keep] for k, v in c.items()
                         if np.ndim(v) == 1 and len(v) == len(c['timestamp'])}
                got.append(piece)
                n += len(piece['timestamp'])
            chunks.clear()
            if n >= n_samples:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"{full_path}: only {n} of {n_samples} samples arrived."
                )
            self._poll((n_samples - n) / rate)

        return {k: np.concatenate([g[k] for g in got])[:n_samples]
                for k in got[0]}