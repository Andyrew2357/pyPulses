"""
This class implements the logging behavior for all devices.
"""

import json

class abstractDevice:
    def __init__(self, logger = None):
        self.logger = logger
        self._registry_name_ = None

    def debug(self, msg):
        """Standard debug message."""
        if self.logger:
            self.logger.debug(msg)

    def info(self, msg):
        """Standard info message."""
        if self.logger:
            self.logger.info(msg)

    def warn(self, msg):
        """Standard warning message."""
        if self.logger:
            self.logger.warning(msg)

    def error(self, msg):
        """Standard error message."""
        if self.logger:
            self.logger.error(msg)

    def __del__(self):
        """
        Anything that should happen when the device is unregistered. By default,
        it just removes all the handlers from the associated logger. If a
        particular device should have additional behavior, this can be achieved
        by decorating the __del__ function.
        """
        if self.logger:
            while self.logger.hasHandlers():
                self.logger.removeHandler(self.logger.handlers[0])

    """Save / Load state by serializing to / deserializing from JSON"""

    def save_state_json(self, path: str):
        """
        Save the instrument state to JSON locally.

        Parameters
        ----------
        path : str
            directory at which to save.
        """

        try:
            state = self._serialize_state()
        except:
            state = {}

        if state == {}:
            print("State serialization failed for ",
                  (self._registry_name_ or "UNREGISTERED_DEVICE"))

        with open(path, 'w') as f:
            json.dump(state, f, indent = 2)

    def load_state_json(self, path: str):
        """
        Load the instrument state from JSON locally.
        
        Parameters
        ----------
        path : str
            directory from which to load.
        """
        with open(path, 'r') as f:
            state = json.load(f)

        try:
            self._deserialize_state(state)
        except:
            print("State deserialization failed for ",
                  (self._registry_name_ or "UNREGISTERED_DEVICE"))

    def _serialize_state(self) -> dict:
        return {}
    
    def _deserialize_state(self, state: dict):
        return
