# This class implements the logging behavior for all devices.

class abstractDevice:
    def __init__(self, logger = None):
        self.logger = logger

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
