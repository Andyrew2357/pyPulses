import pyvisa

class pyvisaDevice:
    def __init__(self, pyvisa_config, logger = None):
        self.logger = logger
        rm = pyvisa.ResourceManager('@py')
        self.device = rm.open_resource(**pyvisa_config)

    def debug(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def warn(self, msg):
        if self.logger:
            self.logger.warning(msg)

    def error(self, msg):
        if self.logger:
            self.logger.error(msg)
