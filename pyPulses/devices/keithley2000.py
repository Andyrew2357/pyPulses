"""
This class is an interface for communicating with the Keithley 2000 multimeter
"""

from .pyvisa_device import pyvisaDevice

class keithley2000(pyvisaDevice):
    """Class interface for controlling the Keithley 2000."""
    def __init__(self, logger: str = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """
        self.pyvisa_config = {
            "resource_name" : "GPIB0::24::INSTR",

            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

    def get_V(self) -> float:
        """
        Query the voltage.
        
        Returns
        -------
        V : float
        """
        """Query the measured voltage."""
        self.device.write(":CONF:VOLT:DC")
        return float(self.device.query(":READ?"))


    def get_I(self) -> float:
        """
        Query the current in amps.
        
        Returns
        -------
        I : float
        """
        self.device.write(":CONF:VOLT:DC")
        return float(self.device.query(":READ?"))
