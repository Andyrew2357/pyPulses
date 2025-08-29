"""
This class is an interface for communicating with the Keithley 2700 multimeter
"""

from .pyvisa_device import pyvisaDevice

class keithley2700(pyvisaDevice):
    """Class interface for controlling the Keithley 2700."""
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
            "resource_name" : "GPIB0::17::INSTR",

            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,

            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
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

        return float(self.query(":MEAS:VOLT:DC?").split("VDC")[0]) # FIX THIS LATER. I JUST NEEDED ANOTHER MULTIMETER QUICKLY
