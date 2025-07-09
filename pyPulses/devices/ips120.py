"""
This class is an interface to the Oxford IPS120 superconducting magnet power
supply.
"""

from .pyvisa_device import pyvisaDevice

import time
from enum import IntEnum

class ips120(pyvisaDevice):
    def __init__(self, logger = None, instrument_id: str = None):
        
        self.pyvisa_config = {
            'resource_name'     : "",
            'read_termination'  : '\r',
            'write_termination' : '\r'
        } 

        super().__init__(self.pyvisa_config, logger, instrument_id)

        self.retries    = 5
        self.max_rate   = 0.5   # T / min
        self.max_B      = 10.0  # T

        self.B_tol_match        = 1e-5  # T (Used when matching output and persistent)
        self.B_tol_assertive    = 1e-4  # T (Used in _goto_B and set_B)

    class parm(IntEnum):
        OUTPUT_CURRENT      = 0,    # A
        OUTPUT_VOLTAGE      = 1,    # V
        MAGNET_CURRENT      = 2,    # A
        SET_CURRENT         = 5,    # A
        SWEEP_RATE          = 6,    # A / min
        OUTPUT_FIELD        = 7,    # T
        SET_FIELD           = 8,    # T
        FIELD_SWEEP_RATE    = 9,    # T / min
        VOLTAGE_LIMIT       = 15,   # V
        PERSISTENT_CURRENT  = 16,   # A
        TRIP_CURRENT        = 17,   # A
        MAGNET_FIELD        = 18,   # T
        TRIP_FIELD          = 19,   # T
        SWITCH_CURRENT      = 20,   # mA
        SAFE_LIMIT_POS      = 21,   # A
        SAFE_LIMIT_NEG      = 22,   # A
        LEAD_RESISTANCE     = 23,   # mOhm
        MAGNET_INDUCTANCE   = 24    # H

    class mode(IntEnum):
        HOLD        = 0,
        GOTOTARGET  = 1,
        GOTOZERO    = 2,
        CLAMP       = 4

    class status(IntEnum):
        HEATER  = 7
        ATREST  = 10,
        MODE    = 3,
        SYSTEM  = 0,
        LIMIT   = 1

    """Base level routines"""

    def _send_cmd(self, cmd: str, retries: int = None) -> str | None:
        """Send command with retry logic"""
        
        retries = retries or self.retries
        for i in range(retries):
            try:
                if cmd.startswith('$'): # silent command
                    self.device.write(cmd)
                    return None
  
                response = self.device.query(cmd)
                if response[0] == cmd[0]: 
                    return response[1:].strip() # strip echo and whitespace

                self.error(f"Command echo mismatch: sent {cmd}, got {response}")
                self.device.clear()

            except:
                self.error(
                    f"Communication error on attempt {i + 1} "
                    f"while sending command: {cmd}"
                )
                if i < retries - 1:
                    time.sleep(0.1)
        
        else:
            self.error(
                f"Maximum number of retries exceeded attempting to send: {cmd}"
            )            

    def get(self, p: 'ips120.parm') -> float | None:
        """Get a parameter from the parameter table"""
        
        resp = self._send_cmd(f"R{p.value}")
        return resp if resp is None else float(resp)
    
    def get_status(self, flag: 'ips120.status') -> str | None:
        """Get the status of the power supply"""

        return self._send_cmd("X")[flag.value]
    
    def _set_mode(self, mode: 'ips120.mode'):
        """Set magnet power supply mode"""

        self._send_cmd(f"A{mode.value}")
        self.info(f"Set power supply mode to {mode.name}")

    def get_mode(self) -> bool:
        """Query the mode"""

        return self.mode(self.get_status(self.status.MODE))

    def _get_output_field(self) -> float | None:
        """Get the output field (not persistent)"""

        return self.get(self.parm.OUTPUT_FIELD)
    
    def _get_persistent_field(self) -> float | None:
        """Get the persistent field"""

        return self.get(self.parm.MAGNET_FIELD)
        
    def _set_target_field(self, field: float):
        """Set the target field in T"""

        if abs(field) > self.max_B:
            field = min(self.max_B, max(-self.max_B, field))
            self.warn(
                f"Target field exceeds maximum field strength; "
                f"clipped to {field} T"
            )

        self._send_cmd(f"J{field:.4f}")
        self.info(f"Set target field strength to {field:.4f} T")
        return True

    def is_at_rest(self) -> bool:
        """Is the power supply state at rest"""

        return self.get_status(self.status.ATREST) == '0'
    
    def is_persistent_mode(self) -> bool:
        """
        We define the controller to be in persistent mode if the heater is off,
        regardless of whether we are at field.
        """

        return self.get_status(self.status.HEATER) in ['0', '2']
    
    def is_heater_on(self) -> bool | None:
        """Query whether the heater is on"""

        c = self.get_status(self.status.HEATER)
        if c == '1':
            return True
        elif c == '5':
            self.error("Heater Fault! (heater on but current is low)")
            return
        elif c in ['0', '2']:
            return False
        else:
            self.warn(f"Unrecognized heater status flag: {c}")
            return

    def _set_heater(self, state: bool, wait_time: float) -> bool:
        """Set the heater state on or off"""

        for _ in range(self.retries):
            if self.get_mode() == self.mode.HOLD:
                break
        else:
            self.error("Cannot set heater state outside of hold mode!")
            return False

        
        for _ in range(self.retries):
            if not self.is_persistent_mode() or \
                abs(self._get_persistent_field() - \
                    self._get_output_field()) < self.B_tol:
                break
            
            self.warn(f"Cannot set heater to H{int(state)}; "
                      f"Lead Current = {self._get_output_field()} T, "
                      f"Mag Current = {self._get_persistent_field()} T ")
            time.sleep(1.0)
        else:        
            self.error(f"Failed to equalize persistent and output fields")
            return False
            
        self.info(f"Setting heater to H{int(state)} abd waiting {wait_time:.4f} s")
        self._send_cmd(f"H{int(state)}")
        time.sleep(wait_time)
        return True

    """Slightly higher level routines"""
        
    def _wait_until_at_rest(self, delay: float = 0.1):
        """Wait until we see the at rest status flag"""

        while not self.is_at_rest():
            time.sleep(delay)

    def _match_currents(self):
        """Match the output current ot the persistent current"""

        self.info("Attempting to match output current to persistent current.")

        if not self.is_heater_on():
            Bper = self._get_persistent_field()
            if Bper != self._get_output_field():
                self._set_mode(self.mode.HOLD)
                self._set_target_field(Bper)
                self._set_mode(self.mode.GOTOTARGET)
                self._wait_until_at_rest()

        self._set_mode(self.mode.HOLD)

    def _goto_B(self, B: float):

        if self.is_persistent_mode():
            if abs(self._get_persistent_field() - B) < self.B_tol_assertive:
                self.info(f"Already at requested field {B}")                
                return

        for _ in range(self.retries):
            # match output and persistent currents
            self._match_currents()

            if not self.is_heater_on():
                # try to turn the heater on (it will check that currents match)
                if self._set_heater(True):
                    break

        else:
            self.error(f"Maximum attempts exceeded; unable to set field.")

        self.info("Output is matched and switch heater is on.")
        self.info(f"Ramping to target field: {B} T")

        self._set_target_field(B)
        self._set_mode(self.mode.GOTOTARGET)
        time.sleep(0.05)
        self._wait_until_at_rest()
        self._set_mode(self.mode.HOLD)

    def _ramp_to_zero(self):
        """ramp the current in the leads to 0"""

        self._set_target_field(0.0)
        self._set_mode(self.mode.GOTOTARGET)
        time.sleep(0.05)
        self._wait_until_at_rest()
        self._set_mode(self.mode.HOLD)
        self.info("Ramped lead current to 0")

    """User level methods"""

    def set_B(self, B: float):
        """Set the field of the magnet"""

        if abs(B) > self.max_B:
            B = min(self.max_B, max(-self.max_B, B))
            self.warn(
                f"Requested field is larger than maximum allowed; clipped to {B} T"
            )

        while True: 
            self._goto_B(B)
            if abs(self.get_B() - B) < self.B_tol_assertive:
                break

        # Turn the heater off
        if not self._set_heater(False):
            self.error(
                "Failed to turn heater off! Aborting part way through set_B"
            )

        # Set the current in the leads to 0
        if abs(self._get_output_field()) != 0:
            self._ramp_to_zero()

    def get_B(self) -> float:
        """
        Query the field value 
        (Persistent if in persistent mode, else output)
        """

        if self.is_persistent_mode():
            return self._get_persistent_field()
        else:
            return self._get_output_field()
        
    def set_sweep_rate(self, rate: float):
        """Set the target sweep rate in T / min"""

        if rate > self.max_rate:
            rate = self.max_rate
            self.warn(
                f"Target rate exceeds maximum ramp rate; clipped to {rate}"
            )

        self._send_cmd(f"T{rate:.4f}")
        self.info(f"Set ramp rate to {rate:.4f} T/m")