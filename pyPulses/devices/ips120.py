from .pyvisa_device import pyvisaDevice
from .registry import register_hardware_class, register_device_class, format_reference, DeferredReference, HardwareRegistry
from .sweepable_channel import AsyncChannel

import time
from enum import IntEnum
from logging import Logger

@register_hardware_class("ips120")
class ips120(pyvisaDevice):
    """
    Class representation of the Oxford IPS120 superconducting magnet power supply.
    """

    DEFAULT_PYVISA_CONFIG = {
        'read_termination'  : '\r',
        'write_termination' : '\r'
    } 

    retries = 5
    max_rate = 0.5   # T / min
    max_B = 10.0  # T

    heater_wait_time = 7 # s
    stabilize_wait_time = 10 # s  pause before/after heater switching to let field stabilize

    B_tol_match = 1e-5  # T (Used when matching output and persistent)
    B_tol_assertive = 1e-4  # T (Used in _goto_B and set_B)

    def __init__(self,
        resource_name: str, 
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_connect: bool = False,
        **kwargs,              
    ):
        """
        Parameters
        ----------
        resource_name : str
            VISA resource name.
        registry_id : str, optional
            Name to register this instance under in the HardwareRegistry
        logger : Logger, optional
            logger used by abstractDevice.
        **kwargs
        """

        super().__init__(resource_name, registry_id, logger, skip_connect, **kwargs)

    class parm(IntEnum):
        OUTPUT_CURRENT      = 0  # A
        OUTPUT_VOLTAGE      = 1  # V
        MAGNET_CURRENT      = 2  # A
        SET_CURRENT         = 5  # A
        SWEEP_RATE          = 6  # A / min
        OUTPUT_FIELD        = 7  # T
        SET_FIELD           = 8  # T
        FIELD_SWEEP_RATE    = 9  # T / min
        VOLTAGE_LIMIT       = 15 # V
        PERSISTENT_CURRENT  = 16 # A
        TRIP_CURRENT        = 17 # A
        MAGNET_FIELD        = 18 # T
        TRIP_FIELD          = 19 # T
        SWITCH_CURRENT      = 20 # mA
        SAFE_LIMIT_POS      = 21 # A
        SAFE_LIMIT_NEG      = 22 # A
        LEAD_RESISTANCE     = 23 # mOhm
        MAGNET_INDUCTANCE   = 24 # H

    class mode(IntEnum):
        HOLD        = 0
        GOTOTARGET  = 1
        GOTOZERO    = 2
        CLAMP       = 4

    class status(IntEnum):
        HEATER  = 7
        ATREST  = 10
        MODE    = 3
        SYSTEM  = 0
        LIMIT   = 1

    """Base level routines"""

    def _send_cmd(self, cmd: str, retries: int = None) -> str | None:
        """Send command with retry logic"""
        # This class has its own bespoke retry logic because I am reticent to
        # change things that appear to have worked in other implementations for
        # an instrument of this nature

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
        """
        Get a parameter from the parameter table
        
        Parameters
        ----------
        p : ips120.parm

        Returns
        -------
        float or None
        """
        
        resp = self._send_cmd(f"R{p.value}")
        return resp if resp is None else float(resp)
    
    def get_status(self, flag: 'ips120.status') -> str | None:
        """
        Get the status of the power supply from the status table
        
        Parameters
        ----------
        flag : ips120.status
            {'HEATER', 'ATREST', 'MODE', 'SYSTEM', 'LIMIT'}

        Returns
        -------
        str or None
        """

        return self._send_cmd("X")[flag.value]
    
    def _set_mode(self, mode: 'ips120.mode'):
        """Set magnet power supply mode"""

        self._send_cmd(f"A{mode.value}")
        self.info(f"Set power supply mode to {mode.name}")

    def get_mode(self) -> bool:
        """
        Query the mode of magnet operation.

        Returns
        -------
        mode : ips120.mode
            {'HOLD', 'GOTOTARGET', 'GOTOZERO', 'CLAMP'}.
        """

        return self.mode(float(self.get_status(self.status.MODE)))

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
        """
        Is the power supply state at rest
        
        Returns
        -------
        bool
        """

        return self.get_status(self.status.ATREST) == '0'
    
    def is_persistent_mode(self) -> bool:
        """
        We define the controller to be in persistent mode if the heater is off,
        regardless of whether we are at field.

        Returns
        -------
        bool
        """

        return self.get_status(self.status.HEATER) in ['0', '2']
    
    def is_heater_on(self) -> bool | None:
        """
        Query whether the heater is on
        
        Returns
        -------
        bool
        """

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

    def _set_heater(self, state: bool) -> bool:
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
                    self._get_output_field()) < self.B_tol_match:
                break
            
            self.warn(f"Cannot set heater to H{int(state)}; "
                      f"Lead Current = {self._get_output_field()} T, "
                      f"Mag Current = {self._get_persistent_field()} T ")
            time.sleep(1.0)
        else:        
            self.error(f"Failed to equalize persistent and output fields")
            return False
        
        wait_time = self.heater_wait_time
        self.info(f"Setting heater to H{int(state)} and waiting {wait_time:.4f} s")
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
            self.error(
                "Maximum attempts exceeded; unable to set field. Ramping back "
                "to 0 lead current for safety and aborting."
            )
            self._set_heater(False)
            self._ramp_to_zero()
            return


        self.info("Output is matched and switch heater is on.")
        self.info(f"Ramping to target field: {B} T")

        self._set_target_field(B)
        self._set_mode(self.mode.GOTOTARGET)
        time.sleep(0.05)
        self._wait_until_at_rest()

        # Poll R7 (output field) directly until target is reached, matching MATLAB
        while True:
            Bout = self._get_output_field()
            if Bout is not None and abs(Bout - B) <= self.B_tol_assertive:
                break
            self.info(f"Waiting to reach target; output field = {Bout:.6f} T, target = {B} T")
            time.sleep(5.0)

        self.info("Output field at target; pausing to stabilize before heater off")
        time.sleep(self.stabilize_wait_time)

        self._set_mode(self.mode.HOLD)

    def _ramp_to_zero(self):
        """Ramp the current in the leads to 0"""

        self._set_mode(self.mode.GOTOZERO)
        time.sleep(0.05)
        self._wait_until_at_rest()

        # Poll output field directly until it reads zero, matching MATLAB behavior
        while True:
            Bout = self._get_output_field()
            if Bout is None or abs(Bout) <= self.B_tol_assertive:
                break
            self.info(f"Waiting for lead current to reach zero; currently {Bout:.6f} T")
            time.sleep(3.0)

        self._set_mode(self.mode.HOLD)
        self.info("Ramped lead current to 0")

    """User level methods"""

    def set_B(self, B: float) -> bool:
        """
        Set the field of the magnet
        
        Parameters
        ----------
        B : float
            target persistent field in T

        Returns
        -------
        success : bool
        """

        if abs(B) > self.max_B:
            B = min(self.max_B, max(-self.max_B, B))
            self.warn(
                f"Requested field is larger than maximum allowed; clipped to {B} T"
            )

        # If we arrive here in driven mode (heater on, leads energized), we
        # must transition to persistence mode safely before proceeding.
        # _exit_driven_mode ramps leads to match coil, turns off heater, and
        # waits for the switch to go superconducting. _goto_B will then match
        # the (now-persistent) coil field before turning the heater on again.
        if self.is_heater_on():
            self.info(
                "set_B called while in driven mode; transitioning to "
                "persistence mode before sweeping."
            )
            if not self._exit_driven_mode():
                self.error(
                    "Failed to exit driven mode safely; aborting set_B."
                )
                return False

        tries = 0
        while True:
            tries += 1
            if tries > self.retries:
                self.warn(
                    "Maximum number of retries exceeded in set_B without "
                    "reaching assertive tolerance; proceeding..."
                )
                break

            self._goto_B(B)
            if abs(self.get_B() - B) < self.B_tol_assertive:
                break

        # Turn the heater off
        if not self._set_heater(False):
            self.error(
                "Failed to turn heater off! Aborting part way through set_B"
            )
            return False

        # Pause to let persistent field stabilize after heater off
        self.info(f"Heater off; pausing {self.stabilize_wait_time} s to stabilize")
        time.sleep(self.stabilize_wait_time)

        # Verify persistent field matches target (R18)
        Bper = self._get_persistent_field()
        if Bper is not None and abs(Bper - B) > self.B_tol_assertive:
            self.warn(
                f"Persistent field {Bper:.6f} T differs from target {B:.6f} T "
                f"after heater off"
            )

        # Ramp leads to zero
        if self._get_output_field() is not None and \
                abs(self._get_output_field()) > self.B_tol_assertive:
            self._ramp_to_zero()

        return True

    def get_B(self) -> float:
        """
        Query the field value in T (Persistent if in persistent mode, else 
        output).

        Returns
        -------
        B : float
        """

        if self.is_persistent_mode():
            return self._get_persistent_field()
        else:
            return self._get_output_field()
        
    def set_sweep_rate(self, rate: float):
        """
        Set the target sweep rate in T / min.

        Parameters
        ----------
        rate: float
            sweep rate in T / min.
        """

        if rate > self.max_rate:
            rate = self.max_rate
            self.warn(
                f"Target rate exceeds maximum ramp rate; clipped to {rate}"
            )

        self._send_cmd(f"T{rate:.4f}")
        self.info(f"Set ramp rate to {rate:.4f} T/m")

    """Driven Mode"""

    def _exit_driven_mode(self) -> bool:
        """
        Safely transition out of driven mode into persistence mode. Ramps the
        leads to match the coil field (if necessary), then turns off the switch
        heater and waits for the switch to go superconducting.
 
        This must be called before any persistence-mode operation when the
        supply may be in driven mode. Turning off the heater with a mismatch
        between lead current and coil current will cause the coil field to
        jump, which risks a quench.
 
        If the heater is already off, this is a no-op.
 
        Returns
        -------
        success : bool
        """
        if not self.is_heater_on():
            self.info("_exit_driven_mode: heater already off; nothing to do.")
            return True
 
        self.info("_exit_driven_mode: heater is on; transitioning to persistence mode.")
 
        # In driven mode R18 (persistent field) is unreliable; the true coil
        # field is the output field R7, since the switch is open (normal).
        # We read it now as the field we will persist.
        B_coil = self._get_output_field()
        if B_coil is None:
            self.error("_exit_driven_mode: could not read output field.")
            return False
 
        self.info(f"_exit_driven_mode: coil field is {B_coil:.6f} T.")
 
        # Hold and turn heater off. _set_heater will verify that persistent
        # and output fields agree before sending H0 — since the heater is on,
        # is_persistent_mode() returns False, so the field-match guard inside
        # _set_heater is skipped and it proceeds directly to H0.
        self._set_mode(self.mode.HOLD)
        if not self._set_heater(False):
            self.error("_exit_driven_mode: failed to turn off switch heater.")
            return False
 
        self.info(
            f"_exit_driven_mode: heater off; pausing {self.stabilize_wait_time} s "
            f"for switch to go superconducting."
        )
        time.sleep(self.stabilize_wait_time)
 
        # Verify the persistent field register now reads the expected value.
        Bper = self._get_persistent_field()
        if Bper is None or abs(Bper - B_coil) > self.B_tol_assertive:
            self.warn(
                f"_exit_driven_mode: persistent field {Bper:.6f} T does not match "
                f"expected coil field {B_coil:.6f} T after heater off."
            )
 
        self.info("_exit_driven_mode: now in persistence mode.")
        return True
 
    def set_B_driven(self, B: float) -> bool:
        """
        Set the field in driven mode. The switch heater is left on and the
        leads remain energized at the target field. This is faster than
        persistence mode but puts a heat load on the pulse tube; the caller
        is responsible for enforcing any field limit (e.g. 4 T) appropriate
        to the system.
 
        If the supply is currently in persistence mode (heater off), the leads
        are matched to the coil field and the heater is turned on before
        ramping. This is the safe entry sequence that prevents a quench from
        closing the switch onto a mismatched field.
 
        Parameters
        ----------
        B : float
            Target field in T.
 
        Returns
        -------
        success : bool
        """
 
        if abs(B) > self.max_B:
            B = min(self.max_B, max(-self.max_B, B))
            self.warn(f"Clipped target field to {B} T.")
 
        self.info(f"set_B_driven: targeting {B:.6f} T.")
 
        # _goto_B unconditionally calls _match_currents before enabling the
        # heater, which is exactly the safe persistent→driven entry sequence:
        # ramp leads to coil field, verify match, then close the switch.
        # If the heater is already on (driven→driven), _match_currents detects
        # this and skips the matching ramp, proceeding directly to the new
        # target.
        self._goto_B(B)
 
        Bout = self._get_output_field()
        if Bout is None or abs(Bout - B) > self.B_tol_assertive:
            self.error(
                f"set_B_driven: output field {Bout} T does not match "
                f"target {B:.6f} T after ramp."
            )
            return False
 
        self.info(
            f"set_B_driven: at {B:.6f} T; heater on, leads energized."
        )
        return True
 
    def get_B_driven(self) -> float | None:
        """
        Query the field in driven mode. Reads the output field register R7
        directly, since R18 (persistent field) is not meaningful while the
        switch heater is on.
 
        Returns
        -------
        B : float or None
        """
        return self._get_output_field()

@register_device_class("IPS120Channel")
class IPS120Channel(AsyncChannel):
    """
    AsyncChannel wrapper for the Oxford IPS120 superconducting magnet power supply.

    Parameters
    ----------
    magnet : ips120
        A connected ips120 instance registered in HardwareRegistry.
    name : str, default 'B'
    long_name : str, default R'$B$'
    unit : str, default 'T'
    registry_id : str, optional
    """

    def __init__(self,
        magnet      : ips120,
        name        : str        = 'B',
        long_name   : str        = R'$B$',
        unit        : str        = 'T',
        registry_id : str | None = None,
    ):
        super().__init__(name=name, long_name=long_name, unit=unit,
                         registry_id=registry_id)
        self._magnet = magnet

    def _get(self) -> float:
        return self._magnet.get_B()

    def _set(self, value: float) -> None:
        self._magnet.set_B(value)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize_state(self) -> dict:
        return {
            'magnet'   : format_reference(self._magnet),
            'name'     : self.name,
            'long_name': self.long_name,
            'unit'     : self.unit,
        }

    def _deserialize_state(self, state: dict) -> None:
        if 'magnet' in state:
            self._magnet = DeferredReference(state['magnet']).unwrap()
        if 'name' in state:
            self.name = state['name']
        if 'long_name' in state:
            self.long_name = state['long_name']
        if 'unit' in state:
            self.unit = state['unit']

    @classmethod
    def from_config(cls, config: dict) -> 'IPS120Channel':
        registry_id = config.pop('registry_id', None)
        magnet_ref  = config.pop('magnet', None)

        instance = cls(
            magnet      = None,  # resolved in _deserialize_state (pass 2)
            name        = config.pop('name', 'B'),
            long_name   = config.pop('long_name', R'$B$'),
            unit        = config.pop('unit', 'T'),
            registry_id = registry_id,
        )
        if magnet_ref is not None:
            instance._magnet = DeferredReference(magnet_ref)
        return instance
    
@register_device_class("IPS120DrivenChannel")
class IPS120DrivenChannel(AsyncChannel):
    """
    AsyncChannel wrapper for the Oxford IPS120 in driven mode. The switch
    heater is left on between sweeps and the leads remain energized. Sweeps
    are faster than in persistence mode, but the continuous heat load on the
    pulse tube limits safe operation to fields below max_driven_B.
 
    Calling _set on this channel while the supply is in persistence mode
    (heater off) will safely match the leads to the coil before closing the
    switch, preventing a quench.
 
    Parameters
    ----------
    magnet : ips120
        A connected ips120 instance registered in HardwareRegistry.
    max_driven_B : float, default 4.0
        Maximum field magnitude permitted in driven mode (T). Requests
        exceeding this limit are refused. Set according to the pulse tube
        heat load limit for the specific system.
    name : str, default 'B'
    long_name : str, default R'$B_\mathrm{driven}$'
    unit : str, default 'T'
    registry_id : str, optional
    """
 
    def __init__(self,
        magnet        : ips120,
        max_driven_B  : float        = 4.0,
        name          : str          = 'B',
        long_name     : str          = R'$B_\mathrm{driven}$',
        unit          : str          = 'T',
        registry_id   : str | None   = None,
    ):
        super().__init__(name=name, long_name=long_name, unit=unit,
                         registry_id=registry_id)
        self._magnet       = magnet
        self.max_driven_B  = max_driven_B
 
    def _get(self) -> float:
        return self._magnet.get_B_driven()
 
    def _set(self, value: float) -> None:
        if abs(value) > self.max_driven_B:
            self._magnet.warn(
                f"IPS120DrivenChannel: requested field {value:.4f} T exceeds "
                f"driven-mode limit of {self.max_driven_B} T; refusing."
            )
            return
        self._magnet.set_B_driven(value)
 
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
 
    def _serialize_state(self) -> dict:
        return {
            'magnet'      : format_reference(self._magnet),
            'max_driven_B': self.max_driven_B,
            'name'        : self.name,
            'long_name'   : self.long_name,
            'unit'        : self.unit,
        }
 
    def _deserialize_state(self, state: dict) -> None:
        if 'magnet' in state:
            self._magnet = DeferredReference(state['magnet']).unwrap()
        if 'max_driven_B' in state:
            self.max_driven_B = state['max_driven_B']
        if 'name' in state:
            self.name = state['name']
        if 'long_name' in state:
            self.long_name = state['long_name']
        if 'unit' in state:
            self.unit = state['unit']
 
    @classmethod
    def from_config(cls, config: dict) -> 'IPS120DrivenChannel':
        registry_id = config.pop('registry_id', None)
        magnet_ref  = config.pop('magnet', None)
 
        instance = cls(
            magnet       = None,  # resolved in _deserialize_state (pass 2)
            max_driven_B = config.pop('max_driven_B', 4.0),
            name         = config.pop('name',      'B'),
            long_name    = config.pop('long_name', R'$B_\mathrm{driven}$'),
            unit         = config.pop('unit',      'T'),
            registry_id  = registry_id,
        )
        if magnet_ref is not None:
            instance._magnet = DeferredReference(magnet_ref)
        return instance