from .pyvisa_device import pyvisaDevice
from .registry import register_hardware_class, register_device_class, format_reference, DeferredReference, HardwareRegistry
from .sweepable_channel import AsyncChannel

import time
from logging import Logger

@register_hardware_class("cryomagnetics4G")
class cryomagnetics4G(pyvisaDevice):
    """
    Class representation of the Cryomagnetics 4G Superconducting Magnet Power Supply.
    """

    DEFAULT_PYVISA_CONFIG = {
        'output_buffer_size' : 512,
        'gpib_eos_mode'     : False,
        'gpib_eos_char'     : ord('\n'),
        'gpib_eoi_mode'     : True,
        'write_termination' : '\n',
        'max_retries': 3,
        'retry_delay': 0.1,
        'min_interval': 0.05
    }

    H_tol_kG = 1e-3 # finite-precision field strength tolerance
    H_sweep_tol = 0.01

    rate_ranges = range(5)  # RATE ranges used by SWEEP UP / DOWN

    # Ramp state; see start_ramp
    _ramp_target_kG = None  # None when no ramp is active
    _ramp_restore_rates = None  # {range: A / s}, rates before the ramp

    def __init__(self,
        resource_name: str, 
        registry_id: str | None = None,
        logger: Logger | None = None,
        skip_connect: bool = False,
        coil_constant: float | None = None,
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
        coil_constant : float, optional
            Magnet coil constant in T / A. Required by start_ramp, since the
            supply takes rates in A / s.
        **kwargs
        """

        self.coil_constant = coil_constant
        super().__init__(resource_name, registry_id, logger, skip_connect, **kwargs)

    def _serialize_state(self) -> dict:
        state = super()._serialize_state()
        state['coil_constant'] = self.coil_constant
        return state

    def get_H(self) -> float:
        """
        Get the current field strength setting.
        
        Returns
        -------
        H : float
            field strength in T.
        """
        imag_ = self.query("IMAG?").strip()
        units = imag_[-2:]
        if not units == 'kG':
            self.error("IMAG? returned incorrect units; expected kilogauss.")
            return
        
        return 0.1 * float(imag_[:-2])

    def sweep_H(self, H_T: float) -> bool:
        """
        Sweeps to a field.
        
        Parameters
        ----------
        H_T : float
            target field in T.

        Returns
        -------
        success : bool
        """
        H_kG        = 10*H_T    # field strength in kilogauss

        self.info(f"CM4G: Requested sweep to {H_T} T.")

        # If the switch heater is on we are in driven mode. Transition safely
        # to persistence mode before proceeding. Attempting to run the
        # persistence sweep with the heater already on would leave the coil
        # field undefined at the point where we close the switch, risking a
        # quench.
        pshtr = self.query("PSHTR?").strip() == '1'
        if pshtr:
            self.info(
                "sweep_H: switch heater is on (driven mode); transitioning to "
                "persistence mode before sweeping."
            )
            if not self._exit_driven_mode():
                self.error("sweep_H: failed to exit driven mode; aborting.")
                return False
                    
        imag_ = self.query("IMAG?").strip()
        units = imag_[-2:]
        if not units == 'kG':
            self.error("IMAG? returned incorrect units; expected kilogauss.")
            return
        
        H_strength_kG = float(imag_[:-2])
        if abs(H_strength_kG - H_kG) < self.H_tol_kG:
            self.info("sweep_H was issued while already at desired field.")
            return
        
        if H_strength_kG > 0:   # UP
            if not self._set_sweep_lim(0.0, H_strength_kG):
                self.error("sweep_H failed.")
                return False
            self.write("SWEEP UP FAST")
        
        elif H_strength_kG < 0: # DOWN
            if not self._set_sweep_lim(H_strength_kG, 0.0):
                self.error("sweep_H failed.")
                return False
            self.write("SWEEP DOWN FAST")

        self._wait_for_field(H_strength_kG, self.H_sweep_tol)

        # pause to stabilize
        self._pause_msg("Pausing to stabilize", 10)
        self.write("PSHTR ON")

        self._pause_msg("Waiting for switch to go normal", 15)
        
        pshtr = self.query("PSHTR?").strip() == '1'
        if not pshtr:
            self.error("Failed to enable switch heater.")
            return False
        
        if H_kG < H_strength_kG:
            if not self._set_verify("LLIM", H_kG, self.H_tol_kG):
                self.error("Error when setting and verifying.")
                return False
            self.write("SWEEP DOWN")
        
        elif H_kG > H_strength_kG:
            if not self._set_verify("ULIM", H_kG, self.H_tol_kG):
                self.error("Error when setting and verifying.")
                return False
            self.write("SWEEP UP")
        
        else:
            self.error("Target and field setpoints identical.")
            return False
        
        self._wait_for_field(H_kG, self.H_tol_kG)
        self.write("PSHTR OFF")
        pshtr = self.query("PSHTR?").strip() == '1'
        if pshtr:
            self.error("Failed to disable switch heater")
            return False
        
        self._pause_msg("Turning heater switch off", 15)
        self.write("SWEEP ZERO FAST")

        self._wait_for_field(0.0, self.H_sweep_tol)

        self.info(f"CM4G: Successfully swept to {H_T} T.")
        return True

    def _set_verify(self, setting: str, H_kG: float, H_tol_kG: float) -> bool:
        self.write(f"{setting} {H_kG}")
        set_H = float(self.query(f"{setting}?").strip()[:-2])

        if abs(set_H - H_kG) > H_tol_kG:
            self.error(f"Error setting {setting}.")
            return False

        self.info(
            f"CM4G: Successfully set {setting} to {H_kG} kG within {H_tol_kG} kG."
        )
        return True

    def _set_sweep_lim(self, H_lo: float, H_hi: float) -> bool:
        for _ in range(3):
            if self._set_sweep_lim_inner(H_lo, H_hi):
                self.info(
                    f"CM4G: Successfully set sweep limits [{H_lo}, {H_hi}]."
                )
                return True
        else:
            self.error("Failed to set sweep limits three times.")
            return False
        
    def _set_sweep_lim_inner(self, H_lo: float, H_hi: float) -> bool:
        self.write(f"ULIM {H_hi}")
        time.sleep(0.5)
        set_H_hi = float(self.query("ULIM?").strip()[:-2])
        time.sleep(0.5)

        self.write(f"LLIM {H_lo}")
        time.sleep(0.5)
        set_H_lo = float(self.query("LLIM?").strip()[:-2])
        time.sleep(0.5)

        if abs(H_hi - set_H_hi) > self.H_sweep_tol:
            self.warn("Failed to set upper limit.")
            return False
        
        if abs(H_lo - set_H_lo) > self.H_sweep_tol:
            self.warn("Failed to set lower limit.")
            return False
        
        return True

    def _wait_for_field(self, H_target_kG: float, H_tol_kG: float):
        while True:
            iout_kG = float(self.query("IOUT?").strip()[:-2])
            self.info(
                f"H_supply = {iout_kG:.5f} kG, H_target = {H_target_kG:.5f} kG"
            )

            if abs(iout_kG - H_target_kG) < H_tol_kG:
                break
            time.sleep(0.5)

    def _pause_msg(self, msg: str, t_s: int):
        """Print a waiting message with countdown"""
        self.info(msg)
        for i in range(t_s):
            self.info(f"{t_s - i} s remaining...")
            time.sleep(1.0)

        self.info("Finished pause.")

    """Driven Mode"""

    def _enter_driven_mode(self) -> bool:
        """
        Safely transition into driven mode. If the heater is already on, this
        is a no-op. If the heater is off (persistence mode), the leads are
        ramped to match the coil field before the heater is enabled.
 
        Turning on the switch heater with a mismatch between lead current and
        coil current will cause the coil field to jump as the switch goes
        normal, which risks a quench. This method enforces the match before
        touching the heater.
 
        Returns
        -------
        success : bool
        """
        pshtr = self.query("PSHTR?").strip() == '1'
        if pshtr:
            self.info("_enter_driven_mode: switch heater already on; already in driven mode.")
            return True
 
        # Heater is off. Read both the coil field (IMAG?) and the lead field
        # (IOUT?) and ramp leads to match coil if necessary before enabling
        # the heater.
        imag_ = self.query("IMAG?").strip()
        if imag_[-2:] != 'kG':
            self.error("_enter_driven_mode: IMAG? returned unexpected units.")
            return False
        H_coil_kG = float(imag_[:-2])
 
        iout_ = self.query("IOUT?").strip()
        if iout_[-2:] != 'kG':
            self.error("_enter_driven_mode: IOUT? returned unexpected units.")
            return False
        H_leads_kG = float(iout_[:-2])
 
        self.info(
            f"_enter_driven_mode: coil = {H_coil_kG:.4f} kG, "
            f"leads = {H_leads_kG:.4f} kG."
        )
 
        if abs(H_coil_kG - H_leads_kG) > self.H_tol_kG:
            self.info(
                "_enter_driven_mode: lead/coil mismatch; ramping leads to "
                "match coil before enabling heater."
            )
            # Ramp leads toward coil field with switch still cold (FAST).
            if H_coil_kG > H_leads_kG:
                if not self._set_sweep_lim(H_leads_kG, H_coil_kG):
                    self.error("_enter_driven_mode: failed to set sweep limits.")
                    return False
                self.write("SWEEP UP FAST")
            else:
                if not self._set_sweep_lim(H_coil_kG, H_leads_kG):
                    self.error("_enter_driven_mode: failed to set sweep limits.")
                    return False
                self.write("SWEEP DOWN FAST")
 
            self._wait_for_field(H_coil_kG, self.H_tol_kG)
 
        self._pause_msg("_enter_driven_mode: pausing to stabilize before heater on", 10)
        self.write("PSHTR ON")
        self._pause_msg("_enter_driven_mode: waiting for switch to go normal", 15)
 
        pshtr = self.query("PSHTR?").strip() == '1'
        if not pshtr:
            self.error("_enter_driven_mode: failed to enable switch heater.")
            return False
 
        self.info("_enter_driven_mode: switch heater on; now in driven mode.")
        return True
 
    def _exit_driven_mode(self) -> bool:
        """
        Safely transition out of driven mode into persistence mode. If the
        heater is already off, this is a no-op. If the heater is on, the leads
        are left at the current field, the heater is turned off, and we wait
        for the switch to go superconducting.
 
        In driven mode IMAG? reflects the actual coil field (the switch is
        open/normal so the coil and leads are the same circuit). IOUT? and
        IMAG? should agree. We turn the heater off directly — there is no
        mismatch to correct since both registers track the same current while
        the switch is normal.
 
        Returns
        -------
        success : bool
        """
        pshtr = self.query("PSHTR?").strip() == '1'
        if not pshtr:
            self.info("_exit_driven_mode: switch heater already off; already in persistence mode.")
            return True

        # The leads must not keep ramping while the switch closes.
        self.stop_ramp()

        # Read the current field for logging. In driven mode IMAG? and IOUT?
        # should agree; use IOUT? as the authoritative lead/coil field.
        iout_ = self.query("IOUT?").strip()
        if iout_[-2:] != 'kG':
            self.error("_exit_driven_mode: IOUT? returned unexpected units.")
            return False
        H_kG = float(iout_[:-2])
 
        self.info(
            f"_exit_driven_mode: turning heater off at {H_kG:.4f} kG; "
            f"field will persist at this value."
        )
 
        self.write("PSHTR OFF")
        self._pause_msg("_exit_driven_mode: waiting for switch to go superconducting", 15)
 
        pshtr = self.query("PSHTR?").strip() == '1'
        if pshtr:
            self.error("_exit_driven_mode: failed to disable switch heater.")
            return False
 
        # Verify IMAG? now reports the correct persistent field.
        imag_ = self.query("IMAG?").strip()
        if imag_[-2:] != 'kG':
            self.error("_exit_driven_mode: IMAG? returned unexpected units after heater off.")
            return False
        H_persist_kG = float(imag_[:-2])
 
        if abs(H_persist_kG - H_kG) > self.H_tol_kG:
            self.warn(
                f"_exit_driven_mode: IMAG? reads {H_persist_kG:.4f} kG after heater off; "
                f"expected {H_kG:.4f} kG."
            )
 
        self.info("_exit_driven_mode: switch heater off; now in persistence mode.")
        return True
 
    def sweep_H_driven(self, H_T: float) -> bool:
        """
        Sweep to a field in driven mode. The switch heater is left on and the
        leads remain energized at the target field. This is faster than
        persistence mode but puts a heat load on the pulse tube; the caller
        is responsible for enforcing any field limit appropriate to the system.
 
        If the supply is currently in persistence mode (heater off), the leads
        are matched to the coil field and the heater is turned on before
        ramping, preventing a quench from closing the switch onto a mismatched
        field.
 
        Parameters
        ----------
        H_T : float
            Target field in T.
 
        Returns
        -------
        success : bool
        """
        H_kG = 10 * H_T
        self.info(f"CM4G driven: requested sweep to {H_T} T ({H_kG:.4f} kG).")

        # Don't take the move at the slow rate of a ramp left in progress.
        if self._ramp_target_kG is not None:
            self.stop_ramp()

        if not self._enter_driven_mode():
            self.error("sweep_H_driven: could not enter driven mode safely; aborting.")
            return False
 
        # At this point the heater is on and leads match the coil. Ramp to
        # the new target field.
        iout_ = self.query("IOUT?").strip()
        H_current_kG = float(iout_[:-2])
 
        if abs(H_current_kG - H_kG) < self.H_tol_kG:
            self.info("sweep_H_driven: already at requested field.")
            return True
 
        if H_kG > H_current_kG:
            if not self._set_verify("ULIM", H_kG, self.H_tol_kG):
                self.error("sweep_H_driven: failed to set upper limit.")
                return False
            self.write("SWEEP UP")
        else:
            if not self._set_verify("LLIM", H_kG, self.H_tol_kG):
                self.error("sweep_H_driven: failed to set lower limit.")
                return False
            self.write("SWEEP DOWN")
 
        self._wait_for_field(H_kG, self.H_tol_kG)
 
        self.info(f"CM4G driven: at {H_T} T; heater on, leads energized.")
        return True
 
    def get_H_driven(self) -> float | None:
        """
        Query the field in driven mode. Reads IOUT? (output/lead current)
        directly, since IMAG? (coil current) is only meaningful in persistence
        mode (heater off).
 
        Returns
        -------
        H : float or None
            Field in T, or None on communication error.
        """
        iout_ = self.query("IOUT?").strip()
        if iout_[-2:] != 'kG':
            self.error("get_H_driven: IOUT? returned unexpected units.")
            return None
        return 0.1 * float(iout_[:-2])

    """Ramping"""

    def start_ramp(self,
        H_T: float,
        rate: float,
        max_driven_H: float = 4.0,
    ) -> bool:
        """
        Start ramping the field toward H_T and return without waiting for it.
        Read the instantaneous field with get_ramp_field, check progress with
        is_ramping, and always finish with stop_ramp, which also restores the
        previous sweep rates.

        If the supply is in persistence mode, this first blocks while the
        leads are matched and the switch heater turns on. The heater stays on
        afterward, so the heat load is the same as driven mode.

        The rate is written to every range, but a range is never raised above
        the rate it is already programmed to.

        Parameters
        ----------
        H_T : float
            Field to ramp to in T.
        rate : float
            Ramp rate in T / min.
        max_driven_H : float, default=4.0
            Largest field magnitude allowed (T); exceeding it is refused.

        Returns
        -------
        success : bool
            True if the sweep was issued.
        """
        if self.coil_constant is None:
            self.error("start_ramp: coil_constant is not set.")
            return False

        if rate <= 0:
            self.error(f"start_ramp: rate must be positive, got {rate}.")
            return False

        if abs(H_T) > max_driven_H:
            self.error(f"start_ramp: {H_T} T exceeds the {max_driven_H} T limit.")
            return False

        if not self._enter_driven_mode():
            self.error("start_ramp: could not enter driven mode; aborting.")
            return False

        self.write("SWEEP PAUSE")

        # A ramp may be replacing another, whose rates are not the ones to keep.
        if self._ramp_restore_rates is None:
            self._ramp_restore_rates = {
                r: float(self.query(f"RATE? {r}").strip())
                for r in self.rate_ranges
            }

        rate_A_s = rate / self.coil_constant / 60
        for r, previous in self._ramp_restore_rates.items():
            self.write(f"RATE {r} {min(rate_A_s, previous):.6f}")

        actual = [float(self.query(f"RATE? {r}").strip()) for r in self.rate_ranges]
        if min(actual) <= 0:
            self.error(f"start_ramp: supply reports sweep rates of {actual}.")
            self.stop_ramp()
            return False
        if any(abs(a - rate_A_s) > 0.01 * rate_A_s for a in actual):
            self.warn(
                f"start_ramp: requested {rate_A_s:.6f} A/s, supply reports "
                f"{actual} A/s."
            )

        H_kG = 10 * H_T
        H_current_kG = 10 * self.get_H_driven()
        self._ramp_target_kG = H_kG

        if abs(H_current_kG - H_kG) < self.H_tol_kG:
            self.info("start_ramp: already at requested field.")
            return True

        setting, direction = ("ULIM", "UP") if H_kG > H_current_kG \
                             else ("LLIM", "DOWN")
        if not self._set_verify(setting, H_kG, self.H_tol_kG):
            self.error(f"start_ramp: failed to set {setting}.")
            self.stop_ramp()
            return False

        self.write(f"SWEEP {direction}")
        self.info(f"start_ramp: ramping to {H_T} T at {rate} T/min.")
        return True

    def get_ramp_field(self) -> float | None:
        """
        Query the instantaneous field during a ramp.

        Returns
        -------
        H : float or None
            Field in T, or None on communication error.
        """
        return self.get_H_driven()

    def is_ramping(self) -> bool:
        """
        Whether a ramp is still on its way to its target. False once it has
        arrived, but stop_ramp is still needed to restore the sweep rates.

        Returns
        -------
        bool
        """
        if self._ramp_target_kG is None:
            return False

        H_T = self.get_H_driven()
        return H_T is None or \
            abs(10 * H_T - self._ramp_target_kG) >= self.H_tol_kG

    def stop_ramp(self) -> bool:
        """
        Hold the field where it is and restore the sweep rates in force before
        the ramp. The heater stays on. Safe to call if no ramp is active.

        Returns
        -------
        success : bool
        """
        self.write("SWEEP PAUSE")
        self._ramp_target_kG = None

        if self._ramp_restore_rates is not None:
            for r, previous in self._ramp_restore_rates.items():
                self.write(f"RATE {r} {previous:.6f}")
            self._ramp_restore_rates = None

        return True


@register_device_class("CM4GChannel")
class CM4GChannel(AsyncChannel):
    """
    AsyncChannel wrapper for the Cryomagnetics 4G superconducting magnet power supply.

    Parameters
    ----------
    magnet : cryomagnetics4G
        A connected cryomagnetics4G instance registered in HardwareRegistry.
    name : str, default 'B'
    long_name : str, default 'Field'
    unit : str, default 'T'
    registry_id : str, optional
    """

    def __init__(self,
        magnet      : cryomagnetics4G,
        name        : str        = 'B',
        long_name   : str        = 'Field',
        unit        : str        = 'T',
        registry_id : str | None = None,
    ):
        super().__init__(name=name, long_name=long_name, unit=unit,
                         registry_id=registry_id)
        self._magnet = magnet

    def _get(self) -> float:
        return self._magnet.get_H()

    def _set(self, value: float) -> None:
        self._magnet.sweep_H(value)

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
    def from_config(cls, config: dict) -> 'CM4GChannel':
        registry_id = config.pop('registry_id', None)
        magnet_ref  = config.pop('magnet', None)

        instance = cls(
            magnet      = None,  # resolved in _deserialize_state (pass 2)
            name        = config.pop('name',      'B'),
            long_name   = config.pop('long_name', 'Field'),
            unit        = config.pop('unit',      'T'),
            registry_id = registry_id,
        )
        if magnet_ref is not None:
            instance._magnet = DeferredReference(magnet_ref)
        return instance
    

@register_device_class("CM4GDrivenChannel")
class CM4GDrivenChannel(AsyncChannel):
    """
    AsyncChannel wrapper for the Cryomagnetics 4G in driven mode. The switch
    heater is left on between sweeps and the leads remain energized. Sweeps
    are faster than in persistence mode, but the continuous heat load on the
    pulse tube limits safe operation to fields below max_driven_B.
 
    Calling _set on this channel while the supply is in persistence mode
    (heater off) will safely match the leads to the coil before closing the
    switch, preventing a quench.
 
    Parameters
    ----------
    magnet : cryomagnetics4G
        A connected cryomagnetics4G instance registered in HardwareRegistry.
    max_driven_H : float, default 4.0
        Maximum field magnitude permitted in driven mode (T). Requests
        exceeding this limit are refused. Set according to the pulse tube
        heat load limit for the specific system.
    name : str, default 'B'
    long_name : str, default 'Field (driven)'
    unit : str, default 'T'
    registry_id : str, optional
    """
 
    def __init__(self,
        magnet        : cryomagnetics4G,
        max_driven_H  : float        = 4.0,
        name          : str          = 'B',
        long_name     : str          = 'Field (driven)',
        unit          : str          = 'T',
        registry_id   : str | None   = None,
    ):
        super().__init__(name=name, long_name=long_name, unit=unit,
                         registry_id=registry_id)
        self._magnet      = magnet
        self.max_driven_H = max_driven_H
 
    def _get(self) -> float:
        return self._magnet.get_H_driven()
 
    def _set(self, value: float) -> None:
        if abs(value) > self.max_driven_H:
            self._magnet.warn(
                f"CM4GDrivenChannel: requested field {value:.4f} T exceeds "
                f"driven-mode limit of {self.max_driven_H} T; refusing."
            )
            return
        self._magnet.sweep_H_driven(value)
 
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
 
    def _serialize_state(self) -> dict:
        return {
            'magnet'      : format_reference(self._magnet),
            'max_driven_H': self.max_driven_H,
            'name'        : self.name,
            'long_name'   : self.long_name,
            'unit'        : self.unit,
        }
 
    def _deserialize_state(self, state: dict) -> None:
        if 'magnet' in state:
            self._magnet = DeferredReference(state['magnet']).unwrap()
        if 'max_driven_H' in state:
            self.max_driven_H = state['max_driven_H']
        if 'name' in state:
            self.name = state['name']
        if 'long_name' in state:
            self.long_name = state['long_name']
        if 'unit' in state:
            self.unit = state['unit']
 
    @classmethod
    def from_config(cls, config: dict) -> 'CM4GDrivenChannel':
        registry_id = config.pop('registry_id', None)
        magnet_ref  = config.pop('magnet', None)
 
        instance = cls(
            magnet       = None,  # resolved in _deserialize_state (pass 2)
            max_driven_H = config.pop('max_driven_H', 4.0),
            name         = config.pop('name',      'B'),
            long_name    = config.pop('long_name', 'Field (driven)'),
            unit         = config.pop('unit',      'T'),
            registry_id  = registry_id,
        )
        if magnet_ref is not None:
            instance._magnet = DeferredReference(magnet_ref)
        return instance


if __name__ == '__main__':
    """Example test of the magnet power supply using dummyResource"""

    import logging
    import sys

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    magnet = cryomagnetics4G(resource_name = 'DEBUG', logger = logger)

    magnet.device.attr['IMAG'] = '0.0kG\n'
    def get_IMAG(obj):
        return obj.attr['IMAG']
    magnet.device.add_command(r'IMAG\?', get_IMAG)

    magnet.device.attr['PSHTR'] = '0\n'
    def get_PSHTR(obj):
        return obj.attr['PSHTR']
    magnet.device.add_command(r'PSHTR\?', get_PSHTR)
    def set_PSHTR(obj, arg):
        if arg in ['1', 'ON']:
            obj.attr['PSHTR'] = '1\n'
        elif arg in ['0', 'OFF']:
            obj.attr['PSHTR'] = '0\n'
    magnet.device.add_command(r'PSHTR (.+)$', set_PSHTR)

    magnet.device.attr['ULIM'] = '0.0kG\n'
    def get_ULIM(obj):
        return obj.attr['ULIM']
    magnet.device.add_command(r'ULIM\?', get_ULIM)
    def set_ULIM(obj, H):
        obj.attr['ULIM'] = f'{H}kG\n'
    magnet.device.add_command(r'ULIM (\d+\.?\d*)', set_ULIM)

    magnet.device.attr['LLIM'] = '0.0kG\n'
    def get_LLIM(obj):
        return obj.attr['LLIM']
    magnet.device.add_command(r'LLIM\?', get_LLIM)
    def set_LLIM(obj, H):
        obj.attr['LLIM'] = f'{H}kG\n'
    magnet.device.add_command(r'LLIM (\d+\.?\d*)', set_LLIM)

    magnet.device.attr['IOUT'] = '0.0kG\n'
    def get_IOUT(obj):
        return obj.attr['IOUT']
    magnet.device.add_command(r'IOUT\?', get_IOUT)

    def sweep(obj, direction, speed):
        match direction:
            case 'ZERO':
                iout = '0.0kG\n'
            case 'UP':
                iout = get_ULIM(obj)
            case 'DOWN':
                iout = get_LLIM(obj)
        
        obj.attr['IOUT'] = iout
        if obj.attr['PSHTR'] == '1\n':
            obj.attr['IMAG'] = iout

    magnet.device.add_command(r"SWEEP (\S+) ?(\S*)", sweep)

    print("MAGNET POWER SUPPLY STATE:")
    for a in magnet.device.attr:
        print(f"{a}: {magnet.device.attr[a]}")

    print("HISTORY:")
    for h in magnet.device.history:
        print(h)

    print("CALLING get_H")
    print(f"result = {magnet.get_H()}")

    print("TESTING SWEEP TO 500 mT")
    magnet.sweep_H(500e-3)

    print("MAGNET POWER SUPPLY STATE:")
    for a in magnet.device.attr:
        print(f"{a}: {magnet.device.attr[a]}")

    print("HISTORY:")
    for h in magnet.device.history:
        print(h)

    print("CALLING get_H")
    print(f"result = {magnet.get_H()}")