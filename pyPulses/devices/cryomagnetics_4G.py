"""
This class is an interface to the Cryomagnetics 4G Superconducting Magnet Power
Supply.
"""

from ._registry import DeviceRegistry
from .pyvisa_device import pyvisaDevice
from typing import Optional
import time

class cryomagnetics4G(pyvisaDevice):
    def __init__(self, logger: Optional[str] = None, 
                 max_step: Optional[float] = 0.05, 
                 wait: Optional[float] = 0.1, 
                 instrument_id: Optional[str] = None):
        
        self.config = {
            "resource_name" : "GPIB0::25::INSTR",

            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }
        if instrument_id: 
            self.config["resource_name"] = instrument_id

        super().__init__(self.config, logger)
        DeviceRegistry.register_device(self.config["resource_name"], self)

        self.H_tol_kG    = 1e-3 # finite-precision field strength tolerance
        self.H_sweep_tol = 0.01

    def get_H(self) -> float:
        """Get the current field strength setting in T."""
        imag_ = self.device.query("IMAG?").strip()
        units = imag_[-2:]
        if not units == 'kG':
            self.error("IMAG? returned incorrect units; expected kilogauss.")
            return
        
        return 0.1 * float(imag_[:-2])


    def sweep_H(self, H_T: float) -> bool:
        """Sweeps to a field H_T in Tesla."""
        H_kG        = 10*H_T    # field strength in kilogauss

        pshtr = self.device.query("PSHTR?").strip() == '1'
        if pshtr:
            self.error("Switch heater is on; expected switch heater to be off.")
            return
        
        imag_ = self.device.query("IMAG?").strip()
        units = imag_[-2:]
        if not units == 'kG':
            self.error("IMAG? returned incorrect units; expected kilogauss.")
            return
        
        H_strength_kG = float(imag_[:-2])
        if abs(H_strength_kG - H_kG) < self.H_tol_kG:
            self.info("sweep_H was issued while already at desired field.")
            return
        
        if H_strength_kG > 0:   # UP
            if not self.set_sweep_lim(0.0, H_strength_kG):
                self.error("sweep_H failed.")
                return False
            self.device.write("SWEEP UP FAST")
        
        elif H_strength_kG < 0: # DOWN
            if not self.set_sweep_lim(H_strength_kG, 0.0):
                self.error("sweep_H failed.")
                return False
            self.device.write("SWEEP DOWN FAST")

        self.wait_for_field(H_strength_kG, self.H_sweep_tol)

        # pause to stabilize
        self.pause_msg("Pausing to stabilize", 10)
        self.device.write("PSHTR ON")

        self.pause_msg("Waiting for switch to go normal", 15)
        
        pshtr = self.device.query("PSHTR?").strip() == '1'
        if not pshtr:
            self.error("Failed to enable switch heater.")
            return False
        
        if H_kG < H_strength_kG:
            if not self.set_verify("LLIM", H_kG, self.H_tol_kG):
                self.error("Error when setting and verifying.")
                return False
            self.device.write("SWEEP DOWN")
        
        elif H_kG > H_strength_kG:
            if not self.set_verify("ULIM", H_kG, self.H_tol_kG):
                self.error("Error when setting and verifying.")
                return False
            self.device.write("SWEEP UP")
        
        else:
            self.error("Target and field setpoints identical.")
            return False
        
        self.wait_for_field(H_kG, self.H_tol_kG)
        self.device.write("PSHTR OFF")
        pshtr = self.device.query("PSHTR?").strip() == '1'
        if pshtr:
            self.error("Failed to disable switch heater")
            return False
        
        self.pause_msg("Turning heater switch off", 15)
        self.device.write("SWEEP ZERO FAST")

        self.wait_for_field(0.0, self.H_sweep_tol)

        return True

    def set_verify(self, setting: str, H_kG: float, H_tol_kG: float) -> bool:
        self.device.write(f"{setting} {H_kG}")
        set_H = float(self.device.query(f"{setting}?").strip()[:-2])

        if abs(set_H - H_kG) > H_tol_kG:
            self.error(f"Error setting {setting}.")
            return False

        return True

    def set_sweep_lim(self, H_lo: float, H_hi: float) -> bool:
        for _ in range(3):
            if self._set_sweep_lim(H_lo, H_hi):
                return True
        else:
            self.error("Failed to set sweep limits three times.")
            return False
        
    def _set_sweep_lim(self, H_lo: float, H_hi: float) -> bool:
        self.device.write(f"ULIM {H_hi}")
        time.sleep(0.5)
        set_H_hi = float(self.device.query("ULIM?").strip()[:-2])
        time.sleep(0.5)

        self.device.write(f"LLIM {H_lo}")
        time.sleep(0.5)
        set_H_lo = float(self.device.query("LLIM?").strip()[:-2])
        time.sleep(0.5)

        if abs(H_hi - set_H_hi) > self.H_sweep_tol:
            self.warn("Failed to set upper limit.")
            return False
        
        if abs(H_lo - set_H_lo) > self.H_sweep_tol:
            self.warn("Failed to set lower limit.")
            return False
        
        return True

    def wait_for_field(self, H_target_kG: float, H_tol_kG: float):
        while True:
            iout_kG = float(self.device.query("IOUT?").strip()[:-2])
            self.info(
                f"H_supply = {iout_kG:.5f} kG, H_target = {H_target_kG:.5f} kG"
            )

            if abs(iout_kG - H_target_kG) < H_tol_kG:
                break

    def pause_msg(self, msg: str, t_s: int):
        """Print a waiting message with countdown"""
        self.info(msg)
        for i in range(t_s):
            self.info(f"{t_s - i} s remaining...")
            time.sleep(1.0)

        self.info("Finished pause.")
