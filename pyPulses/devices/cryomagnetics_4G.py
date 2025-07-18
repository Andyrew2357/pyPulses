"""
This class is an interface to the Cryomagnetics 4G Superconducting Magnet Power
Supply.
"""

from .pyvisa_device import pyvisaDevice
import time

class cryomagnetics4G(pyvisaDevice):
    """
    Class interface for communicating with the Cryomagnetics 4G Superconducting 
    Magnet Power Supply.
    """
    def __init__(self, logger = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice
        instrument_id : str, optional
            VISA resource name
        """

        self.pyvisa_config = {
            "resource_name" : "GPIB0::25::INSTR",

            "output_buffer_size" : 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }

        super().__init__(self.pyvisa_config, logger, instrument_id)

        self.H_tol_kG    = 1e-3 # finite-precision field strength tolerance
        self.H_sweep_tol = 0.01

    def get_H(self) -> float:
        """
        Get the current field strength setting.
        
        Returns
        -------
        H : float
            field strength in T.
        """
        imag_ = self.device.query("IMAG?").strip()
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
            if not self._set_sweep_lim(0.0, H_strength_kG):
                self.error("sweep_H failed.")
                return False
            self.device.write("SWEEP UP FAST")
        
        elif H_strength_kG < 0: # DOWN
            if not self._set_sweep_lim(H_strength_kG, 0.0):
                self.error("sweep_H failed.")
                return False
            self.device.write("SWEEP DOWN FAST")

        self._wait_for_field(H_strength_kG, self.H_sweep_tol)

        # pause to stabilize
        self._pause_msg("Pausing to stabilize", 10)
        self.device.write("PSHTR ON")

        self._pause_msg("Waiting for switch to go normal", 15)
        
        pshtr = self.device.query("PSHTR?").strip() == '1'
        if not pshtr:
            self.error("Failed to enable switch heater.")
            return False
        
        if H_kG < H_strength_kG:
            if not self._set_verify("LLIM", H_kG, self.H_tol_kG):
                self.error("Error when setting and verifying.")
                return False
            self.device.write("SWEEP DOWN")
        
        elif H_kG > H_strength_kG:
            if not self._set_verify("ULIM", H_kG, self.H_tol_kG):
                self.error("Error when setting and verifying.")
                return False
            self.device.write("SWEEP UP")
        
        else:
            self.error("Target and field setpoints identical.")
            return False
        
        self._wait_for_field(H_kG, self.H_tol_kG)
        self.device.write("PSHTR OFF")
        pshtr = self.device.query("PSHTR?").strip() == '1'
        if pshtr:
            self.error("Failed to disable switch heater")
            return False
        
        self._pause_msg("Turning heater switch off", 15)
        self.device.write("SWEEP ZERO FAST")

        self._wait_for_field(0.0, self.H_sweep_tol)

        self.info(f"CM4G: Successfully swept to {H_T} T.")
        return True

    def _set_verify(self, setting: str, H_kG: float, H_tol_kG: float) -> bool:
        self.device.write(f"{setting} {H_kG}")
        set_H = float(self.device.query(f"{setting}?").strip()[:-2])

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

    def _wait_for_field(self, H_target_kG: float, H_tol_kG: float):
        while True:
            iout_kG = float(self.device.query("IOUT?").strip()[:-2])
            self.info(
                f"H_supply = {iout_kG:.5f} kG, H_target = {H_target_kG:.5f} kG"
            )

            if abs(iout_kG - H_target_kG) < H_tol_kG:
                break

    def _pause_msg(self, msg: str, t_s: int):
        """Print a waiting message with countdown"""
        self.info(msg)
        for i in range(t_s):
            self.info(f"{t_s - i} s remaining...")
            time.sleep(1.0)

        self.info("Finished pause.")

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

    magnet = cryomagnetics4G(logger, instrument_id = 'DEBUG')

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
