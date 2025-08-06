
from .abstract_device import abstractDevice
from ..utils import curves
from ..utils.tandem_sweep import ezTandemSweep, SweepResult
from ..utils.param_sweep_measure import SweepMeasure
from ..utils.plot_sweep import plotSweep

import numpy as np
from typing import Callable, Tuple, List

class HEMTCommonSource(abstractDevice):
    def __init__(self, 
                 VG: Callable, 
                 VDD: Callable,
                 IDS: Callable[[], float],
                 RD: float, RS: float,
                 VSS: Callable | str = 'GND', 
                 VM : Callable | None = None,
                 VG_range   : Tuple[float, float] = (-1.2, 0.0),
                 VDD_range  : Tuple[float, float] = (0.0, 4.0),
                 VSS_range  : Tuple[float, float] = (0.0, 0.0),
                 VM_range   : Tuple[float, float] = (-1.2, 0.0),
                 VM_VG_pinch: float = -0.7,
                 RDcold: float = 0.0,
                 RScold: float = 0.0,
                 max_step   : float = 0.005,
                 tolerance  : float = 500e-6,
                 wait       : float = 0.05,
                 logger = None):        
        super().__init__(logger)

        self.VG  = VG
        self.VDD = VDD
        self.VM  = VM

        self.VSS_grounded = VSS == 'GND'
        self.VSS = lambda V: 0.0 if self.VSS_grounded else VSS

        self.IDS = IDS

        self.RD = RD
        self.RS = RS
        self.RDcold = RDcold
        self.RScold = RScold

        self.VGr  = VG_range
        self.VDDr = VDD_range
        self.VSSr = VSS_range
        self.VMr  = VM_range

        self.VM_VG_pinch = VM_VG_pinch

        self.max_step = max_step
        self.tolerance = tolerance
        self.wait = wait

        self.VDD_setpoint = None
        self.power_setpoint = None
        self.calibrated = False
        self.optimized = False

    def parm_list(self) -> List[dict]:
        return np.array([
            {'name': 'VG', 'f': self.VG, 'max_step': self.max_step, 'tolerance': self.tolerance},
            {'name': 'VDD', 'f': self.VDD, 'max_step': self.max_step, 'tolerance': self.tolerance}, 
            {'name': 'VSS', 'f': self.VSS, 'max_step': self.max_step, 'tolerance': self.tolerance}, 
            {'name': 'VM', 'f': self.VM, 'max_step': self.max_step, 'tolerance': self.tolerance}
        ])

    def _pinch_VM(self):
        if not callable(self.VM):
            return
        self.VM(np.clip(self.VG() + self.VM_VG_pinch, *self.VMr))

    def calibrate(self, 
                  time_per_point: float = 0.1, 
                  numpoints: int = 100, 
                  plot: bool = False):
        
        VGlist = np.linspace(*self.VGr[::-1], numpoints)
        calibration_sweep = SweepMeasure(
            measurements = {'f': self.IDS},
            coordinates = [
                {'f': self.VG, 'max_step': self.max_step, 'tolerance': self.tolerance}, # VG
                {'f': self.VM, 'max_step': self.max_step, 'tolerance': self.tolerance}  # VM
            ],
            points = np.stack([VGlist, np.clip(VGlist + self.VM_VG_pinch, *self.VMr)]).T, # Pinch off VM to compensate as we sweep VG
            time_per_point = time_per_point,
            retain_return = True,
            timestamp = False,
            ramp_wait = self.wait,
            ramp_checkpoints = True,
            logger = self.logger,
        )

        if plot:
            self.cal_data, _ = plotSweep(calibration_sweep)
        else:
            self.cal_data = calibration_sweep.run()
        self.cal_data = self.cal_data.flatten()

        self._process_cal_data()

    def _process_cal_data(self):
        pass

    def auto_bias(self):
        pass

    def set_power(self):
        pass

    def set_VDD(self):
        pass

    def get_power(self) -> float:
        pass

    def get_VDD(self) -> float:
        return self.VDD()

    def zero(self) -> SweepResult:
        return ezTandemSweep(
            parms = self.parm_list(),
            target = 0.0,
            wait = self.wait
        )
