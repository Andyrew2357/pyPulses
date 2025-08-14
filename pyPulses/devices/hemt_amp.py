
from .abstract_device import abstractDevice
from ..utils import curves
from ..utils.tandem_sweep import ezTandemSweep, SweepResult
from ..utils.param_sweep_measure import SweepMeasureCut

# REFACTORED PLOTTING...
from .plot_sweep import plotSweep # type: ignore

import json
import numpy as np
from typing import Callable, Tuple, List

class HEMTCommonSource(abstractDevice):
    """
    Software representation of a HEMT amplifier with common source topology.
    
                               VG       VDD
                                ┃        ┃
                             ╻┏━┛        ⌇ RD
                       VM ━━━┫┃          ┣━━━┫┣━━ AC OUT
                             ╹┗━┓     ╻┏━┛
                AC IN ━━━┫┣━━━━━┻━━━━━┫┃
                                      ╹┗━┓
                                         ┃
                                         ⌇ RS
                                         ┃
                                        VSS
    
    """

    def __init__(self,  
                 VDD: Callable,
                 IDS: Callable[[], float],
                 RD: float, RS: float,
                 VG: Callable | str = 'FLO',
                 VM : Callable | None = None,
                 VSS: Callable | str = 'GND', 
                 VG_range   : Tuple[float, float] = (-0.6, 0.0),
                 VDD_range  : Tuple[float, float] = (0.0, 4.0),
                 VSS_range  : Tuple[float, float] = (0.0, 0.0),
                 VM_range   : Tuple[float, float] = (-1.2, 0.0),
                 VM_VG_pinch: float = -0.6,
                 RDcold: float = 0.0,
                 RScold: float = 0.0,
                 tuning_parm: str = 'VDD_VG',
                 max_step   : float = 0.005,
                 tolerance  : float = 500e-6,
                 wait       : float = 0.05,
                 logger = None):
        """
        Parameters
        ----------
        VDD : Callable
            Get and set VDD.
        IDS : Callable
            Get IDS.
        RD : float
            Total drain resistance in Ohms.
        RS : float
            Total source resistance in Ohms.
        VG : Callable or str, default='FLO'
            Get and set VG. Assumed to be floating if not provided
        VM : Callable, optional
            Get and set VM. Assumed to be absent or a static resistor if not 
            provided.
        VSS : Callable or str, default='GND'
            Get or set VSS. Assumed to be cold ground if not provided.
        VG_range : Tuple[float, float], default=(-0.6,0.0)
        VDD_range : Tuple[float, float], default=(0.0,4.0)
        VSS_range : Tuple[float, float], default=(0.0,0.0)
        VM_range : Tuple[float, float], default=(-1.2,0.0)
        VM_VG_pinch : float, default=-0.6
            Volts to pinch off VM - VG.
        RDcold : float, default=0.0
            How much of `RD` is cold.
        RScold : float, default=0.0
            How much of `RS` is cold.
        tuning_parm : str, default='VDD_VG'
            One of {'VDD_VG', 'VDD_VSS'}. Defines the configuration of the amp /
            sets which parameters are tuned.
        max_step : float, default=0.005
            Maximum voltage step to take on amp parameters.
        tolerance : float, default=500e-6
            Sweep tolerance for setting amp parameters.
        wait : float, default=0.05
            Time to wait between stepping to different points while sweeping.
        logger : Logger, optional
        """
            
        super().__init__(logger)

        # bias parameters
        self._VG  = VG
        self._VDD = VDD
        self._VSS = VSS
        self._VM  = VM

        # bias ranges
        self.VGr  = VG_range
        self.VDDr = VDD_range
        self.VSSr = VSS_range
        self.VMr  = VM_range

        # how hard to pinch off bias transistor
        self.VM_VG_pinch = VM_VG_pinch

        # handle static bias resistor
        if VM is None:
            self._VM = lambda *args: 0.0
            self.VMr = (-np.inf, np.inf)
            self.VM_VG_pinch = 0.0

        # handle grounded source
        if VSS == 'GND':
            self._VSS = lambda *args: 0.0
            self.VSSr = (0.0, 0.0)

        # handle floating gate
        if VG == 'FLO':
            self._VG = lambda *args: 0.0
            self._VGr = (0.0, 0.0)

        # current
        self.IDS = IDS

        # resistances
        self.RD = RD
        self.RS = RS
        self.RDcold = RDcold
        self.RScold = RScold

        # ramp settings
        self.max_step = max_step
        self.tolerance = tolerance
        self.wait = wait

        # status information
        self.tuning_parm = tuning_parm
        self.target_RT: float = None
        self.power_setpoint: float = None
        self.calibrated = False
        self.optimized = False
        self.optimize_attempts = 5
        self.percent_tolerance = 10

    def _parm_list(self) -> List[dict]:
        return [
            {'name': 'VG', 'f': self._VG, 'max_step': self.max_step, 'tolerance': self.tolerance},
            {'name': 'VDD', 'f': self._VDD, 'max_step': self.max_step, 'tolerance': self.tolerance}, 
            {'name': 'VSS', 'f': self._VSS, 'max_step': self.max_step, 'tolerance': self.tolerance}, 
            {'name': 'VM', 'f': self._VM, 'max_step': self.max_step, 'tolerance': self.tolerance}
        ]

    def save_state_json(self, path: str):
        """
        Save calibration data and relevant settings.
        
        Parameters
        ----------
        path : str
        """
        super().save_state_json()
    
    def load_state_json(self, path: str):
        """
        Load calibration data and relevant settings.
        
        Parameters
        ----------
        path : str
        """
        super().load_state_json()
    
    def _serialize_state(self) -> dict:
        state = {
            # tuning settings
            'tuning_parm': self.tuning_parm,
            'target_RT': self.target_RT,
            'power_setpoint': self.power_setpoint,
            'calibrated': self.calibrated,
            'optimize_attempts': self.optimize_attempts,
            'percent_tolerance': self.percent_tolerance,
            # resistances
            'RD': self.RD,
            'RS': self.RS,
            'RDcold': self.RDcold,
            'RScold': self.RScold,
            # parameter ranges
            'VGr': self.VGr,
            'VMr': self.VMr,
            'VDDr': self.VDDr,
            'VSSr': self.VSSr,
            # pinch off bias transistor
            'VM_VG_pinch': self.VM_VG_pinch,
            # ramp settings:
            'max_step': self.max_step,
            'tolerance': self.tolerance,
            'wait': self.wait
        }
        if hasattr(self, 'raw_cal_data'):
            state['calibration_data'] = self.raw_cal_data.tolist()
        return state
    
    def _deserialize_state(self, state: dict):
        # tuning settings
        self.optimized = False
        self.tuning_parm = state['tuning_parm']
        self.target_RT = state['target_RT']
        self.power_setpoint = state['power_setpoint']
        self.calibrated = state['calibrated']
        self.optimize_attempts = state['optimize_attempts']
        self.percent_tolerance = state['percent_tolerance']
        # resistances
        self.RD = state['RD']
        self.RS = state['RS']
        self.RDcold = state['RDcold']
        self.RScold = state['RScold']
        # parameter ranges
        self.VGr = state['VGr']
        self.VMr = state['VMr']
        self.VDDr = state['VDDr']
        self.VSSr = state['VSSr']
        # pinch off bias transistor
        self.VM_VG_pinch = state['VM_VG_pinch']
        # ramp settings
        self.max_step = state['max_step']
        self.tolerance = state['tolerance']
        self.wait = state['wait']

        self.raw_cal_data = state.get['calibration_data']
        if self.raw_cal_data is not None:
            self.raw_cal_data = np.array(self.raw_cal_data)
            self._process_cal_data()

    def calibrate(self, 
                  save_path: str = None,
                  time_per_point: float = 0.1, 
                  numpoints: int = 100, 
                  plot: bool = False):
        """
        Perform calibration measurements to characterize the amp.

        Parameters
        ----------
        save_path : str, optional
            Where to save calibration data (if desired).
        time_per_point : float, default=0.1
            Seconds to spend on each measurement.
        numpoints : int, default=100
            Number of points to take in the calibration curve.
        plot : bool, default=False
            Whether to plot data live while measuring.
        """

        self.info("Performing calibration measurements...")

        self.calibrated = False
        self.optimized = False

        if self.tuning_parm == 'VDD_VG':
                
            VGmin = max(self.VGr[0], self.VMr[0] - self.VM_VG_pinch)
            VGmax = min(self.VGr[1], self.VMr[1] - self.VM_VG_pinch)

            VDDcol = self._VDD()
            VSScol = self._VSS()
            calibration_sweep = SweepMeasureCut(
                measurements = [
                    {'f': self.IDS, 'name': 'IDS'}, 
                    {'f': lambda: VDDcol, 'name': 'VDD'}, 
                    {'f': lambda: VSScol, 'name': 'VSS'}
                ],
                coordinates = [
                    {'f': self._VG, 'max_step': self.max_step, 
                     'tolerance': self.tolerance, 'name': 'VG'}, # VG
                    {'f': self._VM, 'max_step': self.max_step, 
                     'tolerance': self.tolerance, 'name': 'VM'}  # VM
                ],
                time_per_point = time_per_point,
                retain_return = True,
                timestamp = False,
                ramp_wait = self.wait,
                ramp_checkpoints = True,
                logger = self.logger,
                # Pinch off VM to compensate as we sweep VG
                numpoints = numpoints,
                start = [VGmax, VGmax + self.VM_VG_pinch],
                end = [VGmin, VGmin + self.VM_VG_pinch]
            )

            if plot:
                cal_data, _ = plotSweep(calibration_sweep)
            else:
                cal_data = calibration_sweep.run()

            raw_VG = np.linspace(VGmax, VGmin, numpoints)
            self.raw_cal_data = np.vstack([raw_VG, 
                                           raw_VG + self.VM_VG_pinch, 
                                           cal_data.T])

        elif self.tuning_parm == 'VDD_VSS':
        
            VDS = self._VDD() - self._VSS()
            if VDS <= 0:
                raise RuntimeError(
                    "Cannot calibrate with no offset between VDD and VSS."
                )
            
            VSSmin = max(self.VSSr[0], self.VDDr[0] - VDS)
            VSSmax = min(self.VSSr[1], self.VDDr[1] - VDS)  
            calibration_sweep = SweepMeasureCut(
                measurements = [{'f': self.IDS, 'name': 'IDS'},],
                coordinates = [
                    {'f': self._VSS, 'max_step': self.max_step, 
                     'tolerance': self.tolerance, 'name': 'VSS'}, # VSS
                    {'f': self._VDD, 'max_step': self.max_step, 
                     'tolerance': self.tolerance, 'name': 'VDD'}  # VDD
                ],
                time_per_point = time_per_point,
                retain_return = True,
                timestamp = False,
                ramp_wait = self.wait,
                ramp_checkpoints = True,
                logger = self.logger,
                # Sweep VSS and VDD maintaining VDS
                numpoints = numpoints,
                start = [VSSmin, VSSmin + VDS],
                end = [VSSmax, VSSmax + VDS]
            )

            if plot:
                cal_data, _ = plotSweep(calibration_sweep)
            else:
                cal_data = calibration_sweep.run()

            raw_VSS = np.linspace(VSSmin, VSSmax, numpoints)
            self.raw_cal_data = np.vstack([raw_VSS, 
                                           raw_VSS + VDS, 
                                           cal_data.T])          

        else:
            raise ValueError(
                f"'{self.tuning_parm}' is not a valid tuning setting "
                 "for HEMTCommonSource."
            )
        
        self.info("Measurements complete. Processing calibration data...")
        self._process_cal_data()
        self.info("Calibration complete.")

        if save_path:
            self.info(f"Saving calibration to {save_path}.")
            self.save_state_json(save_path)

    def _process_cal_data(self):

        if self.tuning_parm == 'VDD_VG':
            VG, VM, IDS, VDD, VSS = self.raw_cal_data
            RT = (VDD - VSS) / IDS - self.RD - self.RS
            VGS = VG - (VDD*(self.RS) + VSS*(RT + self.RD)) / (RT + self.RD + self.RS)
            RTtoVGS_coeffs = curves.pchip_params(*curves.prune_sort(RT, VGS))
            self.RT_to_VGS = curves.pchip_rval_from_params(*RTtoVGS_coeffs)

        elif self.tuning_parm == 'VDD_VSS':
            VSS, VDD, IDS = self.raw_cal_data
            RT = (VDD - VSS) / IDS - self.RS - self.RD
            VS = (VDD*self.RS + VSS*(RT + self.RD)) / (RT + self.RD + self.RS)
            RTtoVGS_coeffs = curves.pchip_params(*curves.prune_sort(RT, -VS))
            self.RT_to_VGS = curves.pchip_rval_from_params(*RTtoVGS_coeffs)

        else:
            raise ValueError(
                f"'{self.tuning_parm}' is not a valid tuning setting "
                 "for HEMTCommonSource."
            )

        self.target_RT = 0.5 * np.max(RT)
        self.calibrated = True

    def get_RT(self):
        """
        Get the amp resistance in Ohms.
        
        Returns
        -------
        float
        """
        return self._VDD() / self.IDS() - self.RS - self.RD

    def auto_tune(self):
        """
        Tune the amp parameters according to the settings.
        """
        
        self.info(f"Automatically tuning in '{self.tuning_parm}' mode...")
        if self.power_setpoint is None:
            raise RuntimeError(
                "Cannot auto_tune HEMTCommonSource without a power set-point."
            )

        if self.tuning_parm == 'VDD_VG':

            # calculate optimal parameters
            R = self.RD + self.RS + self.target_RT
            Rcold = self.target_RT + self.RDcold + self.RScold
            VSS = self._VSS()
            VDD = VSS + R * np.sqrt(self.power_setpoint / Rcold)
            VS = (VDD * self.RS + VSS * (self.RD + self.target_RT)) / R
            VG = VS + self.RT_to_VGS(self.target_RT)
            VM = VG + self.VM_VG_pinch

        elif self.tuning_parm == 'VDD_VSS':

            # determine the effective gate voltage
            RT = self.get_RT()
            VGSeff = self.RT_to_VGS(RT)
            VScurr = (VDD * self.RS + VSS * (self.RD + self.target_RT)) / R
            VGcurr = VScurr + VGSeff

            # calculate optimal parameters
            R = self.RD + self.RS + self.target_RT
            Rcold = self.target_RT + self.RDcold + self.RScold
            VDS = R * np.sqrt(self.power_setpoint / Rcold)
            VS = VGcurr - self.RT_to_VGS(self.target_RT)
            VSS = (VS * R - VDS * self.RS) / R
            VDD = VSS + VDS
            VG = 0.0 # Left floating in this configuration.
            VM = 0.0 # No bias transistor in this configuration

        else:
            raise ValueError(
                f"'{self.tuning_parm}' is not a valid tuning setting "
                 "for HEMTCommonSource."
            )
        
        if VDD > self.VDDr[1] or VDD < self.VDDr[0]:
            raise RuntimeError(
                "Target power would place VDD outside allowed range."
            )
        if VG > self.VGr[1] or VG < self.VGr[0]:
            raise RuntimeError(
                "Required settings would place VG outside allowed range."
            )
        if VM > self.VMr[1] or VG < self.VMr[0]:
            raise RuntimeError(
                "Required settings would place VM outside allowed range."
            )
        
        self.info(f"Tuned Parameters: VG = {VG:.4f}, VM = {VM:.4f}, "
                  f"VDD = {VDD:.4f}, VSS = {VSS:.4f}")
        ezTandemSweep(
            parms  = self._parm_list(),
            target = [VG, VDD, VSS, VM],
            wait   = self.wait
        )

        self.optimize()

    def optimize(self):
        """Fine tune the amp settings to get closer to the target resistance."""
        
        self.info("Fine tuning amp parameters...")
        prev_RT = None
        for i in range(self.optimize_attempts):
            RT = self.get_RT()
            if abs(RT / self.target_RT - 1)<= 0.01 * self.percent_tolerance:
                break

            self.info(f"Performing iteration {i + 1} of optimizations...")

            if self.tuning_parm == 'VDD_VG':
                VG = self._VG()
                VGmin = max(self.VGr[0], self.VMr[0] - self.VM_VG_pinch)
                VGmax = min(self.VGr[1], self.VMr[1] - self.VM_VG_pinch)

                if prev_RT is None:
                    # take a small step in the right direction.
                    VGspan = VGmax - VGmin                   
                    newVG = VG + 0.05 * np.sign(RT - self.target_RT) * VGspan
                else:
                    dRTdVG = (RT - prev_RT) / (VG - prev_VG)
                    newVG = VG + (self.target_RT - RT) / dRTdVG

                if newVG < VGmin or newVG > VGmax:
                    self.warn("Optimized amp settings fall outside range")
                    break

                prev_VG = VG
                self.info(f"Predicted optimum at VG = {newVG:.5f}")
                ezTandemSweep(
                    parms  = self._parm_list(),
                    target = [newVG, self._VDD(), self._VSS(), newVG + self.VM_VG_pinch],
                    wait   = self.wait
                )

            elif self.tuning_parm == 'VDD_VSS':
                VSS = self._VSS()
                VDS = self._VDD() - VSS
                VSSmin = max(self.VSSr[0], self.VDDr[0] - VDS)
                VSSmax = min(self.VSSr[1], self.VDDr[1] - VDS)  

                if prev_RT is None:
                    # take a small step in the right direction.
                    VSSspan = VSSmax - VSSmin                   
                    newVSS = VSS + 0.05 * np.sign(self.target_RT - RT) * VSSspan
                else:
                    dRTdVSS = (RT - prev_RT) / (VSS - prev_VSS)
                    newVSS = VSS + (self.target_RT - RT) / dRTdVSS

                if newVSS < VSSmin or newVSS > VSSmax:
                    self.warn("Optimized amp settings fall outside range")
                    break

                prev_VSS = VSS
                self.info(f"Predicted optimum at VSS = {newVSS:.5f}")
                ezTandemSweep(
                    parms  = self._parm_list(),
                    target = [0.0, newVSS + VDS, newVSS, 0.0],
                    wait   = self.wait
                )

            else:
                raise ValueError(
                    f"'{self.tuning_parm}' is not a valid tuning setting "
                    "for HEMTCommonSource."
                )
            
            prev_RT = RT
            
        self.info("Optimization complete.")
        self.optimized = True

    def set_power(self, power: float):
        """
        Set the target power to dissipate in Watts (for cold circuit elements).
        """
        self.info(f"Set power to {power} W.")
        self.power_setpoint = power
        self.optimized = False

    def get_power(self) -> float:
        """
        Get the power dissipated by cold circuit elements in Watts.

        Returns
        -------
        float
        """
        VDS = self._VDD() - self._VSS() 
        IDS = self.IDS()
        R = VDS / IDS
        ratio = (R - self.RD - self.RD + self.RDcold + self.RScold) / R
        return ratio * IDS * VDS
    
    def VG(self, V: float = None) -> float | None:
        if V is None:
            return self._VG()
        if V < self.VGr[0] or V > self.VGr[1]:
            raise ValueError("Cannot set VG outside allowable range.")
        
        self.info(f"Set VG to {V} V.")
        ezTandemSweep({'name': 'VG', 'f': self._VG, 
                       'max_step': self.max_step, 
                       'tolerance': self.tolerance},
                       V, self.wait)
        self.optimized = False

    def VM(self, V: float = None) -> float | None:
        if V is None:
            return self._VM()
        if V < self.VMr[0] or V > self.VMr[1]:
            raise ValueError("Cannot set VM outside allowable range.")
        
        self.info(f"Set VM to {V} V.")
        ezTandemSweep({'name': 'VM', 'f': self._VM, 
                       'max_step': self.max_step, 
                       'tolerance': self.tolerance},
                       V, self.wait)
        self.optimized = False

    def VDD(self, V: float = None) -> float | None:
        if V is None:
            return self._VDD()
        if V < self.VDDr[0] or V > self.VDDr[1]:
            raise ValueError("Cannot set VDD outside allowable range.")
        
        self.info(f"Set VDD to {V} V.")
        ezTandemSweep({'name': 'VDD', 'f': self._VDD, 
                       'max_step': self.max_step, 
                       'tolerance': self.tolerance},
                       V, self.wait)
        self.optimized = False

    def VSS(self, V: float = None) -> float | None:
        if V is None:
            return self._VSS()
        if V < self.VSSr[0] or V > self.VSSr[1]:
            raise ValueError("Cannot set VSS outside allowable range.")
        
        self.info(f"Set VSS to {V} V.")
        ezTandemSweep({'name': 'VSS', 'f': self._VSS, 
                       'max_step': self.max_step, 
                       'tolerance': self.tolerance},
                       V, self.wait)
        self.optimized = False

    def sweep_parms(self, VG: float = None, VM: float = None, 
                  VDD: float = None, VSS: float = None):
        if VG is None: VG = self._VG()
        if VM is None: VM = self._VM()
        if VDD is None: VDD = self._VDD()
        if VSS is None: VSS = self._VSS()
        self.info(f"Sweeping parameters: VG = {VG:.4f}, VM = {VM:.4f}, "
                  f"VDD = {VDD:.4f}, VSS = {VSS:.4f}...")
        res = ezTandemSweep(
            parms  = self._parm_list(),
            target = [VG, VDD, VSS, VM],
            wait   = self.wait
        )
        self.info("Sweep complete")

    def zero(self) -> SweepResult:
        """Ramp all amp parameters smoothly to 0 V."""

        self.info("Ramping down all amp parameters...")
        res = ezTandemSweep(
            parms  = self._parm_list(),
            target = 0.0,
            wait   = self.wait
        )
        self.info("Successfully ramped to zero.")
        return res
