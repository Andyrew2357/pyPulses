"""
pyPulses.routines.amp_tune
===========================
General SNR-tuning framework for biased amplifier topologies. Operates on
any object satisfying the AmplifierLike protocol (pyPulses.devices.HEMT)
-- not specific to HEMTCommonSource.

Two-stage workflow, both operating on a shared `AmpTuneContext`:

Calibration (`amp_calibrate`)
    Cheap electrical characterization: sweeps the free bias parameters
    over a grid, reading only ID/VGS/dissipated_power_by_stage (no SNR
    calls). Builds a grid-based surrogate (`AmpCalibration`) used to
    cheaply pre-filter candidates for feasibility before the search ever
    touches hardware with a (slow, noisy) SNR measurement.

Optimization (`amp_optimize_snr`)
    Ask/tell Bayesian-optimization search over the free bias parameters,
    restricted to the calibration surrogate's feasible region (transistor
    safety window + per-stage power caps, as hard constraints), maximizing
    snr - lambda_power * power_cost as the search objective.

Typical usage
-------------
    from pyPulses.routines import amp_tune

    ctx = amp_tune.AmpTuneContext(
        amp           = my_hemt,
        snr_estimator = my_snr_function,
        bounds        = {'VG': (-0.6, 0.0), 'VDD': (0.0, 2.0)},
        lambda_power  = 0.1,
    )

    calibration = amp_tune.amp_calibrate(ctx)
    result = amp_tune.amp_optimize_snr(ctx, calibration)
    print(result)

Result dataclasses
------------------
    amp_tune.AmpCalibration, amp_tune.AmpCalibrationPoint
    amp_tune.AmpTuneResult, amp_tune.AmpTuneTrial

Context
-------
    amp_tune.AmpTuneContext
"""

from .context import AmpTuneContext
from .calibrate import amp_calibrate, AmpCalibration, AmpCalibrationPoint
from .optimize import amp_optimize_snr, AmpTuneResult, AmpTuneTrial

__all__ = [
    'AmpTuneContext',
    'amp_calibrate',
    'AmpCalibration',
    'AmpCalibrationPoint',
    'amp_optimize_snr',
    'AmpTuneResult',
    'AmpTuneTrial',
]
