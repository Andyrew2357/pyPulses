"""
pyPulses.routines.cap
=====================
Penetration capacitance measurement routines.

Three measurement modes are supported, all operating on a `CapContext`:

Off-balance (`cap_measure`)
    The bridge is held at the most recent balance point. The lock-in reading
    is converted to (Cex, Closs) using the Kalman filter's cached complex gain.
    Fast and passive — no hardware adjustments, no filter updates.

On-balance (`cap_balance`)
    Iteratively re-balances Vstd at each phase-space point using the Kalman
    filter's gain estimate and the extrapolator's predicted starting point.
    Updates the filter with each (dL, dv) observation.

Differential balance (`differential_balance`)
    Nulls compressibility features along a 1D cut by measuring lock-in readings
    through the excitation and complementary gates separately, then computing a
    complex ratio gamma that sets the complementary amplitude and phase.
    Independent of the Kalman filter machinery.

Balance refinement (`cap_balance_refine`)
    `cap_balance_refine(ctx)` re-nulls the bridge from the current Vstd with
    a small, fixed number of Newton steps through the gain already stored on
    ctx.cap_filter -- the gain itself is not re-estimated (contrast
    cap_balance's continuous Kalman tracking).

Crosstalk compensation (`cap_initialize_crosstalk`, `CapCrosstalkTransition`)
    Subtracts excitation-line crosstalk (signal that bypasses the amplifier
    entirely, e.g. via attenuator-driven upstream coupling) from every
    lock-in reading before any balance math sees it -- a post-hoc phase
    correction can't remove this because it's not a pure rotation.
    cap_initialize_crosstalk measures it once at the current excitation
    (run it before cap_initialize); CapCrosstalkTransition remeasures it
    whenever the excitation changes in a scan. Both wrap ctx.lockin_call in
    a CrosstalkCompensatedLockIn. See crosstalk.py's module docstring for
    the full rationale.

Precomputed calibrations (`CapCalibrationLibrary`, `CapCalibrationTransition`)
    For off-balance work over several excitation states: calibrate crosstalk
    and balance for every state once, parked at one point
    (`cap_build_library`), save the library to JSON, then let
    `CapCalibrationTransition` load the stored state whenever the excitation
    changes -- no reference-point trip or amp cycling mid-scan, so the
    excitation can be an inner scan coordinate. See library.py.

Noise probe (`cap_noise_probe`) and amp tuning (`prep_amp`)
    cap_noise_probe reports the capacitance noise floor (empirical std) and
    non-Gaussianity (empirical excess kurtosis) at the bridge's current
    setting, from a cheap local recalibration (cap_balance_three_point by
    default, or cap_balance_two_point), optional refinement, and repeated
    reads -- no gate sweep, independent of ctx.cap_filter/extrapolator.
    prep_amp wires it into the general SNR-tuning framework
    (pyPulses.routines.amp_tune) to tune a HEMTCommonSource-like amplifier's
    bias for the best capacitance-referred SNR.

Typical usage
-------------
    from pyPulses.routines import cap
    from pyPulses.devices import hf2li

    # Bridge on an HF2LI: reference drive on output 0 / mixer ch 1, sample
    # drive on output 1 / ch 2, measured on demod 0 (see hf2li.configure_bridge)
    hf2 = hf2li('dev1616', registry_id='hf2')
    br = hf2.configure_bridge(std=(0, 1), ex=(1, 2), measure_demod=0,
                              frequency=1e6, output_range=1.0, duration=0.1)
    br['Vex'](0.01)                                  # V peak

    # Build context (Vstd/Vex in the same units: V peak)
    ctx = cap.CapContext(
        Vstd        = br['Vstd'],              # hf2->out0_ch1_amplitude
        Theta       = br['Theta'],             # hf2->out0_ch1_phase
        lockin_call = br['stream_average'],    # hf2->stream_average
        Vstd_range  = 0.1,
        Vex         = 0.01,
        Cstd        = 1e-12,
        settle_time = 0.3,
    )

    # Optional (SRS lock-ins): attach a serializable rescale callback
    ctx.set_rescale_callback(lockin, 'auto_gain')

    # Optional: measure excitation-line crosstalk first (amp off/on)
    cap.cap_initialize_crosstalk(
        ctx, amp_channels=[amp_gate_ch],
        amp_on={'amp_VG': -0.2}, amp_off={'amp_VG': -5.0},
    )

    # Initialize (three-point balance seeds A, P, and K_matrix)
    cfg = cap.cap_initialize(ctx, method='three_point')
    # or two-point:
    cfg = cap.cap_initialize(ctx, method='two_point', dVstd=0.05+0j)

    # Tighten the balance with Newton steps through the stored gain
    cap.cap_balance_refine(ctx, n_refine=2)

    # Off-balance measurement at each phase-space point
    result = cap.cap_measure(ctx)
    result = cap.cap_measure(ctx, use_matrix=True)  # use full K matrix if available

    # On-balance measurement at each phase-space point
    result = cap.cap_balance(ctx)

    # Differential balance along a 1D cut (complementary gate driven from a
    # third mixer channel, e.g. output 1 / ch 3 on the bridge oscillator)
    result = cap.differential_balance(
        ctx        = ctx,
        scan       = scan_1d,
        Vex_amp    = br['Vex'],
        Vexp_amp   = hf2.resolve('out1_ch3_amplitude'),
        Vexp_phase = hf2.resolve('out1_ch3_phase'),
    )

    # Save / restore state
    ctx.save_state_json('cap_state.json')
    ctx.load_state_json('cap_state.json')

    # Reconstruct fully from config
    ctx2 = cap.CapContext.from_json('cap_state.json')

Result dataclasses
------------------
    cap.CapMeasureResult            — off-balance measurement result
    cap.CapBalanceResult            — on-balance measurement result
    cap.ThreePointBalanceResult     — result of three-point initial balance
    cap.TwoPointBalanceResult       — result of two-point initial balance
    cap.CapInitialFilterConfig      — result of cap_initialize / cap_initialize_filter
    cap.DifferentialBalanceResult   — result of differential_balance
    cap.RefineBalanceResult         — result of cap_balance_refine
    cap.CapNoiseProbeResult         — result of cap_noise_probe

Context, filter, and extrapolator
----------------------------------
    cap.CapContext
    cap.CapFilter
    cap.CapExtrapolator

Channel protocols (defined in channel_adapter)
----------------------------------------------
    cap.LockInChannel    — protocol for lockin_call: () -> (mean, cov)
    cap.CommandChannel   — protocol for rescale_callback: () -> None

Config constants
----------------
    from pyPulses.routines.cap.config import *
"""

# Config
from .config import (
    PROCESS_NOISE_COEFF,
    CAP_INIT_P_MULT_THREE_POINT,
    CAP_INIT_P_MULT_TWO_POINT,
)

# Filter and extrapolator
from .cap_filter import CapFilter, CapExtrapolator

# Context
from .context import CapContext

# Initialization subroutines
from .initialize import (
    cap_balance_three_point,
    cap_balance_two_point,
    cap_balance_refine,
    cap_initialize_filter,
    cap_initialize,
    ThreePointBalanceResult,
    TwoPointBalanceResult,
    CapInitialFilterConfig,
    RefineBalanceResult,
)

# Measurement functions
from .measure import cap_measure, CapMeasureResult, CapQuery
from .balance import cap_balance, CapBalanceResult

# Differential balance
from .differential import differential_balance, DifferentialBalanceResult

# Noise probe and amp tuning
from .noise_probe import cap_noise_probe, CapNoiseProbeResult
from .prep_amp import prep_amp

# Transition for scans that change excitation parameters
from .transition import CapRebalanceTransition

# Excitation-line crosstalk compensation
from .crosstalk import (
    CrosstalkCompensatedLockIn,
    CapCrosstalkTransition,
    cap_initialize_crosstalk,
)

# Precomputed per-excitation calibrations
from .library import (
    CapCalibration,
    CapCalibrationLibrary,
    CapCalibrationTransition,
    cap_calibrate_crosstalk,
    cap_calibrate_balance,
    cap_build_library,
)

__all__ = [
    # Config
    'PROCESS_NOISE_COEFF',
    'CAP_INIT_P_MULT_THREE_POINT',
    'CAP_INIT_P_MULT_TWO_POINT',
    # Filter and extrapolator
    'CapFilter',
    'CapExtrapolator',
    # Context
    'CapContext',
    # Initialization
    'cap_balance_three_point',
    'cap_balance_two_point',
    'cap_balance_refine',
    'cap_initialize_filter',
    'cap_initialize',
    'ThreePointBalanceResult',
    'TwoPointBalanceResult',
    'CapInitialFilterConfig',
    'RefineBalanceResult',
    # Measurement
    'cap_measure',
    'CapMeasureResult',
    'cap_balance',
    'CapBalanceResult',
    'CapQuery',
    # Differential balance
    'differential_balance',
    'DifferentialBalanceResult',
    # Noise probe and amp tuning
    'cap_noise_probe',
    'CapNoiseProbeResult',
    'prep_amp',
    # Transition for scans that change excitation parameters
    'CapRebalanceTransition',
    # Excitation-line crosstalk compensation
    'CrosstalkCompensatedLockIn',
    'CapCrosstalkTransition',
    'cap_initialize_crosstalk',
    # Precomputed per-excitation calibrations
    'CapCalibration',
    'CapCalibrationLibrary',
    'CapCalibrationTransition',
    'cap_calibrate_crosstalk',
    'cap_calibrate_balance',
    'cap_build_library',
]