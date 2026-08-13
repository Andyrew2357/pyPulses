"""
pyPulses.routines.cap.prep_amp
================================
prep_amp: cap-specific glue wiring cap_noise_probe into the general
SNR-tuning framework (routines/amp_tune).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context import CapContext
    from .noise_probe import CapNoiseProbeResult

from .noise_probe import cap_noise_probe
from ..amp_tune import AmpTuneContext, AmpTuneResult, amp_calibrate, amp_optimize_snr
from ...devices.HEMT import AmplifierLike

from typing import Dict, Tuple
import logging
import numpy as np


def prep_amp(
    cap_ctx: 'CapContext',
    amp: AmplifierLike,
    bounds: Dict[str, Tuple[float, float]],
    N: int = 15,
    method: str = 'three_point',
    n_refine: int = 1,
    cal_samples: int = 3,
    cal_wait: float | None = None,
    refine_samples: int = 1,
    refine_wait: float | None = None,
    lambda_power: float = 0.0,
    lambda_nongauss: float = 0.0,
    power_cost_weights: Dict[str, float] | None = None,
    n_init: int = 8,
    n_iter: int = 40,
    calibration_numpoints: int = 5,
    random_state: int | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[AmpTuneContext, AmpTuneResult]:
    """
    Tune `amp`'s bias to minimize the capacitance noise floor (and
    non-Gaussian noise) reported by `cap_noise_probe` against `cap_ctx`,
    subject to `amp`'s hard safety window and per-stage power caps, plus
    a soft power penalty (lambda_power).

    The search objective is built entirely in this function's closure --
    the general amp_tune framework (AmpTuneContext.snr_estimator) is an
    opaque callable specifically so cap-specific terms like
    lambda_nongauss stay local here rather than becoming a generic
    AmpTuneContext field. amp_calibrate/amp_optimize_snr from
    routines/amp_tune are used unmodified.

    Parameters
    ----------
    cap_ctx : CapContext
        The capacitance-bridge context to probe (via cap_noise_probe) at
        each candidate bias point. Its cap_filter/extrapolator are
        untouched by this process (see cap_noise_probe).
    amp : AmplifierLike
        The amplifier to tune (e.g. a HEMTCommonSource).
    bounds : dict
        {bias_channel_name: (min, max)} search bounds -- see
        AmpTuneContext.bounds.
    N : int, default=15
        Repeated reads per cap_noise_probe() call -- see its docstring.
    method : {'three_point', 'two_point'}, default='three_point'
        Which local calibration cap_noise_probe uses at each candidate --
        see its docstring. Three-point is the bridge's usual off-balance
        calibration method.
    n_refine : int, default=1
        Number of cap_balance_refine() correction steps cap_noise_probe
        applies before taking its N reads -- keeps the probe centered on
        the true balance point rather than the raw calibration's
        estimate, for a more faithful noise assessment. 0 disables it.
    cal_samples, refine_samples : int
        Passed through to cap_noise_probe's own `cal_samples`/
        `refine_samples`.
    cal_wait, refine_wait : float, optional
        Passed through to cap_noise_probe's own `cal_wait`/`refine_wait`.
        None uses real-hardware-tuned defaults; override for a faster
        synthetic/no-hardware setup.
    lambda_power : float, default=0.0
        Soft power penalty coefficient, same semantics as
        AmpTuneContext.lambda_power. 0.0 disables it.
    lambda_nongauss : float, default=0.0
        Soft penalty coefficient on excess (positive) kurtosis of the
        capacitance noise probe -- catches amp settings whose noise
        variance looks fine but which show slow, non-Gaussian gain jumps
        (e.g. from trap states). 0.0 disables it.
    power_cost_weights : dict, optional
        Passed through to amp.power_cost() inside the objective.
    n_init, n_iter, random_state, logger
        Passed through to AmpTuneContext.
    calibration_numpoints : int, default=5
        Passed through to amp_calibrate()'s `numpoints` -- the
        electrical-feasibility grid has numpoints**len(free_params)
        points. Coarser than this can under-resolve the feasible region
        near a sharp constraint boundary (amp_calibrate's surrogate is a
        nearest-neighbor lookup over this grid); increase it if the
        search seems to stop well short of a hard cap it should be able
        to approach.

    Returns
    -------
    (AmpTuneContext, AmpTuneResult)
        The context that was built (for inspection/reuse) and the
        optimization result. On return, `amp` is sitting at
        `result.best_params` if one was found (see amp_optimize_snr).
    """
    def snr_estimator() -> float:
        probe = cap_noise_probe(
            cap_ctx, N=N, method=method, n_refine=n_refine,
            cal_samples=cal_samples, cal_wait=cal_wait,
            refine_samples=refine_samples, refine_wait=refine_wait,
            logger=logger,
        )
        return (
            -np.log(max(probe.std_Cex, 1e-300))
            - lambda_nongauss * max(0.0, probe.kurtosis_Cex)
        )
        # Only positive (leptokurtic/heavy-tailed) excess kurtosis is
        # penalized -- platykurtic isn't the failure mode this targets.
        # log-scale on std_Cex: it can span orders of magnitude across
        # candidate bias points, same rationale as the power penalty's
        # own objective composition.

    amp_ctx = AmpTuneContext(
        amp=amp,
        snr_estimator=snr_estimator,
        bounds=bounds,
        power_cost_weights=power_cost_weights,
        lambda_power=lambda_power,
        n_init=n_init,
        n_iter=n_iter,
        random_state=random_state,
        logger=logger,
    )
    calibration = amp_calibrate(amp_ctx, numpoints=calibration_numpoints)
    result = amp_optimize_snr(amp_ctx, calibration)
    return amp_ctx, result


if __name__ == '__main__':
    """
    Self-test in two parts, deliberately separated:

    1. The objective composition (-log(std_Cex) - lambda_nongauss *
       max(0, kurtosis)) is checked directly and deterministically at two
       fixed, hand-picked bias points (a "clean" one and a "jumpy" one),
       rather than via a full stochastic Bayesian-optimization search.
       Reasoning: an injected jump inflates *both* the empirical std and
       the kurtosis (a real jump genuinely does add variance, not just
       higher-moment structure -- they aren't cleanly separable), so
       asserting where a noisy, budget-limited BO search converges is
       the wrong thing to test here; the BO mechanics themselves were
       already validated in routines/amp_tune/optimize.py's self-test
       against a clean, deterministic objective. What actually needs
       checking is that lambda_nongauss provably widens the objective
       gap in favor of the clean point as it increases -- an algebraic
       property of the formula, checked directly.
    2. A lighter end-to-end run of the full prep_amp pipeline (synthetic
       AmplifierLike + synthetic bridge, mirroring routines/amp_tune's
       and noise_probe.py's own self-tests) checks it completes, finds
       *some* feasible best point, and that every trial marked feasible
       is actually safe on the amp -- the same safety invariant Phase 2
       checked, now through the cap-specific objective.
    """

    from .context import CapContext
    from ...core.tandem_sweep import SweepResult

    def check(label: str, cond: bool):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        assert cond, label

    class FakeChannel:
        def __init__(self, value: float = 0.0):
            self._v = value
        def __call__(self, v: float | None = None) -> float | None:
            if v is None:
                return self._v
            self._v = v

    class SyntheticAmp:
        """VGS == VG (grounded source); ID = max(VDD, 0) * 1e-3, VDS ==
        VDD; all dissipation on stage '1K'. Safety: VGS in [-0.6, 0.0],
        power <= 5 mW."""
        def __init__(self):
            self._VG, self._VDD = -0.3, 1.0
            self.vgs_min, self.vgs_max = -0.6, 0.0
            self.max_power = 5e-3

        def free_channels(self):
            return {'VG': None, 'VDD': None}

        def ID(self) -> float:
            return max(self._VDD, 0.0) * 1e-3

        def VGS(self) -> float:
            return self._VG

        def VDS(self) -> float:
            return self._VDD

        def dissipated_power_by_stage(self):
            return {'1K': max(self._VDD, 0.0) * 1e-3 * self._VDD}

        def power_cost(self, weights=None):
            by_stage = self.dissipated_power_by_stage()
            if weights is None:
                return sum(by_stage.values())
            return sum(w * by_stage.get(k, 0.0) for k, w in weights.items())

        def check_safe(self) -> bool:
            if not (self.vgs_min <= self.VGS() <= self.vgs_max):
                return False
            return self.dissipated_power_by_stage()['1K'] <= self.max_power

        def move_to(self, panic_behavior='zero', min_wait=None, **targets):
            start = {'VG': self._VG, 'VDD': self._VDD}
            new = {**start, **targets}
            self._VG, self._VDD = new['VG'], new['VDD']
            if not self.check_safe():
                self._VG, self._VDD = start['VG'], start['VDD']
                return SweepResult.PANICKED
            return SweepResult.SUCCEEDED

    A_true = complex(2.0, 0.5)
    V0_true = complex(0.02, -0.01)
    Vex, Cstd, Vstd_range = 1.0, 1e-12, 1.0
    VG_opt = -0.2
    JUMP_VDD_THRESHOLD = 1.5  # noise gets jumpy above this VDD

    def make_lockin_call(amp: SyntheticAmp, Vstd_raw, Theta_raw, rng):
        def lockin_call():
            r, th = Vstd_raw(), np.deg2rad(Theta_raw())
            V = complex(r * np.cos(th), r * np.sin(th))
            dV = V - V0_true
            LX = A_true.real * dV.real - A_true.imag * dV.imag
            LY = A_true.imag * dV.real + A_true.real * dV.imag
            base = 3e-5 / np.sqrt(1.0 + 2.0 * max(amp.VDS(), 0.0))
            sigma = base * (1.0 + 5.0 * (amp.VGS() - VG_opt) ** 2)
            LX += rng.normal(0, sigma)
            LY += rng.normal(0, sigma)
            if amp.VDS() > JUMP_VDD_THRESHOLD and rng.random() < 0.15:
                LX += rng.normal(0, 20 * sigma)
                LY += rng.normal(0, 20 * sigma)
            cov = np.eye(2) * sigma**2  # nominal only, not relied upon
            return np.array([LX, LY]), cov
        return lockin_call

    # -- Part 1: objective composition, checked directly -----------------

    def objective_at(amp: SyntheticAmp, cap_ctx: CapContext,
                      lambda_nongauss: float, N: int = 60) -> tuple[float, CapNoiseProbeResult]:
        probe = cap_noise_probe(cap_ctx, N=N, cal_wait=0.0, refine_wait=0.0)
        obj = -np.log(max(probe.std_Cex, 1e-300)) - lambda_nongauss * max(0.0, probe.kurtosis_Cex)
        return obj, probe

    rng1 = np.random.default_rng(1)
    Vstd_raw1, Theta_raw1 = FakeChannel(0.0), FakeChannel(0.0)
    amp_clean = SyntheticAmp()
    amp_clean.move_to(VG=VG_opt, VDD=0.8)  # below JUMP_VDD_THRESHOLD
    cap_ctx1 = CapContext(
        Vstd=Vstd_raw1, Theta=Theta_raw1,
        lockin_call=make_lockin_call(amp_clean, Vstd_raw1, Theta_raw1, rng1),
        Vstd_range=Vstd_range, Vex=Vex, Cstd=Cstd,
    )

    rng2 = np.random.default_rng(2)
    Vstd_raw2, Theta_raw2 = FakeChannel(0.0), FakeChannel(0.0)
    amp_jumpy = SyntheticAmp()
    amp_jumpy.move_to(VG=VG_opt, VDD=2.0)  # above JUMP_VDD_THRESHOLD
    cap_ctx2 = CapContext(
        Vstd=Vstd_raw2, Theta=Theta_raw2,
        lockin_call=make_lockin_call(amp_jumpy, Vstd_raw2, Theta_raw2, rng2),
        Vstd_range=Vstd_range, Vex=Vex, Cstd=Cstd,
    )

    _, probe_clean = objective_at(amp_clean, cap_ctx1, 0.0)
    _, probe_jumpy = objective_at(amp_jumpy, cap_ctx2, 0.0)
    print(f"clean bias point: {probe_clean}")
    print(f"jumpy bias point: {probe_jumpy}")
    check('the jumpy bias point actually shows elevated kurtosis',
          probe_jumpy.kurtosis_Cex > probe_clean.kurtosis_Cex + 1.0)

    gaps = []
    for lam in (0.0, 1.0, 5.0, 20.0):
        obj_clean = -np.log(max(probe_clean.std_Cex, 1e-300)) - lam * max(0.0, probe_clean.kurtosis_Cex)
        obj_jumpy = -np.log(max(probe_jumpy.std_Cex, 1e-300)) - lam * max(0.0, probe_jumpy.kurtosis_Cex)
        gaps.append(obj_clean - obj_jumpy)
    check('lambda_nongauss monotonically widens the objective gap in '
          "favor of the clean point (given the jumpy point's higher kurtosis)",
          all(gaps[i + 1] >= gaps[i] for i in range(len(gaps) - 1)) and gaps[-1] > gaps[0])

    # -- Part 2: lighter end-to-end pipeline + safety invariant -----------

    rng3 = np.random.default_rng(3)
    Vstd_raw3, Theta_raw3 = FakeChannel(0.0), FakeChannel(0.0)
    amp3 = SyntheticAmp()
    cap_ctx3 = CapContext(
        Vstd=Vstd_raw3, Theta=Theta_raw3,
        lockin_call=make_lockin_call(amp3, Vstd_raw3, Theta_raw3, rng3),
        Vstd_range=Vstd_range, Vex=Vex, Cstd=Cstd,
    )
    bounds = {'VG': (-0.6, 0.0), 'VDD': (0.0, 3.0)}

    print("\nRunning the full prep_amp pipeline end-to-end...")
    ctx, result = prep_amp(
        cap_ctx3, amp3, bounds, N=10, lambda_nongauss=2.0, lambda_power=0.0,
        n_init=5, n_iter=15, calibration_numpoints=8, random_state=0,
        cal_wait=0.0, refine_wait=0.0,  # no real hardware -- skip default settle waits
    )
    print(result)
    check('prep_amp found a best point', result.best_params is not None)
    check('best point respects the VGS safety window',
          amp3.vgs_min <= result.best_params['VG'] <= amp3.vgs_max)
    for t in result.trials:
        if t.feasible:
            vg, vdd = t.params['VG'], t.params['VDD']
            vgs_ok = amp3.vgs_min <= vg <= amp3.vgs_max
            power_ok = (max(vdd, 0.0) * 1e-3) * vdd <= amp3.max_power
            check(f'feasible trial {t.params} is actually safe', vgs_ok and power_ok)

    print("\nAll checks passed.")
