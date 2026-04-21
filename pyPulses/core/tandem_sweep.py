"""
Tandem sweep: move multiple parameters simultaneously while respecting
per-channel step size and rate constraints.

Public API
----------
tandemSweep(channels, target, ...)
    Query current values, derive wait from SweepConfig, then sweep.

Internal workhorse
------------------
_tandemSweep(channels, start, end, wait, ...)
    Accepts explicit start values and a pre-computed wait. Used by
    param_sweep_measure and other callers that maintain their own position
    cache and want to bypass the initial hardware query.
"""

from .job import checkpoint
from ..devices.sweepable_channel import SweepableChannel

from enum import Enum, auto
from math import ceil
from typing import Any, Callable, Dict, List
import numpy as np
import time


class SweepResult(Enum):
    SUCCEEDED = auto()
    FAILED = auto()
    PANICKED = auto()


"""Wait derivation"""

def _derive_wait(channels: List[SweepableChannel], min_wait: float | None) -> float:
    """
    Derive the shared per-step wait time from channel SweepConfigs.

    For each channel the effective wait contribution is:
        max_step / max_rate + settle_time   (when max_rate is set)
        settle_time                         (when only settle_time is set)
        0                                   (when neither is set)

    The shared wait is the maximum across all channels, floored by min_wait.

    Raises
    ------
    ValueError
        If no channel contributes a wait constraint and min_wait is not
        provided, since the sweep would have no cadence information.
    """

    contributions = []
    for ch in channels:
        cfg = ch.config
        w = cfg.settle_time
        if cfg.max_rate is not None and cfg.max_step is not None:
            w += cfg.max_step / cfg.max_rate
        contributions.append(w)

    any_constrained = any(
        ch.config.max_rate is not None or ch.config.settle_time > 0
        for ch in channels
    )

    if not any_constrained and min_wait is None:
        raise ValueError(
            "No channel has a max_rate or settle_time, and no min_wait was "
            "supplied. Cannot determine sweep cadence."
        )

    derived = max(contributions) if contributions else 0.0
    if min_wait is not None:
        derived = max(derived, min_wait)
    return derived


"""Workhorse"""

def _tandemSweep(
    channels         : List[SweepableChannel],
    start            : np.ndarray,
    end              : np.ndarray,
    wait             : float,
    callback         : Callable[[np.ndarray], Any] | None = None,
    panic_condition  : Callable[..., bool] = lambda *_,**__: False,
    panic_behavior   : str | Callable[..., Any] | None = 'zero',
    handle_exceptions: bool = True,
    ignore_checkpoints: bool = True,
    verbose          : bool = False
) -> SweepResult:

    """
    Sweep channels from explicit start values to end values.

    This is the internal workhorse. Callers that maintain a position cache
    (e.g. param_sweep_measure) should call this directly to avoid an
    unnecessary hardware query at the start of every step.

    Parameters
    ----------
    channels : list of SweepableChannel
        Channels to sweep. Step constraints are read from each channel's
        SweepConfig.
    start : ndarray
        Starting values, one per channel. Caller is responsible for accuracy.
    end : ndarray
        Target values, one per channel.
    wait : float
        Per-step wait in seconds. Caller is responsible for deriving this
        (see _derive_wait).
    callback : Callable, optional
        Called once per step with the current settings array after all
        channels have been set for that step.
    panic_condition : Callable, optional
        Called with the current settings array after each step. Returns True
        if a panic condition is detected.
    panic_behavior : str or Callable or None, default='zero'
        'zero'  — sweep back to start and return PANICKED.
        'stop'  — return PANICKED immediately without sweeping back.
        Callable — called with the current settings; sweep continues.
    handle_exceptions : bool, default=True
        If a setter raises, attempt to sweep back to start before returning
        FAILED. Set False for the recursive recovery sweep to prevent loops.
    ignore_checkpoints : bool, default=True
        If False, honor ThreadJob checkpoints on each iteration.
    verbose : bool, default=False
        Print current settings on each step.

    Returns
    -------
    SweepResult
    """

    N = len(channels)
    start = np.asarray(start, dtype=float)
    end   = np.asarray(end,   dtype=float)

    if start.shape != (N,) or end.shape != (N,):
        raise IndexError("start/end length must match number of channels.")

    # Unpack constraints from SweepConfigs
    max_step  = np.array([np.inf if c.config.max_step  is None
                           else c.config.max_step  for c in channels])
    min_step  = np.array([0.0   if c.config.min_step  is None
                           else c.config.min_step  for c in channels])
    tolerance = np.array([c.config.tolerance for c in channels])

    # min_step should never be smaller than the tolerance
    min_step = np.maximum(min_step, tolerance)

    if np.any(max_step - min_step < 0):
        raise ValueError("'min_step' cannot exceed 'max_step' for any channel.")
    if np.any(min_step < 0):
        raise ValueError("'min_step' values cannot be negative.")

    # Number of steps driven by the bottleneck channel
    M = ceil(np.max(np.abs(end - start) / max_step)) if np.any(np.isfinite(max_step)) else 1

    if verbose:
        names = [ch.name or f'ch{i}' for i, ch in enumerate(channels)]
        print("".join(f'\t{n:>12}' for n in names))

    prev = start.copy()

    for j in range(M):
        if not ignore_checkpoints:
            checkpoint()

        new = start + j * (end - start) / M

        # Skip channels that haven't moved enough to bother setting
        similar = np.abs(new - prev) <= min_step
        new[similar] = prev[similar]

        # Set all channels for this step
        for i, ch in enumerate(channels):
            if prev[i] == new[i]:
                continue
            try:
                success = ch.set_output(new[i])
                # Setters may signal failure by returning False
                if isinstance(success, bool) and not success:
                    raise RuntimeError(f"setter for '{ch.name}' returned False")
            except Exception:
                name = ch.name or f'ch{i}'
                print(f"_tandemSweep: error setting '{name}'")
                if handle_exceptions:
                    print("Attempting to sweep back to start...")
                    _tandemSweep(channels, prev, start, wait,
                                 handle_exceptions=False,
                                 ignore_checkpoints=ignore_checkpoints)
                    return SweepResult.FAILED
                else:
                    raise RuntimeError(
                        f"_tandemSweep unable to recover while setting '{name}'"
                    )
            prev[i] = new[i]

        # Panic check and callback fire once per step, on the full settled state
        if panic_condition(new):
            if panic_behavior == 'zero':
                _tandemSweep(channels, prev, start, wait,
                             handle_exceptions=False,
                             ignore_checkpoints=ignore_checkpoints)
                return SweepResult.PANICKED
            elif panic_behavior == 'stop':
                return SweepResult.PANICKED
            elif callable(panic_behavior):
                panic_behavior(new)
            else:
                raise ValueError(
                    f"Invalid panic_behavior: {panic_behavior!r}"
                )

        if callback:
            callback(new.copy())

        if verbose:
            print("".join(f'\t{v:12g}' for v in new), end='\r')

        time.sleep(wait)

    # Final nudge to make sure we land within tolerance of the target
    for i, ch in enumerate(channels):
        if np.abs(prev[i] - end[i]) > tolerance[i]:
            ch.set_output(end[i])

    if verbose:
        print("".join(f'\t{v:12g}' for v in end))

    return SweepResult.SUCCEEDED

"""Public API"""

def tandemSweep(
    channels         : List[SweepableChannel],
    target           : np.ndarray | List[float] | Dict[str, float],
    min_wait         : float | None = None,
    callback         : Callable[[np.ndarray], Any] | None = None,
    panic_condition  : Callable[..., bool] = lambda *_,**__: False,
    panic_behavior   : str | Callable[..., Any] | None = 'zero',
    handle_exceptions: bool = True,
    ignore_checkpoints: bool = False,
    verbose          : bool = False
) -> SweepResult:

    """
    Sweep a list of SweepableChannels to a target, querying start values
    from hardware and deriving the step cadence from each channel's SweepConfig.

    Parameters
    ----------
    channels : list of SweepableChannel
        Channels to sweep.
    target : array-like or dict
        Target values. If a dict, keys must match channel names and missing
        channels are held at their current values. If array-like, must be
        one value per channel.
    min_wait : float, optional
        Floor on the derived per-step wait in seconds. Required if no channel
        has a max_rate or settle_time set.
    callback : Callable, optional
        Called once per step with the current settings array after all
        channels have been set.
    panic_condition : Callable, optional
        Called with the current settings array after each step. Returns True
        to trigger panic behavior.
    panic_behavior : str or Callable or None, default='zero'
        'zero'  — sweep back to start and return PANICKED.
        'stop'  — return PANICKED immediately.
        Callable — called with current settings; sweep continues.
    handle_exceptions : bool, default=True
        If a setter raises, attempt to sweep back to start.
    ignore_checkpoints : bool, default=False
        If False, honor ThreadJob checkpoints on each iteration.
    verbose : bool, default=False
        Print current settings on each step.

    Returns
    -------
    SweepResult
    """

    N = len(channels)

    # Query start values from hardware
    start = np.array([ch.get_output() for ch in channels], dtype=float)

    # Resolve target
    if isinstance(target, dict):
        names = [ch.name for ch in channels]
        if any(k not in names for k in target):
            bad = [k for k in target if k not in names]
            raise ValueError(f"Target keys not found in channel names: {bad}")
        end = np.array(
            [target.get(ch.name, start[i]) for i, ch in enumerate(channels)],
            dtype=float,
        )
    else:
        end = np.asarray(target, dtype=float)
        if end.shape != (N,):
            raise IndexError("target length must match number of channels.")

    wait = _derive_wait(channels, min_wait)

    return _tandemSweep(
        channels = channels,
        start = start,
        end = end,
        wait = wait,
        callback = callback,
        panic_condition = panic_condition,
        panic_behavior = panic_behavior,
        handle_exceptions = handle_exceptions,
        ignore_checkpoints = ignore_checkpoints,
        verbose = verbose,
    )