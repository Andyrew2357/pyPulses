"""
Typing support for the shared, vendor-neutral lock-in interface.
"""

from __future__ import annotations
from typing import Protocol, Tuple

import numpy as np


class LockinLike(Protocol):
    """
    Structural interface that `lockin.gui.LockinGUI` and the channel adapters
    in `lockin.channels` actually depend on, derived from their real usage of
    `self._instrument` / `parent` (not from guesswork).

    Any lock-in driver -- SRS SR8xx/SR850, AMETEK/Signal Recovery 7280, or
    otherwise -- that implements these members can be used with the shared
    GUI and channel-adapter layer.

    A number of additional attributes are optional and merely probed with
    `hasattr()` rather than required here: `sens_vals`, `tau_vals`,
    `irng_vals`, and the various per-model setting names in
    `gui._ALL_SETTINGS` / `gui._ACTION_METHODS` (e.g. `input_configuration`,
    `time_constant`, `reference_trigger`, ...).
    """

    out_aux_channels: list

    def get_xy(self) -> Tuple[float, float]: ...
    def get_rt(self) -> Tuple[float, float]: ...
    def aux_output(self, idx: int, value: float) -> None: ...
    def input_sensitivity(self, val: float = None, units: str = "V") -> float: ...
    def get_average(self, auto_rescale: bool = False) -> Tuple[np.ndarray, np.ndarray]: ...
    def get_average_series_correlated(
        self, auto_rescale: bool = False, L: int = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    def _serialize_state(self) -> dict: ...
