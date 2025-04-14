from ..utils import balance1d, BalanceConfig, RootFinderStatus, RootFinderState
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union

@dataclass
class PulsedR():
    Vg_predictor        : object
    Vy_predictor        : object
    Vg_balance_config   : BalanceConfig
    Vy_balance_config   : BalanceConfig
    set_Vy              : Callable[[float], Any]
    set_Vg              : Callable[[float], Any]
    background_mean     : float = 0.0
    background_std      : float = 0.0
    logger              : Optional[object] = None

    def __post_init__(self):
        self.get_y = self.Vy_balance_config.get_y
        if not self.background_mean == 0:
            self.set_background(self.background_mean)

    def balance(self, p: Union[float, Tuple[float, ...]]) -> bool:
        self.info(
            f"Attempting balance at point:\n    |"
            + ''.join([f"{x:.5f}|" for x in p])
        )

        Vy_guess = self.Vy_predictor.predict0(p)
        Vg_guess = self.Vg_predictor.predict0(p)

        self.info(
            f"Predicted balance point:\n"
          + f"    Vy_guess = {Vy_guess:.5f} V\n"
          + f"    Vg_guess = {Vg_guess:.5f} V"
        )

        min_Vy, max_Vy = self.Vy_balance_config.search_range
        min_Vg, max_Vg = self.Vg_balance_config.search_range
        if (not min_Vy <= Vy_guess <= max_Vy) or \
            (not min_Vg <= Vg_guess <= max_Vg):

            self.info("Predicted balance out of bounds; Truncating...")

            Vy_guess = max(min(max_Vy, Vy_guess), min_Vy)
            Vg_guess = max(min(max_Vg, Vg_guess), min_Vg)

            self.info(
                f"Truncated predicted balance:\n"
              + f"    Vy_guess = {Vy_guess:.5f} V\n"
              + f"    Vg_guess = {Vg_guess:.5f} V"
            )

        self.set_Vy(Vy_guess)
        self.set_Vg(Vg_guess)

        self.info("Attempting to balance using pulse heights...")
        Vy_balance_state = balance1d(p, self.Vy_balance_config)
        self.info("Result:\n" + str(Vy_balance_state))

        if Vy_balance_state.status == RootFinderStatus.CONVERGED:
            self.info(
                "Successfully balanced using pulse heights.\n"
              + f"    Vy = {Vy_balance_state.root}\n"
              + f"    Vg = {Vg_guess}"
            )

            self.Vy_predictor.update(p, Vy_balance_state.root)
            self.Vg_predictor.update(p, Vg_guess)
            return True

        self.info(
            "Unable to balance using pulse heights alone.\n"
          + "Setting Vy back to Vy_guess and Rebalancing gate..."
        )

        self.set_Vy(Vy_guess)
        Vg_balance_state = balance1d(p, self.Vg_balance_config)
        self.info("Result:\n" + str(Vg_balance_state))

        if not Vg_balance_state.status == RootFinderStatus.CONVERGED:
            self.info("Failed to rebalance the gate.")
            return False
        
        self.info("Successfully rebalanced the gate.")

        if Vy_balance_state.best_value < self.Vy_balance_config.y_tolerance:
            self.info(
                "Rebalanced y is within allowed tolerance; Forgo refinement."
            )

            Vy_balance_state = RootFinderState(
                status      = RootFinderStatus.CONVERGED,
                point       = Vy_guess,
                root        = Vy_guess,
                iterations  = 0,
                message     = "Converged without having to fine tune",
                best_value  = Vg_balance_state.best_value
            )
        else:
            self.info("Attempting to balance using pulse heights...")
            Vy_balance_state = balance1d(p, self.Vy_balance_config)

        self.info("Result:\n" + str(Vy_balance_state))

        if not Vy_balance_state == RootFinderStatus.CONVERGED:
            self.info("Failed to balance.")
            return False
        
        self.info(
            "Successfully balanced using pulse heights.\n"
          + f"    Vy = {Vy_balance_state.root}\n"
          + f"    Vg = {Vg_guess}"
        )

        self.Vy_predictor.update(p, Vy_balance_state.root)
        self.Vg_predictor.update(p, Vg_guess)
        return True

    def calibrate_background(self, samples: int = 300):
        """
        Take samples using self.get_y to determine the subtracted background
        """
        if self.logger:
            self.logger.info(f"Calibrating background ({samples} samples)...")

        # take samples of get_y and find the mean and standard deviation
        background = np.array([self.get_y() for _ in 
                               range(samples)])
        self.background_mean = background.mean()
        self.background_std  = background.std()
        
        if self.logger:
            self.logger.info(
                f"Calibrated background\n"
              + f"    mean = {self.background_mean}\n"
              + f"    std  = {self.background_std}"
            )
        
        # update the balance configs
        self.set_background(self.background_mean)

    def set_background(self, background_mean: float):
        """
        Modify the balance config get_y functions to subtract off background
        """
        # update the balance configs
        self.VY1_balance_config.get_y   = self.get_y() - background_mean
        self.VGref_balance_config.get_y = self.get_y() - background_mean

    def info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
