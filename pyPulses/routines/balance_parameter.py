from ..utils import getSetter

from abc import abstractmethod
from typing import Any, Callable, Tuple


class balanceKnob():
    def __init__(self,
        bounds: Tuple[float, float],
        guess: float,
        l: float, 
        h: float,
        f: Callable[[float | None], float | None],
        set: Callable[[float], Any] | None = None,
        absolute: bool = True,
        guess_independent: bool = False,
    ):
      
        self.guess = guess
        self.l = l
        self.h = h
        self.absolute = absolute
        self.guess_independent = guess_independent

        self.min = min(bounds)
        self.max = max(bounds)
        
        if set is None:
            self.f = f
        else:
            self.f = getSetter(f, set)

    def set_val(self, v: float):
        if v < self.min or v > self.max:
            raise ValueError(f"Value {v} out of bounds [{self.l}, {self.h}]")
        return self.f(v)

    def get_val(self) -> float:
        return self.f()
    
    def set_low(self):
        if self.guess_independent:
            self.f(self.l)
        else:
            if self.absolute:
                self.f(self.guess + self.l)
            else:
                self.f(self.guess * self.l)

    def set_high(self):
        if self.guess_independent:
            self.f(self.h)
        else:
            if self.absolute:
                self.f(self.guess + self.h)
            else:
                self.f(self.guess * self.h)

    def set_bool(self, high: bool):
        if high:
            self.set_high()
        else:
            self.set_low()

    def update_guess(self, v: float):
        self.guess = v

class balanceError():
    def __init__(self):
        self.error: float | None = None
        self.error_variance: float | None = None

    @abstractmethod
    def __call__(self) -> Tuple[float, float]: ...

    def get_error(self) -> float | None:
        return self.error
    
    def get_variance(self) -> float | None:
        return self.error_variance