import numpy as np
from typing import Any, Callable, Protocol, runtime_checkable, Tuple

@runtime_checkable
class BoolChannel(Protocol):
    """A channel that returns a boolean value."""
    def get_output(self) -> bool: ...
    def set_output(self, value: bool) -> None: ...
    def __call__(self, value: bool | None = None) -> bool: ...

@runtime_checkable
class ScalarChannel(Protocol):
    """Anything that can get/set a scalar value."""
    def get_output(self) -> float: ...
    def set_output(self, value: float) -> None: ...
    def __call__(self, v: float | None = None) -> float | None: ...

@runtime_checkable
class ChannelizedHardware(Protocol):
    """Protocol for hardware that exposes channels."""
    
    def resolve(self, accessor: str) -> ScalarChannel: ...

@runtime_checkable 
class SwitchableChannel(Protocol):
    """Extends ScalarChannel with enable/disable."""
    def get_output(self) -> float: ...
    def set_output(self, value: float) -> None: ...
    def get_enable(self) -> bool: ...
    def set_enable(self, enabled: bool) -> None: ...
    def __call__(self, v: float | None = None) -> float | None: ...

@runtime_checkable 
class CompPair(Protocol):
    """Complementary Digital Timing Pairs (e.g. dtgCompPair)"""

    def Xlow(self, v: float | None = None) -> float | None: ...
    def Xhigh(self, v: float | None = None) -> float | None: ...
    def Ylow(self, v: float | None = None) -> float | None: ...
    def Yhigh(self, v: float | None = None) -> float | None: ...
    def xpolarity(self, pos: bool | None = None) -> bool | None: ...
    def ypolarity(self, pos: bool | None = None) -> bool | None: ...
    def enable(self, on: bool | None = None) -> bool | None: ...
    def ldelay(self, v: float | None = None) -> float | None: ...
    def ldoff(self, v: float | None = None) -> float | None: ...
    def tdelay(self, v: float | None = None) -> float | None: ...
    def tdoff(self, v: float | None = None) -> float | None: ...
    def width(self, v: float | None = None) -> float | None: ...

@runtime_checkable
class ScopeChannel(Protocol):
    """A scope channel that returns (waveform, dt, t0)."""
    def __call__(self) -> Tuple[np.ndarray, float, float]: ...

@runtime_checkable
class ErrorChannel(Protocol):
    """A channel that returns (error, variance) for balancing."""
    def __call__(self) -> Tuple[float, float]: ...

@runtime_checkable
class OffsetChannel(Protocol):
    """A channel that returns a dynamic offset value."""
    def __call__(self) -> float: ...

@runtime_checkable
class LockInChannel(Protocol):
    """
    A channel that returns a lock-in mean reading and its covariance.
 
    Returns
    -------
    mean : ndarray, shape (2,)
        [X, Y] mean reading.
    cov : ndarray, shape (2, 2)
        Covariance matrix of the mean.
    """
    def __call__(self) -> Tuple[np.ndarray, np.ndarray]: ...
 
@runtime_checkable
class CommandChannel(Protocol):
    """
    A channel that executes an immediate command with no arguments and
    no meaningful return value. Typical use: triggering a lock-in
    auto-gain or auto-range adjustment.
    """
    def __call__(self) -> None: ...

class ChannelAdapter():
    """
    Base class for channel adapters.
    """
    def __init__(self, parent, accessor: str):
        self._parent = parent
        self._accessor = accessor

    @property
    def parent(self):
        return self._parent
    
    @property
    def accessor(self) -> str:
        return self._accessor

    def format_ref(self):
        return self._parent, self._accessor

class BoolChannelAdapter(ChannelAdapter):
    
    def __init__(self, 
        parent, 
        accessor: str, 
        getter: Callable[[], bool], 
        setter: Callable[[bool], Any]
    ):
        super().__init__(parent, accessor)
        self._getter = getter
        self._setter = setter

    def get_output(self) -> bool:
        return self._getter()
    
    def set_output(self, value: bool):
        if self._setter is None:
            return
        self._setter(value)
    
    def __call__(self, value: bool | None = None) -> bool | None:
        if value is None:
            return self._getter()
        self._setter(value)

class ScalarChannelAdapter(ChannelAdapter):
    
    def __init__(self, parent, accessor: str):
        super().__init__(parent, accessor)
    
    def get_output(self) -> float: ...
    def set_output(self, value: float) -> None: ...

    def __call__(self, value: float | None = None) -> float | None:
        if value is None:
            return self.get_output()
        self.set_output(value)

class ScopeChannelAdapter(ChannelAdapter):
    
    def __init__(self, parent, accessor: str):
        super().__init__(parent, accessor)

    def __call__(self) -> Tuple[np.ndarray, float, float]: ...

class LockInChannelAdapter(ChannelAdapter):
    """
    Base class for lock-in channel adapters.
    Subclasses implement __call__ to return (mean, cov).
    """
    def __init__(self, parent, accessor: str):
        super().__init__(parent, accessor)
 
    def __call__(self) -> Tuple[np.ndarray, np.ndarray]: ...
 
 
class CommandChannelAdapter(ChannelAdapter):
    """
    Base class for command channel adapters.
    Subclasses implement __call__ to execute an immediate command.
    """
    def __init__(self, parent, accessor: str):
        super().__init__(parent, accessor)
 
    def __call__(self) -> None: ...