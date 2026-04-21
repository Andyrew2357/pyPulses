"""
Job: run a callable in a background thread with cooperative pause/stop control.

The pause/stop mechanism uses a ContextVar to propagate a Control object into
the running thread. Any code in the call stack can call checkpoint() to
cooperatively pause or stop without needing an explicit reference to the job.
"""

from __future__ import annotations

import threading
import time
import traceback
from contextvars import ContextVar
from typing import Any, Callable, List

"""Stop signal"""

class StopRequested(Exception):
    """Raised inside the job thread when stop() has been called."""
    pass

"""Control object"""

class Control:
    """
    Pause/stop state for a running job.

    Held by the Job and propagated into the thread via a ContextVar so that
    checkpoint() can access it from anywhere in the call stack.
    """

    def __init__(self):
        self._pause = threading.Event()
        self._stop  = threading.Event()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def wait_if_paused_or_stopped(self) -> None:
        """
        Block while paused. Raise StopRequested if stop has been called.
        Called by checkpoint().
        """
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.05)
        if self._stop.is_set():
            raise StopRequested()

"""ContextVar plumbing"""

_current_control: ContextVar[Control | None] = ContextVar(
    '_current_control', default=None
)

def checkpoint() -> None:
    """
    Cooperatively pause or stop the current job at this point.
    No-op if called outside a Job thread.
    """
    ctrl = _current_control.get()
    if ctrl is not None:
        ctrl.wait_if_paused_or_stopped()

"""Job"""

class Job:
    """
    Runs a callable in a background thread with cooperative pause/stop control.

    Each Job instance is single-use. Construct a new Job to re-run a function.

    Parameters
    ----------
    func : Callable
        The function to run. Called as func(*args, **kwargs).
    *args, **kwargs
        Passed through to func.

    Callbacks
    ---------
    on_start : list of Callable[[Job], Any]
        Called (in the job thread) just before func begins.
    on_finish : list of Callable[[Job, result], Any]
        Called (in the job thread) after func returns normally.
    on_stop : list of Callable[[Job], Any]
        Called (in the job thread) if the job is stopped via stop().
    on_error : list of Callable[[Job, str], Any]
        Called (in the job thread) if func raises an unexpected exception.
        The second argument is the formatted traceback string.

    All callbacks are called from the job thread. Keep them short and 
    thread-safe. Exceptions in callbacks are printed but do not propagate.

    Attributes
    ----------
    result : Any
        Return value of func, populated after normal completion.
    exc : BaseException or None
        Exception raised by func, populated after an error.
    control : Control
        The pause/stop control object. Prefer calling job.pause() / job.stop()
        rather than accessing this directly.
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any):
        self.func    = func
        self.args    = args
        self.kwargs  = kwargs
        self.control = Control()

        self.result: Any              = None
        self.exc   : BaseException | None = None

        self.on_start  : List[Callable] = []
        self.on_finish : List[Callable] = []
        self.on_stop   : List[Callable] = []
        self.on_error  : List[Callable] = []

        self._thread: threading.Thread | None = None
        self._started = False

    """Control interface"""

    def pause(self) -> None:
        """Pause the job at the next checkpoint."""
        self.control.pause()

    def resume(self) -> None:
        """Resume a paused job."""
        self.control.resume()

    def stop(self) -> None:
        """Request the job to stop at the next checkpoint."""
        self.control.stop()

    @property
    def is_paused(self) -> bool:
        return self.control.is_paused

    @property
    def is_stopped(self) -> bool:
        return self.control.is_stopped

    def is_alive(self) -> bool:
        """Return True if the job thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    """Lifecycle"""
    
    def start(self) -> 'Job':
        """
        Start the job in a background thread.

        Returns self for chaining:
            job = Job(func).start()
        """
        if self._started:
            raise RuntimeError(
                "This Job has already been started. "
                "Construct a new Job to re-run the function."
            )
        self._started = True

        from .sidecar import Sidecar
        sidecar = Sidecar.instance()
        if sidecar is not None:
            sidecar.register_job(self)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        """
        Wait for the job to finish.

        Uses a short-timeout polling loop so the call remains interruptible
        via KeyboardInterrupt and does not block widget comm delivery.

        Parameters
        ----------
        timeout : float or None
            Maximum seconds to wait. None means wait indefinitely.
        """
        if self._thread is None:
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._thread.is_alive():
            remaining = None if deadline is None \
                else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining == 0.0:
                break
            self._thread.join(timeout=min(0.1, remaining) if remaining is not None else 0.1)

    """Internal"""

    def _fire(self, callbacks: List[Callable], *args: Any) -> None:
        """Fire a list of callbacks, printing but not propagating exceptions."""
        for cb in callbacks:
            try:
                cb(*args)
            except Exception:
                traceback.print_exc()

    def _run(self) -> None:
        token = _current_control.set(self.control)
        try:
            self._fire(self.on_start, self)
            self.result = self.func(*self.args, **self.kwargs)
            self._fire(self.on_finish, self, self.result)

        except StopRequested:
            self._fire(self.on_stop, self)

        except Exception as e:
            self.exc = e
            self._fire(self.on_error, self, traceback.format_exc())

        finally:
            _current_control.reset(token)