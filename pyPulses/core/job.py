"""
Job + JobQueue: serial execution of background work with cooperative
pause/stop and an automatic FIFO queue.

Public surface
--------------
Job(func, *args, **kwargs)         — wrap a callable
job.start()                        — submit to the global JobQueue
JobQueue.instance()                — the session-wide singleton
JobQueue.instance().current        — the contextually relevant job
JobQueue.instance().pending        — list of jobs waiting to run
checkpoint()                       — cooperative pause/stop point
"""

from __future__ import annotations

import threading
import time
import traceback
from contextvars import ContextVar
from enum import Enum
from typing import Any, Callable, List

"""
Stop signal + control
"""

class StopRequested(Exception):
    """Raised inside the job thread when stop() has been called."""
    pass


class JobState(Enum):
    PENDING   = 'pending'     # in the queue, not yet running
    RUNNING   = 'running'
    PAUSED    = 'paused'
    FINISHED  = 'finished'    # normal completion
    STOPPED   = 'stopped'     # user-initiated stop
    ERRORED   = 'errored'
    CANCELLED = 'cancelled'   # removed from queue before it ran


class Control:
    """Pause/stop state for a running job."""

    def __init__(self):
        self._pause = threading.Event()
        self._stop  = threading.Event()

    def pause(self)  -> None: self._pause.set()
    def resume(self) -> None: self._pause.clear()
    def stop(self)   -> None: self._stop.set()

    @property
    def is_paused(self)  -> bool: return self._pause.is_set()
    @property
    def is_stopped(self) -> bool: return self._stop.is_set()

    def wait_if_paused_or_stopped(self) -> None:
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.05)
        if self._stop.is_set():
            raise StopRequested()


"""
ContextVar plumbing
"""

_current_control: ContextVar[Control | None] = ContextVar(
    '_current_control', default=None
)


def checkpoint() -> None:
    """Cooperatively pause or stop the current job. No-op outside a Job."""
    ctrl = _current_control.get()
    if ctrl is not None:
        ctrl.wait_if_paused_or_stopped()


"""
Job
"""

class Job:
    """
    A unit of work submitted to the global JobQueue.

    Calling .start() does NOT immediately spawn a thread — it submits the job
    to JobQueue.instance(), which runs jobs serially in submission order.
    A queued job stays in state PENDING until the queue pops it.

    Parameters
    ----------
    func : Callable
    *args, **kwargs : passed to func
    name : str, optional
        Human-readable label shown in the sidecar. Defaults to func.__name__.

    Callbacks (all called on the job thread; exceptions are printed not raised):
        on_start  : Callable[[Job], Any]
        on_finish : Callable[[Job, result], Any]
        on_stop   : Callable[[Job], Any]
        on_error  : Callable[[Job, traceback_str], Any]
    """

    def __init__(self,
        func: Callable,
        *args: Any,
        name: str | None = None,
        **kwargs: Any,
    ):
        self.func    = func
        self.args    = args
        self.kwargs  = kwargs
        self.name    = name or getattr(func, '__name__', 'job')
        self.control = Control()

        self.state : JobState = JobState.PENDING
        self.result: Any                  = None
        self.exc   : BaseException | None = None

        self.submitted_at: float | None = None
        self.started_at  : float | None = None
        self.finished_at : float | None = None

        self.on_start  : List[Callable] = []
        self.on_finish : List[Callable] = []
        self.on_stop   : List[Callable] = []
        self.on_error  : List[Callable] = []

        self._thread : threading.Thread | None = None
        self._submitted = False

    """Convenience pass-throughs"""

    def pause(self) -> None:
        """Pause the job at the next checkpoint."""
        self.control.pause()
        if self.state == JobState.RUNNING:
            self.state = JobState.PAUSED
            JobQueue.instance()._notify()

    def resume(self) -> None:
        """Resume a paused job."""
        self.control.resume()
        if self.state == JobState.PAUSED:
            self.state = JobState.RUNNING
            JobQueue.instance()._notify()

    def stop(self)   -> None: self.control.stop()

    @property
    def is_paused(self) -> bool:
        return self.state == JobState.PAUSED or self.control.is_paused
    @property
    def is_stopped(self) -> bool:
        return self.control.is_stopped
    
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        """Seconds spent in the RUNNING/PAUSED state (0 if not yet started)."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    @property
    def queued_for(self) -> float:
        """Seconds spent in the queue before starting."""
        if self.submitted_at is None:
            return 0.0
        end = self.started_at if self.started_at is not None else time.time()
        return end - self.submitted_at

    """Lifecycle"""

    def start(self) -> 'Job':
        """Submit to the global JobQueue. Returns self for chaining."""
        if self._submitted:
            raise RuntimeError(
                "This Job has already been submitted. "
                "Construct a new Job to re-run the function."
            )
        self._submitted = True
        JobQueue.instance().submit(self)
        return self

    def cancel(self) -> bool:
        """
        Remove a still-pending job from the queue.

        Returns True if the job was cancelled, False if it had already started.
        """
        return JobQueue.instance().cancel(self)

    def join(self, timeout: float | None = None) -> None:
        """
        Wait for the job to finish (running or queued).

        Polls in 0.1 s slices so the call stays interruptible.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.state in (JobState.PENDING, JobState.RUNNING, JobState.PAUSED):
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(0.1)

    """Internal"""

    def _fire(self, callbacks: List[Callable], *args: Any) -> None:
        for cb in callbacks:
            try:
                cb(*args)
            except Exception:
                traceback.print_exc()

    def _run(self) -> None:
        """Executed on the queue's worker thread."""
        token = _current_control.set(self.control)
        self.started_at = time.time()
        self.state = JobState.RUNNING
        JobQueue.instance()._notify()
        try:
            self._fire(self.on_start, self)
            self.result = self.func(*self.args, **self.kwargs)
            self.state = JobState.FINISHED
            self._fire(self.on_finish, self, self.result)

        except StopRequested:
            self.state = JobState.STOPPED
            self._fire(self.on_stop, self)

        except Exception as e:
            self.exc = e
            self.state = JobState.ERRORED
            self._fire(self.on_error, self, traceback.format_exc())

        finally:
            self.finished_at = time.time()
            _current_control.reset(token)


"""
JobQueue
"""

class JobQueue:
    """
    Session-wide serial job queue.

    Singleton. Access via JobQueue.instance(). Jobs submitted via Job.start()
    run one at a time in submission order. Each successor is launched only
    if its predecessor finished normally (FINISHED). On STOPPED or ERRORED
    the queue drains by default; set continue_on_error / continue_on_stop
    to override.

    Observers are callables taking no arguments, invoked whenever queue state
    changes (current swap, pending mutate, job state transition).
    """

    _instance: 'JobQueue | None' = None

    def __init__(self):
        self._lock = threading.RLock()
        self._current: Job | None = None
        self._pending: List[Job]  = []
        self._observers: List[Callable[[], None]] = []
        self.continue_on_error: bool = False
        self.continue_on_stop : bool = False

    @classmethod
    def instance(cls) -> 'JobQueue':
        if cls._instance is None:
            cls._instance = JobQueue()
        return cls._instance

    """Public API"""

    @property
    def current(self) -> Job | None:
        """The contextually relevant job."""
        return self._current

    @property
    def pending(self) -> List[Job]:
        with self._lock:
            return list(self._pending)

    def submit(self, job: Job) -> None:
        """Append a job to the queue and start it if no job is running."""
        with self._lock:
            job.submitted_at = time.time()
            job.state = JobState.PENDING

            # Wire terminal handlers so the queue advances automatically.
            job.on_finish.append(lambda j, _r: self._on_done(j, success=True))
            job.on_stop.append(lambda j: self._on_done(j, success=False, reason='stopped'))
            job.on_error.append(lambda j, _t: self._on_done(j, success=False, reason='errored'))

            if self._current is None:
                self._launch(job)
            else:
                self._pending.append(job)

        self._notify()

    def cancel(self, job: Job) -> bool:
        """Remove a pending job. No effect on a running job; use job.stop()."""
        with self._lock:
            if job in self._pending:
                self._pending.remove(job)
                job.state = JobState.CANCELLED
                self._notify()
                return True
            return False

    def clear_pending(self) -> int:
        """Cancel every pending job. Returns the number cancelled."""
        with self._lock:
            n = len(self._pending)
            for j in self._pending:
                j.state = JobState.CANCELLED
            self._pending.clear()
        self._notify()
        return n

    def move(self, job: Job, new_index: int) -> bool:
        """
        Reorder a pending job. `new_index` is its target position within the
        pending list (0 = next to run). Indices are clamped to the valid range.
        Returns True if the move was applied, False if the job is not pending.
        """
        with self._lock:
            if job not in self._pending:
                return False
            self._pending.remove(job)
            new_index = max(0, min(new_index, len(self._pending)))
            self._pending.insert(new_index, job)
        self._notify()
        return True

    def find_pending(self, job_id: int) -> Job | None:
        """Look up a pending job by id(job)."""
        with self._lock:
            for j in self._pending:
                if id(j) == job_id:
                    return j
        return None

    """Observer plumbing"""

    def add_observer(self, f: Callable[[], None]) -> None:
        with self._lock:
            self._observers.append(f)

    def remove_observer(self, f: Callable[[], None]) -> None:
        with self._lock:
            try:
                self._observers.remove(f)
            except ValueError:
                pass

    def _notify(self) -> None:
        """Fire all observers. Exceptions are printed, never propagated."""
        with self._lock:
            obs = list(self._observers)
        for f in obs:
            try:
                f()
            except Exception:
                traceback.print_exc()

    """Internal lifecycle"""

    def _launch(self, job: Job) -> None:
        """Mark `job` as current and spawn its worker thread. Caller holds lock."""
        self._current = job
        job._thread = threading.Thread(
            target=job._run,
            daemon=True,
            name=f'Job:{job.name}',
        )
        job._thread.start()

    def _on_done(self, job: Job, success: bool, reason: str | None = None) -> None:
        """
        Terminal-callback hook fired by Job._run. Decide whether to advance
        the queue and, if so, launch the next pending job.
        """
        with self._lock:
            if self._current is job:
                self._current = None

            should_advance = (
                success
                or (reason == 'stopped'  and self.continue_on_stop)
                or (reason == 'errored'  and self.continue_on_error)
            )

            if not should_advance:
                # Drain pending jobs.
                for j in self._pending:
                    j.state = JobState.CANCELLED
                self._pending.clear()
            elif self._pending:
                next_job = self._pending.pop(0)
                self._launch(next_job)

        self._notify()