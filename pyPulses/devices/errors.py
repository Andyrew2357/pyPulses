"""
Recoverability contract shared by instrument drivers and the scan runner.

Only the layer that performed a recovery knows whether the instrument is
back in a consistent state, so that layer decides. An exception is
*recoverable* when its raiser guarantees:

    1. The data from this call is lost.
    2. The instrument and link have been restored to a known, consistent
       state (buffers cleared, no stale responses queued, host-side state
       such as an acquisition flag reset).

Under that guarantee it is safe for a caller to retry the call or to move on
without it. Everything else, including exceptions that merely *look*
transient, is treated as fatal. This is an allowlist: a new failure mode is
fatal until a driver explicitly opts it in.

Two ways to opt in:

    raise SomeRecoverableError(...)     # subclass RecoverableError
    raise mark_recoverable(exc)         # annotate an existing exception
                                        # without changing its type
"""

from __future__ import annotations


class RecoverableError(Exception):
    """Base class for failures after which the instrument is known-good."""


def mark_recoverable(exc: BaseException) -> BaseException:
    """
    Flag an existing exception as recoverable without changing its type, so
    callers that catch e.g. pyvisa.errors.VisaIOError keep working.
    Returns the exception so it can be used as `raise mark_recoverable(e)`.
    """
    exc.recoverable = True
    return exc


def is_recoverable(exc: BaseException) -> bool:
    """True if `exc` carries the recoverability guarantee above."""
    if not isinstance(exc, Exception):
        # KeyboardInterrupt, SystemExit, job-cancellation BaseExceptions, ...
        return False
    return isinstance(exc, RecoverableError) or getattr(exc, 'recoverable', False) is True