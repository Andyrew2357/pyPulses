"""
Measurement abstractions for parameter sweep experiments.

QuerySignature
    Protocol that any measurement source must satisfy.

Query
    Concrete adapter wrapping a bare callable. The everyday way to define a 
    measurement.

Measurement
    Orchestrates a list of Query objects at a single point in parameter space: 
    waits for settling, fires pre-callbacks, collects readings (threading lazy 
    queries concurrently with eager ones), fires post-callbacks, and returns a 
    flat numpy array.
"""

from __future__ import annotations

from ..devices.registry import format_reference, resolve_reference

import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import time
import numpy as np
from typing import Protocol, runtime_checkable

@runtime_checkable
class QuerySignature(Protocol):
    """
    Protocol satisfied by anything that can act as a measurement source.

    Implementing this protocol (rather than subclassing a base class) means any 
    callable-wrapping object can participate without inheriting from anything. 
    The Query dataclass satisfies this automatically.
    """

    def measure(self) -> float | np.ndarray:
        """
        Take a reading and return it.

        Returns
        -------
        float or ndarray
            Scalar for single-column sources. 1-D array of length N for sources 
            that produce N columns simultaneously (e.g. a lock-in returning X 
            and Y).
        """
        ...

    @property
    def name(self) -> str | List[str]:
        """
        Column name(s).

        A plain string for scalar sources. A list of strings, one per output 
        column, for multi-column sources. Length must match the length of the 
        array returned by measure().
        """
        ...

    @property
    def long_name(self) -> str | List[str] | None:
        """
        Display name(s), may contain LaTeX syntax for plot axis labels. Mirrors 
        the structure of `name`.
        """
        ...

    @property
    def unit(self) -> str | List[str] | None:
        """
        Physical unit string(s) (e.g. 'V', r'\\mu T'). Mirrors the structure of 
        `name`.
        """
        ...

    @property
    def lazy(self) -> bool:
        """
        If True, this query is dispatched concurrently with other lazy queries 
        while eager queries run on the main thread. Useful for slow instruments 
        where the read latency dominates.
        """
        ...

@dataclass
class Query:
    """
    Wraps a bare callable as a QuerySignature.

    This is the everyday way to define a measurement. Any object that satisfies 
    QuerySignature (e.g. a custom instrument class with a measure() method and 
    the required properties) can be used directly without going through Query.

    Parameters
    ----------
    f : Callable
        Zero-argument callable that returns a float or 1-D ndarray.
    name : str or list of str
        Column name(s). Use a list for multi-column sources.
    long_name : str or list of str or None
        Display name(s), may contain LaTeX. Mirrors structure of `name`.
    unit : str or list of str or None
        Physical unit string(s). Mirrors structure of `name`.
    lazy : bool
        Whether to dispatch this query concurrently with other lazy queries 
        during measurement.
    """

    f: Callable[[], float | np.ndarray]
    name: str | List[str] = 'unnamed'
    long_name: str | List[str] | None = None
    unit: str | List[str] | None = None
    lazy: bool = False

    def measure(self) -> float | np.ndarray:
        return self.f()

    def to_dict(self) -> dict:
        try:
            f_ref = format_reference(self.f)
        except Exception:
            warnings.warn(
                f"Query '{self.name}': callable 'f' could not be serialized "
                f"({type(self.f).__name__}). It will be stored as None and must "
                f"be restored manually."
            )
            f_ref = None
        return {
            'f': f_ref,
            'name': self.name,
            'long_name': self.long_name,
            'unit': self.unit,
            'lazy': self.lazy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Query':
        f_ref = d.get('f')
        if f_ref is not None:
            f = resolve_reference(f_ref)
            if f is None:
                warnings.warn(
                    f"Query '{d.get('name')}': could not resolve reference "
                    f"'{f_ref}'. 'f' will be None — reassign before measuring."
                )
        else:
            warnings.warn(
                f"Query '{d.get('name')}': no callable reference stored. "
                f"'f' will be None — reassign before measuring."
            )
            f = None
        return cls(
            f = f,
            name = d['name'],
            long_name = d.get('long_name'),
            unit = d.get('unit'),
            lazy = d.get('lazy', False),
        )

class Measurement:
    """
    Orchestrates a collection of Query objects at a single point in parameter 
    space.

    Responsibilities
    ----------------
    - Wait `time_per_point` seconds before measuring (settling time).
    - Fire pre-callbacks before taking readings.
    - Dispatch lazy queries concurrently with eager ones via a thread pool.
    - Collect all readings into a flat ndarray in declaration order.
    - Fire post-callbacks with the results.
    - Expose column metadata (names, long names, units) for use by the runner 
      when writing file headers or setting up live plots.

    Parameters
    ----------
    queries : list of QuerySignature
        Measurement sources in declaration order. Each may be a Query instance 
        or any object satisfying QuerySignature.
    time_per_point : float
        Seconds to wait before taking readings at each point.
    pre_callbacks : list of Callable, optional
        Called before readings are taken. Signature:
            f(idx, coords, timestamp) -> Any
        where coords is a Dict[str, float] keyed by coordinate name.
    post_callbacks : list of Callable, optional
        Called after readings are taken. Signature:
            f(idx, coords, measured, timestamp) -> Any
        where measured is a Dict[str, float] keyed by column name.
    """

    def __init__(self,
        queries: List[QuerySignature],
        time_per_point: float = 0.0,
        pre_callbacks: List[Callable] | None = None,
        post_callbacks: List[Callable] | None = None,
    ):
        self._queries = list(queries)
        self.time_per_point = time_per_point
        self._pre_callbacks = list(pre_callbacks  or [])
        self._post_callbacks = list(post_callbacks or [])

        # Parse column layout once at construction time so measure() is fast
        self._layout = self._parse_layout(self._queries)

    """Column metadata"""

    @staticmethod
    def _parse_layout(
        queries: List[QuerySignature],
    ) -> List[Tuple[int, int, QuerySignature]]:
        """
        Return a list of (start_col, width, query) tuples describing how each 
        query maps into the flat output array.
        """
        layout = []
        col = 0
        for q in queries:
            n = q.name
            width = 1 if isinstance(n, str) else len(n)
            layout.append((col, width, q))
            col += width
        return layout

    @property
    def num_cols(self) -> int:
        """Total number of output columns across all queries."""
        if not self._layout:
            return 0
        last_start, last_width, _ = self._layout[-1]
        return last_start + last_width

    @property
    def col_names(self) -> List[str]:
        """Flat list of column names in declaration order."""
        names = []
        for _, _, q in self._layout:
            n = q.name
            if isinstance(n, str):
                names.append(n)
            else:
                names.extend(n)
        return names

    @property
    def col_long_names(self) -> List[str | None]:
        """Flat list of long (display) names, None where unset."""
        result = []
        for _, width, q in self._layout:
            ln = q.long_name
            if ln is None:
                result.extend([None] * width)
            elif isinstance(ln, str):
                result.append(ln)
            else:
                result.extend(ln)
        return result

    @property
    def col_units(self) -> List[str | None]:
        """Flat list of unit strings, None where unset."""
        result = []
        for _, width, q in self._layout:
            u = q.unit
            if u is None:
                result.extend([None] * width)
            elif isinstance(u, str):
                result.append(u)
            else:
                result.extend(u)
        return result

    """Callback management"""

    def add_pre_callback(self, f: Callable) -> None:
        """Append a pre-callback."""
        self._pre_callbacks.append(f)

    def remove_pre_callback(self, f: Callable) -> None:
        """Remove a pre-callback by identity."""
        self._pre_callbacks.remove(f)

    def add_post_callback(self, f: Callable) -> None:
        """Append a post-callback."""
        self._post_callbacks.append(f)

    def remove_post_callback(self, f: Callable) -> None:
        """Remove a post-callback by identity."""
        self._post_callbacks.remove(f)

    """Core measurement"""

    def _collect(self) -> np.ndarray:
        """
        Take readings from all queries, threading lazy ones concurrently with 
        eager ones. Returns a flat ndarray of length num_cols.
        """
        result = np.full(self.num_cols, fill_value=np.nan, dtype=float)

        eager = [(s, w, q) for s, w, q in self._layout if not q.lazy]
        lazy = [(s, w, q) for s, w, q in self._layout if q.lazy]

        if lazy:
            with ThreadPoolExecutor(max_workers=len(lazy)) as ex:
                futures = [(s, w, ex.submit(q.measure)) for s, w, q in lazy]

                # Run eager queries on the main thread while lazy ones execute
                for s, w, q in eager:
                    result[s:s + w] = np.atleast_1d(q.measure())

                for s, w, fut in futures:
                    result[s:s + w] = np.atleast_1d(fut.result())
        else:
            for s, w, q in eager:
                result[s:s + w] = np.atleast_1d(q.measure())

        return result

    def measure(self,
        idx: np.ndarray,
        coords: Dict[str, float],
        timestamp: datetime.datetime | None,
    ) -> np.ndarray:
        """
        Settle, fire pre-callbacks, collect readings, fire post-callbacks.

        Parameters
        ----------
        idx : ndarray
            Index of the current point within the sweep, as provided by the 
            sweep iterator.
        coords : dict
            Current coordinate values keyed by name, as provided by the runner.
        timestamp : datetime or None
            Timestamp for this point, as provided by the runner. None if the 
            runner has timestamps disabled.

        Returns
        -------
        ndarray
            Flat array of length `num_cols` with all readings in declaration 
            order.
        """

        # Settling wait
        if self.time_per_point > 0:
            time.sleep(self.time_per_point)

        # Pre-callbacks
        for cb in self._pre_callbacks:
            cb(idx, coords, timestamp)

        # Collect readings
        measured = self._collect()

        # Build named dict for post-callbacks
        measured_dict: Dict[str, float] = dict(zip(self.col_names, measured))

        # Post-callbacks
        for cb in self._post_callbacks:
            cb(idx, coords, measured_dict, timestamp)

        return measured
    
    """Serialization"""

    def to_dict(self) -> dict:
        serialized_callbacks = []
        for cb_list, label in [
            (self._pre_callbacks, 'pre_callback'),
            (self._post_callbacks, 'post_callback'),
        ]:
            refs = []
            for cb in cb_list:
                try:
                    refs.append(format_reference(cb))
                except Exception:
                    warnings.warn(
                        f"Measurement {label} '{getattr(cb, '__name__', repr(cb))}' "
                        f"could not be serialized and will be omitted."
                    )
            serialized_callbacks.append(refs)

        pre_refs, post_refs = serialized_callbacks
        return {
            'queries': [q.to_dict() for q in self._queries],
            'time_per_point': self.time_per_point,
            'pre_callbacks': pre_refs,
            'post_callbacks': post_refs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Measurement':
        queries = [Query.from_dict(q) for q in d['queries']]

        def _resolve_callbacks(refs: list, label: str) -> list:
            callbacks = []
            for r in refs:
                cb = resolve_reference(r)
                if cb is None:
                    warnings.warn(
                        f"Measurement {label}: could not resolve reference '{r}'. "
                        f"This callback will be skipped."
                    )
                else:
                    callbacks.append(cb)
            return callbacks

        return cls(
            queries = queries,
            time_per_point = d.get('time_per_point', 0.0),
            pre_callbacks = _resolve_callbacks(d.get('pre_callbacks',  []), 'pre_callback'),
            post_callbacks = _resolve_callbacks(d.get('post_callbacks', []), 'post_callback'),
        )