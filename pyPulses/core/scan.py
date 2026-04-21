"""
Scan objects: describe and execute traversals through parameter space.

Architecture
------------
ScanBase (abstract)
    Owns channels, position cache, move_to, space_mask, and the __add__ / 
    __mul__ algebra. Subclasses implement _iter() and expose npoints / 
    dimensions.

Interior nodes (constructed only via arithmetic):
    _ScanSum      result of scan_a + scan_b  (concatenation)
    _ScanProduct  result of scan_a * scan_b  (outer product)

Leaf nodes (constructed directly by the user):
    ScanCut            linear sweep from start to end
    ScanPoints         explicit list of points
    ScanGrid           outer product of per-channel axes
    ScanParallelepiped parallelepiped in parameter space
    ScanArbitrary      user-supplied iterator

The runner calls scan.move_to(coords) to move hardware, and scan._iter() to walk 
through the trajectory. Masks are callables Dict[str, float] -> bool and may be 
attached to any node.
"""

from __future__ import annotations

from .tandem_sweep import _tandemSweep, _derive_wait
from .mask import Expr
from ..devices.sweepable_channel import SweepableChannel
from ..devices.registry import format_reference, resolve_reference

import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Tuple
import numpy as np

"""Type aliases"""

Coords    = Dict[str, float]
IdxCoords = Tuple[np.ndarray, Coords]

_SCAN_REGISTRY: Dict[str, type] = {}

def _register_scan(tag: str):
    def decorator(cls):
        cls._scan_type = tag
        _SCAN_REGISTRY[tag] = cls
        return cls
    return decorator

"""Abstract base"""

class ScanBase(ABC):
    """
    Abstract base class for all scan objects.

    Parameters
    ----------
    channels : list of SweepableChannel
        Hardware channels swept by this scan. Names must be unique.
    space_mask : Callable[[Coords], bool], optional
        Called for each candidate point inside _iter. If it returns False the 
        point is skipped entirely. Because masking happens inside the iterator, 
        masks on child nodes remain active after composition.
    ramp_kwargs : dict, optional
        Extra keyword arguments forwarded to _tandemSweep (e.g.
        panic_condition, panic_behavior).
    """

    _scan_type: str

    def __init__(self,
        channels: List[SweepableChannel],
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        self._channels   = list(channels)
        self._validate_channel_names(self._channels)
        self.space_mask = space_mask
        self.ramp_kwargs = ramp_kwargs or {}

        # Position cache: keyed by channel name, None until first move
        self._current_coords: Coords | None = None

    """Channel access"""

    @property
    def channels(self) -> List[SweepableChannel]:
        return list(self._channels)

    @property
    def coord_names(self) -> List[str]:
        """Ordered list of coordinate names, one per channel."""
        return [ch.name for ch in self._channels]

    def _channel_map(self) -> Dict[str, SweepableChannel]:
        return {ch.name: ch for ch in self._channels}

    @staticmethod
    def _validate_channel_names(channels: List[SweepableChannel]) -> None:
        names = [ch.name for ch in channels]
        if len(names) != len(set(names)):
            seen = set()
            dupes = [n for n in names if n in seen or seen.add(n)]
            raise ValueError(f"Duplicate channel names in scan: {dupes}")
        if any(n is None for n in names):
            raise ValueError(
                "All channels in a scan must have a name. "
                "Set the 'name' attribute on each SweepableChannel."
            )

    """Abstract interface"""

    @property
    @abstractmethod
    def npoints(self) -> int:
        """Total number of points in the scan (before masking)."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> np.ndarray:
        """Shape of the result array (before masking), as a 1-D int array."""
        ...

    @abstractmethod
    def _iter_unmasked(self) -> Iterator[IdxCoords]:
        """
        Yield (idx, coords) pairs for every point in the scan without applying 
        this node's space_mask. Child nodes may have already applied their own 
        masks via their own _iter calls.

        idx    : np.ndarray of int, index into the result array
        coords : Dict[str, float], target coordinate values
        """
        ...

    def _iter(self) -> Iterator[IdxCoords]:
        """
        Yield (idx, coords) pairs, skipping any point rejected by this node's 
        space_mask. Child masks have already been applied inside
        _iter_unmasked.
        """
        for idx, coords in self._iter_unmasked():
            if self.space_mask is None or self.space_mask(coords):
                yield idx, coords

    """Movement"""

    def _get_current_coords(self) -> Coords:
        """
        Return current coordinates, querying hardware only if cache is cold.
        """
        if self._current_coords is None:
            self._current_coords = {
                ch.name: ch.get_output() for ch in self._channels
            }
        return self._current_coords

    def move_to(self, target: Coords, min_wait: float | None = None) -> None:
        """
        Move all channels to the target coordinates using _tandemSweep.

        Uses the cached position as the start to avoid hardware queries.
        Updates the cache after moving.

        Parameters
        ----------
        target : dict
            Target coordinate values keyed by channel name. All channel
            names must be present.
        min_wait : float, optional
            Floor on derived wait time, forwarded to _tandemSweep.
        """
        start = self._get_current_coords()
        ordered_channels = self._channels
        start_arr = np.array([start[ch.name]  for ch in ordered_channels])
        end_arr = np.array([target[ch.name] for ch in ordered_channels])
        wait = _derive_wait(ordered_channels, min_wait)

        _tandemSweep(
            channels = ordered_channels,
            start = start_arr,
            end = end_arr,
            wait = wait,
            **self.ramp_kwargs,
        )

        self._current_coords = dict(target)

    def invalidate_cache(self) -> None:
        """Force the next move_to to query hardware for the start position."""
        self._current_coords = None

    """Preview"""

    def preview(self, coords: List[str], use_mask: bool = True, **kwargs):
        """
        Plot a preview of the path swept out by this scan, projected onto
        1, 2, or 3 named coordinate axes.

        Parameters
        ----------
        coords : list of str
            Names of the coordinate axes to project onto. Must be 1, 2, or 3
            names drawn from coord_names.
        use_mask : bool, default=True
            If True, uses _iter() so all masks in the AST are applied.
            If False, uses _iter_unmasked() to show the full unfiltered path.
        **kwargs
            Forwarded to the matplotlib plot call. Defaults: marker='o', 
            color='r', alpha=0.5.
        """
        import matplotlib.pyplot as plt

        if not 1 <= len(coords) <= 3:
            raise ValueError("coords must have 1, 2, or 3 elements.")

        all_names = self.coord_names
        for c in coords:
            if c not in all_names:
                raise ValueError(f"'{c}' is not a coordinate of this scan.")

        source   = self._iter() if use_mask else self._iter_unmasked()
        raw      = list(source)
        if len(raw) == 0:
            print("Warning: no points to preview (all masked or empty scan).")
            return np.empty((0, len(coords))), (fig, ax)
        points   = np.array([[c[name] for name in coords] for _, c in raw])

        defaults = {'marker': 'o', 'color': 'r', 'alpha': 0.5}
        kwargs   = {**defaults, **kwargs}

        fig = plt.figure(figsize=(12, 12))
        match len(coords):
            case 1:
                ax = fig.add_subplot(111)
                ax.plot(points[:, 0], **kwargs)
                ax.set_title(coords[0])
            case 2:
                ax = fig.add_subplot(111)
                ax.plot(points[:, 0], points[:, 1], **kwargs)
                ax.set_xlabel(coords[0])
                ax.set_ylabel(coords[1])
            case 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2], **kwargs)
                ax.set_xlabel(coords[0])
                ax.set_ylabel(coords[1])
                ax.set_zlabel(coords[2])

        return points, (fig, ax)

    """Algebra"""

    def __add__(self, other: ScanBase) -> _ScanSum:
        """
        Concatenate two scans along the outermost axis.

        Both scans must have identical channel name sets (same names, same
        channel objects). The left operand is traversed first.
        """
        if not isinstance(other, ScanBase):
            return NotImplemented
        self._check_sum_compatibility(other)
        return _ScanSum(self, other)

    def __mul__(self, other: ScanBase) -> _ScanProduct:
        """
        Form the outer product of two scans.

        The left operand forms the outer loop, the right forms the inner loop. 
        Channel name sets must be disjoint.
        """
        if not isinstance(other, ScanBase):
            return NotImplemented
        self._check_product_compatibility(other)
        return _ScanProduct(self, other)

    def _check_sum_compatibility(self, other: ScanBase) -> None:
        a_names = set(self.coord_names)
        b_names = set(other.coord_names)
        if a_names != b_names:
            raise ValueError(
                f"Cannot add scans with different coordinate sets.\n"
                f"  Left:  {sorted(a_names)}\n"
                f"  Right: {sorted(b_names)}"
            )
        a_map = self._channel_map()
        b_map = other._channel_map()
        mismatched = [n for n in a_names if a_map[n] is not b_map[n]]
        if mismatched:
            raise ValueError(
                f"Scans share coordinate names but use different channel "
                f"objects for: {mismatched}. Concatenation requires identical "
                f"hardware channels."
            )
        if not np.array_equal(self.dimensions[1:], other.dimensions[1:]):
            raise ValueError(
                f"Cannot add scans whose inner dimensions differ.\n"
                f"  Left:  {self.dimensions}\n"
                f"  Right: {other.dimensions}"
            )

    def _check_product_compatibility(self, other: ScanBase) -> None:
        a_names = set(self.coord_names)
        b_names = set(other.coord_names)
        overlap = a_names & b_names
        if overlap:
            raise ValueError(
                f"Cannot multiply scans with overlapping coordinate names: "
                f"{sorted(overlap)}"
            )

    """Serialization"""

    def _serialize_mask(self) -> dict | None:
        if self.space_mask is None:
            return None
        if isinstance(self.space_mask, Expr):
            return self.space_mask.to_dict()
        raise TypeError(
            f"space_mask on {type(self).__name__} is a plain callable and cannot "
            f"be serialized. Use an Expr mask from masks.py instead."
        )

    @staticmethod
    def _deserialize_mask(d: dict | None) -> Expr | None:
        if d is None:
            return None
        return Expr.from_dict(d)

    def _serialize_channels(self) -> List[str]:
        return [format_reference(ch) for ch in self._channels]

    @staticmethod
    def _deserialize_channels(refs: List[str]) -> List[SweepableChannel]:
        channels = []
        for r in refs:
            ch = resolve_reference(r)
            if ch is None:
                raise ValueError(f"Could not resolve channel reference: '{r}'")
            channels.append(ch)
        return channels

    def to_dict(self) -> dict:
        """
        Serialize this scan to a plain dict. The result is JSON-compatible and 
        can be restored with ScanBase.from_dict().

        Raises TypeError if the scan contains a non-Expr space_mask or
        (for ScanArbitrary) a user-supplied iterator.
        """
        return {
            'scan_type': self._scan_type,
            'channels': self._serialize_channels(),
            'space_mask': self._serialize_mask(),
            'ramp_kwargs': self.ramp_kwargs,
            'state': self._serialize_state(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ScanBase':
        """
        Reconstruct a scan from a dict produced by to_dict().
        Channels are resolved from the DeviceRegistry / HardwareRegistry, so 
        those must be populated before calling this.
        """
        tag = d['scan_type']
        if tag not in _SCAN_REGISTRY:
            raise ValueError(f"Unknown scan type: '{tag}'")
        scan_cls = _SCAN_REGISTRY[tag]

        channels = ScanBase._deserialize_channels(d['channels'])
        space_mask = ScanBase._deserialize_mask(d.get('space_mask'))
        ramp_kwargs = d.get('ramp_kwargs') or {}

        return scan_cls._from_dict(
            d['state'],
            channels = channels,
            space_mask = space_mask,
            ramp_kwargs = ramp_kwargs,
        )

    @abstractmethod
    def _serialize_state(self) -> dict:
        """Serialize subclass-specific state. Called by to_dict."""
        ...

    @classmethod
    @abstractmethod
    def _from_dict(cls, state: dict, channels, space_mask, ramp_kwargs) -> 'ScanBase':
        """Reconstruct from subclass-specific state. Called by from_dict."""
        ...

"""Interior nodes"""

@_register_scan("sum")
class _ScanSum(ScanBase):
    """
    Concatenation of two scans along the outermost axis.
    Constructed only via ScanBase.__add__.
    """

    def __init__(self, left: ScanBase, right: ScanBase):
        super().__init__(
            channels = left.channels,
            ramp_kwargs = left.ramp_kwargs or right.ramp_kwargs,
        )
        self._left = left
        self._right = right

    @property
    def npoints(self) -> int:
        return self._left.npoints + self._right.npoints

    @property
    def dimensions(self) -> np.ndarray:
        d = self._left.dimensions.copy()
        d[0] += self._right.dimensions[0]
        return d

    def _iter_unmasked(self) -> Iterator[IdxCoords]:
        # Delegate to children via _iter so their masks are applied
        offset = self._left.dimensions[0]
        yield from self._left._iter()
        for idx, coords in self._right._iter():
            shifted = idx.copy()
            shifted[0] += offset
            yield shifted, coords

    def _serialize_state(self) -> dict:
        return {
            'left': self._left.to_dict(),
            'right': self._right.to_dict(),
        }

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        left = ScanBase.from_dict(state['left'])
        right = ScanBase.from_dict(state['right'])
        node = cls(left, right)
        node.space_mask = space_mask
        if ramp_kwargs:
            node.ramp_kwargs = ramp_kwargs
        return node

@_register_scan("prod")
class _ScanProduct(ScanBase):
    """
    Outer product of two scans.
    Constructed only via ScanBase.__mul__.
    """

    def __init__(self, outer: ScanBase, inner: ScanBase):
        super().__init__(
            channels = outer.channels + inner.channels,
            ramp_kwargs = outer.ramp_kwargs or inner.ramp_kwargs,
        )
        self._outer = outer
        self._inner = inner

    @property
    def npoints(self) -> int:
        return self._outer.npoints * self._inner.npoints

    @property
    def dimensions(self) -> np.ndarray:
        return np.concatenate([self._outer.dimensions, self._inner.dimensions])

    def _iter_unmasked(self) -> Iterator[IdxCoords]:
        for outer_idx, outer_coords in self._outer._iter():
            for inner_idx, inner_coords in self._inner._iter():
                idx = np.concatenate([outer_idx, inner_idx])
                coords = {**outer_coords, **inner_coords}
                yield idx, coords

    def _serialize_state(self) -> dict:
        return {
            'outer': self._outer.to_dict(),
            'inner': self._inner.to_dict(),
        }

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        outer = ScanBase.from_dict(state['outer'])
        inner = ScanBase.from_dict(state['inner'])
        node = cls(outer, inner)
        node.space_mask = space_mask
        if ramp_kwargs:
            node.ramp_kwargs = ramp_kwargs
        return node

"""Leaf nodes"""

class _ScanLeaf(ScanBase, ABC):
    """Abstract base for all leaf (concrete traversal) scan types."""

    def __init__(self,
        channels: List[SweepableChannel],
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(channels, space_mask, ramp_kwargs)

    def _iter_unmasked(self) -> Iterator[IdxCoords]:
        """
        Leaf nodes generate raw points with no masking. Masking is applied by 
        the inherited _iter wrapper in ScanBase.
        """
        yield from self._raw_iter()

    @abstractmethod
    def _raw_iter(self) -> Iterator[IdxCoords]:
        """Generate raw (idx, coords) pairs with no mask applied."""
        ...

@_register_scan("cut")
class ScanCut(_ScanLeaf):
    """
    Sweep channels linearly from start to end, taking numpoints steps.

    Parameters
    ----------
    channels : list of SweepableChannel
    start : dict
        Starting coordinate values keyed by channel name.
    end : dict
        Ending coordinate values keyed by channel name.
    numpoints : int
        Number of points including both endpoints.
    space_mask : Callable[[Coords], bool], optional
    ramp_kwargs : dict, optional
    """

    def __init__(self,
        channels: List[SweepableChannel],
        start: Coords,
        end: Coords,
        numpoints: int,
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(channels, space_mask, ramp_kwargs)
        names = self.coord_names
        _require_keys(start, names, 'start')
        _require_keys(end, names, 'end')
        self._start = {n: float(start[n]) for n in names}
        self._end = {n: float(end[n]) for n in names}
        self._numpoints = numpoints

    @property
    def npoints(self) -> int:
        return self._numpoints

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self._numpoints], dtype=int)

    def _raw_iter(self) -> Iterator[IdxCoords]:
        names = self.coord_names
        start = np.array([self._start[n] for n in names])
        end = np.array([self._end[n] for n in names])
        for i in range(self._numpoints):
            t = i / (self._numpoints - 1) if self._numpoints > 1 else 0.0
            vals = start + t * (end - start)
            coords = dict(zip(names, vals))
            yield np.array([i]), coords

    def _serialize_state(self) -> dict:
        return {
            'start': self._start,
            'end': self._end,
            'numpoints': self._numpoints,
        }

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        return cls(
            channels = channels,
            start = state['start'],
            end = state['end'],
            numpoints = state['numpoints'],
            space_mask = space_mask,
            ramp_kwargs = ramp_kwargs,
        )
    
@_register_scan("points")
class ScanPoints(_ScanLeaf):
    """
    Visit an explicit sequence of points in parameter space.

    Parameters
    ----------
    channels : list of SweepableChannel
    points : list of dict or ndarray of shape (npoints, nchannels)
        Each row/element is a dict of coordinate values or an array ordered to 
        match channels.
    space_mask : Callable[[Coords], bool], optional
    ramp_kwargs : dict, optional
    """

    def __init__(self,
        channels: List[SweepableChannel],
        points: List[Coords] | np.ndarray,
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(channels, space_mask, ramp_kwargs)
        names = self.coord_names

        if isinstance(points, np.ndarray):
            if points.ndim == 1:
                points = points.reshape(-1, 1)
            if points.shape[1] != len(names):
                raise ValueError(
                    f"points array has {points.shape[1]} columns but scan "
                    f"has {len(names)} channels."
                )
            self._points: List[Coords] = [
                dict(zip(names, row)) for row in points
            ]
        else:
            for p in points:
                _require_keys(p, names, 'point')
            self._points = [{n: float(p[n]) for n in names} for p in points]

    @property
    def npoints(self) -> int:
        return len(self._points)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([len(self._points)], dtype=int)

    def _raw_iter(self) -> Iterator[IdxCoords]:
        for i, coords in enumerate(self._points):
            yield np.array([i]), coords

    def _serialize_state(self) -> dict:
        return {'points': self._points}

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        return cls(
            channels = channels,
            points = state['points'],
            space_mask = space_mask,
            ramp_kwargs = ramp_kwargs,
        )
    
@_register_scan("grid")
class ScanGrid(_ScanLeaf):
    """
    Sweep over the full outer product of per-channel axes.

    Parameters
    ----------
    channels : list of SweepableChannel
    axes : dict
        Mapping from channel name to 1-D array of values for that channel.
    serpentine : bool, default=False
        If True, reverse the fastest axis on alternating passes.
    space_mask : Callable[[Coords], bool], optional
    ramp_kwargs : dict, optional
    """

    def __init__(self,
        channels: List[SweepableChannel],
        axes: Dict[str, np.ndarray],
        serpentine: bool = False,
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(channels, space_mask, ramp_kwargs)
        names = self.coord_names
        _require_keys(axes, names, 'axes')
        self._axes = {n: np.asarray(axes[n], dtype=float) for n in names}
        self._serpentine = serpentine
        self._dims = np.array([len(self._axes[n]) for n in names], dtype=int)

    @property
    def npoints(self) -> int:
        return int(self._dims.prod())

    @property
    def dimensions(self) -> np.ndarray:
        return self._dims.copy()

    def _raw_iter(self) -> Iterator[IdxCoords]:
        names = self.coord_names
        dim = len(names)
        for raw_idx in itertools.product(*[range(d) for d in self._dims]):
            idx = np.array(raw_idx)
            if self._serpentine:
                sidx = np.array(raw_idx)
                for k in range(1, dim):
                    if sidx[:k].sum() % 2 == 1:
                        sidx[k] = self._dims[k] - 1 - raw_idx[k]
                idx = sidx
            coords = {n: float(self._axes[n][idx[k]])
                      for k, n in enumerate(names)}
            yield idx, coords

    def _serialize_state(self) -> dict:
        return {
            'axes': {n: a.tolist() for n, a in self._axes.items()},
            'serpentine': self._serpentine,
        }

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        return cls(
            channels = channels,
            axes = {n: np.array(a) for n, a in state['axes'].items()},
            serpentine = state['serpentine'],
            space_mask = space_mask,
            ramp_kwargs = ramp_kwargs,
        )

@_register_scan("parallelepiped")
class ScanParallelepiped(_ScanLeaf):
    """
    Sweep over a parallelepiped in parameter space.

    Parameters
    ----------
    channels : list of SweepableChannel
    origin : dict
        Starting corner, keyed by channel name.
    endpoints : dict
        Other corners of the parallelepiped. Each key is a channel name, each 
        value is a 1-D array of length n_axes giving the endpoint coordinate 
        along each axis. Fastest axis last.
    shape : list of int
        Number of points along each axis.
    serpentine : bool, default=False
    space_mask : Callable[[Coords], bool], optional
    ramp_kwargs : dict, optional
    """

    def __init__(self,
        channels: List[SweepableChannel],
        origin: Coords,
        endpoints: Dict[str, np.ndarray],
        shape: List[int],
        serpentine: bool = False,
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(channels, space_mask, ramp_kwargs)
        names = self.coord_names
        n_axes = len(shape)

        _require_keys(origin, names, 'origin')
        _require_keys(endpoints, names, 'endpoints')

        self._origin = np.array([float(origin[n]) for n in names])
        self._A = np.column_stack([
            np.array([float(endpoints[n][k]) for n in names]) - self._origin
            for k in range(n_axes)
        ])
        self._shape = list(shape)
        self._dims = np.array(shape, dtype=int)
        self._serpentine = serpentine

    @property
    def npoints(self) -> int:
        return int(np.prod(self._shape))

    @property
    def dimensions(self) -> np.ndarray:
        return self._dims.copy()

    def _raw_iter(self) -> Iterator[IdxCoords]:
        names = self.coord_names
        n_axes = len(self._shape)
        for raw_idx in itertools.product(*[range(d) for d in self._shape]):
            idx = np.array(raw_idx)
            if self._serpentine:
                sidx = np.array(raw_idx)
                for k in range(1, n_axes):
                    if sidx[:k].sum() % 2 == 1:
                        sidx[k] = self._shape[k] - 1 - raw_idx[k]
                idx = sidx
            t = idx / np.maximum(self._dims - 1, 1)
            point = self._origin + self._A @ t
            coords = dict(zip(names, point))
            yield idx, coords

    def _serialize_state(self) -> dict:
        names = self.coord_names
        return {
            'origin': dict(zip(names, self._origin.tolist())),
            'A_cols': self._A.tolist(),
            'shape': self._shape,
            'serpentine': self._serpentine,
        }

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        names  = [ch.name for ch in channels]
        origin = state['origin']
        A      = np.array(state['A_cols'])
        shape  = state['shape']
        n_axes = A.shape[1]
        origin_arr = np.array([origin[n] for n in names])
        endpoints = {
            n: [float(origin_arr[i] + A[i, k]) for k in range(n_axes)]
            for i, n in enumerate(names)
        }
        return cls(
            channels = channels,
            origin = origin,
            endpoints = {n: np.array(v) for n, v in endpoints.items()},
            shape = shape,
            serpentine = state['serpentine'],
            space_mask = space_mask,
            ramp_kwargs = ramp_kwargs,
        )

@_register_scan("arbitrary")
class ScanArbitrary(_ScanLeaf):
    """
    Scan defined by a user-supplied iterator.

    The iterator must yield (idx, coords) pairs where idx is a 1-D int array and 
    coords is a Dict[str, float] whose keys match the channel names. The scan 
    trusts the user iterator to be consistent.

    Parameters
    ----------
    channels : list of SweepableChannel
    iterate : Callable[[], Iterator[IdxCoords]]
        Zero-argument callable that returns a fresh iterator each time it is 
        called. Called once per scan execution.
    npoints : int
        Total number of points the iterator will yield.
    dimensions : array-like of int
        Shape of the result array.
    space_mask : Callable[[Coords], bool], optional
    ramp_kwargs : dict, optional
    """

    def __init__(self,
        channels: List[SweepableChannel],
        iterate: Callable[[], Iterator[IdxCoords]],
        npoints: int,
        dimensions: List[int] | np.ndarray,
        space_mask: Callable[[Coords], bool] | None = None,
        ramp_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(channels, space_mask, ramp_kwargs)
        self._iterate = iterate
        self._npoints = npoints
        self._dimensions = np.asarray(dimensions, dtype=int)

    @property
    def npoints(self) -> int:
        return self._npoints

    @property
    def dimensions(self) -> np.ndarray:
        return self._dimensions.copy()

    def _raw_iter(self) -> Iterator[IdxCoords]:
        yield from self._iterate()

    def _serialize_state(self) -> dict:
        raise TypeError(
            "ScanArbitrary cannot be serialized because its iterator is a "
            "user-supplied callable. Reconstruct it manually after loading."
        )

    @classmethod
    def _from_dict(cls, state, channels, space_mask, ramp_kwargs):
        raise TypeError(
            "ScanArbitrary cannot be deserialized. Reconstruct it manually."
        )

"""Helpers"""

def _require_keys(d: dict, names: List[str], label: str) -> None:
    missing = [n for n in names if n not in d]
    extra   = [k for k in d if k not in names]
    if missing:
        raise ValueError(f"'{label}' is missing keys: {missing}")
    if extra:
        raise ValueError(f"'{label}' has unexpected keys: {extra}")