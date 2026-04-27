"""
DatabaseLogger
==============
An observer that streams parameter sweep results into a SQLite database as
they are produced, compatible with the ez_data sqlite_to_xarray reader.

Usage
-----
    logger = DatabaseLogger.from_runner(runner, "my_experiment.db")
    runner.add_observer(logger)
    runner.run()
    logger.close()

Or as a context manager:

    with DatabaseLogger.from_runner(runner, "my_experiment.db") as logger:
        runner.add_observer(logger)
        runner.run()

"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .runner import Runner

import json
import sqlite3
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import datetime
import numpy as np

def _to_builtin(obj):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if hasattr(obj, 'item'):   # numpy scalar
        return obj.item()
    return obj

def _json(v: Any) -> str:
    return json.dumps(_to_builtin(v))

"""Checkpoint mode"""

class CheckpointMode(Enum):
    NONE  = 'none'
    POINT = 'point'   # after every point
    ROWS  = 'rows'    # after every N flush calls
    SIZE  = 'size'    # after every N points written

def _parse_checkpoint_mode(mode) -> tuple[CheckpointMode, int | None]:
    """
    Parse a checkpoint mode specification.

    Accepts:
        CheckpointMode enum value
        'none', 'point'
        'rows:N'  — checkpoint after every N flush calls
        'size:N'  — checkpoint after every N points written
    """
    if isinstance(mode, CheckpointMode):
        return mode, None
    if not isinstance(mode, str):
        raise TypeError("checkpoint_mode must be a str or CheckpointMode")
    if mode in ('none', 'point'):
        return CheckpointMode(mode), None
    for prefix, cm in (('rows:', CheckpointMode.ROWS), ('size:', CheckpointMode.SIZE)):
        if mode.startswith(prefix):
            try:
                param = int(mode[len(prefix):])
            except ValueError:
                raise ValueError(f"Invalid checkpoint_mode parameter in '{mode}'")
            return cm, param
    raise ValueError(f"Unrecognised checkpoint_mode: '{mode!r}'")

"""DatabaseLogger"""

class DatabaseLogger:
    """
    Streams parameter sweep results into a SQLite database point by point.

    Acts as a runner observer: calling an instance with
    `(idx, coords, measured, timestamp)` writes one row.

    Parameters
    ----------
    path : str or Path
        Output file path. The `.db` extension is added automatically.
    dims : list of str
        Names of the scan index dimensions (e.g. ['dim0', 'dim1']).
    shape : list of int or None
        Expected size along each dimension. Stored as metadata only.
    coord_names : list of str
        Names of the swept coordinate columns.
    data_names : list of str
        Names of the measured data columns.
    timestamp : bool
        Whether to record a timestamp column.
    use_buffer : bool
        If True, rows are accumulated in memory and flushed in batches. Reduces 
        write overhead at the cost of potential data loss on crash.
    buffer_size : int
        Number of rows to accumulate before flushing (only used when 
        use_buffer=True).
    checkpoint_mode : str or CheckpointMode
        Controls WAL checkpointing frequency. See CheckpointMode.
    """

    def __init__(self,
        path: str | Path,
        dims: List[str],
        shape: List[int] | None,
        coord_names: List[str],
        data_names: List[str],
        timestamp: bool = True,
        use_buffer: bool = False,
        buffer_size: int  = 100,
        checkpoint_mode: str | CheckpointMode = CheckpointMode.NONE,
    ):
        
        self.path = Path(path).with_suffix('.db')
        self.dims = list(dims)
        self.shape = list(shape) if shape is not None else None
        self.coord_names = list(coord_names)
        self.data_names = list(data_names)
        self.timestamp = timestamp
        self.use_buffer = use_buffer
        self.buffer_size = buffer_size

        self._checkpoint_mode, self._checkpoint_param = \
            _parse_checkpoint_mode(checkpoint_mode)

        self._buffer: List[tuple] = []
        self._flush_count: int = 0
        self._point_count: int = 0

        self._conn = sqlite3.connect(self.path, timeout=30)
        self._cur = self._conn.cursor()

        # WAL mode: readers never block writers and vice versa
        self._cur.execute("PRAGMA journal_mode=WAL")
        self._cur.execute("PRAGMA synchronous=NORMAL")
        self._cur.execute("PRAGMA temp_store=MEMORY")

        if self._has_tables():
            self._load_schema()
        else:
            self._init_tables()
            self._store_schema()

        self._insert_sql = self._make_insert_sql()

    """Construction helpers"""

    @classmethod
    def from_runner(cls,
        runner: Runner,
        path: str | Path,
        use_buffer: bool = False,
        buffer_size: int  = 100,
        checkpoint_mode: str | CheckpointMode = CheckpointMode.NONE,
    ) -> 'DatabaseLogger':
        
        """
        Construct a DatabaseLogger from a Runner instance.

        Pulls dimension names, shape, coordinate names, and measurement column 
        names directly from the runner's scan and measurement objects. Also 
        stores per-variable metadata (long_name, unit) from the measurement's 
        Query objects and the scan's SweepableChannels.
        """
        
        scan = runner.scan
        measurement = runner.measurement

        if scan is not None:
            dims = scan.dimensions
            dim_names = [f'dim{i}' for i in range(len(dims))]
            shape = dims.tolist()
            coord_names = scan.coord_names
        else:
            dim_names = ['dim0']
            shape = None
            coord_names = []

        data_names = measurement.col_names

        instance = cls(
            path = path,
            dims = dim_names,
            shape = shape,
            coord_names = coord_names,
            data_names = data_names,
            timestamp = runner.timestamp,
            use_buffer = use_buffer,
            buffer_size = buffer_size,
            checkpoint_mode = checkpoint_mode,
        )

        # Global attrs
        instance.set_global_attrs({
            'time_per_point': measurement.time_per_point,
        })

        # Per-coordinate metadata from SweepableChannels
        if scan is not None:
            for ch in scan.channels:
                attrs = {}
                if ch.long_name is not None:
                    attrs['long_name'] = ch.long_name
                if ch.unit is not None:
                    attrs['unit'] = ch.unit
                if attrs:
                    instance.set_variable_attrs(ch.name, attrs)

        # Per-measurement metadata from Query objects
        col_long_names = measurement.col_long_names
        col_units = measurement.col_units
        col_names = measurement.col_names
        for name, long_name, unit in zip(col_names, col_long_names, col_units):
            attrs = {}
            if long_name is not None:
                attrs['long_name'] = long_name
            if unit is not None:
                attrs['unit'] = unit
            if attrs:
                instance.set_variable_attrs(name, attrs)

        return instance

    """Schema"""

    def _has_tables(self) -> bool:
        self._cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return bool(self._cur.fetchall())

    def _init_tables(self):
        idx_cols = [f'{d} INTEGER' for d in self.dims]
        coord_cols = [f'"{n}" REAL'  for n in self.coord_names]
        data_cols = [f'"{n}" REAL'  for n in self.data_names]
        ts_col = ['timestamp REAL'] if self.timestamp else []

        all_cols = idx_cols + coord_cols + data_cols + ts_col
        self._cur.execute(
            f"CREATE TABLE IF NOT EXISTS sweep "
            f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(all_cols)})"
        )
        self._cur.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT)"
        )
        self._cur.execute(
            "CREATE TABLE IF NOT EXISTS var_metadata "
            "(var_name TEXT, key TEXT, value TEXT, PRIMARY KEY (var_name, key))"
        )
        self._conn.commit()

    def _store_schema(self):
        schema = {
            'dims': self.dims,
            'shape': self.shape,
            'coord_names': self.coord_names,
            'data_names': self.data_names,
            'timestamp': self.timestamp,
            'version': 1,
        }
        self._cur.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('__schema__', _json(schema))
        )
        self._conn.commit()

    def _load_schema(self):
        """Load schema from an existing database, filling instance attributes."""
        self._cur.execute("SELECT value FROM metadata WHERE key='__schema__'")
        row = self._cur.fetchone()
        if not row:
            raise RuntimeError("Existing database has no __schema__ entry")
        schema = json.loads(row[0])
        self.dims = schema['dims']
        self.shape = schema.get('shape')
        self.coord_names = schema['coord_names']
        self.data_names = schema['data_names']
        self.timestamp = schema.get('timestamp', False)

    def _make_insert_sql(self) -> str:
        cols = self.dims + self.coord_names + self.data_names
        if self.timestamp:
            cols.append('timestamp')
        quoted = [f'"{c}"' for c in cols]
        placeholders = ', '.join(['?'] * len(cols))
        return f"INSERT INTO sweep ({', '.join(quoted)}) VALUES ({placeholders})"

    """Observer interface"""

    def __call__(self,
        idx: np.ndarray,
        coords: Dict[str, float],
        measured: Dict[str, float],
        timestamp: datetime.datetime | None,
    ):
        """
        Observer callback. Called by the runner after each point.

        Parameters
        ----------
        idx : ndarray of int
            Scan index for this point.
        coords : dict
            Coordinate values keyed by channel name.
        measured : dict
            Measurement values keyed by column name.
        timestamp : datetime or None
        """

        coord_vals = [coords[n]   for n in self.coord_names]
        data_vals = [measured[n] for n in self.data_names]
        ts_val = [timestamp.timestamp() if timestamp is not None else None] \
                  if self.timestamp else []

        row = tuple(int(i) for i in idx) + \
              tuple(coord_vals) + \
              tuple(data_vals) + \
              tuple(ts_val)

        self._point_count += 1

        if self.use_buffer:
            self._buffer.append(row)
            if len(self._buffer) >= self.buffer_size:
                self._flush()
        else:
            self._cur.execute(self._insert_sql, row)
            self._conn.commit()
            self._maybe_checkpoint()

    """Flushing and checkpointing"""

    def _flush(self):
        """Flush the buffer to disk and maybe checkpoint."""
        if self._buffer:
            self._cur.executemany(self._insert_sql, self._buffer)
            self._conn.commit()
            self._buffer.clear()
        self._flush_count += 1
        self._maybe_checkpoint()

    def _maybe_checkpoint(self):
        cm, cp = self._checkpoint_mode, self._checkpoint_param
        if cm == CheckpointMode.NONE:
            return
        if cm == CheckpointMode.POINT:
            self._wal_checkpoint()
        elif cm == CheckpointMode.SIZE and self._point_count >= cp:
            self._wal_checkpoint()
            self._point_count = 0
        elif cm == CheckpointMode.ROWS and self._flush_count >= cp:
            self._wal_checkpoint()
            self._flush_count = 0

    def _wal_checkpoint(self):
        busy, log_frames, ckpt_frames = self._cur.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        self._conn.commit()
        if busy:
            warnings.warn(
                f"WAL checkpoint could not complete (busy). "
                f"Frames in WAL: {log_frames}, checkpointed: {ckpt_frames}. "
                "Readers may see stale data until a later checkpoint succeeds."
            )

    """Metadata"""

    def set_global_attrs(self, attrs: Dict[str, Any]):
        """
        Store global key-value metadata. Values must be JSON-serializable.
        None values are silently skipped.
        """
        for k, v in attrs.items():
            if v is None:
                continue
            self._cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (k, _json(v)),
            )
        self._conn.commit()

    def set_variable_attrs(self, var: str, attrs: Dict[str, Any]):
        """
        Store per-variable metadata for a coordinate or data column.
        None values are silently skipped.

        Parameters
        ----------
        var : str
            Column name. Must be a known coordinate or data variable.
        attrs : dict
            Metadata to attach. Values must be JSON-serializable.
        """

        known = set(self.coord_names) | set(self.data_names)
        if self.timestamp:
            known.add('timestamp')
        if var not in known:
            raise ValueError(
                f"'{var}' is not a known variable. "
                f"Known: {sorted(known)}"
            )
        for k, v in attrs.items():
            if v is None:
                continue
            self._cur.execute(
                "INSERT OR REPLACE INTO var_metadata "
                "(var_name, key, value) VALUES (?, ?, ?)",
                (var, k, _json(v)),
            )
        self._conn.commit()

    """Lifecycle"""

    def close(self):
        """Flush any buffered rows and close the database connection."""
        if self.use_buffer:
            self._flush()
        self._conn.close()

    def __enter__(self) -> 'DatabaseLogger':
        return self

    def __exit__(self, *_):
        self.close()