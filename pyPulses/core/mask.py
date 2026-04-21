"""
Serializable mask expressions for scan space_mask.

A mask is a callable Dict[str, float] -> bool. This module provides a small
expression tree that can be constructed with natural Python syntax, evaluated
as a function, and serialized to/from a plain dict for persistence.

Usage
-----
    from .masks import C

    # Circular region in (Bx, By) space
    mask = (C('Bx')**2 + C('By')**2) < C_const(1.0)**2

    # Wedge in (V, B) space
    mask = (C('V') > 0.1) & (C('V') < 2 * C('B') + 1.0)

    # Serialize / deserialize
    d    = mask.to_dict()
    mask = Expr.from_dict(d)

C(name)       — reference to a coordinate by name
C_const(val)  — a literal float constant

Arithmetic : +  -  *  /  **  unary-
Comparison  : <  <=  >  >=
Boolean     : &  |  ~
Math        : abs(), via Abs node
"""

from __future__ import annotations
from typing import Dict

Coords = Dict[str, float]

"""Base expression node"""

class Expr:
    """
    Abstract base for all mask expression nodes.

    Every node is callable as a function of a coordinate dict, and supports
    the full set of arithmetic, comparison, and boolean operators so that
    expressions can be built with natural Python syntax.
    """

    def __call__(self, coords: Coords) -> float | bool:
        raise NotImplementedError

    # -- Arithmetic --

    def __add__(self, other) -> 'Add':
        return Add(self, _coerce(other))

    def __radd__(self, other) -> 'Add':
        return Add(_coerce(other), self)

    def __sub__(self, other) -> 'Sub':
        return Sub(self, _coerce(other))

    def __rsub__(self, other) -> 'Sub':
        return Sub(_coerce(other), self)

    def __mul__(self, other) -> 'Mul':
        return Mul(self, _coerce(other))

    def __rmul__(self, other) -> 'Mul':
        return Mul(_coerce(other), self)

    def __truediv__(self, other) -> 'Div':
        return Div(self, _coerce(other))

    def __rtruediv__(self, other) -> 'Div':
        return Div(_coerce(other), self)

    def __pow__(self, other) -> 'Pow':
        return Pow(self, _coerce(other))

    def __rpow__(self, other) -> 'Pow':
        return Pow(_coerce(other), self)

    def __neg__(self) -> 'Neg':
        return Neg(self)

    def __abs__(self) -> 'Abs':
        return Abs(self)

    # -- Comparison --

    def __lt__(self, other) -> 'Lt':
        return Lt(self, _coerce(other))

    def __le__(self, other) -> 'Le':
        return Le(self, _coerce(other))

    def __gt__(self, other) -> 'Gt':
        return Gt(self, _coerce(other))

    def __ge__(self, other) -> 'Ge':
        return Ge(self, _coerce(other))

    # -- Boolean --
    # Use & | ~ rather than and/or/not since Python doesn't allow overloading
    # the latter. Comparisons already return Expr nodes so & and | chain
    # naturally: (C('x') > 0) & (C('y') < 1)

    def __and__(self, other) -> 'And':
        return And(self, _coerce(other))

    def __rand__(self, other) -> 'And':
        return And(_coerce(other), self)

    def __or__(self, other) -> 'Or':
        return Or(self, _coerce(other))

    def __ror__(self, other) -> 'Or':
        return Or(_coerce(other), self)

    def __invert__(self) -> 'Not':
        return Not(self)

    # -- Serialization --

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> 'Expr':
        tag = d['op']
        if tag not in _REGISTRY:
            raise ValueError(f"Unknown mask expression op: '{tag}'")
        return _REGISTRY[tag]._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> 'Expr':
        raise NotImplementedError

# Registry (populated by _register decorator below)
_REGISTRY: Dict[str, type] = {}

def _register(tag: str):
    def decorator(cls):
        cls._op_tag = tag
        _REGISTRY[tag] = cls
        return cls
    return decorator

# Helpers
def _coerce(x) -> Expr:
    """Wrap a bare float/int as a Const node; pass Expr through unchanged."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return Const(float(x))
    raise TypeError(f"Cannot coerce {type(x).__name__} to a mask Expr.")

def _child(d: dict, key: str) -> Expr:
    return Expr.from_dict(d[key])

"""Leaf nodes"""

@_register('coord')
class Coord(Expr):
    """Reference to a named coordinate: evaluates to coords[name]."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, coords: Coords) -> float:
        try:
            return float(coords[self.name])
        except KeyError:
            raise KeyError(
                f"Mask references coordinate '{self.name}' which is not "
                f"present in the coordinate dict."
            )

    def to_dict(self) -> dict:
        return {'op': 'coord', 'name': self.name}

    @classmethod
    def _from_dict(cls, d: dict) -> 'Coord':
        return cls(d['name'])


@_register('const')
class Const(Expr):
    """A literal float constant."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, coords: Coords) -> float:
        return self.value

    def to_dict(self) -> dict:
        return {'op': 'const', 'value': self.value}

    @classmethod
    def _from_dict(cls, d: dict) -> 'Const':
        return cls(d['value'])

"""Unary nodes"""

class _Unary(Expr):
    def __init__(self, operand: Expr):
        self.operand = operand

    def to_dict(self) -> dict:
        return {'op': self._op_tag, 'operand': self.operand.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> '_Unary':
        return cls(_child(d, 'operand'))


@_register('neg')
class Neg(_Unary):
    def __call__(self, coords: Coords) -> float:
        return -self.operand(coords)


@_register('abs')
class Abs(_Unary):
    def __call__(self, coords: Coords) -> float:
        return abs(self.operand(coords))


@_register('not')
class Not(_Unary):
    def __call__(self, coords: Coords) -> bool:
        return not self.operand(coords)

"""Binary nodes"""

class _Binary(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left  = left
        self.right = right

    def to_dict(self) -> dict:
        return {
            'op'   : self._op_tag,
            'left' : self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict) -> '_Binary':
        return cls(_child(d, 'left'), _child(d, 'right'))

# Arithmetic

@_register('add')
class Add(_Binary):
    def __call__(self, coords: Coords) -> float:
        return self.left(coords) + self.right(coords)

@_register('sub')
class Sub(_Binary):
    def __call__(self, coords: Coords) -> float:
        return self.left(coords) - self.right(coords)

@_register('mul')
class Mul(_Binary):
    def __call__(self, coords: Coords) -> float:
        return self.left(coords) * self.right(coords)

@_register('div')
class Div(_Binary):
    def __call__(self, coords: Coords) -> float:
        return self.left(coords) / self.right(coords)

@_register('pow')
class Pow(_Binary):
    def __call__(self, coords: Coords) -> float:
        return self.left(coords) ** self.right(coords)

# Comparison

@_register('lt')
class Lt(_Binary):
    def __call__(self, coords: Coords) -> bool:
        return self.left(coords) < self.right(coords)

@_register('le')
class Le(_Binary):
    def __call__(self, coords: Coords) -> bool:
        return self.left(coords) <= self.right(coords)

@_register('gt')
class Gt(_Binary):
    def __call__(self, coords: Coords) -> bool:
        return self.left(coords) > self.right(coords)

@_register('ge')
class Ge(_Binary):
    def __call__(self, coords: Coords) -> bool:
        return self.left(coords) >= self.right(coords)

# Boolean

@_register('and')
class And(_Binary):
    def __call__(self, coords: Coords) -> bool:
        return bool(self.left(coords)) and bool(self.right(coords))

@_register('or')
class Or(_Binary):
    def __call__(self, coords: Coords) -> bool:
        return bool(self.left(coords)) or bool(self.right(coords))

"""Public convenience aliases"""

def C(name: str) -> Coord:
    """Shorthand for Coord(name). The primary way to reference coordinates."""
    return Coord(name)

def C_const(value: float) -> Const:
    """Shorthand for Const(value). Useful when you want an explicit node."""
    return Const(value)