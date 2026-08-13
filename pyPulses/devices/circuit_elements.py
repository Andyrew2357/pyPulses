"""
Reusable circuit-element / thermal-stage abstraction for amplifier bias
networks.

Power dissipated in an amplifier's bias network doesn't all matter equally:
a drain-side bias resistor anchored at the 1K pot has a very different
thermal budget than the transistor die sitting next to the sample. This
module lets each physical component in a bias network be tagged with where
it physically sits (a ThermalStage) so power can be tracked and capped per
stage rather than only as a single lumped total.

Topologies (e.g. HEMTCommonSource in HEMT.py) wire these elements together
as concrete series chains per branch -- this module intentionally does not
attempt to be a general circuit-graph solver. A branch is just
`ResistiveElement | list[ResistiveElement]`; since the same current flows
through every element of a series branch, each element's dissipation
(I**2 * R_i) is automatically that element's resistance fraction of the
branch's total power -- a direct, N-stage generalization of the old
hemt_amp.py's RDcold/RScold 2-way split.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ThermalStage:
    """
    A named location in the fridge with an optional cooling-power budget.

    Parameters
    ----------
    name : str
        Identifier for the stage (e.g. '4K', 'still', 'MC', 'sample'). Two
        elements that should share a power budget must use ThermalStage
        instances with the same name. Stage accounting is scoped per
        device -- there is no cross-device ledger tracking total
        dissipation at a stage shared across multiple amplifiers.
    max_power : float, optional
        Hard cap on power dissipated at this stage, in Watts. None means
        this stage's power is tracked/reported but not capped.
    """
    name: str
    max_power: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'max_power': self.max_power}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ThermalStage':
        return cls(name=d['name'], max_power=d.get('max_power'))


class CircuitElement(ABC):
    """A component in a bias network that dissipates power somewhere."""

    stage: ThermalStage

    @abstractmethod
    def power(self, current: float, voltage: float | None = None) -> float:
        """Power dissipated in this element, in Watts."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...


@dataclass
class ResistiveElement(CircuitElement):
    """
    A fixed resistor. Power is inferred purely from the current through it
    (I**2 * R); a resistor spanning two physical locations should be
    represented as two ResistiveElements in series, each tagged with the
    stage it actually sits at.
    """
    resistance: float
    stage: ThermalStage

    def power(self, current: float, voltage: float | None = None) -> float:
        return current**2 * self.resistance

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resistance': self.resistance,
            'stage': self.stage.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResistiveElement':
        return cls(
            resistance = d['resistance'],
            stage      = ThermalStage.from_dict(d['stage']),
        )


@dataclass
class TransistorElement(CircuitElement):
    """
    The active device itself. Carries its own safe-operating-region spec
    so it can be reused verbatim by any future topology built from the
    same transistor model (e.g. a cascode stage using two of the same
    part).

    Parameters
    ----------
    stage : ThermalStage
        Where the transistor die physically sits (often the coldest stage
        in the system, since it usually sits close to the sample).
    vgs_min, vgs_max : float
        Safe gate-source voltage window, inclusive. Defaults reflect the
        common HEMT model this framework was written for, which tolerates
        essentially no positive V_GS -- override for other parts.
    id_max : float, optional
        Optional hard cap on |I_D|, in Amps. None disables the check.
    """
    stage: ThermalStage
    vgs_min: float = -np.inf
    vgs_max: float = 0.0
    id_max: float | None = None

    def power(self, current: float, voltage: float | None = None) -> float:
        if voltage is None:
            raise ValueError(
                "TransistorElement.power requires the drain-source voltage."
            )
        return current * voltage  # I_D * V_DS

    def is_safe(self, vgs: float, id_: float | None = None) -> bool:
        if not (self.vgs_min <= vgs <= self.vgs_max):
            return False
        if self.id_max is not None and id_ is not None and abs(id_) > self.id_max:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage'  : self.stage.to_dict(),
            'vgs_min': self.vgs_min,
            'vgs_max': self.vgs_max,
            'id_max' : self.id_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TransistorElement':
        return cls(
            stage   = ThermalStage.from_dict(d['stage']),
            vgs_min = d.get('vgs_min', -np.inf),
            vgs_max = d.get('vgs_max', 0.0),
            id_max  = d.get('id_max'),
        )


"""Branch helpers: a branch is a ResistiveElement, a list of them, or None."""

Branch = ResistiveElement | List[ResistiveElement] | None


def as_element_list(branch: Branch) -> List[ResistiveElement]:
    """Normalize a branch to a list, for iteration/serialization."""
    if branch is None:
        return []
    if isinstance(branch, ResistiveElement):
        return [branch]
    return list(branch)


def branch_resistance(branch: Branch) -> float:
    """Total resistance of a series branch. None/[] -> 0.0 (a direct short)."""
    return sum(e.resistance for e in as_element_list(branch))


def branch_power_by_stage(branch: Branch, current: float) -> Dict[str, float]:
    """
    Power dissipated by each element of a series branch, keyed by stage
    name and accumulated across elements that share a stage.
    """
    out: Dict[str, float] = {}
    for e in as_element_list(branch):
        out[e.stage.name] = out.get(e.stage.name, 0.0) + e.power(current)
    return out
