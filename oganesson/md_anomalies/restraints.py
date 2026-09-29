"""Flat-bottom harmonic restraints: containment spheres, exclusion spheres and slabs.

Anchors are fractional coordinates, so they move with the cell during constant-pressure
MD and strain triggers. The calculator returns the restraints' virial stress, so
barostats see the full pressure when it is combined with MatGL through SumCalculator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ase.stress import full_3x3_to_voigt_6_stress


@dataclass
class Sphere:
    indices: List[int]
    center_frac: List[float]
    radius: float
    k: float  # eV/A^2
    inside: bool = True  # True keeps atoms inside the sphere; False keeps them out


@dataclass
class Slab:
    indices: List[int]
    axis: int
    lo_frac: float
    hi_frac: float
    k: float  # eV/A^2


@dataclass
class RestraintGroup:
    name: str
    terms: List[Union[Sphere, Slab]]
    scale: float = 1.0  # multiplies every spring constant; 0 releases the group

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scale": self.scale,
            "terms": [{"type": type(t).__name__, **asdict(t)} for t in self.terms],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RestraintGroup":
        terms: List[Union[Sphere, Slab]] = []
        for t in d["terms"]:
            t = dict(t)
            kind = t.pop("type")
            terms.append(Sphere(**t) if kind == "Sphere" else Slab(**t))
        return cls(name=d["name"], terms=terms, scale=d.get("scale", 1.0))


class RestraintCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, groups: Sequence[RestraintGroup]):
        super().__init__()
        names = [g.name for g in groups]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate RestraintGroup name(s): {dupes}")
        self.groups: Dict[str, RestraintGroup] = {g.name: g for g in groups}

    def set_scale(self, name: str, scale: float) -> None:
        self.groups[name].scale = float(scale)
        self.reset()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        cell = self.atoms.cell.array
        pbc = self.atoms.pbc
        energy = 0.0
        forces = np.zeros_like(pos)
        virial = np.zeros((3, 3))
        for group in self.groups.values():
            if group.scale == 0.0:
                continue
            for term in group.terms:
                k = term.k * group.scale
                idx = np.asarray(term.indices, dtype=int)
                if isinstance(term, Sphere):
                    center = np.asarray(term.center_frac, dtype=float) @ cell
                    u, d = find_mic(pos[idx] - center, cell, pbc)
                    excess = d - term.radius if term.inside else term.radius - d
                    active = excess > 0
                    if not np.any(active):
                        continue
                    e = excess[active]
                    unit = u[active] / np.maximum(d[active], 1e-12)[:, None]
                    sign = 1.0 if term.inside else -1.0
                    grad = (sign * k * e)[:, None] * unit  # dE/du
                    energy += 0.5 * k * float(np.sum(e**2))
                    forces[idx[active]] -= grad
                    virial += u[active].T @ grad
                else:
                    grad_f = np.linalg.inv(cell)[:, term.axis]  # d(fractional coordinate)/dr
                    height = 1.0 / np.linalg.norm(grad_f)
                    frac = pos[idx] @ grad_f
                    rel = frac - 0.5 * (term.lo_frac + term.hi_frac)
                    rel -= np.round(rel)
                    excess = np.abs(rel) - 0.5 * (term.hi_frac - term.lo_frac)
                    active = excess > 0
                    if not np.any(active):
                        continue
                    e = excess[active]
                    sgn = np.sign(rel[active])
                    energy += 0.5 * k * float(np.sum((height * e) ** 2))
                    grad = (k * height**2 * e * sgn)[:, None] * grad_f[None, :]  # dE/dr
                    forces[idx[active]] -= grad
                    w = (height**2 * e * sgn)[:, None] * grad_f[None, :]  # wall-to-atom vector
                    virial += w.T @ grad
        stress = 0.5 * (virial + virial.T) / abs(np.linalg.det(cell))
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "stress": full_3x3_to_voigt_6_stress(stress),
        }
