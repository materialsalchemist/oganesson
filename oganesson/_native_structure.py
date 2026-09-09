"""Native, dependency-light data model backing OgStructure.

Owns the atomic data (cartesian coordinates, atomic numbers, lattice matrix,
periodic boundary flags) as plain numpy arrays, and builds genuine
pymatgen.core.Structure / ase.Atoms objects from it on demand. This lets
OgStructure expose pymatgen- and ASE-shaped reads directly, computed from
one native store, while still handing real Structure/Atoms instances to
code (bsym, spglib-backed symmetry, XRDCalculator, ase.ga, diep/matgl, ...)
that requires them.

Both pymatgen's Lattice.matrix and ASE's Atoms.cell use the same row-vector
convention (rows are the a/b/c lattice vectors; cartesian = fractional @
matrix), so a single matrix and a single set of formulas serve both sides.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols
from pymatgen.core import Lattice, Structure


@dataclass
class NativeStructure:
    cart_coords: np.ndarray
    lattice_matrix: np.ndarray
    atomic_numbers: np.ndarray
    pbc: Tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self):
        self.cart_coords = np.array(self.cart_coords, dtype=float).reshape(-1, 3)
        self.lattice_matrix = np.array(self.lattice_matrix, dtype=float).reshape(3, 3)
        self.atomic_numbers = np.array(self.atomic_numbers, dtype=int).reshape(-1)
        if len(self.atomic_numbers) != len(self.cart_coords):
            raise ValueError(
                "cart_coords and atomic_numbers must have matching length"
            )

    # -- construction from / conversion to the real libraries --

    @classmethod
    def from_pymatgen(cls, structure: Structure) -> "NativeStructure":
        return cls(
            cart_coords=np.array(structure.cart_coords, dtype=float),
            lattice_matrix=np.array(structure.lattice.matrix, dtype=float),
            atomic_numbers=np.array(structure.atomic_numbers, dtype=int),
        )

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "NativeStructure":
        pbc = atoms.pbc
        if not hasattr(pbc, "__len__"):
            pbc = (pbc, pbc, pbc)
        return cls(
            cart_coords=np.array(atoms.get_positions(), dtype=float),
            lattice_matrix=np.array(atoms.cell.array, dtype=float),
            atomic_numbers=np.array(atoms.get_atomic_numbers(), dtype=int),
            pbc=tuple(bool(x) for x in pbc),
        )

    def to_pymatgen(self) -> Structure:
        return Structure(
            lattice=Lattice(self.lattice_matrix),
            species=self.atomic_numbers.tolist(),
            coords=self.cart_coords,
            coords_are_cartesian=True,
        )

    def to_ase(self) -> Atoms:
        a, b, c, alpha, beta, gamma = self.cell_lengths_and_angles
        return Atoms(
            positions=self.cart_coords,
            numbers=self.atomic_numbers,
            pbc=True,
            cell=Cell.fromcellpar([a, b, c, alpha, beta, gamma]),
        )

    def copy(self) -> "NativeStructure":
        return NativeStructure(
            cart_coords=self.cart_coords.copy(),
            lattice_matrix=self.lattice_matrix.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            pbc=self.pbc,
        )

    # -- geometry, computed directly from the native arrays --

    @property
    def frac_coords(self) -> np.ndarray:
        return self.cart_coords @ np.linalg.inv(self.lattice_matrix)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice_matrix)))

    @property
    def cell_lengths_and_angles(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        m = self.lattice_matrix
        a, b, c = (float(np.linalg.norm(v)) for v in m)

        def angle(u: np.ndarray, v: np.ndarray) -> float:
            cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

        alpha = angle(m[1], m[2])
        beta = angle(m[0], m[2])
        gamma = angle(m[0], m[1])
        return a, b, c, alpha, beta, gamma

    @property
    def symbols(self):
        return [chemical_symbols[z] for z in self.atomic_numbers]

    def get_all_distances(self, mic: bool = True) -> np.ndarray:
        if not mic:
            diff = self.cart_coords[:, None, :] - self.cart_coords[None, :, :]
            return np.linalg.norm(diff, axis=-1)
        inv = np.linalg.inv(self.lattice_matrix)
        frac = self.cart_coords @ inv
        diff_frac = frac[:, None, :] - frac[None, :, :]
        diff_frac -= np.round(diff_frac)
        diff_cart = diff_frac @ self.lattice_matrix
        return np.linalg.norm(diff_cart, axis=-1)
