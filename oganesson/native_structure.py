"""
oganesson.native_structure
==========================

This module implements a **native (dependency-light)** crystal/atomic structure
representation used by :class:`oganesson.ogstructure.OgStructure`.

Why this exists
---------------
Historically, ``OgStructure`` stored either an ASE ``Atoms`` or a pymatgen
``Structure`` instance and performed frequent conversions between them.
That made import-time heavy (ASE/pymatgen/matplotlib/sympy pulled in),
increased memory churn, and complicated running Og in minimal environments.

The classes below provide:

- A compact data model (lattice + fractional coordinates + species).
- Fast vectorized coordinate transforms.
- A small subset of the pymatgen ``Structure`` API that Og uses internally
  (``frac_coords``, ``cart_coords``, ``atomic_numbers``, ``get_neighbors``,
  ``translate_sites``, ``remove_sites``, ``replace``, ``make_supercell``,
  ``get_sorted_structure``, ``sort``, ``formula``).

The goal is not to re-implement full crystallography; it is to provide the
operations Og actually needs while keeping adapters to ASE/pymatgen available.

Notes
-----
- Neighbor search is currently O(N^2) with a minimal-image convention in
  fractional space. This is typically fine for the small/medium structures
  encountered in evolutionary search and descriptor generation. If you need
  millions of neighbor queries for large cells, consider adding a cell-list
  or k-d tree.
- Atomic numbers are resolved from ``oganesson.utilities.atomic_data.symbols``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from oganesson.utilities import atomic_data


# ---------- helpers ----------

_SYMBOL_TO_Z = {sym: i for i, sym in enumerate(atomic_data.symbols)}


def symbol_to_Z(symbol: str) -> int:
    """Convert an element symbol (e.g., ``'Si'``) to atomic number (Z)."""
    try:
        return int(_SYMBOL_TO_Z[symbol])
    except KeyError as e:
        raise ValueError(f"Unknown element symbol: {symbol!r}") from e


def wrap_frac(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coords into [0, 1)."""
    return frac - np.floor(frac)


def frac_deltas_min_image(dfrac: np.ndarray) -> np.ndarray:
    """Apply minimal image convention to fractional deltas into [-0.5, 0.5)."""
    return dfrac - np.round(dfrac)


# ---------- core data model ----------

@dataclass(frozen=True)
class OgLattice:
    """A 3x3 lattice matrix in row-vector form (same convention as ASE)."""

    matrix: np.ndarray  # shape (3,3)

    @staticmethod
    def from_matrix(matrix: Union[np.ndarray, Sequence[Sequence[float]]]) -> "OgLattice":
        m = np.asarray(matrix, dtype=float)
        if m.shape != (3, 3):
            raise ValueError(f"lattice matrix must be shape (3,3), got {m.shape}")
        return OgLattice(m)

    @property
    def inv_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    @property
    def lengths(self) -> Tuple[float, float, float]:
        a, b, c = (np.linalg.norm(self.matrix[i]) for i in range(3))
        return float(a), float(b), float(c)

    @property
    def a(self) -> float:
        return self.lengths[0]

    @property
    def b(self) -> float:
        return self.lengths[1]

    @property
    def c(self) -> float:
        return self.lengths[2]

    @property
    def alpha(self) -> float:
        """Angle (degrees) between b and c."""
        b, c = self.matrix[1], self.matrix[2]
        return float(np.degrees(np.arccos(np.clip(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c)), -1.0, 1.0))))

    @property
    def beta(self) -> float:
        """Angle (degrees) between a and c."""
        a, c = self.matrix[0], self.matrix[2]
        return float(np.degrees(np.arccos(np.clip(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)), -1.0, 1.0))))

    @property
    def gamma(self) -> float:
        """Angle (degrees) between a and b."""
        a, b = self.matrix[0], self.matrix[1]
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


@dataclass(frozen=True)
class OgSpecie:
    """Lightweight element wrapper compatible with the bits of pymatgen used in Og."""

    symbol: str

    @property
    def number(self) -> int:
        return symbol_to_Z(self.symbol)


@dataclass(frozen=True)
class OgSite:
    """A site with a specie and fractional coordinates."""

    specie: OgSpecie
    frac_coords: np.ndarray  # shape (3,)
    _lattice_matrix: np.ndarray  # shape (3,3), stored for coords transform

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates."""
        return self.frac_coords @ self._lattice_matrix

    @property
    def species_string(self) -> str:
        return self.specie.symbol


@dataclass(frozen=True)
class OgNeighbor:
    """Neighbor wrapper returned from ``get_neighbors``."""

    site: OgSite
    distance: float

    # Provide pymatgen-like shortcuts used elsewhere in Og
    @property
    def specie(self) -> OgSpecie:
        return self.site.specie

    @property
    def frac_coords(self) -> np.ndarray:
        return self.site.frac_coords

    @property
    def coords(self) -> np.ndarray:
        return self.site.coords


class OgNativeStructure:
    """Native structure container.

    Parameters
    ----------
    lattice
        Lattice as 3x3 matrix with row vectors.
    species
        Iterable of element symbols (e.g., ``['C','O','O']``).
    frac_coords
        Nx3 fractional coordinates.
    pbc
        Periodic boundary conditions. If a single bool, applied to all axes.
    """

    def __init__(
        self,
        lattice: Union[OgLattice, np.ndarray, Sequence[Sequence[float]]],
        species: Sequence[str],
        frac_coords: Union[np.ndarray, Sequence[Sequence[float]]],
        pbc: Union[bool, Sequence[bool]] = True,
    ) -> None:
        self.lattice = lattice if isinstance(lattice, OgLattice) else OgLattice.from_matrix(lattice)
        self._frac = np.asarray(frac_coords, dtype=float)
        if self._frac.ndim != 2 or self._frac.shape[1] != 3:
            raise ValueError(f"frac_coords must be shape (N,3), got {self._frac.shape}")
        self._frac = wrap_frac(self._frac)
        self._species = list(species)
        if len(self._species) != self._frac.shape[0]:
            raise ValueError("species and frac_coords must have same length")
        if isinstance(pbc, bool):
            self.pbc = (pbc, pbc, pbc)
        else:
            p = tuple(bool(x) for x in pbc)
            if len(p) != 3:
                raise ValueError("pbc must have length 3")
            self.pbc = p

    # ---- basic API ----

    def __len__(self) -> int:
        return int(self._frac.shape[0])

    def copy(self) -> "OgNativeStructure":
        return OgNativeStructure(self.lattice.matrix.copy(), list(self._species), self._frac.copy(), self.pbc)

    @property
    def frac_coords(self) -> np.ndarray:
        return self._frac

    @property
    def cart_coords(self) -> np.ndarray:
        return self._frac @ self.lattice.matrix

    @property
    def species(self) -> List[OgSpecie]:
        return [OgSpecie(s) for s in self._species]

    @property
    def atomic_numbers(self) -> List[int]:
        return [symbol_to_Z(s) for s in self._species]

    @property
    def formula(self) -> str:
        # Simple hill-like formula without charges/oxidation states.
        from collections import Counter
        c = Counter(self._species)
        # Keep deterministic order by atomic number
        items = sorted(c.items(), key=lambda kv: symbol_to_Z(kv[0]))
        out = []
        for sym, n in items:
            out.append(sym)
            if n != 1:
                out.append(str(n))
        return "".join(out)

    # ---- iteration / sites ----

    def __getitem__(self, idx: int) -> OgSite:
        m = self.lattice.matrix
        i = int(idx)
        return OgSite(OgSpecie(self._species[i]), np.asarray(self._frac[i], dtype=float), m)

    def __iter__(self):
        m = self.lattice.matrix
        for sym, f in zip(self._species, self._frac):
            yield OgSite(OgSpecie(sym), np.asarray(f, dtype=float), m)

    def append(
        self,
        specie: Union[str, OgSpecie],
        coords: Sequence[float],
        coords_are_cartesian: bool = True,
    ) -> "OgNativeStructure":
        """Return a new structure with an appended site.

        Mirrors a small part of pymatgen's ``Structure.append``.
        """
        sym = specie.symbol if isinstance(specie, OgSpecie) else str(specie)
        c = np.asarray(coords, dtype=float).reshape(3)
        if coords_are_cartesian:
            frac = c @ self.lattice.inv_matrix
        else:
            frac = c
        frac = wrap_frac(frac)
        new_species = self._species + [sym]
        new_frac = np.vstack([self._frac, frac[None, :]])
        return OgNativeStructure(self.lattice.matrix.copy(), new_species, new_frac, self.pbc)

# ---- mutation-like operations ----

    def sort(self) -> None:
        """In-place sort by atomic number then fractional coordinates."""
        Z = np.array([symbol_to_Z(s) for s in self._species], dtype=int)
        order = np.lexsort((self._frac[:, 2], self._frac[:, 1], self._frac[:, 0], Z))
        self._frac = self._frac[order]
        self._species = [self._species[i] for i in order]

    def get_sorted_structure(self) -> "OgNativeStructure":
        s = self.copy()
        s.sort()
        return s

    def translate_sites(self, indices: Iterable[int], vector: Sequence[float], frac_coords: bool = False) -> "OgNativeStructure":
        v = np.asarray(vector, dtype=float)
        if v.shape != (3,):
            v = v.reshape(3)
        s = self.copy()
        idx = np.fromiter(indices, dtype=int)
        if frac_coords:
            s._frac[idx] = wrap_frac(s._frac[idx] + v)
        else:
            # cart -> frac delta
            dfrac = v @ s.lattice.inv_matrix
            s._frac[idx] = wrap_frac(s._frac[idx] + dfrac)
        return s

    def remove_sites(self, indices: Sequence[int]) -> "OgNativeStructure":
        idx = np.array(sorted(set(int(i) for i in indices)), dtype=int)
        mask = np.ones(len(self), dtype=bool)
        mask[idx] = False
        s = OgNativeStructure(self.lattice.matrix.copy(), list(np.array(self._species)[mask]), self._frac[mask].copy(), self.pbc)
        return s

    def replace(self, i: int, specie: Union[str, OgSpecie]) -> "OgNativeStructure":
        s = self.copy()
        sym = specie.symbol if isinstance(specie, OgSpecie) else str(specie)
        s._species[int(i)] = sym
        return s

    def make_supercell(self, scaling_matrix: Union[int, Sequence[int], np.ndarray]) -> "OgNativeStructure":
        """Create a supercell.

        Supports:
        - int k -> k*k*k scaling
        - (kx,ky,kz)
        - 3x3 integer matrix (only diagonal recommended)
        """
        if isinstance(scaling_matrix, int):
            kx = ky = kz = int(scaling_matrix)
            S = np.diag([kx, ky, kz])
        else:
            S = np.asarray(scaling_matrix, dtype=int)
            if S.shape == (3,):
                S = np.diag(S)
            if S.shape != (3, 3):
                raise ValueError("scaling_matrix must be int, length-3, or 3x3")
        # Only support diagonal for now (common use in Og)
        if not np.all(S == np.diag(np.diagonal(S))):
            raise NotImplementedError("Non-diagonal supercells are not supported in native mode yet.")
        kx, ky, kz = map(int, np.diagonal(S))
        # new lattice
        new_lat = S @ self.lattice.matrix
        # replicate
        shifts = np.array([[i, j, k] for i in range(kx) for j in range(ky) for k in range(kz)], dtype=float)
        base = self._frac
        reps = (base[None, :, :] + shifts[:, None, :]) / np.array([kx, ky, kz], dtype=float)[None, None, :]
        reps = reps.reshape(-1, 3)
        species = self._species * (kx * ky * kz)
        return OgNativeStructure(new_lat, species, reps, self.pbc)

    # ---- neighbor search ----

    def get_neighbors(self, site: OgSite, r: float) -> List[OgNeighbor]:
        """Return neighbors within radius ``r`` (in Å).

        This mirrors the small subset of pymatgen's ``Structure.get_neighbors``
        that Og uses.
        """
        # compute dcart to all sites with minimal image
        fc0 = np.asarray(site.frac_coords, dtype=float)
        dfrac = self._frac - fc0[None, :]
        dfrac = frac_deltas_min_image(dfrac)
        dcart = dfrac @ self.lattice.matrix
        dist = np.linalg.norm(dcart, axis=1)
        mask = (dist < float(r)) & (dist > 1e-12)
        idx = np.where(mask)[0]
        m = self.lattice.matrix
        out: List[OgNeighbor] = []
        for j in idx:
            # neighbor site should carry the wrapped frac coords of the image we used
            neigh_frac = wrap_frac(fc0 + dfrac[j])
            neigh_site = OgSite(OgSpecie(self._species[j]), neigh_frac, m)
            out.append(OgNeighbor(neigh_site, float(dist[j])))
        return out

    # ---- adapters (optional deps) ----

    def to_pymatgen(self):
        """Convert to pymatgen ``Structure`` (requires pymatgen)."""
        try:
            from pymatgen.core import Structure  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("pymatgen is required for to_pymatgen()") from e
        return Structure(
            lattice=self.lattice.matrix,
            species=self._species,
            coords=self.cart_coords,
            coords_are_cartesian=True,
        )

    def to_ase_atoms(self):
        """Convert to ASE ``Atoms`` (requires ase)."""
        try:
            from ase import Atoms  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("ase is required for to_ase_atoms()") from e
        return Atoms(
            scaled_positions=self._frac.copy(),
            symbols=list(self._species),
            cell=self.lattice.matrix.copy(),
            pbc=True,
        )

    def to(self, fmt: str) -> str:
        """Serialize structure to a format string (currently: ``'cif'`` only)."""
        fmt = fmt.lower().strip()
        if fmt != "cif":
            raise ValueError(f"Unsupported format: {fmt!r}. Only 'cif' is supported.")
        pmg = self.to_pymatgen()
        return pmg.to(fmt="cif")
