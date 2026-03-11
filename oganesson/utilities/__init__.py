"""oganesson.utilities

Lightweight utilities used across the package.

This module historically imported ASE/pymatgen unconditionally, which made the
whole package fail to import in minimal environments. Those imports are now
guarded; functionality that requires ASE/pymatgen will raise if used without the
dependency installed.
"""

import numpy as np
import oganesson.utilities.atomic_data  # re-export side-effect table

# Optional deps
try:
    from ase.atoms import Atoms, Cell  # type: ignore
    from ase.io import read, write  # type: ignore
    from ase.db import connect  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Atoms = Cell = read = write = connect = None  # type: ignore

try:
    from pymatgen.core import Structure  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Structure = None  # type: ignore


def formula(structure):
    """Return a list of element symbols in the given structure.

    Works with pymatgen `Structure`, ASE `Atoms`, or Og native structures.
    """
    # OgNativeStructure / OgStructure expose species strings via iteration
    try:
        return [site.species_string if hasattr(site, "species_string") else str(site.specie.symbol) for site in structure]
    except Exception:
        pass

    # ASE Atoms
    if Atoms is not None and isinstance(structure, Atoms):
        return list(structure.get_chemical_symbols())

    # pymatgen Structure
    if Structure is not None and isinstance(structure, Structure):
        return [str(s) for s in structure.species]

    raise TypeError(f"Unsupported structure type for formula(): {type(structure)!r}")


def get_index_positions(list_of_elems, element):
    return np.where(np.array(list_of_elems) == element)


def epsilon(a, b):
    return abs(a - b) < 1e-6
