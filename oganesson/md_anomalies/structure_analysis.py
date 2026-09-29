"""Structure-analysis helpers shared by the event detectors.

Time averaging over unwrapped trajectories, OVITO pipelines (Wigner-Seitz defect counts), and covalent
bond graphs / molecule identity from ASE neighbour lists. OVITO is imported lazily, only by the functions
that need it; install it with ``pip install oganesson[md-events]``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def bond_cutoff(a: str, b: str, factor: float = 1.2) -> float:
    return factor * (covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]])


def unwrapped_scaled_positions(frames: Sequence[Atoms]) -> np.ndarray:
    """(n_frames, n_atoms, 3) scaled positions made continuous across periodic boundaries."""
    s = np.stack([f.get_scaled_positions(wrap=False) for f in frames])
    steps = np.diff(s, axis=0)
    steps -= np.round(steps)
    return np.concatenate([s[:1], s[:1] + np.cumsum(steps, axis=0)])


def averaged_atoms(frames: Sequence[Atoms], unwrapped: np.ndarray, t: int, half_window: int,
                   indices: Sequence[int]) -> Atoms:
    lo, hi = max(0, t - half_window), min(len(frames), t + half_window + 1)
    idx = np.asarray(list(indices), dtype=int)
    out = frames[t][idx]
    out.set_scaled_positions(unwrapped[lo:hi, idx].mean(axis=0))
    return out


def _pipeline(atoms: Atoms):
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource

    return Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))


def defect_counts(atoms: Atoms, reference: Atoms) -> Tuple[int, int]:
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import WignerSeitzAnalysisModifier
    from ovito.pipeline import StaticSource

    ws = WignerSeitzAnalysisModifier(affine_mapping=WignerSeitzAnalysisModifier.AffineMapping.ToReference)
    ws.reference = StaticSource(data=ase_to_ovito(reference))
    pipe = _pipeline(atoms)
    pipe.modifiers.append(ws)
    attributes = pipe.compute().attributes
    return int(attributes["WignerSeitz.vacancy_count"]), int(attributes["WignerSeitz.interstitial_count"])


def bond_pairs(atoms: Atoms, indices: Sequence[int], factor: float = 1.2) -> set:
    idx = np.asarray(list(indices), dtype=int)
    sub = atoms[idx]
    species = sorted(set(sub.get_chemical_symbols()))
    cutoffs = {(x, y): bond_cutoff(x, y, factor) for x in species for y in species}
    i, j = neighbor_list("ij", sub, cutoffs)
    keep = i < j
    return set(zip(idx[i[keep]].tolist(), idx[j[keep]].tolist()))


def _molecules_from_pairs(indices: Sequence[int], pairs: set) -> set:
    """Connected components of `indices` under `pairs`, as frozensets of atom indices."""
    idx = np.asarray(list(indices), dtype=int)
    local = {g: n for n, g in enumerate(idx.tolist())}
    rows = [local[p] for p, _ in pairs]
    cols = [local[q] for _, q in pairs]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(idx), len(idx)))
    _, labels = connected_components(graph, directed=False)
    return {frozenset(idx[labels == c].tolist()) for c in np.unique(labels)}


def molecules(atoms: Atoms, indices: Sequence[int], factor: float = 1.2) -> set:
    """Covalently bonded groups among `indices`, as frozensets of atom indices."""
    return _molecules_from_pairs(indices, bond_pairs(atoms, indices, factor))


def _robust_bond_pairs(frames: Sequence[Atoms], indices: Sequence[int], window: int, factor: float = 1.2) -> set:
    """Bond pairs present in a majority of the first `window` frames, so one frame's transient thermal
    bond-length fluctuation can't permanently poison a reference molecule topology (see module docstring)."""
    n = min(window, len(frames))
    counts: Dict[Tuple[int, int], int] = {}
    for frame in frames[:n]:
        for pair in bond_pairs(frame, indices, factor):
            counts[pair] = counts.get(pair, 0) + 1
    return {pair for pair, count in counts.items() if count > n / 2}
