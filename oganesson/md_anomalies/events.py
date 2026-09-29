"""Event detectors for MD trajectories: discrete, localized changes in structure or chemistry.

Each detector returns a size per frame, compared against a reference structure taken from the
trajectory itself (for example just before a trigger fires):

- ``melt_cluster``: atoms in the largest connected cluster that PTM no longer recognises (local melting)
- ``foreign_phase_cluster``: largest cluster in a crystal structure other than the host's
- ``vacancies`` / ``vacancies_2``: Wigner-Seitz vacancies on one or two sublattices
- ``molecule_changes``: molecules whose covalent topology changed (bond breaking, fragmentation)
- ``void_radius``: radius of the largest empty sphere (cavitation)
- ``dendrite_cluster``: largest connected cluster of atoms protruding above a surface plane

Detectors deliberately avoid state variables such as temperature or potential energy, which a
perturbation changes directly. `event_masks` turns sizes into events with a threshold and a minimum
duration (`EventRule`). PTM and Wigner-Seitz need OVITO: ``pip install oganesson[md-events]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .structure_analysis import (
    _molecules_from_pairs,
    _pipeline,
    _robust_bond_pairs,
    averaged_atoms,
    defect_counts,
    molecules,
    unwrapped_scaled_positions,
)

# PTM structure codes (ovito PolyhedralTemplateMatchingModifier.Type)
OTHER, FCC, HCP, BCC, ICO = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class EventRule:
    threshold: float  # the event fires when the detector's value is >= threshold
    persist_frames: int  # ... for at least this many consecutive frames


# Default thresholds. Calibrate them per system on unperturbed runs: raise a threshold until no normal
# run fires. void_radius is 4.0 A because packed water (0.9 g/cm^3) already has a 3.43 A empty sphere
# to the nearest O. Pass your own `rules` to event_masks() to override.
EVENT_RULES: Dict[str, EventRule] = {
    "melt_cluster": EventRule(50, 20),
    "foreign_phase_cluster": EventRule(10, 20),
    "vacancies": EventRule(1, 50),
    "vacancies_2": EventRule(1, 50),
    "molecule_changes": EventRule(1, 20),
    "void_radius": EventRule(4.0, 20),
    "dendrite_cluster": EventRule(10, 20),
}


def ptm_types(atoms: Atoms) -> np.ndarray:
    """Per-atom PTM structure type (OTHER, FCC, HCP, BCC or ICO)."""
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM

    pipe = _pipeline(atoms)
    pipe.modifiers.append(PTM())
    return np.asarray(pipe.compute().particles["Structure Type"], dtype=int)


def largest_cluster(atoms: Atoms, mask: np.ndarray, cutoff: float) -> int:
    """Size of the largest group of `mask`ed atoms connected by neighbour distances below `cutoff`."""
    idx = np.flatnonzero(mask)
    if idx.size <= 1:
        return int(idx.size)
    i, j = neighbor_list("ij", atoms[idx], cutoff)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(idx.size, idx.size))
    _, labels = connected_components(graph, directed=False)
    return int(np.bincount(labels).max())


def melt_cluster(types: np.ndarray, reference_types: np.ndarray, atoms: Atoms, cutoff: float) -> int:
    """Largest cluster of atoms that PTM no longer recognises but did in the reference."""
    return largest_cluster(atoms, (types == OTHER) & (reference_types != OTHER), cutoff)


def foreign_phase_cluster(types: np.ndarray, host_type: int, atoms: Atoms, cutoff: float) -> int:
    """Largest cluster of atoms in a crystal structure other than the host's (e.g. FCC or HCP in bcc)."""
    return largest_cluster(atoms, (types != OTHER) & (types != host_type), cutoff)


def largest_void_radius(atoms: Atoms, indices: Sequence[int], spacing: float = 0.5) -> float:
    """Radius of the largest empty sphere: the largest distance from a grid point to its nearest atom."""
    cell = atoms.cell.array
    if not np.allclose(cell, np.diag(np.diag(cell))):
        raise ValueError("largest_void_radius needs an orthorhombic cell")
    lengths = np.diag(cell)
    pos = np.mod(atoms.positions[np.asarray(list(indices), dtype=int)], lengths)
    tree = cKDTree(pos, boxsize=lengths)
    axes = [np.arange(0.0, length, spacing) for length in lengths]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return float(tree.query(grid)[0].max())


def dendrite_cluster(atoms: Atoms, indices: Sequence[int], surface_z: float, height: float, cutoff: float) -> int:
    """Largest cluster of `indices` atoms more than `height` above the reference surface plane."""
    idx = np.asarray(list(indices), dtype=int)
    mask = np.zeros(len(atoms), dtype=bool)
    mask[idx[atoms.positions[idx, 2] > surface_z + height]] = True
    return largest_cluster(atoms, mask, cutoff)


@dataclass
class EventConfig:
    """Which detectors a system runs, on which atoms. `None` switches a detector off."""
    ptm_indices: Optional[List[int]] = None  # crystalline atoms (host sublattice): melt, phase, vacancies
    host_structure: Optional[int] = None  # PTM type of the intact host, for foreign_phase_cluster
    cluster_cutoff: float = 3.4  # A, neighbour distance that connects a cluster of crystalline atoms
    second_sublattice: Optional[List[int]] = None  # e.g. O in MgO: its own vacancy count
    molecular_indices: Optional[List[int]] = None  # atoms whose molecule identity is tracked
    void_indices: Optional[List[int]] = None  # atoms that bound empty space (e.g. water O)
    dendrite_indices: Optional[List[int]] = None  # mobile Li atoms
    dendrite_height: float = 2.0  # A above the reference surface plane
    # A, Li-Li distance that connects a protrusion: between bcc Li's second (3.44 A, the in-plane spacing of
    # a (100) layer) and third (4.87 A) neighbour shells, so atoms lifted from one layer stay connected.
    dendrite_cutoff: float = 4.1
    half_window: int = 15  # 31-frame (300 fs) structure averaging


def reference_surface_z(atoms: Atoms, indices: Sequence[int], layer_tolerance: float = 0.5) -> float:
    """Mean z of the top atomic layer of `indices` (atoms within `layer_tolerance` of the highest)."""
    z = atoms.positions[np.asarray(list(indices), dtype=int), 2]
    return float(z[z >= z.max() - layer_tolerance].mean())


def compute_events(frames: Sequence[Atoms], cfg: EventConfig, reference_frame: int) -> Dict[str, np.ndarray]:
    """Per-frame detector values. The reference is the structure averaged over the window ending at
    `reference_frame` (the trigger's first frame, or the matching frame of a normal run). Frames within
    `half_window` of either end, where the structure average is truncated, get NaN for the structural
    detectors and are never labelled."""
    n = len(frames)
    hw = cfg.half_window
    out: Dict[str, np.ndarray] = {}
    ref_t = max(hw, min(reference_frame, n - 1) - hw)
    needs_average = any(x is not None for x in (cfg.ptm_indices, cfg.second_sublattice, cfg.dendrite_indices))
    unwrapped = unwrapped_scaled_positions(frames) if needs_average else None
    interior = np.zeros(n, dtype=bool)
    interior[hw:n - hw] = True

    if cfg.ptm_indices is not None:
        reference = averaged_atoms(frames, unwrapped, ref_t, hw, cfg.ptm_indices)
        reference_types = ptm_types(reference)
        melt, phase, vac = (np.full(n, np.nan) for _ in range(3))
        for t in np.flatnonzero(interior):
            averaged = averaged_atoms(frames, unwrapped, t, hw, cfg.ptm_indices)
            types = ptm_types(averaged)
            melt[t] = melt_cluster(types, reference_types, averaged, cfg.cluster_cutoff)
            if cfg.host_structure is not None:
                phase[t] = foreign_phase_cluster(types, cfg.host_structure, averaged, cfg.cluster_cutoff)
            vac[t] = defect_counts(averaged, reference)[0]
        out["melt_cluster"] = melt
        if cfg.host_structure is not None:
            out["foreign_phase_cluster"] = phase
        out["vacancies"] = vac
    if cfg.second_sublattice is not None:
        reference_2 = averaged_atoms(frames, unwrapped, ref_t, hw, cfg.second_sublattice)
        vac2 = np.full(n, np.nan)
        for t in np.flatnonzero(interior):
            vac2[t] = defect_counts(averaged_atoms(frames, unwrapped, t, hw, cfg.second_sublattice), reference_2)[0]
        out["vacancies_2"] = vac2
    if cfg.molecular_indices is not None:
        # Reference topology by majority vote over the pre-reference window, so one frame's thermally
        # stretched bond is not baked in as a broken molecule (see structure_analysis._robust_bond_pairs).
        lo = max(0, ref_t - hw)
        ref_mol = _molecules_from_pairs(cfg.molecular_indices,
                                        _robust_bond_pairs(frames[lo:ref_t + 1], cfg.molecular_indices, 2 * hw + 1))
        out["molecule_changes"] = np.array([len(ref_mol - molecules(f, cfg.molecular_indices)) for f in frames], dtype=float)
    if cfg.void_indices is not None:
        out["void_radius"] = np.array([largest_void_radius(f, cfg.void_indices) for f in frames])
    if cfg.dendrite_indices is not None:
        surface = reference_surface_z(averaged_atoms(frames, unwrapped, ref_t, hw, range(len(frames[0]))),
                                      cfg.dendrite_indices)
        out["dendrite_cluster"] = np.array([dendrite_cluster(f, cfg.dendrite_indices, surface, cfg.dendrite_height,
                                                             cfg.dendrite_cutoff) for f in frames], dtype=float)
    return out


def event_masks(values: Dict[str, np.ndarray], rules: Optional[Dict[str, EventRule]] = None) -> Dict[str, np.ndarray]:
    """Per-detector boolean masks: value >= threshold for at least `persist_frames` consecutive frames.
    NaN (truncated-average frames) never counts."""
    rules = rules or EVENT_RULES
    out = {}
    for name, series in values.items():
        rule = rules[name]
        hit = np.nan_to_num(np.asarray(series, dtype=float), nan=-np.inf) >= rule.threshold
        out[name] = _persistent(hit, rule.persist_frames)
    return out


def _persistent(mask: np.ndarray, frames: int) -> np.ndarray:
    out = np.zeros_like(mask)
    start = None
    for t, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = t
        elif not value and start is not None:
            if t - start >= frames:
                out[start:t] = True
            start = None
    return out


def events_from_masks(masks: Dict[str, np.ndarray], values: Dict[str, np.ndarray]) -> List[Dict]:
    """Contiguous event intervals per detector, with the detector's peak value in each."""
    events: List[Dict] = []
    for name, mask in masks.items():
        padded = np.concatenate([[False], mask, [False]]).astype(int)
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1) - 1
        for s, e in zip(starts, ends):
            events.append({"type": name, "start_frame": int(s), "end_frame": int(e),
                           "peak": float(np.nanmax(values[name][s:e + 1]))})
    return sorted(events, key=lambda ev: (ev["start_frame"], ev["type"]))
