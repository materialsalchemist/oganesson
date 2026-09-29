"""Event detectors find planted events of the right size and stay silent on intact or merely perturbed structures."""

import numpy as np
import pytest

pytest.importorskip("ovito")

from ase import Atoms  # noqa: E402
from ase.build import bulk, molecule  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from oganesson.md_anomalies.events import (  # noqa: E402
    BCC,
    EVENT_RULES,
    EventConfig,
    EventRule,
    compute_events,
    dendrite_cluster,
    event_masks,
    events_from_masks,
    foreign_phase_cluster,
    largest_cluster,
    largest_void_radius,
    melt_cluster,
    ptm_types,
    reference_surface_z,
)


def _water_box(n=5, spacing=3.1, seed=0):
    """n^3 randomly rotated water molecules on a cubic grid (about liquid density at 3.1 A spacing)."""
    rng = np.random.default_rng(seed)
    template = molecule("H2O")
    template.positions -= template.get_center_of_mass()
    atoms = Atoms(cell=[n * spacing] * 3, pbc=True)
    for ijk in np.ndindex(n, n, n):
        centre = (np.array(ijk) + 0.5) * spacing
        pos = Rotation.random(random_state=rng).apply(template.positions) + centre
        atoms += Atoms(template.get_chemical_symbols(), positions=pos)
    return atoms


def _li_slab():
    """bcc Li(100) slab, 4x4 in plane and 8 sublayers thick, with vacuum above and below."""
    slab = bulk("Li", "bcc", a=3.44, cubic=True).repeat((4, 4, 4))
    slab.cell[2, 2] += 20.0
    slab.positions[:, 2] += 10.0
    return slab


def _bcc_fe(n=8):
    return bulk("Fe", "bcc", a=2.87, cubic=True).repeat(n)


def _scramble_sphere(atoms, radius, rng, amplitude=0.7):
    """Randomly displace every atom within `radius` of the cell centre, destroying local crystal order."""
    out = atoms.copy()
    centre = out.cell.array.sum(axis=0) / 2
    inside = np.linalg.norm(out.positions - centre, axis=1) < radius
    out.positions[inside] += rng.uniform(-amplitude, amplitude, (inside.sum(), 3))
    return out, int(inside.sum())


def test_largest_cluster_counts_only_connected_masked_atoms():
    atoms = _bcc_fe(4)
    mask = np.zeros(len(atoms), dtype=bool)
    assert largest_cluster(atoms, mask, 3.4) == 0
    mask[:] = True
    assert largest_cluster(atoms, mask, 3.4) == len(atoms)
    # two atoms far apart are two clusters of one
    far = np.zeros(len(atoms), dtype=bool)
    far[0] = True
    far[int(np.argmax(atoms.get_distances(0, range(len(atoms)), mic=True)))] = True  # farthest under periodic wrapping
    assert largest_cluster(atoms, far, 3.4) == 1


def test_melt_cluster_finds_a_scrambled_region_and_ignores_the_intact_crystal():
    rng = np.random.default_rng(0)
    crystal = _bcc_fe()
    reference_types = ptm_types(crystal)
    assert (reference_types == BCC).all()
    assert melt_cluster(ptm_types(crystal), reference_types, crystal, 3.4) == 0

    melted, n_scrambled = _scramble_sphere(crystal, 6.0, rng)
    assert n_scrambled >= 60
    size = melt_cluster(ptm_types(melted), reference_types, melted, 3.4)
    assert size >= EVENT_RULES["melt_cluster"].threshold  # >= 50 atoms: an event

    small, n_small = _scramble_sphere(crystal, 3.0, rng)
    assert 0 < melt_cluster(ptm_types(small), reference_types, small, 3.4) < EVENT_RULES["melt_cluster"].threshold


def test_foreign_phase_cluster_sees_fcc_in_a_bcc_host_but_not_bcc_itself():
    fcc = bulk("Fe", "fcc", a=3.6, cubic=True).repeat(4)
    assert foreign_phase_cluster(ptm_types(fcc), BCC, fcc, 3.0) == len(fcc)
    bcc = _bcc_fe(4)
    assert foreign_phase_cluster(ptm_types(bcc), BCC, bcc, 3.4) == 0


def test_void_radius_grows_when_molecules_are_removed_from_water():
    atoms = _water_box()
    oxygens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
    packed = largest_void_radius(atoms, oxygens)
    assert packed < EVENT_RULES["void_radius"].threshold

    centre = atoms.cell.array.sum(axis=0) / 2
    d = np.linalg.norm(atoms.positions[oxygens] - centre, axis=1)
    kept_o = [o for o, dist in zip(oxygens, d) if dist > 5.0]
    assert largest_void_radius(atoms, kept_o) >= 4.5  # a 5 A cavity, sampled on a 0.5 A grid


def test_largest_void_radius_rejects_a_triclinic_cell():
    atoms = bulk("Fe", "bcc", a=2.87)  # primitive, non-orthorhombic
    with pytest.raises(ValueError):
        largest_void_radius(atoms, range(len(atoms)))


def test_dendrite_cluster_counts_lifted_surface_patches():
    atoms = _li_slab()
    mobile_li = list(range(len(atoms)))
    surface = reference_surface_z(atoms, mobile_li)
    assert dendrite_cluster(atoms, mobile_li, surface, 2.0, 4.1) == 0

    # lift the 12 surface atoms nearest the middle of the top layer by 2.5 A: one connected protrusion
    z = atoms.positions[mobile_li, 2]
    top = np.array(mobile_li)[z >= z.max() - 0.5]
    middle = atoms.positions[top].mean(axis=0)
    order = top[np.argsort(np.linalg.norm(atoms.positions[top, :2] - middle[:2], axis=1))]
    lifted = atoms.copy()
    lifted.positions[order[:12], 2] += 2.5
    assert dendrite_cluster(lifted, mobile_li, surface, 2.0, 4.1) == 12

    # a 1.5 A lift stays below the 2.0 A height: surface roughness, not a dendrite
    rough = atoms.copy()
    rough.positions[order[:12], 2] += 1.5
    assert dendrite_cluster(rough, mobile_li, surface, 2.0, 4.1) == 0


def test_event_masks_need_persistence_and_ignore_nan_frames():
    rules = {"x": EventRule(threshold=5, persist_frames=3)}
    values = {"x": np.array([np.nan, 6, 6, 0, 6, 6, 6, 6, 0, np.nan])}
    mask = event_masks(values, rules)["x"]
    assert mask.tolist() == [False, False, False, False, True, True, True, True, False, False]
    [event] = events_from_masks({"x": mask}, values)
    assert event == {"type": "x", "start_frame": 4, "end_frame": 7, "peak": 6.0}


def test_compute_events_on_a_melting_trajectory_starts_at_the_melt_not_before():
    """Frames 0-59 are an intact (thermally noisy) crystal, frames 60+ carry a scrambled 60-atom
    region. The melt detector must stay at zero before the melt, fire after it, and never label the
    truncated-average frames at either end."""
    rng = np.random.default_rng(1)
    crystal = _bcc_fe()
    centre = crystal.cell.array.sum(axis=0) / 2
    inside = np.linalg.norm(crystal.positions - centre, axis=1) < 6.0
    melted_offsets = rng.uniform(-0.7, 0.7, (inside.sum(), 3))
    frames = []
    for t in range(100):
        frame = crystal.copy()
        frame.positions += rng.normal(0.0, 0.05, frame.positions.shape)
        if t >= 60:
            frame.positions[inside] += melted_offsets
        frames.append(frame)
    cfg = EventConfig(ptm_indices=list(range(len(crystal))), host_structure=BCC)
    values = compute_events(frames, cfg, reference_frame=40)
    melt = values["melt_cluster"]
    assert np.isnan(melt[:15]).all() and np.isnan(melt[-15:]).all()
    assert np.nanmax(melt[15:45]) == 0
    assert np.nanmin(melt[76:85]) >= EVENT_RULES["melt_cluster"].threshold
    mask = event_masks(values)["melt_cluster"]
    first = int(np.flatnonzero(mask)[0])
    assert 45 <= first <= 76  # onset lies in the averaging window around the melt at frame 60
    assert np.nanmax(values["vacancies"][15:45]) == 0
