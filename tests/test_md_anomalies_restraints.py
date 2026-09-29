"""Restraints must be exact derivatives (forces and virial) and move with the cell."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress

from oganesson.md_anomalies.restraints import RestraintCalculator, RestraintGroup, Slab, Sphere

CELLS = {
    "orthorhombic": np.diag([12.0, 13.0, 14.0]),
    "triclinic": np.array([[12.0, 0.0, 0.0], [2.0, 13.0, 0.0], [1.0, 1.5, 14.0]]),
}


def _atoms(cell, seed=0):
    rng = np.random.default_rng(seed)
    return Atoms("Ar20", scaled_positions=rng.random((20, 3)), cell=cell, pbc=True)


def _groups():
    return [
        RestraintGroup("contain", [Sphere(list(range(8)), [0.5, 0.5, 0.5], 2.0, 3.0, True)]),
        RestraintGroup("exclude", [Sphere(list(range(8, 14)), [0.2, 0.3, 0.4], 4.0, 2.0, False)]),
        RestraintGroup("wall", [Slab(list(range(14, 20)), 2, 0.3, 0.6, 4.0)]),
    ]


@pytest.mark.parametrize("cell_name", sorted(CELLS))
def test_forces_and_stress_match_finite_differences(cell_name):
    atoms = _atoms(CELLS[cell_name])
    atoms.calc = RestraintCalculator(_groups())
    assert atoms.get_potential_energy() > 0.0
    np.testing.assert_allclose(atoms.get_forces(), calculate_numerical_forces(atoms, 1e-5), atol=1e-6)
    np.testing.assert_allclose(atoms.get_stress(), calculate_numerical_stress(atoms, 1e-6), atol=1e-8)


def test_released_groups_exert_nothing():
    atoms = _atoms(CELLS["triclinic"])
    calc = RestraintCalculator(_groups())
    atoms.calc = calc
    for name in calc.groups:
        calc.set_scale(name, 0.0)
    assert atoms.get_potential_energy() == 0.0
    assert np.abs(atoms.get_forces()).max() == 0.0


def test_anchors_follow_affine_cell_changes():
    atoms = Atoms("Ar", scaled_positions=[[0.5, 0.5, 0.5]], cell=np.diag([10.0, 10.0, 10.0]), pbc=True)
    atoms.calc = RestraintCalculator([RestraintGroup("r", [Sphere([0], [0.5, 0.5, 0.5], 0.1, 5.0, True)])])
    atoms.set_cell(np.array([[11.0, 0.0, 0.0], [1.0, 10.0, 0.0], [0.0, 0.0, 9.0]]), scale_atoms=True)
    assert atoms.get_potential_energy() == 0.0


def test_group_dict_roundtrip():
    for group in _groups():
        assert RestraintGroup.from_dict(group.to_dict()) == group




def test_duplicate_group_names_raise_error():
    """Duplicate RestraintGroup names should raise ValueError during initialization."""
    with pytest.raises(ValueError, match="duplicate RestraintGroup name"):
        RestraintCalculator([
            RestraintGroup("same_name", [Sphere([0], [0.5, 0.5, 0.5], 1.0, 2.0, True)]),
            RestraintGroup("same_name", [Sphere([1], [0.3, 0.3, 0.3], 1.0, 2.0, True)]),
        ])


def test_empty_indices_sphere_and_slab():
    """Sphere and Slab with empty indices should contribute zero energy and force."""
    atoms = _atoms(CELLS["orthorhombic"])
    calc = RestraintCalculator([
        RestraintGroup("empty_sphere", [Sphere([], [0.5, 0.5, 0.5], 2.0, 3.0, True)]),
        RestraintGroup("empty_slab", [Slab([], 2, 0.3, 0.6, 4.0)]),
    ])
    atoms.calc = calc
    assert atoms.get_potential_energy() == 0.0
    assert np.abs(atoms.get_forces()).max() == 0.0
