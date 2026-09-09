"""Regression tests for OgStructure's native data model.

Covers: round trips between the native array store and genuine
pymatgen.core.Structure / ase.Atoms objects, the additive dual-API surface
(pymatgen- and ASE-shaped reads computed directly from the native arrays),
and the mutation-persistence fixes required once `OgStructure.structure`
became a computed property instead of a stored attribute (a bare
`self.structure.mutate()` with no reassignment now discards the mutation).
"""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from pymatgen.core import Lattice, Structure

from oganesson._native_structure import NativeStructure
from oganesson.ogstructure import OgStructure

TRICLINIC_MATRIX = np.array([[3.0, 0, 0], [0.5, 3.5, 0], [0.2, 0.3, 4.0]])


def _triclinic_structure() -> Structure:
    return Structure(
        lattice=Lattice(TRICLINIC_MATRIX),
        species=[14, 8],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=True,
    )


def test_round_trip_from_pymatgen():
    s0 = _triclinic_structure()
    s1 = NativeStructure.from_pymatgen(s0).to_pymatgen()
    assert np.allclose(s0.frac_coords, s1.frac_coords)
    assert np.allclose(s0.cart_coords, s1.cart_coords)
    assert np.allclose(s0.lattice.matrix, s1.lattice.matrix)
    assert tuple(s0.atomic_numbers) == tuple(s1.atomic_numbers)


def test_round_trip_from_ase():
    a0 = Atoms(
        numbers=[14, 8],
        positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=TRICLINIC_MATRIX,
        pbc=True,
    )
    native = NativeStructure.from_ase(a0)
    assert np.allclose(a0.get_positions(), native.cart_coords)
    assert np.allclose(a0.cell.array, native.lattice_matrix)
    assert np.array_equal(a0.get_atomic_numbers(), native.atomic_numbers)


def test_geometry_helpers_match_pymatgen():
    s0 = _triclinic_structure()
    native = NativeStructure.from_pymatgen(s0)
    assert abs(native.volume - s0.lattice.volume) < 1e-9
    assert np.allclose(
        native.cell_lengths_and_angles,
        [s0.lattice.a, s0.lattice.b, s0.lattice.c,
         s0.lattice.alpha, s0.lattice.beta, s0.lattice.gamma],
    )
    assert np.allclose(native.frac_coords, s0.frac_coords)


def test_get_all_distances_matches_ase():
    a0 = Atoms(
        numbers=[14, 8],
        positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=TRICLINIC_MATRIX,
        pbc=True,
    )
    native = NativeStructure.from_ase(a0)
    assert np.allclose(native.get_all_distances(mic=True), a0.get_all_distances(mic=True))


def test_to_ase_matches_legacy_pymatgen_to_ase():
    """The fast to_ase() path must match the old convert-on-demand behavior,
    including its lossy cellpar reorientation for skewed lattices -- this is
    a deliberate bug-for-bug preservation, not a bug in the new code."""
    og = OgStructure(_triclinic_structure())
    fast = og.to_ase()
    slow = OgStructure.pymatgen_to_ase(og.structure)
    assert np.allclose(fast.cell.array, slow.cell.array)
    assert np.allclose(fast.get_positions(), slow.get_positions())
    assert np.array_equal(fast.get_atomic_numbers(), slow.get_atomic_numbers())


def test_dual_api_surface_on_ogstructure():
    cu = OgStructure(bulk("Cu", "fcc", a=3.6))
    assert np.allclose(cu.frac_coords, cu.structure.frac_coords)
    assert np.allclose(cu.cart_coords, cu.structure.cart_coords)
    assert abs(cu.volume - cu.structure.volume) < 1e-9
    assert np.allclose([cu.a, cu.b, cu.c],
                        [cu.structure.lattice.a, cu.structure.lattice.b, cu.structure.lattice.c])
    assert np.allclose(cu.positions, cu.to_ase().get_positions())
    assert np.array_equal(cu.numbers, cu.to_ase().get_atomic_numbers())
    assert cu.symbols == ["Cu"]
    assert np.allclose(cu.cell.cellpar(), cu.to_ase().cell.cellpar())
    assert cu.pbc == (True, True, True)
    assert np.allclose(cu.get_cell_lengths_and_angles(), cu.to_ase().cell.cellpar())
    assert np.allclose(cu.get_all_distances(mic=True), cu.to_ase().get_all_distances(mic=True))
    assert np.array_equal(cu.atomic_numbers, cu.structure.atomic_numbers)
    assert len(cu) == 1


def test_ogstructure_wrapping_ogstructure_no_longer_crashes():
    cu = OgStructure(bulk("Cu", "fcc", a=3.6))
    wrapped = OgStructure(cu)
    assert wrapped.structure.composition.reduced_formula == "Cu"


def test_init_sort_persists():
    disordered = Structure(
        lattice=Lattice.cubic(4.0),
        species=[8, 3],
        coords=[[0.5, 0.5, 0.5], [0, 0, 0]],
        coords_are_cartesian=False,
    )
    og = OgStructure(disordered)
    assert [str(sp) for sp in og.structure.species] == ["Li", "O"]


def test_substitutions_random_persists():
    alloy = OgStructure(bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2)))
    alloy.substitutions_random("Cu", {"Al": 4, "Cu": 4})
    comp = alloy.structure.composition.as_dict()
    assert comp.get("Al", 0) == 4
    assert comp.get("Cu", 0) == 4


def test_add_interstitial_rejection_reverts_append():
    host = OgStructure(bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3)))
    n_before = len(host)
    result = host.add_interstitial("H", divisions=[4, 4, 4], threshold=2)
    assert result is False
    assert len(host) == n_before


def test_add_interstitial_success_persists_append():
    host = OgStructure(bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3)))
    n_before = len(host)
    result = host.add_interstitial("H", divisions=[4, 4, 4], threshold=0.5)
    assert result is not False
    assert len(host) == n_before + 1
    assert "H" in host.structure.composition.as_dict()


def test_add_interstitial_fill_multiple_spots_stays_consistent():
    host = OgStructure(bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3)))
    n_before = len(host)
    host.add_interstitial("H", divisions=[4, 4, 4], threshold=0.5, fill_multiple_spots=True)
    assert len(host) >= n_before
    assert len(host.structure) == len(host)


def test_fracture_pattern_translate_sites_persists():
    """Reproduces the exact `s = self.structure; s.translate_sites(...);
    self.structure = s` pattern used in the fracture-family methods."""
    og = OgStructure(bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2)))
    before_frac = og.structure.frac_coords.copy()
    lattice = og.structure.lattice.matrix.copy()
    expected_delta = np.array([0, 0, 0.1]) @ np.linalg.inv(lattice)
    s = og.structure
    s.translate_sites(indices=range(len(og)), vector=[0, 0, 0.1], frac_coords=False)
    og.structure = s
    after_frac = og.structure.frac_coords
    delta = (after_frac - before_frac) % 1.0
    assert np.allclose(delta[0], expected_delta % 1.0)
