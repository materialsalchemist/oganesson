"""Clock, frame writer and in-place integrator changes used by triggers and equilibration."""

import math

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
from ase.io import Trajectory, read
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.constraints import FixAtoms

from oganesson.md_anomalies.integrator import (
    FrameWriter,
    SimClock,
    bussi_temperature,
    frozen_atom_indices,
    invalidate_calculator,
    remove_net_momentum,
    set_bussi_temperature,
    set_bussi_timestep,
)
from oganesson.md_anomalies.restraints import RestraintCalculator, RestraintGroup, Sphere


def _cu_md(repeat=(4, 4, 4), dt_fs=2.0, temperature=600.0, seed=0):
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(repeat)
    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    dyn = Bussi(atoms, dt_fs * units.fs, temperature_K=temperature, taut=100 * units.fs, rng=np.random.default_rng(seed))
    clock = SimClock(dyn)
    dyn.attach(clock, interval=1)
    return atoms, dyn, clock


def test_clock_survives_timestep_change():
    _, dyn, clock = _cu_md()
    dyn.run(10)
    set_bussi_timestep(dyn, 1.0)
    dyn.run(5)
    assert math.isclose(clock.time_fs, 25.0)


def test_clock_stays_correct_when_dt_changes_from_an_attached_observer():
    """SimClock must be attached before observers that change dt mid-run.

    This test ensures that when dt changes from within an attached observer
    during a single dyn.run(), the clock still computes time correctly.
    The clock is attached first (by _cu_md), and a dt-switching observer
    is attached after it; ASE's FIFO order ensures clock fires first each step.
    """
    _, dyn, clock = _cu_md()
    switched = {"done": False}

    def _switch_dt_at_step_5():
        # After 5 steps at dt=2.0 fs, switch to dt=1.0 fs for remaining steps
        if not switched["done"] and dyn.nsteps == 5:
            set_bussi_timestep(dyn, 1.0)
            switched["done"] = True

    dyn.attach(_switch_dt_at_step_5, interval=1)
    dyn.run(10)
    # Expected: 5 steps * 2 fs + 5 steps * 1 fs = 15 fs
    assert math.isclose(clock.time_fs, 15.0)


def test_frame_writer_writes_on_simulation_time(tmp_path):
    atoms, dyn, clock = _cu_md()
    traj = Trajectory(str(tmp_path / "frames.traj"), "w")
    writer = FrameWriter(atoms, clock, traj, frame_fs=10.0, n_frames=6)
    dyn.attach(writer, interval=1)
    while not writer.done:
        dyn.run(1)
    traj.close()
    assert writer.times == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    frames = read(str(tmp_path / "frames.traj"), ":")
    assert len(frames) == 6
    assert np.isfinite(frames[-1].get_potential_energy())


def test_bussi_setters_update_every_dependent_attribute():
    _, dyn, _ = _cu_md()
    set_bussi_temperature(dyn, 900.0)
    assert math.isclose(bussi_temperature(dyn), 900.0)
    assert math.isclose(dyn.target_kinetic_energy, 0.5 * dyn.temp * dyn.ndof)
    set_bussi_timestep(dyn, 0.5)
    assert math.isclose(dyn.dt, 0.5 * units.fs)
    assert math.isclose(dyn._exp_term, math.exp(-dyn.dt / dyn.taut))




def test_invalidate_calculator_refreshes_sum_calculator():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(3)
    atoms.positions[0] += [0.4, 0.0, 0.0]
    restraints = RestraintCalculator([RestraintGroup("reservoir:0", [Sphere([0], [0.0, 0.0, 0.0], 0.1, 5.0, True)])])
    atoms.calc = SumCalculator([EMT(), restraints])
    before = atoms.get_forces()[0].copy()
    restraints.set_scale("reservoir:0", 0.0)
    invalidate_calculator(atoms)
    assert not np.allclose(before, atoms.get_forces()[0])


def test_frozen_atom_indices_reads_fixatoms_and_is_empty_without_one():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
    assert frozen_atom_indices(atoms) == set()
    atoms.set_constraint(FixAtoms(indices=[0, 3, 5]))
    assert frozen_atom_indices(atoms) == {0, 3, 5}


def test_remove_net_momentum():
    atoms, _, _ = _cu_md()
    p = atoms.get_momenta()
    p[0] += [5.0, 0.0, 0.0]
    atoms.set_momenta(p)
    remove_net_momentum(atoms)
    np.testing.assert_allclose(atoms.get_momenta().sum(axis=0), 0.0, atol=1e-10)
