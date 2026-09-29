"""Each trigger acts on schedule, keeps the atom count and records the interval it was active."""

import math

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
from ase.geometry import get_distances
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from oganesson.md_anomalies.integrator import SimClock, bussi_temperature
from oganesson.md_anomalies.restraints import RestraintCalculator, RestraintGroup, Sphere
from oganesson.md_anomalies.triggers import (
    HeatRamp,
    HotSpot,
    InterstitialPush,
    Recoil,
    ReservoirRelease,
    StrainRamp,
    TriggerContext,
    make_trigger,
)


def _context(temperature=600.0, dt_fs=2.0, restraints=None, seed=0, attach_clock=True):
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4))
    atoms.calc = EMT() if restraints is None else SumCalculator([EMT(), restraints])
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    dyn = Bussi(atoms, dt_fs * units.fs, temperature_K=temperature, taut=100 * units.fs, rng=np.random.default_rng(seed))
    clock = SimClock(dyn)
    if attach_clock:
        dyn.attach(clock, interval=1)
    ctx = TriggerContext(
        atoms=atoms,
        dyn=dyn,
        clock=clock,
        rng=np.random.default_rng(seed + 1),
        temperature_run=temperature,
        base_dt_fs=dt_fs,
        host_indices=np.arange(len(atoms)),
        reactant_indices=np.array([], dtype=int),
        restraints=restraints,
    )
    return ctx


def _run(ctx, trigger, steps):
    ctx.dyn.attach(lambda: trigger(ctx), interval=1)
    n_atoms = len(ctx.atoms)
    ctx.dyn.run(steps)
    assert len(ctx.atoms) == n_atoms


def test_heat_ramp_reaches_target_and_records_interval():
    ctx = _context()
    trigger = HeatRamp(10.0, 20.0, 900.0)
    _run(ctx, trigger, 20)
    assert math.isclose(bussi_temperature(ctx.dyn), 900.0)
    record = trigger.record()
    assert (record["family"], record["subtype"], record["t_on_fs"], record["t_off_fs"]) == ("thermal", "heat", 10.0, 30.0)


def test_strain_ramp_reaches_target_strain():
    ctx = _context()
    z0 = ctx.atoms.cell.lengths()[2]
    trigger = StrainRamp(0.0, 20.0, "tension", 0.05, 2)
    _run(ctx, trigger, 15)
    assert math.isclose(ctx.atoms.cell.lengths()[2], 1.05 * z0)
    assert trigger.record()["t_off_fs"] == 20.0


def test_hotspot_heats_a_sphere_once():
    ctx = _context()
    trigger = HotSpot(4.0, 5.0, 4.0)
    _run(ctx, trigger, 5)
    record = trigger.record()
    assert record["n_heated"] > 0
    assert record["t_on_fs"] == record["t_off_fs"] == 4.0


def test_recoil_is_capped_limits_step_displacement_and_restores_timestep():
    ctx = _context()
    trigger = Recoil(4.0, 50.0)
    previous = [ctx.atoms.get_positions().copy()]
    largest = [0.0]

    def track():
        pos = ctx.atoms.get_positions()
        largest[0] = max(largest[0], float(np.linalg.norm(pos - previous[0], axis=1).max()))
        previous[0] = pos.copy()

    ctx.dyn.attach(lambda: trigger(ctx), interval=1)
    ctx.dyn.attach(track, interval=1)
    while ctx.clock.time_fs < 600.0 and trigger.t_off_fs is None:
        ctx.dyn.run(1)
    record = trigger.record()
    cap = 0.75 * len(ctx.atoms) * units.kB * 600.0
    assert math.isclose(record["applied_energy_eV"], min(50.0, cap))
    assert record["t_off_fs"] is not None
    assert math.isclose(ctx.dyn.dt, 2.0 * units.fs)
    assert largest[0] < 0.08


def test_recoil_clock_stays_correct_with_adaptive_timestep():
    """SimClock must be attached before Recoil to track time correctly during adaptive dt.

    Recoil changes dt on nearly every step during its fast phase, so this is the organic,
    state-dependent counterpart to integrator tests' test_clock_stays_correct_when_dt_changes_
    from_an_attached_observer (a single hand-picked dt switch pinned to an exact expected
    time_fs). Recoil's dt sequence can't be hand-derived the same way, so instead this proves
    order-sensitivity directly: run the identical scenario (same seed, so Recoil's random atom
    pick, kick direction and resulting dt sequence are bit-for-bit identical between the two
    runs -- confirmed empirically while writing this test) once with clock attached before the
    trigger's observer (correct, matches _context()'s default) and once attached after it (the
    bug SimClock's and Recoil's docstrings warn about), and assert the two final clock readings
    differ by a large, unmistakable amount. A clock implementation that can't tell correct from
    reversed attach order would show ~0 fs difference here.
    """
    max_steps = 50

    def run(attach_clock_first: bool):
        ctx = _context(attach_clock=attach_clock_first)
        trigger = Recoil(0.0, 100.0)
        if attach_clock_first:
            ctx.dyn.attach(lambda: trigger(ctx), interval=1)
        else:
            # Reversed order: attach the trigger's observer first, then the clock -- ASE's FIFO
            # order then makes the clock read dt *after* Recoil has already adapted it for the
            # next step, misattributing that step's elapsed time.
            ctx.dyn.attach(lambda: trigger(ctx), interval=1)
            ctx.dyn.attach(ctx.clock, interval=1)
        clock_times = []
        ctx.dyn.attach(lambda: clock_times.append(ctx.clock.time_fs), interval=1)
        n_atoms = len(ctx.atoms)
        steps_run = 0
        while trigger.t_off_fs is None and steps_run < max_steps:
            ctx.dyn.run(1)
            steps_run += 1
        assert len(ctx.atoms) == n_atoms
        return ctx, trigger, clock_times

    # Correct, production-realistic order.
    ctx, trigger, clock_times = run(attach_clock_first=True)

    assert trigger.t_off_fs is not None, "trigger should have finished within max_steps"
    assert len(clock_times) > 10, "trigger should have run for multiple adaptive-dt steps"
    for earlier, later in zip(clock_times, clock_times[1:]):
        assert later >= earlier, f"time went backwards: {earlier} -> {later}"
    assert 1.0 < ctx.clock.time_fs < max_steps * 2.5

    # Reversed (buggy) order, same seed.
    ctx_rev, trigger_rev, _ = run(attach_clock_first=False)
    assert trigger_rev.t_off_fs is not None

    # The order-sensitivity check that actually matters: with identical physics in both runs,
    # a correct clock and a reversed-attach clock must disagree by a large, unmistakable margin.
    drift_fs = abs(ctx.clock.time_fs - ctx_rev.clock.time_fs)
    assert drift_fs > 0.5, f"clock.time_fs should differ measurably by attach order, got drift={drift_fs} fs"


def test_interstitial_push_reaches_interstitial_sites():
    ctx = _context(temperature=300.0, dt_fs=1.0)
    lattice = ctx.atoms.positions.copy()
    trigger = InterstitialPush(1.0, 3, 1.0)
    _run(ctx, trigger, 110)
    others = np.setdiff1d(np.arange(len(lattice)), trigger.atoms_moved)
    clearance = get_distances(trigger.target, lattice[others], cell=ctx.atoms.cell, pbc=True)[1].min(axis=1)
    assert np.all((clearance > 1.5) & (clearance < 1.9))  # fcc Cu octahedral site clearance is 1.80 A
    assert np.all(np.linalg.norm(trigger.target - trigger.start, axis=1) >= trigger.d_nn - 1e-6)


def _reservoir(atoms_frac):
    groups = [RestraintGroup(f"reservoir:{u}", [Sphere([u], atoms_frac[u].tolist(), 0.05, 5.0, True)]) for u in range(3)]
    groups.append(RestraintGroup("exclusion", [Sphere([10, 11], [0.5, 0.5, 0.5], 1.0, 5.0, False)]))
    return RestraintCalculator(groups)


def test_release_frees_a_fraction_of_units_and_the_exclusion():
    frac = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4)).get_scaled_positions()
    ctx = _context(restraints=_reservoir(frac))
    trigger = ReservoirRelease(10.0, 20.0, release_fraction=0.5, final_scale=0.0)
    _run(ctx, trigger, 20)
    scales = {name: group.scale for name, group in ctx.restraints.groups.items()}
    assert sum(scales[f"reservoir:{u}"] == 0.0 for u in range(3)) == 2
    assert scales["exclusion"] == 0.0
    assert len(trigger.record()["released"]) == 3


def test_weak_release_halves_every_spring():
    frac = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4)).get_scaled_positions()
    ctx = _context(restraints=_reservoir(frac))
    _run(ctx, ReservoirRelease(0.0, 10.0, release_fraction=1.0, final_scale=0.5), 10)
    assert {group.scale for group in ctx.restraints.groups.values()} == {0.5}


@pytest.mark.parametrize(
    "subtype,params,expected",
    [
        ("heat", {"target_factor": 1.5}, HeatRamp),
        ("quench", {"target_factor": 0.5}, HeatRamp),
        ("hotspot", {"radius": 5.0, "factor": 3.0}, HotSpot),
        ("shear", {"strain": 0.02, "axis": 0}, StrainRamp),
        ("recoil", {"energy_eV": 20.0}, Recoil),
        ("interstitial", {"k": 2, "fraction": 0.3}, InterstitialPush),
        ("release", {"release_fraction": 1.0, "final_scale": 0.0}, ReservoirRelease),
    ],
)
def test_make_trigger_builds_each_subtype(subtype, params, expected):
    spec = {"family": "x", "subtype": subtype, "t_start_fs": 1000.0, "ramp_fs": 200.0, "weak": False, "params": params}
    trigger = make_trigger(spec, temperature_run=800.0)
    assert isinstance(trigger, expected)
    assert trigger.subtype == subtype
    if subtype in ("heat", "quench"):
        assert math.isclose(trigger.target_K, params["target_factor"] * 800.0)
