"""Small helpers around ASE's Bussi NVT integrator.

ASE's MolecularDynamics.get_time() is nsteps * dt, which is wrong once a trigger changes the
timestep, so runs keep their own SimClock. The Bussi setters change private attributes; they
match ASE 3.29's Bussi and are covered by tests/test_md_anomalies_integrator.py.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from ase import Atoms, units
from ase.md.bussi import Bussi


class SimClock:
    """Elapsed simulation time in fs.

    MUST be attached (interval=1) before any observer that may change `dyn.dt` mid-run (e.g. a
    trigger that ramps temperature or timestep). ASE calls attached observers in FIFO order each
    step; if a dt-changing observer fires before this one in a given step, that step's elapsed
    time is silently computed with the wrong (post-change) dt, permanently skewing time_fs by one
    step's dt delta with no way to detect it after the fact.
    """

    def __init__(self, dyn):
        self.dyn = dyn
        self.time_fs = 0.0
        self._last_nsteps = dyn.nsteps

    def __call__(self) -> None:
        n = self.dyn.nsteps
        if n > self._last_nsteps:
            self.time_fs += (n - self._last_nsteps) * self.dyn.dt / units.fs
            self._last_nsteps = n


def bussi_temperature(dyn: Bussi) -> float:
    return dyn.temp / units.kB


def set_bussi_temperature(dyn: Bussi, temperature_K: float) -> None:
    dyn.temp = temperature_K * units.kB
    dyn.target_kinetic_energy = 0.5 * dyn.temp * dyn.ndof


def set_bussi_timestep(dyn: Bussi, dt_fs: float) -> None:
    dyn.dt = dt_fs * units.fs
    dyn._exp_term = math.exp(-dyn.dt / dyn.taut)


class FrameWriter:
    """Writes a frame each time simulation time reaches the next multiple of frame_fs."""

    def __init__(self, atoms: Atoms, clock: SimClock, trajectory, frame_fs: float, n_frames: int):
        self.atoms = atoms
        self.clock = clock
        self.trajectory = trajectory
        self.frame_fs = frame_fs
        self.n_frames = n_frames
        self.times: List[float] = []

    @property
    def done(self) -> bool:
        return len(self.times) >= self.n_frames

    def __call__(self) -> None:
        if not self.done and self.clock.time_fs + 1e-6 >= len(self.times) * self.frame_fs:
            self.trajectory.write(self.atoms)
            self.times.append(self.clock.time_fs)


def invalidate_calculator(atoms: Atoms) -> None:
    """Drop cached results so the next force call sees changed restraint springs (SumCalculator has no reset())."""
    calc = atoms.calc
    if hasattr(calc, "reset"):
        calc.reset()
    else:
        calc.results = {}


def remove_net_momentum(atoms: Atoms) -> None:
    p = atoms.get_momenta()
    m = atoms.get_masses()
    atoms.set_momenta(p - np.outer(m / m.sum(), p.sum(axis=0)))


def frozen_atom_indices(atoms: Atoms) -> set:
    """Every index covered by a FixAtoms (or otherwise `get_indices()`-exposing) constraint on `atoms`."""
    frozen: set = set()
    for constraint in atoms.constraints:
        if hasattr(constraint, "get_indices"):
            frozen.update(int(i) for i in constraint.get_indices())
    return frozen
