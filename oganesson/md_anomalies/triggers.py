"""Scripted triggers applied during production MD.

A trigger is an MD observer. Once simulation time reaches t_start_fs it calls begin() once, then
update() every step with the ramp fraction until update() returns True, then end(). It records
the interval it was active (`record()`), e.g. for labelling trigger frames. No trigger changes the
atom count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from ase import Atoms, units
from ase.geometry import get_distances
from ase.md.bussi import Bussi

from .integrator import (
    SimClock,
    bussi_temperature,
    invalidate_calculator,
    remove_net_momentum,
    set_bussi_temperature,
    set_bussi_timestep,
)
from .restraints import RestraintCalculator


@dataclass
class TriggerContext:
    atoms: Atoms
    dyn: Bussi
    clock: SimClock
    rng: np.random.Generator
    temperature_run: float
    base_dt_fs: float
    host_indices: np.ndarray
    reactant_indices: np.ndarray
    restraints: Optional[RestraintCalculator] = None


class Trigger:
    family = ""
    subtype = ""

    def __init__(self, t_start_fs: float, ramp_fs: float):
        self.t_start_fs = float(t_start_fs)
        self.ramp_fs = float(ramp_fs)
        self.t_on_fs: Optional[float] = None
        self.t_off_fs: Optional[float] = None

    def __call__(self, ctx: TriggerContext) -> None:
        t = ctx.clock.time_fs
        if self.t_off_fs is not None or t + 1e-9 < self.t_start_fs:
            return
        if self.t_on_fs is None:
            self.t_on_fs = t
            self.begin(ctx)
        frac = 1.0 if self.ramp_fs <= 0 else min(1.0, (t - self.t_on_fs) / self.ramp_fs)
        if self.update(ctx, frac):
            self.t_off_fs = t
            self.end(ctx)

    def begin(self, ctx: TriggerContext) -> None:
        pass

    def update(self, ctx: TriggerContext, frac: float) -> bool:
        return frac >= 1.0

    def end(self, ctx: TriggerContext) -> None:
        pass

    def details(self) -> Dict:
        return {}

    def record(self) -> Dict:
        return {
            "family": self.family,
            "subtype": self.subtype,
            "t_start_fs": self.t_start_fs,
            "ramp_fs": self.ramp_fs,
            "t_on_fs": self.t_on_fs,
            "t_off_fs": self.t_off_fs,
            **self.details(),
        }


class HeatRamp(Trigger):
    family = "thermal"

    def __init__(self, t_start_fs: float, ramp_fs: float, target_K: float, subtype: str = "heat"):
        super().__init__(t_start_fs, ramp_fs)
        self.target_K = float(target_K)
        self.subtype = subtype

    def begin(self, ctx):
        self.start_K = bussi_temperature(ctx.dyn)

    def update(self, ctx, frac):
        set_bussi_temperature(ctx.dyn, self.start_K + frac * (self.target_K - self.start_K))
        return frac >= 1.0

    def details(self):
        return {"target_K": self.target_K}


class HotSpot(Trigger):
    family, subtype = "thermal", "hotspot"

    def __init__(self, t_start_fs: float, radius: float, factor: float):
        super().__init__(t_start_fs, 0.0)
        self.radius = float(radius)
        self.factor = float(factor)
        self.n_heated = 0

    def begin(self, ctx):
        a = ctx.atoms
        host = np.asarray(ctx.host_indices)
        centre = int(ctx.rng.choice(host))
        d = get_distances(a.positions[host], a.positions[centre][None], cell=a.cell, pbc=a.pbc)[1][:, 0]
        selected = host[d <= self.radius]
        p = a.get_momenta()
        p[selected] *= math.sqrt(self.factor)  # kinetic energy scales with the square of momentum
        a.set_momenta(p)
        remove_net_momentum(a)
        self.n_heated = int(len(selected))

    def details(self):
        return {"radius": self.radius, "factor": self.factor, "n_heated": self.n_heated}


class StrainRamp(Trigger):
    family = "mechanical"

    def __init__(self, t_start_fs: float, ramp_fs: float, mode: str, strain: float, axis: int):
        super().__init__(t_start_fs, ramp_fs)
        if mode not in ("tension", "compression", "shear"):
            raise ValueError(f"unknown strain mode {mode!r}")
        self.subtype = mode
        self.strain = float(strain)
        self.axis = int(axis)

    def begin(self, ctx):
        self.cell0 = ctx.atoms.cell.array.copy()

    def update(self, ctx, frac):
        F = np.eye(3)
        value = frac * self.strain
        if self.subtype == "tension":
            F[self.axis, self.axis] += value
        elif self.subtype == "compression":
            F[self.axis, self.axis] -= value
        else:
            F[self.axis, (self.axis + 1) % 3] += value
        ctx.atoms.set_cell(self.cell0 @ F, scale_atoms=True)
        return frac >= 1.0

    def details(self):
        return {"strain": self.strain, "axis": self.axis}


class Recoil(Trigger):
    """Radiation-damage cascade: kicks one atom, then adaptively shrinks the MD timestep every
    step to bound per-step displacement, restoring it in end().

    Like any trigger that changes dt mid-run, this depends on SimClock being attached to the
    dynamics object before this trigger's observer (see SimClock's docstring in integrator.py) --
    otherwise time_fs silently drifts by one step's dt delta each time dt changes.
    """
    family, subtype = "defect", "recoil"
    max_step_displacement = 0.05  # A per step while the cascade is fast
    min_dt_fs = 0.01

    def __init__(self, t_start_fs: float, energy_eV: float, max_duration_fs: float = 500.0):
        super().__init__(t_start_fs, max_duration_fs)
        self.requested_energy_eV = float(energy_eV)
        self.applied_energy_eV = 0.0
        self.atom = -1

    def _adapt_timestep(self, ctx) -> float:
        speed = float(np.linalg.norm(ctx.atoms.get_velocities(), axis=1).max()) * units.fs  # A per fs
        if speed == 0.0:
            dt_fs = ctx.base_dt_fs
        else:
            dt_fs = min(ctx.base_dt_fs, max(self.max_step_displacement / speed, self.min_dt_fs))
        set_bussi_timestep(ctx.dyn, dt_fs)
        return dt_fs

    def begin(self, ctx):
        a = ctx.atoms
        # Cap so the global temperature rises by at most 50%: small cells cannot absorb a large cascade.
        cap = 0.75 * len(a) * units.kB * ctx.temperature_run
        self.applied_energy_eV = min(self.requested_energy_eV, cap)
        self.atom = int(ctx.rng.choice(np.asarray(ctx.host_indices)))
        direction = ctx.rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        p = a.get_momenta()
        p[self.atom] = direction * math.sqrt(2.0 * a.get_masses()[self.atom] * self.applied_energy_eV)
        a.set_momenta(p)
        remove_net_momentum(a)
        self._adapt_timestep(ctx)

    def update(self, ctx, frac):
        return self._adapt_timestep(ctx) >= ctx.base_dt_fs or frac >= 1.0

    def end(self, ctx):
        set_bussi_timestep(ctx.dyn, ctx.base_dt_fs)

    def details(self):
        return {"requested_energy_eV": self.requested_energy_eV, "applied_energy_eV": self.applied_energy_eV, "atom": self.atom}


class InterstitialPush(Trigger):
    family, subtype = "defect", "interstitial"

    def __init__(self, t_start_fs: float, k: int, fraction: float = 1.0, path_fs: float = 100.0):
        super().__init__(t_start_fs, path_fs)
        self.k = int(k)
        self.fraction = float(fraction)  # the weak variant moves atoms 30% of the way, then releases them

    def begin(self, ctx):
        a = ctx.atoms
        host = np.asarray(ctx.host_indices)
        centre = int(ctx.rng.choice(host))
        d = get_distances(a.positions[host], a.positions[centre][None], cell=a.cell, pbc=a.pbc)[1][:, 0]
        self.atoms_moved = host[np.argsort(d)[: self.k]]
        dists = get_distances(a.positions[host], cell=a.cell, pbc=a.pbc)[1]
        np.fill_diagonal(dists, np.inf)
        self.d_nn = float(np.median(dists.min(axis=1)))
        self.start = a.positions[self.atoms_moved].copy()
        # Every current site blocks, including the moved atoms' own, so atoms land in interstitial
        # space instead of swapping into a site another moved atom just left.
        blockers = a.positions.copy()
        targets = []
        for i in self.atoms_moved:
            dirs = ctx.rng.normal(size=(512, 3))
            dirs /= np.linalg.norm(dirs, axis=1)[:, None]
            candidates = np.concatenate([a.positions[i] + f * self.d_nn * dirs for f in (1.0, 1.25, 1.5)])
            clearance = get_distances(candidates, blockers, cell=a.cell, pbc=a.pbc)[1].min(axis=1)
            best = candidates[int(np.argmax(clearance))]
            targets.append(best)
            blockers = np.vstack([blockers, best])
        self.target = np.array(targets)

    def update(self, ctx, frac):
        a = ctx.atoms
        pos = a.get_positions()
        pos[self.atoms_moved] = self.start + frac * self.fraction * (self.target - self.start)
        a.set_positions(pos)
        p = a.get_momenta()
        p[self.atoms_moved] = 0.0
        a.set_momenta(p)
        return frac >= 1.0

    def details(self):
        return {"k": self.k, "fraction": self.fraction, "atoms_moved": [int(i) for i in getattr(self, "atoms_moved", [])]}


class ReservoirRelease(Trigger):
    family, subtype = "chemical", "release"

    def __init__(self, t_start_fs: float, ramp_fs: float, release_fraction: float = 1.0, final_scale: float = 0.0):
        super().__init__(t_start_fs, ramp_fs)
        self.release_fraction = float(release_fraction)
        self.final_scale = float(final_scale)
        self.released = []

    def begin(self, ctx):
        units_ = sorted(name for name in ctx.restraints.groups if name.startswith("reservoir:"))
        k = min(len(units_), max(1, math.ceil(self.release_fraction * len(units_)))) if units_ else 0
        self.released = sorted(str(x) for x in ctx.rng.choice(units_, size=k, replace=False)) if k else []
        # Solvent must be free to reach any released reactant, so the exclusion sphere goes with it.
        if "exclusion" in ctx.restraints.groups:
            self.released.append("exclusion")

    def update(self, ctx, frac):
        scale = 1.0 + frac * (self.final_scale - 1.0)
        for name in self.released:
            ctx.restraints.set_scale(name, scale)
        invalidate_calculator(ctx.atoms)
        return frac >= 1.0

    def details(self):
        return {"release_fraction": self.release_fraction, "final_scale": self.final_scale, "released": list(self.released)}


def make_trigger(spec: Dict, temperature_run: float) -> Trigger:
    """Build a trigger from a plain-dict spec.

    `spec` has keys `subtype`, `t_start_fs`, `ramp_fs` and `params`; `params` holds the subtype's
    strength, e.g. ``{"target_factor": 1.5}`` for heat/quench (a multiple of `temperature_run`),
    ``{"strain": 0.05, "axis": 0}`` for tension/compression/shear, ``{"energy_eV": 50.0}`` for recoil.
    """
    p = spec["params"]
    t0, ramp, subtype = spec["t_start_fs"], spec["ramp_fs"], spec["subtype"]
    if subtype in ("heat", "quench"):
        return HeatRamp(t0, ramp, p["target_factor"] * temperature_run, subtype)
    if subtype == "hotspot":
        return HotSpot(t0, p["radius"], p["factor"])
    if subtype in ("tension", "compression", "shear"):
        return StrainRamp(t0, ramp, subtype, p["strain"], p["axis"])
    if subtype == "recoil":
        return Recoil(t0, p["energy_eV"])
    if subtype == "interstitial":
        return InterstitialPush(t0, p["k"], p["fraction"])
    if subtype == "release":
        return ReservoirRelease(t0, ramp, p["release_fraction"], p["final_scale"])
    raise ValueError(f"unknown trigger subtype {subtype!r}")
