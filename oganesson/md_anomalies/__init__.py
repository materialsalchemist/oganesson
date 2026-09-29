"""Scripted perturbations (triggers) and event detectors for anomaly studies in molecular dynamics.

Triggers are ASE MD observers that perturb a running simulation on schedule: thermostat heat/quench
ramps, hot spots, strain ramps, recoil cascades, interstitial pushes, and release of flat-bottom
restraints. Event detectors find what the system actually did in response: local melting, foreign
crystal phases, surviving vacancies, broken molecules, cavitation and surface protrusions.

    from oganesson.md_anomalies import HeatRamp, TriggerContext, compute_events, event_masks

The detectors that use polyhedral template matching and Wigner-Seitz analysis need OVITO
(``pip install oganesson[md-events]``); everything else needs only ASE, NumPy and SciPy.
"""

from .events import (
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
from .integrator import SimClock, bussi_temperature, frozen_atom_indices, set_bussi_temperature, set_bussi_timestep
from .restraints import RestraintCalculator, RestraintGroup, Slab, Sphere
from .triggers import (
    HeatRamp,
    HotSpot,
    InterstitialPush,
    Recoil,
    ReservoirRelease,
    StrainRamp,
    Trigger,
    TriggerContext,
    make_trigger,
)

__all__ = [
    "EVENT_RULES", "EventConfig", "EventRule", "compute_events", "dendrite_cluster", "event_masks",
    "events_from_masks", "foreign_phase_cluster", "largest_cluster", "largest_void_radius", "melt_cluster",
    "ptm_types", "reference_surface_z",
    "SimClock", "bussi_temperature", "frozen_atom_indices", "set_bussi_temperature", "set_bussi_timestep",
    "RestraintCalculator", "RestraintGroup", "Slab", "Sphere",
    "HeatRamp", "HotSpot", "InterstitialPush", "Recoil", "ReservoirRelease", "StrainRamp", "Trigger",
    "TriggerContext", "make_trigger",
]
