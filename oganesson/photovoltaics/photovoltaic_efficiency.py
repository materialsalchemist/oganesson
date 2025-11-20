#!/usr/bin/env python3
"""
pv_efficiency_from_vasp_with_slme.py

Compute photovoltaic efficiencies for a material described by VASP output:

1) Shockley–Queisser (SQ) detailed-balance efficiency limit
   -> Uses only the band gap (from vasprun.xml or user input).

2) Spectroscopic Limited Maximum Efficiency (SLME)
   -> Uses the energy-dependent absorption from a VASP optical calculation
      (vasprun with LOPTICS = .TRUE.), and the method implemented in
      pymatgen.analysis.solar.slme.
"""

import argparse
import math
from typing import Tuple, Optional

import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.analysis.solar.slme import (
    absorption_coefficient,
    get_dir_indir_gap,
    slme as slme_function,
)

# ======================= Physical constants (SI units) =======================

h = 6.62607015e-34       # Planck constant, J s
c = 2.99792458e8         # Speed of light, m / s
kB = 1.380649e-23        # Boltzmann constant, J / K
q = 1.602176634e-19      # Elementary charge, C

R_SUN = 6.9634e8         # Solar radius, m
D_AU = 1.495978707e11    # Astronomical unit, m

F_GEOM = (R_SUN / D_AU) ** 2  # Geometric dilution factor Sun -> Earth

T_SUN_DEFAULT = 5778.0   # Effective temperature of the Sun, K
T_CELL_DEFAULT = 300.0   # Cell temperature, K

# Energy grid for numerical integration (in Joule)
E_MIN_EV = 0.001         # eV
E_MAX_EV = 5.0           # eV
N_E = 5000               # grid points


# ========================= VASP band gap extraction ==========================

def get_bandgap_from_vasprun(path: str) -> Tuple[float, bool]:
    """
    Read band gap (in eV) from a VASP vasprun.xml using pymatgen.

    Returns
    -------
    (gap_eV, is_direct)
    """
    vrun = Vasprun(path, parse_eigen=True)
    gap, cbm, vbm, is_direct = vrun.eigenvalue_band_properties
    if gap <= 0:
        raise ValueError(
            f"Extracted band gap is {gap:.4f} eV (non-positive). "
            "Check your calculation (metallic or no eigenvalues parsed?)."
        )
    return float(gap), bool(is_direct)


# ==================== Shockley–Queisser efficiency model =====================

def _build_energy_grid() -> np.ndarray:
    """Energy grid [J] for integration."""
    e_min = E_MIN_EV * q
    e_max = E_MAX_EV * q
    return np.linspace(e_min, e_max, N_E)


def _photon_flux_sun(E: np.ndarray, T_sun: float) -> np.ndarray:
    """
    Blackbody photon flux from the Sun at Earth (AM0-like, before atmosphere).

    Parameters
    ----------
    E : np.ndarray
        Photon energies [J].
    T_sun : float
        Solar temperature [K].

    Returns
    -------
    np.ndarray
        Photon flux in [photons / (m^2 s J)].
    """
    return F_GEOM * (2.0 * E ** 2 / (h ** 3 * c ** 2)) / np.expm1(E / (kB * T_sun))


def _photon_flux_cell(E: np.ndarray, T_cell: float) -> np.ndarray:
    """
    Blackbody photon flux emitted by the cell (radiative recombination).

    Parameters
    ----------
    E : np.ndarray
        Photon energies [J].
    T_cell : float
        Cell temperature [K].

    Returns
    -------
    np.ndarray
        Photon flux in [photons / (m^2 s J)].
    """
    return (2.0 * E ** 2 / (h ** 3 * c ** 2)) / np.expm1(E / (kB * T_cell))


def compute_sq_efficiency(
    Eg_eV: float,
    T_sun: float = T_SUN_DEFAULT,
    T_cell: float = T_CELL_DEFAULT,
) -> dict:
    """
    Compute Shockley–Queisser detailed-balance efficiency for a given band gap.

    Parameters
    ----------
    Eg_eV : float
        Band gap in eV.
    T_sun : float
        Effective Sun temperature [K].
    T_cell : float
        Cell temperature [K].

    Returns
    -------
    dict with keys:
        - bandgap_eV
        - efficiency (dimensionless, e.g. 0.31 for 31%)
        - Jsc_A_m2
        - J0_A_m2
        - Voc_V
        - Vmp_V
        - Pmax_W_m2
        - FF
        - P_in_W_m2
    """
    Eg_J = Eg_eV * q
    E_grid = _build_energy_grid()

    # Solar + cell photon flux
    Phi_sun = _photon_flux_sun(E_grid, T_sun)
    Phi_cell = _photon_flux_cell(E_grid, T_cell)

    # Raw incident solar power density (W/m^2)
    P_sun_raw = np.trapz(E_grid * Phi_sun, E_grid)

    # Scale to AM1.5-like 1000 W/m^2
    scale = 1000.0 / P_sun_raw
    Phi_sun_scaled = Phi_sun * scale
    P_sun = P_sun_raw * scale

    # Integrals above Eg
    mask = E_grid >= Eg_J

    # Short-circuit current density: J_sc = q ∫ Phi_sun(E) dE (E >= Eg)
    Phi_abs = np.trapz(Phi_sun_scaled[mask], E_grid[mask])
    Jsc = q * Phi_abs  # [A/m^2]

    # Radiative dark saturation current: J0 = q ∫ Phi_cell(E) dE (E >= Eg)
    Phi_em = np.trapz(Phi_cell[mask], E_grid[mask])
    J0 = q * Phi_em  # [A/m^2]

    # J(V) = J_sc - J0 (exp(qV/kT) - 1)
    q_over_kT = q / (kB * T_cell)
    V_grid = np.linspace(0.0, Eg_eV, 400)  # up to Eg/q ~ Eg_eV volts
    P_arr = np.empty_like(V_grid)

    for i, V in enumerate(V_grid):
        J = Jsc - J0 * (math.exp(q_over_kT * V) - 1.0)
        P_arr[i] = V * J  # power density [W/m^2]

    idx_max = int(np.argmax(P_arr))
    Pmax = float(P_arr[idx_max])
    Vmp = float(V_grid[idx_max])

    Voc = (kB * T_cell / q) * math.log(Jsc / J0 + 1.0)

    eta = Pmax / P_sun
    FF = Pmax / (Jsc * Voc) if Jsc * Voc > 0 else float("nan")

    return {
        "bandgap_eV": Eg_eV,
        "efficiency": eta,
        "Jsc_A_m2": Jsc,
        "J0_A_m2": J0,
        "Voc_V": Voc,
        "Vmp_V": Vmp,
        "Pmax_W_m2": Pmax,
        "FF": FF,
        "P_in_W_m2": P_sun,
    }


# ====================== SLME from VASP optical data ==========================

def compute_slme_from_vasprun(
    optic_vasprun_path: str,
    thickness_m: float,
    T_cell: float = T_CELL_DEFAULT,
) -> dict:
    """
    Compute SLME from a VASP optical calculation using pymatgen's implementation.

    Parameters
    ----------
    optic_vasprun_path : str
        Path to vasprun.xml from an optical calculation (LOPTICS = .TRUE.).
    thickness_m : float
        Absorber thickness in meters.
    T_cell : float
        Cell temperature (K), used by the SLME routine.

    Returns
    -------
    dict with keys:
        - efficiency       (SLME, dimensionless, e.g. 0.28 for 28%)
        - Eg_dir_eV        (direct allowed band gap)
        - Eg_indir_eV      (indirect gap)
        - energies_eV      (energy grid used for absorption, np.ndarray)
        - alpha_m_inv      (absorption coefficient in m^-1, np.ndarray)
    """
    vrun = Vasprun(optic_vasprun_path)
    dielectric = vrun.dielectric
    if dielectric is None:
        raise ValueError(
            "No dielectric data found in optic vasprun.xml. "
            "Make sure the calculation was run with LOPTICS = .TRUE."
        )

    # dielectric: [energies, eps_real, eps_imag]
    energies_eV = np.array(dielectric[0])

    # Absorption coefficient α(E) in cm^-1 -> convert to m^-1
    alpha_cm_inv = absorption_coefficient(dielectric)
    alpha_m_inv = np.array(alpha_cm_inv) * 100.0

    # Direct and indirect gaps (in eV) from the same vasprun
    Eg_dir_eV, Eg_indir_eV = get_dir_indir_gap(optic_vasprun_path)

    # SLME: pymatgen returns maximum efficiency (fraction, e.g. 0.3 for 30%)
    eta_slme = slme_function(
        material_energy_for_absorbance_data=energies_eV,
        material_absorbance_data=alpha_m_inv,
        material_direct_allowed_gap=Eg_dir_eV,
        material_indirect_gap=Eg_indir_eV,
        thickness=thickness_m,
        temperature=T_cell,
        absorbance_in_inverse_centimeters=False,
        cut_off_absorbance_below_direct_allowed_gap=True,
        plot_current_voltage=False,
    )

    return {
        "efficiency": float(eta_slme),
        "Eg_dir_eV": float(Eg_dir_eV),
        "Eg_indir_eV": float(Eg_indir_eV),
        "energies_eV": energies_eV,
        "alpha_m_inv": alpha_m_inv,
    }


# ============================== CLI interface ================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute ideal photovoltaic efficiencies (Shockley–Queisser and SLME) "
            "for a material from VASP outputs."
        )
    )
    parser.add_argument(
        "--vasprun",
        type=str,
        default=None,
        help="Path to vasprun.xml (to read band gap for SQ).",
    )
    parser.add_argument(
        "--bandgap",
        type=float,
        default=None,
        help="Band gap in eV (if supplied, overrides --vasprun for SQ).",
    )
    parser.add_argument(
        "--Tcell",
        type=float,
        default=T_CELL_DEFAULT,
        help=f"Cell temperature in K (default: {T_CELL_DEFAULT}).",
    )
    parser.add_argument(
        "--Tsun",
        type=float,
        default=T_SUN_DEFAULT,
        help=f"Sun temperature in K for SQ model (default: {T_SUN_DEFAULT}).",
    )
    parser.add_argument(
        "--slme-optic-vasprun",
        type=str,
        default=None,
        help=(
            "Path to vasprun.xml from an optical calculation (LOPTICS = .TRUE.) "
            "for SLME computation."
        ),
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=5e-7,
        help="Absorber thickness in meters for SLME (default: 5e-7 m = 500 nm).",
    )

    args = parser.parse_args()

    # --------- Decide how to get the band gap for SQ ---------
    is_direct: Optional[bool] = None
    if args.bandgap is not None:
        Eg_eV = float(args.bandgap)
        print(f"Using user-supplied band gap for SQ: Eg = {Eg_eV:.4f} eV")
    elif args.vasprun is not None:
        Eg_eV, is_direct = get_bandgap_from_vasprun(args.vasprun)
        print(
            f"Band gap for SQ from {args.vasprun}: Eg = {Eg_eV:.4f} eV "
            f"({'direct' if is_direct else 'indirect'})"
        )
    else:
        raise SystemExit(
            "Error: please provide either --bandgap or --vasprun for SQ.\n"
            "  Example: python pv_efficiency_from_vasp_with_slme.py --vasprun vasprun.xml"
        )

    # --------- Shockley–Queisser efficiency ---------
    res_sq = compute_sq_efficiency(Eg_eV, T_sun=args.Tsun, T_cell=args.Tcell)

    print("\n=== Shockley–Queisser PV efficiency (radiative limit) ===")
    print(f"Band gap (SQ) : {res_sq['bandgap_eV']:.4f} eV")
    if is_direct is not None:
        print(f"Gap type      : {'direct' if is_direct else 'indirect'}")
    print(f"Incident P_in  : {res_sq['P_in_W_m2']:.2f} W/m^2 (scaled BB Sun)")
    print(f"J_sc           : {res_sq['Jsc_A_m2']:.3f} A/m^2")
    print(f"J_0            : {res_sq['J0_A_m2']:.3e} A/m^2")
    print(f"V_oc           : {res_sq['Voc_V']:.3f} V")
    print(f"V_mp           : {res_sq['Vmp_V']:.3f} V")
    print(f"P_max          : {res_sq['Pmax_W_m2']:.2f} W/m^2")
    print(f"Fill factor    : {res_sq['FF']:.3f}")
    print(f"SQ efficiency  : {100.0 * res_sq['efficiency']:.2f} %")

    # --------- SLME efficiency (optional) ---------
    if args.slme_optic_vasprun is not None:
        print(
            "\nComputing SLME using optical data from: "
            f"{args.slme_optic_vasprun}"
        )
        res_slme = compute_slme_from_vasprun(
            optic_vasprun_path=args.slme_optic_vasprun,
            thickness_m=args.thickness,
            T_cell=args.Tcell,
        )
        print("\n=== Spectroscopic Limited Maximum Efficiency (SLME) ===")
        print(
            f"Direct gap (optical)    : {res_slme['Eg_dir_eV']:.4f} eV\n"
            f"Indirect gap (optical)  : {res_slme['Eg_indir_eV']:.4f} eV"
        )
        print(f"Thickness used for SLME : {args.thickness:.3e} m")
        print(f"SLME efficiency         : {100.0 * res_slme['efficiency']:.2f} %")
    else:
        print(
            "\n(No --slme-optic-vasprun supplied -> SLME not computed. "
            "Provide an optical vasprun.xml with LOPTICS=.TRUE. to get SLME.)"
        )


if __name__ == "__main__":
    main()
