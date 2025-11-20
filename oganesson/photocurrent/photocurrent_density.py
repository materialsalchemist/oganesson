#!/usr/bin/env python3
"""
Compute short-circuit photocurrent density J_sc from a VASP optical calculation.

Inputs:
    - vasprun.xml from a calculation with LOPTICS = .TRUE.
    - Solar spectrum file with at least two columns:
        wavelength (nm), spectral irradiance (W/m^2/nm)
      (e.g. ASTM G173 AM1.5G spectrum from NREL)

Output:
    J_sc in A/m^2 and mA/cm^2
"""

import argparse
import numpy as np
from scipy import constants as const
from pymatgen.io.vasp import Vasprun


# ---------- Step 1: Read dielectric function from vasprun.xml ----------

def read_dielectric_from_vasprun(vasprun_path):
    """
    Read the frequency-dependent dielectric function from vasprun.xml
    using pymatgen's Vasprun.dielectric interface.

    Returns
    -------
    energies_eV : (N,) np.ndarray
        Photon energies in eV.
    eps1_iso : (N,) np.ndarray
        Isotropic (xx+yy+zz)/3 real part of dielectric function.
    eps2_iso : (N,) np.ndarray
        Isotropic (xx+yy+zz)/3 imaginary part of dielectric function.
    """
    vr = Vasprun(
        vasprun_path,
        parse_dos=False,
        parse_eigen=False,
        ionic_step_skip=1
    )

    # Vasprun.dielectric returns:
    # (energies, [real tensors], [imag tensors]),
    # where each tensor row is [xx, yy, zz, xy, xz, yz].
    energies, eps_real, eps_imag = vr.dielectric

    energies = np.array(energies, dtype=float)
    eps_real = np.array(eps_real, dtype=float)
    eps_imag = np.array(eps_imag, dtype=float)

    # Isotropic average over diagonal components xx, yy, zz
    eps1_iso = (eps_real[:, 0] + eps_real[:, 1] + eps_real[:, 2]) / 3.0
    eps2_iso = (eps_imag[:, 0] + eps_imag[:, 1] + eps_imag[:, 2]) / 3.0

    return energies, eps1_iso, eps2_iso


# ---------- Step 2: Absorption coefficient α(E) from ε1, ε2 ----------

def absorption_coefficient_from_dielectric(E_eV, eps1, eps2):
    """
    Compute absorption coefficient α(ω) from dielectric function.

    Formula (standard optics / as in many VASP post-processing tools):
        α(ω) = sqrt(2) * ω / c * sqrt( sqrt(ε1^2 + ε2^2) - ε1 )

    where ω is angular frequency and c is speed of light.

    Parameters
    ----------
    E_eV : (N,) np.ndarray
        Photon energies in eV.
    eps1, eps2 : (N,) np.ndarray
        Real and imaginary parts of dielectric function.

    Returns
    -------
    alpha : (N,) np.ndarray
        Absorption coefficient in 1/m.
    """
    # Convert energy to angular frequency: ω = E / ħ
    E_J = E_eV * const.e  # eV -> J
    omega = E_J / const.hbar  # rad/s

    term = np.sqrt(eps1 ** 2 + eps2 ** 2)
    inside = term - eps1
    inside = np.clip(inside, 0.0, None)  # avoid small negative due to rounding

    alpha = np.sqrt(2.0) * omega / const.c * np.sqrt(inside)  # 1/m
    return alpha


# ---------- Step 3: Absorptivity of a film ----------

def absorptivity(alpha, thickness_nm, n_passes=2.0):
    """
    Compute absorptivity A(E) of a thin film of thickness L with absorption
    coefficient α(E).

    Simple model:
        A(E) = 1 - exp(-n_passes * α(E) * L)

    where:
        L = thickness in meters
        n_passes = effective number of passes (2 ≈ back-reflector)

    Parameters
    ----------
    alpha : (N,) np.ndarray
        Absorption coefficient in 1/m.
    thickness_nm : float
        Film thickness in nm.
    n_passes : float
        Effective number of passes (1 single, ~2 for perfect back reflector).

    Returns
    -------
    A : (N,) np.ndarray
        Absorptivity (0–1).
    """
    L_m = thickness_nm * 1e-9
    A = 1.0 - np.exp(-n_passes * alpha * L_m)
    return np.clip(A, 0.0, 1.0)


# ---------- Step 4: Solar spectrum and photon flux ----------

def load_solar_spectrum(file_path, wl_col=0, irr_col=1, skiprows=0):
    """
    Load solar spectrum from a text file.

    Expected format by default:
        wavelength[nm]  irradiance[W/m^2/nm]

    Parameters
    ----------
    file_path : str
        Path to solar spectrum file.
    wl_col : int
        Index of wavelength column (0-based).
    irr_col : int
        Index of irradiance column (0-based).
    skiprows : int
        How many header rows to skip.

    Returns
    -------
    wl_nm : (M,) np.ndarray
        Wavelengths in nm.
    irr_W_m2_nm : (M,) np.ndarray
        Spectral irradiance in W/m^2/nm.
    """
    data = np.loadtxt(file_path, comments="#", skiprows=skiprows)
    wl_nm = data[:, wl_col]
    irr = data[:, irr_col]
    return wl_nm, irr


def photon_flux_per_nm(wl_nm, irr_W_m2_nm):
    """
    Convert spectral irradiance I_lambda(λ) (W/m^2/nm) to photon flux Φ_λ(λ).

    Φ_λ(λ) = I_λ(λ) / E_photon
           = I_λ(λ) * λ / (h * c)

    Units of Φ_λ: photons / (m^2 * s * nm)

    Parameters
    ----------
    wl_nm : (M,) np.ndarray
        Wavelengths in nm.
    irr_W_m2_nm : (M,) np.ndarray
        Spectral irradiance in W/m^2/nm.

    Returns
    -------
    phi_lambda : (M,) np.ndarray
        Photon flux per nm.
    """
    wl_m = wl_nm * 1e-9
    E_photon_J = const.h * const.c / wl_m
    phi = irr_W_m2_nm / E_photon_J
    return phi


# ---------- Step 5: Glue everything together to get J_sc ----------

def compute_Jsc_from_vasp(
    vasprun_path,
    solar_spectrum_path,
    thickness_nm=500.0,
    spectrum_wl_col=0,
    spectrum_irr_col=1,
    spectrum_skiprows=0,
    n_passes=2.0,
):
    """
    Compute short-circuit current density J_sc from VASP optics + solar spectrum.

    Parameters
    ----------
    vasprun_path : str
        Path to vasprun.xml from a LOPTICS-compatible run.
    solar_spectrum_path : str
        Path to solar spectrum file (wavelength[nm], irradiance[W/m^2/nm]).
    thickness_nm : float
        Absorber thickness in nm.
    spectrum_wl_col : int
        Column index for wavelength in solar spectrum file.
    spectrum_irr_col : int
        Column index for irradiance in solar spectrum file.
    spectrum_skiprows : int
        Number of header lines to skip in solar spectrum file.
    n_passes : float
        Effective number of passes (1 single pass, ~2 with perfect back reflector).

    Returns
    -------
    J_sc_A_m2 : float
        Short-circuit current density in A/m^2.
    J_sc_mA_cm2 : float
        Short-circuit current density in mA/cm^2.
    """
    # 1) Dielectric from VASP
    E_eV, eps1, eps2 = read_dielectric_from_vasprun(vasprun_path)

    # 2) Absorption coefficient and absorptivity
    alpha = absorption_coefficient_from_dielectric(E_eV, eps1, eps2)
    absorp_E = absorptivity(alpha, thickness_nm, n_passes=n_passes)

    # Convert energy grid to wavelength grid for interpolation:
    # E[eV] = 1240 / λ[nm]  ->  λ[nm] = 1240 / E[eV]
    wl_nm_dielectric = 1240.0 / E_eV

    # Sort by increasing wavelength for interpolation
    sort_idx = np.argsort(wl_nm_dielectric)
    wl_sorted = wl_nm_dielectric[sort_idx]
    A_sorted = absorp_E[sort_idx]

    # 3) Solar spectrum + photon flux
    wl_nm_sun, irr_W_m2_nm = load_solar_spectrum(
        solar_spectrum_path,
        wl_col=spectrum_wl_col,
        irr_col=spectrum_irr_col,
        skiprows=spectrum_skiprows,
    )
    phi_lambda = photon_flux_per_nm(wl_nm_sun, irr_W_m2_nm)

    # 4) Interpolate absorptivity onto solar spectrum wavelength grid
    A_interp = np.interp(
        wl_nm_sun,
        wl_sorted,
        A_sorted,
        left=0.0,
        right=0.0,
    )

    # 5) Integrate absorbed photon flux over wavelength
    # integrand units: photons / (m^2 s nm)
    integrand = A_interp * phi_lambda
    photons_absorbed_per_m2_s = np.trapz(integrand, wl_nm_sun)  # integrate over nm

    # 6) Convert to current density J_sc = q * absorbed photon flux
    J_sc_A_m2 = const.e * photons_absorbed_per_m2_s  # A/m^2

    # 1 A/m^2 = 0.1 mA/cm^2
    J_sc_mA_cm2 = J_sc_A_m2 * 0.1

    return J_sc_A_m2, J_sc_mA_cm2


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Compute J_sc from VASP vasprun.xml and a solar spectrum."
    )
    parser.add_argument("vasprun", help="Path to vasprun.xml (LOPTICS run).")
    parser.add_argument(
        "spectrum",
        help=(
            "Path to solar spectrum file with columns: "
            "wavelength[nm] irradiance[W/m^2/nm]"
        ),
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=500.0,
        help="Absorber thickness in nm (default: 500 nm).",
    )
    parser.add_argument(
        "--passes",
        type=float,
        default=2.0,
        help="Effective number of passes (default: 2.0; use 1.0 for single-pass).",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=0,
        help="Number of header lines to skip in solar spectrum file.",
    )
    parser.add_argument(
        "--wl-col",
        type=int,
        default=0,
        help="Index of wavelength column in solar spectrum file (default: 0).",
    )
    parser.add_argument(
        "--irr-col",
        type=int,
        default=1,
        help="Index of irradiance column in solar spectrum file (default: 1).",
    )

    args = parser.parse_args()

    J_A_m2, J_mA_cm2 = compute_Jsc_from_vasp(
        vasprun_path=args.vasprun,
        solar_spectrum_path=args.spectrum,
        thickness_nm=args.thickness,
        spectrum_wl_col=args.wl_col,
        spectrum_irr_col=args.irr_col,
        spectrum_skiprows=args.skiprows,
        n_passes=args.passes,
    )

    print("===== Short-circuit photocurrent density =====")
    print(f"Thickness          : {args.thickness:.1f} nm")
    print(f"Effective passes   : {args.passes:g}")
    print(f"J_sc               : {J_A_m2:.3f} A/m^2")
    print(f"J_sc               : {J_mA_cm2:.3f} mA/cm^2")


if __name__ == "__main__":
    main()
