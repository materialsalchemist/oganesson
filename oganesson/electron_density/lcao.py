#!/usr/bin/env python3
"""
Periodic LCAO initial electron density from CIF (H–Bi), output as CUBE or CHGCAR.

Periodic enforcement:
- Each AO φ_mu(r) is evaluated using *wrapped fractional displacements*:
    Δf = wrap(f_grid - f_atom) into [-0.5, 0.5)
  then Δr_cart = Δf_x * a + Δf_y * b + Δf_z * c.
  This yields a strictly lattice-periodic φ_mu(r) and hence periodic ρ(r).

Optionally, you can sum over a small set of lattice images (e.g. {-1,0,1}^3)
instead of the minimum-image displacement; set --pbc sum and --images 1.

Usage
-----
python lcao_density_pbc.py structure.cif --fmt cube --out rho.cube
python lcao_density_pbc.py structure.cif --fmt chgcar --out CHGCAR --nx 60 --ny 60 --nz 60
"""

import json
import math
import numpy as np
from ase.io import read as ase_read
from ase.data import atomic_numbers

# ----------------- Tunables -----------------
GRID_N = (40, 40, 40)
K_WOLFSBERG = 1.75
FERMI_TOL = 1e-9
OUTPUT_FMT = "cube"  # or "chgcar"
OUTPUT_FILE = "rho_init.cube"
PBC_MODE = "minimum"  # "minimum" (default) or "sum"
N_IMAGES = 0  # sum over translations in [-N..N]^3 if PBC_MODE=="sum"
ALPHA_L_SCALE = {0: 1.00, 1: 0.90, 2: 0.85}
RY_EV = 13.605693009
# --------------------------------------------

AUFBAU = [
    (1, 0, 2),
    (2, 0, 2),
    (2, 1, 6),
    (3, 0, 2),
    (3, 1, 6),
    (4, 0, 2),
    (3, 2, 10),
    (4, 1, 6),
    (5, 0, 2),
    (4, 2, 10),
    (5, 1, 6),
    (6, 0, 2),
    (4, 3, 14),
    (5, 2, 10),
    (6, 1, 6),
    (7, 0, 2),
    (5, 3, 14),
    (6, 2, 10),
    (7, 1, 6),
]
L_LABEL = {0: "s", 1: "p", 2: "d", 3: "f"}


def electron_configuration(Z):
    occ, rem = {}, Z
    for n, l, cap in AUFBAU:
        if rem <= 0:
            break
        fill = min(cap, rem)
        occ[(n, l)] = occ.get((n, l), 0) + fill
        rem -= fill
    return occ


def slater_Z_eff(Z, occ, n, l):
    by_n = {}
    for (nq, lq), e in occ.items():
        by_n[nq] = by_n.get(nq, 0) + e
    same_n = by_n.get(n, 0)
    lower = sum(v for k, v in by_n.items() if k < n)
    if l in (0, 1):
        S_same = (0.30 if n == 1 else 0.35) * max(0, same_n - 1)
        S_nm1 = 0.85 * by_n.get(n - 1, 0)
        S_low = 1.00 * sum(v for k, v in by_n.items() if k <= n - 2)
        S = S_same + S_nm1 + S_low
    else:
        same_nl = occ.get((n, l), 0)
        S_same = 0.35 * max(0, same_nl - 1)
        S = S_same + 1.00 * lower
    return max(1.0, Z - S)


def guess_valence_shells(Z, occ):
    n_max = max(n for (n, l) in occ if occ[(n, l)] > 0)
    shells = []
    if occ.get((n_max, 0), 0) > 0:
        shells.append((n_max, 0))
    if occ.get((n_max, 1), 0) > 0:
        shells.append((n_max, 1))
    if occ.get((n_max - 1, 2), 0) > 0:
        shells.append((n_max - 1, 2))
    if not shells:
        top = max(occ.items(), key=lambda kv: (kv[0][0], kv[0][1]))[0]
        shells = [top]
    return shells


def double_factorial(n):
    if n <= 0:
        return 1
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def normalize_gto(alpha, lxyz):
    lx, ly, lz = lxyz
    pref = (2 * alpha / np.pi) ** 1.5
    Nx = (4 * alpha) ** (lx / 2) / math.sqrt(double_factorial(2 * lx - 1))
    Ny = (4 * alpha) ** (ly / 2) / math.sqrt(double_factorial(2 * ly - 1))
    Nz = (4 * alpha) ** (lz / 2) / math.sqrt(double_factorial(2 * lz - 1))
    return math.sqrt(pref) * Nx * Ny * Nz


def cartesian_grid_with_frac(cell, ngrid):
    """
    cell: 3x3 with rows a,b,c (Å)
    Returns:
      R_cart: (Nx,Ny,Nz,3)
      F_frac: (Nx,Ny,Nz,3) fractional coords in [0,1)
      vol, dV
    """
    a, b, c = cell
    Nx, Ny, Nz = ngrid
    us = (np.arange(Nx) + 0.5) / Nx
    vs = (np.arange(Ny) + 0.5) / Ny
    ws = (np.arange(Nz) + 0.5) / Nz
    U, V, W = np.meshgrid(us, vs, ws, indexing="ij")
    F = np.stack([U, V, W], axis=-1)  # frac
    R = U[..., None] * a + V[..., None] * b + W[..., None] * c  # cart
    vol = abs(np.linalg.det(cell))
    dV = vol / (Nx * Ny * Nz)
    return R, F, vol, dV


def wrap01(x):
    return x - np.floor(x)


def wrapm05p05(x):
    # wrap into [-0.5, 0.5)
    return ((x + 0.5) % 1.0) - 0.5


def eval_gto_on_grid_pbc_minimum(cell, Fgrid, f_center, alpha, lxyz):
    """
    Periodic AO by minimum-image wrapped fractional displacement.
    """
    a, b, c = cell
    dF = wrapm05p05(Fgrid - f_center[None, None, None, :])  # (Nx,Ny,Nz,3)
    # Δr_cart = dFx*a + dFy*b + dFz*c
    dR = (
        dF[..., 0, None] * a[None, None, None, :]
        + dF[..., 1, None] * b[None, None, None, :]
        + dF[..., 2, None] * c[None, None, None, :]
    )
    dx, dy, dz = dR[..., 0], dR[..., 1], dR[..., 2]
    lx, ly, lz = lxyz
    r2 = dx * dx + dy * dy + dz * dz
    N = normalize_gto(alpha, lxyz)
    return N * (dx**lx) * (dy**ly) * (dz**lz) * np.exp(-alpha * r2)


def eval_gto_on_grid_pbc_sum(cell, Fgrid, f_center, alpha, lxyz, nimg):
    """
    Periodic AO by explicit lattice sum over translations t in [-nimg..nimg]^3.
    nimg=0 reduces to the home cell only (not periodic) – use >=1 to be meaningful.
    """
    a, b, c = cell
    Nx, Ny, Nz, _ = Fgrid.shape
    phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    N = normalize_gto(alpha, lxyz)
    lx, ly, lz = lxyz
    for i in range(-nimg, nimg + 1):
        for j in range(-nimg, nimg + 1):
            for k in range(-nimg, nimg + 1):
                dF = Fgrid - (f_center + np.array([i, j, k]))[None, None, None, :]
                dR = (
                    dF[..., 0, None] * a[None, None, None, :]
                    + dF[..., 1, None] * b[None, None, None, :]
                    + dF[..., 2, None] * c[None, None, None, :]
                )
                dx, dy, dz = dR[..., 0], dR[..., 1], dR[..., 2]
                r2 = dx * dx + dy * dy + dz * dz
                phi += N * (dx**lx) * (dy**ly) * (dz**lz) * np.exp(-alpha * r2)
    return phi


def assemble_matrices_from_grid(
    cell, Fgrid, dV, basis, K=K_WOLFSBERG, pbc_mode="minimum", nimg=1
):
    nbf = len(basis)
    shp = Fgrid.shape[:3]
    npts = np.prod(shp)
    PHI = np.zeros((nbf, npts), dtype=np.float64)
    eps = np.zeros(nbf, dtype=np.float64)
    for i, bf in enumerate(basis):
        if pbc_mode == "minimum":
            phi = eval_gto_on_grid_pbc_minimum(
                cell, Fgrid, bf["fcenter"], bf["alpha"], bf["lxyz"]
            )
        else:
            phi = eval_gto_on_grid_pbc_sum(
                cell, Fgrid, bf["fcenter"], bf["alpha"], bf["lxyz"], nimg
            )
        PHI[i] = phi.reshape(-1)
        eps[i] = bf["eps"]
    S = dV * (PHI @ PHI.T)
    H = np.empty_like(S)
    np.fill_diagonal(H, eps)
    avg = 0.5 * (eps[:, None] + eps[None, :])
    H += K * S * avg
    np.fill_diagonal(H, eps)
    return S, H, PHI, eps


def generalized_eig(H, S):
    se, U = np.linalg.eigh(S)
    se = np.clip(se, 1e-10, None)
    Sminushalf = (U / np.sqrt(se)) @ U.T
    H_o = Sminushalf @ H @ Sminushalf
    e, Ctilde = np.linalg.eigh(H_o)
    C = Sminushalf @ Ctilde
    return e, C


def estimate_valence_electrons(symbols):
    total = 0
    for sym in symbols:
        Z = atomic_numbers[sym]
        occ = electron_configuration(Z)
        n_max = max(n for (n, l) in occ if occ[(n, l)] > 0)
        val = (
            occ.get((n_max, 0), 0) + occ.get((n_max, 1), 0) + occ.get((n_max - 1, 2), 0)
        )
        total += val
    return int(total)


def basis_for_atom(Z, f_center, occ):
    shells = guess_valence_shells(Z, occ)
    bas = []
    for n, l in shells:
        Zeff = slater_Z_eff(Z, occ, n, l)
        eps_ev = -RY_EV * (Zeff**2) / (n**2)
        alpha = (Zeff / n) ** 2 * ALPHA_L_SCALE.get(l, 0.8)
        if l == 0:
            bas.append(
                {
                    "fcenter": f_center,
                    "lxyz": (0, 0, 0),
                    "alpha": alpha,
                    "eps": eps_ev,
                    "label": f"{n}s",
                }
            )
        elif l == 1:
            for lxyz, lab in [
                ((1, 0, 0), "p_x"),
                ((0, 1, 0), "p_y"),
                ((0, 0, 1), "p_z"),
            ]:
                bas.append(
                    {
                        "fcenter": f_center,
                        "lxyz": lxyz,
                        "alpha": alpha,
                        "eps": eps_ev,
                        "label": f"{n}p{lab}",
                    }
                )
        elif l == 2:
            for lxyz, lab in [
                ((2, 0, 0), "d_xx"),
                ((0, 2, 0), "d_yy"),
                ((0, 0, 2), "d_zz"),
                ((1, 1, 0), "d_xy"),
                ((1, 0, 1), "d_xz"),
                ((0, 1, 1), "d_yz"),
            ]:
                bas.append(
                    {
                        "fcenter": f_center,
                        "lxyz": lxyz,
                        "alpha": alpha,
                        "eps": eps_ev,
                        "label": f"{n}d_{lab}",
                    }
                )
    return bas


def build_basis_all_atoms(symbols, frac_positions):
    basis = []
    for sym, fpos in zip(symbols, frac_positions):
        Z = atomic_numbers[sym]
        if Z < 1 or Z > 83:
            raise ValueError(f"Element {sym} Z={Z} not in [1,83].")
        occ = electron_configuration(Z)
        basis.extend(basis_for_atom(Z, fpos, occ))
    return basis


def write_cube(filename, rho_flat, cell, grid_shape, atom_syms, atom_cart):
    Nx, Ny, Nz = grid_shape
    a, b, c = cell
    with open(filename, "w") as f:
        f.write("LCAO periodic initial density\n")
        f.write("Numerical extended-Hückel (Γ-only)\n")
        f.write(f"{len(atom_syms):5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{Nx:5d} {a[0]/Nx:12.6f} {a[1]/Nx:12.6f} {a[2]/Nx:12.6f}\n")
        f.write(f"{Ny:5d} {b[0]/Ny:12.6f} {b[1]/Ny:12.6f} {b[2]/Ny:12.6f}\n")
        f.write(f"{Nz:5d} {c[0]/Nz:12.6f} {c[1]/Nz:12.6f} {c[2]/Nz:12.6f}\n")
        for sym, pos in zip(atom_syms, atom_cart):
            Z = atomic_numbers[sym]
            f.write(
                f"{Z:5d} {float(Z):12.6f} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n"
            )
        rho = rho_flat.reshape(grid_shape)
        for ix in range(Nx):
            for iy in range(Ny):
                line = []
                for iz in range(Nz):
                    line.append(f"{rho[ix,iy,iz]:13.5e}")
                    if len(line) == 6:
                        f.write("".join(line) + "\n")
                        line = []
                if line:
                    f.write("".join(line) + "\n")


def write_chgcar(filename, rho_flat, cell, atom_syms, atom_cart, grid_shape):
    Nx, Ny, Nz = grid_shape
    a, b, c = cell
    # order species by first appearance
    order = []
    counts = {}
    for s in atom_syms:
        if s not in counts:
            order.append(s)
        counts[s] = counts.get(s, 0) + 1
    species = order
    num = [counts[s] for s in species]
    # fractional positions (r = f @ cell; f = r @ cell^{-1})
    Minv = np.linalg.inv(cell)
    frac = atom_cart @ Minv
    with open(filename, "w") as f:
        f.write("LCAO periodic initial density (CHGCAR)\n")
        f.write("  1.00000000000000\n")
        for v in cell:
            f.write(f"  {v[0]:.16f}  {v[1]:.16f}  {v[2]:.16f}\n")
        f.write("  " + "  ".join(species) + "\n")
        f.write("  " + "  ".join(str(x) for x in num) + "\n")
        f.write("Direct\n")
        for r in frac:
            rr = r - np.floor(r)
            f.write(f"  {rr[0]:.16f}  {rr[1]:.16f}  {rr[2]:.16f}\n")
        f.write("\n")
        f.write(f"   {Nx:5d}   {Ny:5d}   {Nz:5d}\n")
        vals = rho_flat.reshape(grid_shape, order="C").ravel(order="F")
        for i in range(0, vals.size, 5):
            f.write(" ".join(f"{x:.11E}" for x in vals[i : i + 5]) + "\n")


def compute_density(atoms, ngrid, K, pbc_mode, nimg):
    cell = np.array(atoms.cell)  # rows a,b,c
    symbols = atoms.get_chemical_symbols()
    cart = atoms.get_positions()
    Minv = np.linalg.inv(cell)
    frac = cart @ Minv  # N x 3, not wrapped
    Rgrid, Fgrid, vol, dV = cartesian_grid_with_frac(cell, ngrid)
    # build periodic basis in fractional coordinates
    basis = []
    for sym, fpos in zip(symbols, frac):
        Z = atomic_numbers[sym]
        occ = electron_configuration(Z)
        basis.extend(basis_for_atom(Z, fpos, occ))
    # matrices
    S, H, PHI, eps = assemble_matrices_from_grid(
        cell, Fgrid, dV, basis, K=K, pbc_mode=pbc_mode, nimg=nimg
    )
    e, C = generalized_eig(H, S)
    nelec_val = estimate_valence_electrons(symbols)
    occ = np.zeros_like(e)
    nfull = min(len(e), nelec_val // 2)
    occ[:nfull] = 2.0
    if nelec_val % 2 == 1 and nfull < len(e):
        occ[nfull] = 1.0
    PSI = C.T @ PHI
    rho_flat = (occ[:, None] * (PSI * PSI)).sum(axis=0)  # spin-summed
    # scale density to match the electron count (numerical consistency)
    Nx, Ny, Nz = ngrid
    vol = abs(np.linalg.det(cell))
    dV = vol / (Nx * Ny * Nz)
    Q_num = rho_flat.sum() * dV
    if Q_num > 0:
        rho_flat *= nelec_val / Q_num
    summary = {
        "nelec_valence": int(nelec_val),
        "nbasis": int(len(basis)),
        "nmo": int(len(e)),
        "occupied_MOs": int(np.count_nonzero(occ > FERMI_TOL)),
        "grid": list(ngrid),
        "first10_e_ev": [float(x) for x in e[: min(10, len(e))]],
        "pbc_mode": pbc_mode,
        "images": nimg,
    }
    return rho_flat, cell, symbols, cart, summary


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Periodic LCAO initial density from CIF (H–Bi)."
    )
    p.add_argument("cif")
    p.add_argument("--nx", type=int, default=GRID_N[0])
    p.add_argument("--ny", type=int, default=GRID_N[1])
    p.add_argument("--nz", type=int, default=GRID_N[2])
    p.add_argument("--K", type=float, default=K_WOLFSBERG)
    p.add_argument("--fmt", choices=["cube", "chgcar"], default=OUTPUT_FMT)
    p.add_argument("--out", default=OUTPUT_FILE)
    p.add_argument(
        "--pbc",
        choices=["minimum", "sum"],
        default=PBC_MODE,
        help="Periodic AO evaluation method",
    )
    p.add_argument(
        "--images",
        type=int,
        default=N_IMAGES,
        help="If --pbc sum, sum over [-images..images]^3",
    )
    args = p.parse_args()

    rho_flat, cell, symbols, cart, summary = compute_density(
        args.cif,
        ngrid=(args.nx, args.ny, args.nz),
        K=args.K,
        pbc_mode=args.pbc,
        nimg=args.images,
    )
    if args.fmt == "cube":
        write_cube(args.out, rho_flat, cell, (args.nx, args.ny, args.nz), symbols, cart)
    else:
        write_chgcar(
            args.out, rho_flat, cell, symbols, cart, (args.nx, args.ny, args.nz)
        )
    print(
        json.dumps(
            {"summary": summary, "output": {"format": args.fmt, "file": args.out}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
