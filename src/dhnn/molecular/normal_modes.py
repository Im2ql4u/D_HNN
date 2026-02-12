"""
Normal mode analysis from a trained potential.

Given a differentiable energy model, compute the Hessian at equilibrium,
mass-weight and diagonalise to obtain normal mode coordinates.
Provides projection utilities to transform between Cartesian and
normal-mode representations.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

# ── Atomic masses (amu) for common elements ──────────────────────────
ATOMIC_MASSES: dict[int, float] = {
    1: 1.008,     # H
    6: 12.011,    # C
    7: 14.007,    # N
    8: 15.999,    # O
    9: 18.998,    # F
    15: 30.974,   # P
    16: 32.060,   # S
    17: 35.450,   # Cl
}


class NormalModeResult(NamedTuple):
    """Output of :func:`normal_mode_analysis`."""
    frequencies_cm1: np.ndarray   # [3N-6] in cm⁻¹ (sorted ascending)
    mode_vectors: np.ndarray      # [3N, 3N-6] mass-weighted eigenvectors
    hessian: np.ndarray           # [3N, 3N] Cartesian Hessian (kcal/mol/ų)
    eq_positions: np.ndarray      # [N, 3] equilibrium geometry (Å)
    masses: np.ndarray            # [N] atomic masses (amu)


# ── Hessian computation ──────────────────────────────────────────────

def compute_hessian(
    model,
    z: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Compute the Cartesian Hessian  H_ij = ∂²E/∂r_i∂r_j  via autograd.

    Parameters
    ----------
    model : SchNetHamiltonian (or any model with ``forward(z, pos, batch) → energy``)
    z : LongTensor [N_atoms]
    pos : FloatTensor [N_atoms, 3]  — equilibrium geometry
    batch : LongTensor [N_atoms], optional

    Returns
    -------
    hessian : ndarray [3N, 3N]
    """
    model.eval()
    pos = pos.clone().detach().float().requires_grad_(True)
    z = z.clone().detach()
    if batch is not None:
        batch = batch.clone().detach()

    n_coords = pos.numel()  # 3N

    energy = model(z, pos, batch)
    # First derivatives (forces with sign)
    grad = torch.autograd.grad(
        energy.sum(), pos, create_graph=True,
    )[0]  # [N, 3]

    grad_flat = grad.reshape(-1)  # [3N]

    # Second derivatives — row by row
    hess_rows = []
    for i in range(n_coords):
        row = torch.autograd.grad(
            grad_flat[i], pos, retain_graph=True,
        )[0]  # [N, 3]
        hess_rows.append(row.reshape(-1).detach().cpu().numpy())

    hessian = np.stack(hess_rows, axis=0)  # [3N, 3N]

    # Symmetrise (numerical noise)
    hessian = 0.5 * (hessian + hessian.T)
    return hessian


# ── Normal mode analysis ─────────────────────────────────────────────

def normal_mode_analysis(
    model,
    z: torch.Tensor,
    eq_pos: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> NormalModeResult:
    """
    Full vibrational analysis: Hessian → mass-weighted diagonalisation.

    Removes 6 translational + rotational modes (3N-5 for linear molecules)
    by discarding modes with near-zero frequency.

    Parameters
    ----------
    model : SchNetHamiltonian
    z : LongTensor [N_atoms]
    eq_pos : FloatTensor [N_atoms, 3]

    Returns
    -------
    NormalModeResult
    """
    n_atoms = z.shape[0]
    masses_amu = np.array([ATOMIC_MASSES[int(zi)] for zi in z])

    # ── Hessian in Cartesian coordinates ─────────────────────────────
    hessian = compute_hessian(model, z, eq_pos, batch)

    # ── Mass-weight the Hessian ──────────────────────────────────────
    #   H_mw = M^{-1/2} H M^{-1/2}
    # where M is diag(m1, m1, m1, m2, m2, m2, ...)
    mass_vec = np.repeat(masses_amu, 3)              # [3N]
    inv_sqrt_m = 1.0 / np.sqrt(mass_vec)             # [3N]
    hessian_mw = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)

    # ── Diagonalise ──────────────────────────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_mw)
    # eigenvalues are in kcal/mol/ų/amu — convert to cm⁻¹

    # Conversion factor:  ω (cm⁻¹) = sqrt(λ · conv) / (2π c)
    # λ in kcal/mol/(amu·Å²)
    # 1 kcal/mol = 4184 J/mol → per molecule = 4184/Nₐ J
    # 1 amu = 1.66054e-27 kg,  1 Å = 1e-10 m
    avogadro = 6.02214076e23
    conv = (4184.0 / avogadro) / (1.66054e-27 * (1e-10)**2)  # → s⁻²
    c_cm = 2.99792458e10                   # speed of light in cm/s

    freqs_s2 = eigenvalues * conv          # Hz²
    # Some eigenvalues are negative (translation/rotation) — take abs then sign
    signs = np.sign(freqs_s2)
    freqs_cm1 = signs * np.sqrt(np.abs(freqs_s2)) / (2.0 * np.pi * c_cm)

    # ── Remove translation + rotation modes (|ν| < 10 cm⁻¹) ────────
    internal_mask = np.abs(freqs_cm1) > 10.0
    freqs_internal = freqs_cm1[internal_mask]
    modes_internal = eigenvectors[:, internal_mask]   # [3N, n_internal]

    # Sort by frequency
    order = np.argsort(freqs_internal)
    freqs_internal = freqs_internal[order]
    modes_internal = modes_internal[:, order]

    # Un-mass-weight the mode vectors for projection in Cartesian space
    # Q_k = Σ_i (m_i^{1/2} · Δr_i) · e_k_i  (mass-weighted displacement)
    # We store mass-weighted eigenvectors so project_to_modes uses them directly

    return NormalModeResult(
        frequencies_cm1=freqs_internal,
        mode_vectors=modes_internal,
        hessian=hessian,
        eq_positions=eq_pos.detach().cpu().numpy(),
        masses=masses_amu,
    )


# ── Projection utilities ─────────────────────────────────────────────

def project_to_modes(
    positions: np.ndarray,
    result: NormalModeResult,
) -> np.ndarray:
    """
    Project Cartesian positions → normal mode coordinates.

    Parameters
    ----------
    positions : ndarray [N_atoms, 3] or [n_configs, N_atoms, 3]
    result : NormalModeResult

    Returns
    -------
    Q : ndarray [..., n_modes]   normal mode coordinates
    """
    eq = result.eq_positions             # [N, 3]
    masses = result.masses               # [N]
    U = result.mode_vectors              # [3N, K]

    single = positions.ndim == 2
    if single:
        positions = positions[np.newaxis]  # [1, N, 3]

    n_configs = positions.shape[0]
    delta = positions - eq[np.newaxis]   # [n, N, 3]
    delta_flat = delta.reshape(n_configs, -1)  # [n, 3N]

    # Mass-weight the displacements
    mass_vec = np.repeat(masses, 3)      # [3N]
    sqrt_m = np.sqrt(mass_vec)
    delta_mw = delta_flat * sqrt_m[np.newaxis]  # [n, 3N]

    Q = delta_mw @ U                     # [n, K]
    return Q[0] if single else Q


def project_from_modes(
    Q: np.ndarray,
    result: NormalModeResult,
) -> np.ndarray:
    """
    Reconstruct Cartesian positions from normal mode coordinates.

    Parameters
    ----------
    Q : ndarray [n_modes] or [n_configs, n_modes]
    result : NormalModeResult

    Returns
    -------
    positions : ndarray [N_atoms, 3] or [n_configs, N_atoms, 3]
    """
    eq = result.eq_positions
    masses = result.masses
    U = result.mode_vectors              # [3N, K]
    n_atoms = eq.shape[0]

    single = Q.ndim == 1
    if single:
        Q = Q[np.newaxis]

    # Un-mass-weight:  Δr_flat = M^{-1/2} · U · Q^T
    mass_vec = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(mass_vec)

    delta_mw = Q @ U.T                   # [n, 3N]
    delta_flat = delta_mw * inv_sqrt_m[np.newaxis]
    delta = delta_flat.reshape(-1, n_atoms, 3)

    positions = delta + eq[np.newaxis]
    return positions[0] if single else positions
