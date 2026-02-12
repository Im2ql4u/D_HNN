"""
SOAP-like Spherical Harmonics × Radial Basis expansion.

For each atom i in a molecular frame, compute:
    c^i_{n, lm} = Σ_{j≠i} g_n(r_ij) · Y_l^m(r̂_ij) · f_cut(r_ij)

where g_n are Gaussian radial basis functions, Y_l^m are real solid
harmonics, and f_cut is a smooth cosine cutoff.

The full per-atom descriptor is a tensor of shape (n_max, (l_max+1)²).
Flattened and concatenated over atoms this gives a per-frame descriptor.

The SOAP power spectrum (rotation-invariant) is also available:
    p^i_{nn'l} = Σ_m c^i_{n,l,m} · c^i_{n',l,m}
"""

from __future__ import annotations

import math
import numpy as np
import torch


# ── Real solid harmonics up to l = 4 ────────────────────────────────

def real_solid_harmonics(l_max: int, r_hat: torch.Tensor) -> list[torch.Tensor]:
    """
    Real solid harmonics on unit vectors.

    Parameters
    ----------
    l_max : int  (0 ≤ l_max ≤ 4)
    r_hat : (..., 3) unit direction vectors

    Returns
    -------
    list of tensors, one per l, each of shape (..., 2l+1)
    """
    x = r_hat[..., 0]
    y = r_hat[..., 1]
    z = r_hat[..., 2]
    results: list[torch.Tensor] = []

    # l = 0
    results.append(
        torch.ones_like(x).unsqueeze(-1) * 0.2820947917738781
    )  # 1 / (2√π)

    if l_max >= 1:
        c = 0.4886025119029199  # √(3/4π)
        results.append(torch.stack([c * y, c * z, c * x], dim=-1))

    if l_max >= 2:
        results.append(torch.stack([
            1.0925484305920792 * x * y,                     # m=-2
            1.0925484305920792 * y * z,                     # m=-1
            0.3153915652525200 * (2 * z * z - x * x - y * y),  # m=0
            1.0925484305920792 * x * z,                     # m=1
            0.5462742152960396 * (x * x - y * y),           # m=2
        ], dim=-1))

    if l_max >= 3:
        x2, y2, z2 = x * x, y * y, z * z
        results.append(torch.stack([
            0.5900435899266435 * y * (3 * x2 - y2),        # m=-3
            2.8906114426405538 * x * y * z,                 # m=-2
            0.4570457994644658 * y * (4 * z2 - x2 - y2),   # m=-1
            0.3731763325901154 * z * (2 * z2 - 3 * x2 - 3 * y2),  # m=0
            0.4570457994644658 * x * (4 * z2 - x2 - y2),   # m=1
            1.4453057213202769 * z * (x2 - y2),             # m=2
            0.5900435899266435 * x * (x2 - 3 * y2),        # m=3
        ], dim=-1))

    if l_max >= 4:
        x2, y2, z2 = x * x, y * y, z * z
        results.append(torch.stack([
            2.5033429417967046 * x * y * (x2 - y2),                # m=-4
            1.7701307697799304 * y * z * (3 * x2 - y2),           # m=-3
            0.9461746957575601 * x * y * (7 * z2 - 1),            # m=-2
            0.6690465435572892 * y * z * (7 * z2 - 3),            # m=-1
            0.1057855469152043 * (35 * z2 * z2 - 30 * z2 + 3),   # m=0
            0.6690465435572892 * x * z * (7 * z2 - 3),            # m=1
            0.4730873478787801 * (x2 - y2) * (7 * z2 - 1),        # m=2
            1.7701307697799304 * x * z * (x2 - 3 * y2),           # m=3
            0.6258357354491761 * (x2 * x2 - 6 * x2 * y2 + y2 * y2),  # m=4
        ], dim=-1))

    return results


# ── Radial basis ─────────────────────────────────────────────────────

def gaussian_rbf(
    distances: torch.Tensor,
    n_max: int,
    cutoff: float,
) -> torch.Tensor:
    """
    Gaussian radial basis functions with cosine cutoff envelope.

    Parameters
    ----------
    distances : (...) tensor of distances
    n_max : number of radial basis functions
    cutoff : cutoff radius in Å

    Returns
    -------
    (..., n_max) tensor of basis values
    """
    centers = torch.linspace(0.5, cutoff - 0.5, n_max, device=distances.device)
    width = 1.0 / ((cutoff / n_max) ** 2)

    d = distances.unsqueeze(-1)                 # (..., 1)
    rbf = torch.exp(-width * (d - centers) ** 2)

    # Cosine cutoff envelope
    envelope = 0.5 * (1.0 + torch.cos(math.pi * d / cutoff))
    envelope = envelope * (d < cutoff).float()

    return rbf * envelope


# ── Main SOAP expansion ─────────────────────────────────────────────

def compute_soap_coefficients(
    positions: np.ndarray,
    *,
    n_max: int = 8,
    l_max: int = 4,
    cutoff: float = 5.0,
    batch_size: int = 5000,
) -> np.ndarray:
    """
    Compute SOAP-like expansion coefficients for molecular frames.

    Parameters
    ----------
    positions : (N_frames, N_atoms, 3) numpy array
    n_max : number of radial basis functions
    l_max : max angular momentum (0 ≤ l_max ≤ 4)
    cutoff : cutoff radius in Å
    batch_size : frames per batch (memory control)

    Returns
    -------
    coeffs : (N_frames, N_atoms, n_max, (l_max+1)²)  numpy array
    """
    N_frames, N_atoms, _ = positions.shape
    L_total = (l_max + 1) ** 2

    all_coeffs = np.zeros((N_frames, N_atoms, n_max, L_total), dtype=np.float32)

    for start in range(0, N_frames, batch_size):
        end = min(start + batch_size, N_frames)
        pos = torch.tensor(positions[start:end], dtype=torch.float32)  # (B, N, 3)
        B = pos.shape[0]

        # Pairwise vectors:  r_ij = pos_i - pos_j   (B, N, N, 3)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1).clamp(min=1e-10)  # (B, N, N)

        # Mask: exclude self + beyond cutoff
        mask = (dist > 0.01) & (dist < cutoff)  # (B, N, N)

        # Unit direction vectors
        r_hat = diff / dist.unsqueeze(-1)

        # Radial basis: (B, N, N, n_max)
        rbf = gaussian_rbf(dist, n_max, cutoff)
        rbf = rbf * mask.unsqueeze(-1).float()

        # Spherical harmonics per l
        ylm_list = real_solid_harmonics(l_max, r_hat)

        # Accumulate coefficients per l via einsum:
        #   c^i_{n,lm} = Σ_j rbf[b,i,j,n] · Y_lm[b,i,j,m]
        coeffs_parts = []
        for Y_l in ylm_list:
            # Y_l: (B, N, N, 2l+1),  rbf: (B, N, N, n_max)
            c_l = torch.einsum("bijn,bijm->binm", rbf, Y_l)  # (B, N, n_max, 2l+1)
            coeffs_parts.append(c_l)

        # Concatenate angular channels: (B, N, n_max, L_total)
        coeffs_batch = torch.cat(coeffs_parts, dim=-1)
        all_coeffs[start:end] = coeffs_batch.numpy()

    return all_coeffs


def soap_power_spectrum(
    coeffs: np.ndarray,
    l_max: int = 4,
) -> np.ndarray:
    """
    SOAP power spectrum (rotation-invariant).

    p^i_{nn'l} = Σ_m c^i_{n,l,m} · c^i_{n',l,m}

    Parameters
    ----------
    coeffs : (N_frames, N_atoms, n_max, (l_max+1)²)

    Returns
    -------
    ps : (N_frames, N_atoms, n_max*(n_max+1)/2 * (l_max+1))
    """
    N, A, n_max, _ = coeffs.shape
    ps_parts = []

    offset = 0
    for l in range(l_max + 1):
        dim_l = 2 * l + 1
        c_l = coeffs[..., offset:offset + dim_l]  # (N, A, n_max, 2l+1)
        offset += dim_l

        # p_{nn'l} = c_{n,:} · c_{n',:}  over m
        # (N, A, n_max, 2l+1) @ (N, A, 2l+1, n_max) → (N, A, n_max, n_max)
        p_l = np.einsum("...nm,...km->...nk", c_l, c_l)

        # Take upper triangle (including diagonal) for uniqueness
        idx_n, idx_k = np.triu_indices(n_max)
        p_l_flat = p_l[..., idx_n, idx_k]  # (N, A, n_unique)
        ps_parts.append(p_l_flat)

    return np.concatenate(ps_parts, axis=-1)
