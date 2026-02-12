"""
SchNet-based Hamiltonian backbone for molecular systems.

Wraps PyG's SchNet to predict a scalar potential energy E(z, r).
Forces are obtained as  F = -∇_r E  via autograd, guaranteeing
energy conservation by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet


class SchNetHamiltonian(nn.Module):
    """
    SchNet that outputs a shifted / scaled scalar energy.

    Parameters
    ----------
    hidden_channels : int
        Embedding + interaction channel width.
    num_interactions : int
        Number of message-passing layers.
    num_gaussians : int
        Radial basis functions for distance expansion.
    cutoff : float
        Radial cutoff (Å).
    mean, std : float
        Energy shift/scale (computed from training data).
    max_num_neighbors : int
        Neighbour list cap.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        mean: float = 0.0,
        std: float = 1.0,
        max_num_neighbors: int = 64,
    ):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout="add",
        )
        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("_std",  torch.tensor(std,  dtype=torch.float))

    # ── Forward: scalar energy ───────────────────────────────────────

    def forward(self, z, pos, batch=None):
        """
        Predict total energy (de-normalised).

        Parameters
        ----------
        z : LongTensor [N_atoms]
            Atomic numbers.
        pos : FloatTensor [N_atoms, 3]
            Cartesian positions (Å).
        batch : LongTensor [N_atoms], optional
            Batch assignment vector.

        Returns
        -------
        energy : FloatTensor [n_molecules]
        """
        raw = self.schnet(z, pos, batch)          # [n_mol, 1] or [n_mol]
        raw = raw.squeeze(-1)                     # [n_mol]
        return raw * self._std + self._mean

    # ── Forces via autograd ──────────────────────────────────────────

    def forces(self, z, pos, batch=None):
        """
        Compute  F = -∇_r E  via autograd.

        Returns
        -------
        energy : FloatTensor [n_mol]
        forces : FloatTensor [N_atoms, 3]
        """
        pos = pos.requires_grad_(True)
        energy = self.forward(z, pos, batch)
        grad = torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]
        forces = -grad
        return energy, forces

    # ── Convenience: full (q,p) Hamiltonian interface ────────────────

    def hamiltonian(self, z, pos, momenta, masses, batch=None):
        """
        H = T + V  where  T = Σ p²/(2m),  V = E_SchNet(z, r).

        Parameters
        ----------
        momenta : FloatTensor [N_atoms, 3]
        masses  : FloatTensor [N_atoms] or [N_atoms, 1]

        Returns
        -------
        H : FloatTensor [n_mol]
        """
        if masses.dim() == 1:
            masses = masses.unsqueeze(-1)         # [N, 1]
        kinetic = 0.5 * (momenta ** 2 / masses).sum(dim=-1)   # [N]

        # Aggregate kinetic energy per molecule
        if batch is not None:
            from torch_geometric.utils import scatter
            kinetic = scatter(kinetic, batch, dim=0, reduce="add")
        else:
            kinetic = kinetic.sum(dim=0, keepdim=True)

        potential = self.forward(z, pos, batch)
        return kinetic + potential
