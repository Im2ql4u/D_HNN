"""
Autoencoder for SOAP-like expansion coefficients.

Architecture (two-level, weight-shared across atoms):

  Encoder:
    per-atom:  (n_max × L_total) → d_atom          [shared weights]
    global:    (N_atoms × d_atom) → d_hidden → (q, p)

  Decoder:
    global:    (q, p) → d_hidden → (N_atoms × d_atom)
    per-atom:  d_atom → (n_max × L_total)            [shared weights]

The latent space is split into (q, p) for Hamiltonian dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SOAPAutoencoder(nn.Module):
    """
    Two-level autoencoder for SOAP expansion coefficients.

    Parameters
    ----------
    n_atoms : int
        Number of atoms per frame (fixed ordering).
    d_soap : int
        SOAP descriptor dimension per atom (n_max × (l_max+1)²).
    d_atom : int
        Per-atom hidden dimension after the shared atom-encoder.
    d_hidden : int
        Global hidden layer dimension.
    d_latent : int
        Latent dimension for q (and p).  Total latent = 2 × d_latent.
    """

    def __init__(
        self,
        n_atoms: int = 9,
        d_soap: int = 200,
        d_atom: int = 64,
        d_hidden: int = 256,
        d_latent: int = 24,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.d_soap = d_soap
        self.d_atom = d_atom
        self.d_latent = d_latent

        # ── Per-atom encoder (shared across atoms) ───────────────────
        self.atom_encoder = nn.Sequential(
            nn.Linear(d_soap, d_atom * 2),
            nn.SiLU(),
            nn.Linear(d_atom * 2, d_atom),
            nn.SiLU(),
        )

        # ── Global encoder ───────────────────────────────────────────
        d_global = n_atoms * d_atom
        self.global_encoder = nn.Sequential(
            nn.Linear(d_global, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 2 * d_latent),
        )

        # ── Global decoder ───────────────────────────────────────────
        self.global_decoder = nn.Sequential(
            nn.Linear(2 * d_latent, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_global),
        )

        # ── Per-atom decoder (shared across atoms) ───────────────────
        self.atom_decoder = nn.Sequential(
            nn.Linear(d_atom, d_atom * 2),
            nn.SiLU(),
            nn.Linear(d_atom * 2, d_soap),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode SOAP coefficients to latent (q, p).

        Parameters
        ----------
        x : (batch, n_atoms, d_soap)  or  (batch, n_atoms * d_soap)

        Returns
        -------
        q, p : each (batch, d_latent)
        """
        if x.dim() == 2 and x.shape[-1] == self.n_atoms * self.d_soap:
            x = x.reshape(-1, self.n_atoms, self.d_soap)

        B = x.shape[0]

        # Per-atom encoding (shared weights)
        h = self.atom_encoder(x)  # (B, n_atoms, d_atom)

        # Flatten and encode globally
        h_flat = h.reshape(B, -1)   # (B, n_atoms * d_atom)
        z = self.global_encoder(h_flat)  # (B, 2*d_latent)

        q, p = z.chunk(2, dim=-1)
        return q, p

    def decode(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Decode latent (q, p) to reconstructed SOAP coefficients.

        Returns
        -------
        x_recon : (batch, n_atoms, d_soap)
        """
        z = torch.cat([q, p], dim=-1)   # (B, 2*d_latent)
        h_flat = self.global_decoder(z)  # (B, n_atoms * d_atom)

        B = h_flat.shape[0]
        h = h_flat.reshape(B, self.n_atoms, self.d_atom)

        # Per-atom decoding (shared weights)
        x_recon = self.atom_decoder(h)  # (B, n_atoms, d_soap)
        return x_recon

    def forward(self, x: torch.Tensor):
        """
        Full autoencoder pass.

        Returns
        -------
        x_recon, q, p
        """
        q, p = self.encode(x)
        x_recon = self.decode(q, p)
        return x_recon, q, p
