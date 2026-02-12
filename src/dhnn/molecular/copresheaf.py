"""
Copresheaf Neural Network for molecular energy/force prediction.

Mathematical foundation
-----------------------
A **copresheaf** on the molecular graph assigns:

* 0-cells (atoms)  →  feature stalks  F(v) ∈ R^d_node
* 1-cells (edges)  →  interaction stalks  F(e) ∈ R^d_stalk

with **extension maps** from atom stalks into the shared edge stalk:

    F_{v→e} : R^d_node → R^d_stalk

For efficiency the maps are *factored*:

    F_{v→e} = diag(φ_v(rbf)) · W_v

where  W_v ∈ R^{d_stalk × d_node}  is a learned basis projection (one
for sender, one for receiver) and  φ_v(rbf) ∈ R^d_stalk  is a
distance-conditioned modulation from Gaussian RBF expansion.

The copresheaf message from atom j to atom i through edge e is:

    m_{j→i} = W_recv^T · diag(φ(rbf)) · W_send · x_j

where φ(rbf) is a single distance-conditioned modulation in the stalk
space. The asymmetry W_send ≠ W_recv captures the copresheaf structure —
sender and receiver "see" the interaction through different extension maps.

Why copresheaf > standard GNN:
  - The extension maps capture *how* each atom's properties extend to
    define a pairwise interaction (different perspectives per endpoint).
  - The product φ_recv ⊙ φ_send gives a *bilinear* distance filter —
    more expressive than SchNet's single continuous filter.
  - The low-rank structure (via d_stalk) provides implicit regularisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Distance expansion & envelope ────────────────────────────────────

class GaussianRBF(nn.Module):
    """Expand scalar distances into Gaussian radial basis functions."""

    def __init__(self, n_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        offset = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.coeff * (dist.unsqueeze(-1) - self.offset) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine envelope → 0 at cutoff."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(math.pi * dist / self.cutoff)) * (
            dist < self.cutoff
        ).float()


# ── Pure-PyTorch radius graph (no torch_cluster dependency) ──────────

def _build_edges(
    pos: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor | None,
) -> torch.Tensor:
    """
    Build radius-graph edge_index via cdist.

    Works on CPU/MPS/CUDA without torch_cluster.
    """
    if batch is None:
        batch = pos.new_zeros(pos.shape[0], dtype=torch.long)

    N = pos.shape[0]
    with torch.no_grad():
        dist = torch.cdist(pos.detach(), pos.detach())   # [N, N]
        same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)
        mask = same_mol & (dist < cutoff) & ~torch.eye(
            N, dtype=torch.bool, device=pos.device
        )
        src, tgt = torch.where(mask)

    return torch.stack([src, tgt], dim=0)


# ── Copresheaf layer ─────────────────────────────────────────────────

class CopresheafLayer(nn.Module):
    """
    Copresheaf diffusion layer with factored extension maps.

    For each directed edge (sender j → receiver i):
        proj_j   = W_send · x_j                        (into stalk)
        stalk_m  = φ(rbf) ⊙ proj_j                     (distance-modulated)
        msg      = W_recv^T · stalk_m                   (back to node)

    The copresheaf structure arises from the asymmetric basis
    projections W_send ≠ W_recv — sender and receiver "see" the
    interaction through different extension maps — combined with a
    distance-conditioned filter φ that modulates the stalk.

    Aggregated messages are passed through a gated residual update.
    """

    def __init__(self, d_node: int, d_stalk: int, n_rbf: int):
        super().__init__()
        self.d_node = d_node
        self.d_stalk = d_stalk

        # ── Extension map basis (shared across edges) ────────────────
        self.W_send = nn.Linear(d_node, d_stalk, bias=False)
        self.W_recv = nn.Linear(d_node, d_stalk, bias=False)

        # ── Distance-conditioned stalk modulation ────────────────────
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_stalk),
        )

        # ── Gated residual update ────────────────────────────────────
        self.gate = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_node),
        )
        self.norm = nn.LayerNorm(d_node)

        # Zero-init filter output → messages start at zero, but
        # gradient of φ·proj_j w.r.t. filter params is non-zero
        # (proj_j ≠ 0 from atom embeddings), so learning starts
        nn.init.zeros_(self.filter_net[-1].weight)
        nn.init.zeros_(self.filter_net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        envelope: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : [N, d_node]    node features
        edge_index: [2, E]         (src=sender j, tgt=receiver i)
        rbf       : [E, n_rbf]     radial basis features
        envelope  : [E]            smooth cutoff values
        """
        src, tgt = edge_index
        N = x.shape[0]

        # Distance-dependent stalk modulation
        phi = self.filter_net(rbf)                           # [E, d_stalk]
        phi = phi * envelope.unsqueeze(-1)

        # Project sender into stalk, modulate, project back to receiver
        proj_j = self.W_send(x[src])                         # [E, d_stalk]
        stalk_msg = phi * proj_j                              # [E, d_stalk]
        msg = stalk_msg @ self.W_recv.weight                  # [E, d_node]

        # Aggregate at receiver nodes
        agg = x.new_zeros(N, self.d_node)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand_as(msg), msg)

        # Pre-norm gated residual
        return self.norm(x + self.gate(agg))


# ── Full network ─────────────────────────────────────────────────────

class CopresheafNet(nn.Module):
    """
    Copresheaf Neural Network.

    atom embeddings → Gaussian RBF → N copresheaf layers → per-atom
    energy → sum-pooling → molecular energy.
    """

    def __init__(
        self,
        d_node: int = 128,
        d_stalk: int = 16,
        n_layers: int = 6,
        n_rbf: int = 50,
        cutoff: float = 5.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.atom_emb = nn.Embedding(max_z, d_node)
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.envelope = CosineCutoff(cutoff)

        self.layers = nn.ModuleList(
            [CopresheafLayer(d_node, d_stalk, n_rbf) for _ in range(n_layers)]
        )

        self.readout = nn.Sequential(
            nn.Linear(d_node, d_node // 2),
            nn.SiLU(),
            nn.Linear(d_node // 2, 1),
        )

        # Small readout init → initial raw energy ≈ 0
        nn.init.xavier_uniform_(self.readout[0].weight, gain=0.1)
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-molecule raw energy (before shift/scale)."""
        if batch is None:
            batch = pos.new_zeros(pos.shape[0], dtype=torch.long)

        # ── Build graph ──────────────────────────────────────────────
        edge_index = _build_edges(pos, self.cutoff, batch)
        src, tgt = edge_index

        # ── Interatomic distances (differentiable) ───────────────────
        diff = pos[tgt] - pos[src]
        dist = diff.norm(dim=-1)
        rbf = self.rbf(dist)
        env = self.envelope(dist)

        # ── Atom embeddings → copresheaf diffusion ───────────────────
        x = self.atom_emb(z)
        for layer in self.layers:
            x = layer(x, edge_index, rbf, env)

        # ── Per-atom energy → sum → per-molecule ─────────────────────
        atom_e = self.readout(x).squeeze(-1)               # [N_atoms]
        n_mol = batch.max().item() + 1
        energy = torch.zeros(n_mol, device=atom_e.device, dtype=atom_e.dtype)
        energy = energy.index_add(0, batch, atom_e)        # out-of-place
        return energy


# ── Hamiltonian wrapper (same API as SchNetHamiltonian) ──────────────

class CopresheafHamiltonian(nn.Module):
    """
    Copresheaf network wrapped as a differentiable Hamiltonian.

    ``forward(z, pos, batch) → energy``  (shifted & scaled)
    ``forces(z, pos, batch) → (energy, F = -∇E)`` via autograd.
    """

    def __init__(
        self,
        d_node: int = 128,
        d_stalk: int = 16,
        n_layers: int = 6,
        n_rbf: int = 50,
        cutoff: float = 5.0,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__()
        self.net = CopresheafNet(
            d_node=d_node,
            d_stalk=d_stalk,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
        )
        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("_std", torch.tensor(std, dtype=torch.float))

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.net(z, pos, batch)
        return raw * self._std + self._mean

    def forces(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ):
        """Compute energy & conservative forces F = -∇_r E."""
        pos = pos.requires_grad_(True)
        # enable_grad() overrides any outer no_grad() context (e.g. val loop)
        with torch.enable_grad():
            energy = self.forward(z, pos, batch)
            grad = torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        return energy, -grad
