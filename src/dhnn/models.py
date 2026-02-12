"""
Neural network architectures for Hamiltonian learning.

- BaselineNN:  standard MLP  →  (dq/dt, dp/dt)
- HNN:         learns H_θ   →  dynamics via Hamilton's equations
- DHNN:        learns H_θ + D_θ  →  conservative + dissipative dynamics
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Helpers ──────────────────────────────────────────────────────────

def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int,
              n_hidden: int = 2, activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Build a simple MLP with *n_hidden* hidden layers."""
    layers: list[nn.Module] = []
    dims = [input_dim] + [hidden_dim] * n_hidden + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:            # no activation on output
            layers.append(activation())
    return nn.Sequential(*layers)


# ── Baseline ─────────────────────────────────────────────────────────

class BaselineNN(nn.Module):
    """Standard MLP that directly predicts (dq/dt, dp/dt)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 200,
                 output_dim: int = 2, n_hidden: int = 2):
        super().__init__()
        self.net = _make_mlp(input_dim, hidden_dim, output_dim, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── HNN ──────────────────────────────────────────────────────────────

class HNN(nn.Module):
    """
    Hamiltonian Neural Network.

    Learns a scalar H_θ(q, p) and extracts dynamics via:
        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 200,
                 n_hidden: int = 2):
        super().__init__()
        self.net = _make_mlp(input_dim, hidden_dim, 1, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar Hamiltonian value(s)."""
        return self.net(x)

    def time_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute (dq/dt, dp/dt) = (∂H/∂p, -∂H/∂q) via autograd."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.net(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=self.training)[0]
        n = x.shape[-1] // 2
        dq_dt = dH[..., n:]       # ∂H/∂p
        dp_dt = -dH[..., :n]      # -∂H/∂q
        return torch.cat([dq_dt, dp_dt], dim=-1)


# ── D-HNN ────────────────────────────────────────────────────────────

class DHNN(nn.Module):
    """
    Dissipative Hamiltonian Neural Network.

    Two scalar networks:
        H_θ  — conservative (Hamiltonian) part
        D_θ  — dissipative part

    Dynamics:   ẋ = J·∇H + ρ·∇D
    where J is the canonical symplectic matrix.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 200,
                 n_hidden: int = 2):
        super().__init__()
        self.H_net = _make_mlp(input_dim, hidden_dim, 1, n_hidden)
        self.D_net = _make_mlp(input_dim, hidden_dim, 1, n_hidden)

    # ── scalar outputs ───────────────────────────────────────────────

    def forward_H(self, x: torch.Tensor) -> torch.Tensor:
        return self.H_net(x)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return self.D_net(x)

    # ── dynamics ─────────────────────────────────────────────────────

    def time_derivative(self, x: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
        """Full dynamics: J·∇H + ρ·∇D."""
        return self.conservative_part(x) + self.dissipative_part(x, rho)

    def conservative_part(self, x: torch.Tensor) -> torch.Tensor:
        """Symplectic part: J·∇H  →  (∂H/∂p, -∂H/∂q)."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.H_net(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=self.training)[0]
        n = x.shape[-1] // 2
        return torch.cat([dH[..., n:], -dH[..., :n]], dim=-1)

    def dissipative_part(self, x: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
        """Gradient part: ρ·∇D."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            D = self.D_net(x)
            dD = torch.autograd.grad(D.sum(), x, create_graph=self.training)[0]
        return rho * dD
