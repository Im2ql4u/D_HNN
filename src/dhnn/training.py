"""
Training loops and loss functions for HNN / D-HNN / Baseline models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


# =====================================================================
#  Loss functions
# =====================================================================

def loss_baseline(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE between MLP output and true derivatives."""
    return ((model(x) - y) ** 2).mean()


def loss_hnn(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """HNN loss: autograd-derived dynamics vs true derivatives."""
    pred = model.time_derivative(x)
    return ((pred - y) ** 2).mean()


def loss_dhnn(model, x: torch.Tensor, y: torch.Tensor,
              rho: float = 1.0) -> torch.Tensor:
    """D-HNN loss: (J·∇H + ρ·∇D) vs true derivatives."""
    pred = model.time_derivative(x, rho=rho)
    return ((pred - y) ** 2).mean()


# =====================================================================
#  Data preparation
# =====================================================================

def _prepare_tensors(x_data: np.ndarray, y_data: np.ndarray):
    """Flatten (n_sims, n_steps, 2) → (N, 2) float32 tensors."""
    X = torch.tensor(x_data.reshape(-1, 2), dtype=torch.float32)
    Y = torch.tensor(y_data.reshape(-1, 2), dtype=torch.float32)
    return X, Y


# =====================================================================
#  Generic training loop
# =====================================================================

def train_model(
    model: nn.Module,
    loss_fn,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose_every: int = 500,
) -> tuple[list[float], list[float]]:
    """
    Train *model* using *loss_fn* with Adam + cosine annealing.

    Parameters
    ----------
    model       : nn.Module
    loss_fn     : callable(model, x, y) → scalar loss
    x_train     : (n_sims, n_steps, 2) ndarray
    y_train     : (n_sims, n_steps, 2) ndarray
    x_test, y_test : same shapes

    Returns
    -------
    train_losses, test_losses : lists of per-epoch scalars
    """
    X_tr, Y_tr = _prepare_tensors(x_train, y_train)
    X_te, Y_te = _prepare_tensors(x_test, y_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses: list[float] = []
    test_losses:  list[float] = []
    n = len(X_tr)

    for epoch in tqdm(range(1, epochs + 1), desc="Training", leave=False):
        idx = torch.randperm(n)[:batch_size]
        loss = loss_fn(model, X_tr[idx], Y_tr[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            tl = loss_fn(model, X_tr, Y_tr).item()
            vl = loss_fn(model, X_te, Y_te).item()
        train_losses.append(tl)
        test_losses.append(vl)

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} │ Train {tl:.6f} │ Test {vl:.6f}")

    return train_losses, test_losses


def train_dhnn(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose_every: int = 500,
    rho: float = 1.0,
) -> tuple[list[float], list[float]]:
    """
    Train a DHNN model (both H and D networks jointly).

    Uses separate optimisers for H_net and D_net, matching
    the original Julia implementation.
    """
    X_tr, Y_tr = _prepare_tensors(x_train, y_train)
    X_te, Y_te = _prepare_tensors(x_test, y_test)

    opt_H = torch.optim.Adam(model.H_net.parameters(), lr=lr)
    opt_D = torch.optim.Adam(model.D_net.parameters(), lr=lr)
    sched_H = torch.optim.lr_scheduler.CosineAnnealingLR(opt_H, T_max=epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs)

    train_losses: list[float] = []
    test_losses:  list[float] = []
    n = len(X_tr)

    for epoch in tqdm(range(1, epochs + 1), desc="Training D-HNN", leave=False):
        idx = torch.randperm(n)[:batch_size]
        x_b, y_b = X_tr[idx], Y_tr[idx]

        # Update H
        loss_H = loss_dhnn(model, x_b, y_b, rho=rho)
        opt_H.zero_grad()
        loss_H.backward()
        opt_H.step()

        # Update D
        loss_D = loss_dhnn(model, x_b, y_b, rho=rho)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        sched_H.step()
        sched_D.step()

        with torch.no_grad():
            tl = loss_dhnn(model, X_tr, Y_tr, rho=rho).item()
            vl = loss_dhnn(model, X_te, Y_te, rho=rho).item()
        train_losses.append(tl)
        test_losses.append(vl)

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} │ Train {tl:.6f} │ Test {vl:.6f}")

    return train_losses, test_losses
