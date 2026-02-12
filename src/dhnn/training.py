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
              rho: float = 0.5) -> torch.Tensor:
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


def _prepare_tensors_with_rho(
    x_data: np.ndarray,
    y_data: np.ndarray,
    rho_per_traj: np.ndarray,
):
    """
    Flatten multi-ρ data and replicate ρ per time-step.

    x_data       : (n_traj, n_steps, 2)
    y_data       : (n_traj, n_steps, 2)
    rho_per_traj : (n_traj,)

    Returns X (N,2), Y (N,2), R (N,)
    """
    n_traj, n_steps, _ = x_data.shape
    X = torch.tensor(x_data.reshape(-1, 2), dtype=torch.float32)
    Y = torch.tensor(y_data.reshape(-1, 2), dtype=torch.float32)
    R = torch.tensor(
        np.repeat(rho_per_traj, n_steps), dtype=torch.float32
    )
    return X, Y, R


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

        model.eval()
        tl = loss_fn(model, X_tr, Y_tr).item()
        vl = loss_fn(model, X_te, Y_te).item()
        model.train()
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
    epochs: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 512,
    verbose_every: int = 1000,
    rho: float | None = 0.5,
    rho_train: np.ndarray | None = None,
    rho_test: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
    """
    Train a DHNN model (H and D networks *jointly*).

    Supports two modes:
    1. Single-ρ:  set ``rho`` (scalar).  All data assumed at that ρ.
    2. Multi-ρ:   supply ``rho_train`` and ``rho_test`` arrays, one
       ρ per trajectory.  Each mini-batch sample uses its own ρ.

    Parameters
    ----------
    rho       : scalar ρ  (used if rho_train is None)
    rho_train : (n_traj_train,) per-trajectory ρ
    rho_test  : (n_traj_test,)  per-trajectory ρ
    """
    multi_rho = rho_train is not None

    if multi_rho:
        X_tr, Y_tr, R_tr = _prepare_tensors_with_rho(x_train, y_train, rho_train)
        X_te, Y_te, R_te = _prepare_tensors_with_rho(x_test, y_test, rho_test)
    else:
        X_tr, Y_tr = _prepare_tensors(x_train, y_train)
        X_te, Y_te = _prepare_tensors(x_test, y_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses: list[float] = []
    test_losses:  list[float] = []
    n = len(X_tr)

    for epoch in tqdm(range(1, epochs + 1), desc="Training D-HNN", leave=False):
        model.train()
        idx = torch.randperm(n)[:batch_size]

        if multi_rho:
            # Compute per-sample loss with individual ρ
            x_b, y_b, r_b = X_tr[idx], Y_tr[idx], R_tr[idx]
            unique_rhos = r_b.unique()
            loss = torch.tensor(0.0)
            for ur in unique_rhos:
                mask = r_b == ur
                pred = model.time_derivative(x_b[mask], rho=ur.item())
                loss = loss + ((pred - y_b[mask]) ** 2).sum()
            loss = loss / len(x_b)
        else:
            loss = loss_dhnn(model, X_tr[idx], Y_tr[idx], rho=rho)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── eval ──
        model.eval()
        if multi_rho:
            # Use average across all ρ's
            tl_acc, vl_acc = 0.0, 0.0
            for ur in R_tr.unique():
                mask_tr = R_tr == ur
                mask_te = R_te == ur
                pred_tr = model.time_derivative(X_tr[mask_tr], rho=ur.item())
                pred_te = model.time_derivative(X_te[mask_te], rho=ur.item())
                tl_acc += ((pred_tr - Y_tr[mask_tr]) ** 2).sum().item()
                vl_acc += ((pred_te - Y_te[mask_te]) ** 2).sum().item()
            tl = tl_acc / len(X_tr)
            vl = vl_acc / len(X_te)
        else:
            tl = loss_dhnn(model, X_tr, Y_tr, rho=rho).item()
            vl = loss_dhnn(model, X_te, Y_te, rho=rho).item()
        model.train()

        train_losses.append(tl)
        test_losses.append(vl)

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} │ Train {tl:.6f} │ Test {vl:.6f}")

    return train_losses, test_losses
