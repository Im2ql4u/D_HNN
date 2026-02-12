"""
Training loop for the SchNet Hamiltonian potential.

Standard energy + force loss:
    L = α · MSE(E_pred, E_true)  +  MSE(F_pred, F_true)

Force-dominated (α ≪ 1) following NequIP / MACE convention.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from dhnn.molecular.schnet_hamiltonian import SchNetHamiltonian


CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train_schnet(
    model: SchNetHamiltonian,
    train_loader,
    val_loader,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    energy_weight: float = 0.01,
    device: str | torch.device = "cpu",
    patience: int = 30,
    checkpoint_name: str = "schnet_best.pt",
    verbose: bool = True,
) -> dict:
    """
    Train a SchNetHamiltonian on energy + force targets.

    Parameters
    ----------
    model : SchNetHamiltonian
    train_loader, val_loader : PyG DataLoader
    epochs : int
    lr : float
    energy_weight : float
        Weight α for the energy MSE term (forces weight = 1).
    device : str
    patience : int
        Early-stopping patience on validation force MAE.
    checkpoint_name : str
        Filename for the best-model checkpoint.
    verbose : bool

    Returns
    -------
    dict with keys:
        train_loss, val_loss          — per-epoch total loss
        train_energy_mae, val_energy_mae
        train_force_mae,  val_force_mae
    """
    model = model.to(device)
    optimiser = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5,
                                   patience=15, min_lr=1e-6)

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_energy_mae", "val_energy_mae",
        "train_force_mae", "val_force_mae",
    ]}

    best_val_fmae = float("inf")
    wait = 0
    ckpt_path = CHECKPOINT_DIR / checkpoint_name

    iterator = tqdm(range(epochs), desc="SchNet", disable=not verbose)
    for epoch in iterator:
        # ── Train ────────────────────────────────────────────────────
        model.train()
        tot_loss = 0.0
        tot_emae = 0.0
        tot_fmae = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            energy_pred, force_pred = model.forces(
                batch.z, batch.pos, batch.batch,
            )
            energy_true = batch.y.squeeze()
            force_true  = batch.force

            loss_e = nn.functional.mse_loss(energy_pred, energy_true)
            loss_f = nn.functional.mse_loss(force_pred, force_true)
            loss   = energy_weight * loss_e + loss_f

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimiser.step()

            with torch.no_grad():
                tot_loss  += loss.item()
                tot_emae  += (energy_pred - energy_true).abs().mean().item()
                tot_fmae  += (force_pred - force_true).abs().mean().item()
                n_batches += 1

        history["train_loss"].append(tot_loss / n_batches)
        history["train_energy_mae"].append(tot_emae / n_batches)
        history["train_force_mae"].append(tot_fmae / n_batches)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        tot_loss = 0.0
        tot_emae = 0.0
        tot_fmae = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                energy_pred, force_pred = model.forces(
                    batch.z, batch.pos, batch.batch,
                )
                energy_true = batch.y.squeeze()
                force_true  = batch.force

                loss_e = nn.functional.mse_loss(energy_pred, energy_true)
                loss_f = nn.functional.mse_loss(force_pred, force_true)
                loss   = energy_weight * loss_e + loss_f

                tot_loss  += loss.item()
                tot_emae  += (energy_pred - energy_true).abs().mean().item()
                tot_fmae  += (force_pred - force_true).abs().mean().item()
                n_batches += 1

        val_loss = tot_loss / n_batches
        val_emae = tot_emae / n_batches
        val_fmae = tot_fmae / n_batches

        history["val_loss"].append(val_loss)
        history["val_energy_mae"].append(val_emae)
        history["val_force_mae"].append(val_fmae)

        scheduler.step(val_fmae)

        # ── Early stopping ───────────────────────────────────────────
        if val_fmae < best_val_fmae:
            best_val_fmae = val_fmae
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"  ⏹  Early stop @ epoch {epoch}  "
                               f"(best val force MAE: {best_val_fmae:.4f})")
                break

        if verbose:
            iterator.set_postfix({
                "lr": optimiser.param_groups[0]["lr"],
                "E_mae": f"{val_emae:.4f}",
                "F_mae": f"{val_fmae:.4f}",
            })

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    if verbose:
        print(f"  ✓ Best val force MAE: {best_val_fmae:.4f} kcal/mol/Å")
        print(f"  ✓ Checkpoint: {ckpt_path}")

    return history
