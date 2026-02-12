"""
rMD17 data loading and preprocessing.

Provides :func:`get_rmd17` which returns ready-to-use PyG DataLoaders
with proper train/val/test splits following the literature protocol
(≤ 1 000 training samples).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph

# ── Constants ────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

RMD17_MOLECULES: dict[str, dict] = {
    "ethanol":  {"name": "revised ethanol",  "n_atoms": 9,  "n_configs": 100_000},
    "aspirin":  {"name": "revised aspirin",  "n_atoms": 21, "n_configs": 100_000},
    "toluene":  {"name": "revised toluene",  "n_atoms": 15, "n_configs": 100_000},
    "benzene":  {"name": "revised benzene",  "n_atoms": 12, "n_configs": 100_000},
    "uracil":   {"name": "revised uracil",   "n_atoms": 12, "n_configs": 100_000},
    "naphthalene": {"name": "revised naphthalene", "n_atoms": 18, "n_configs": 100_000},
    "salicylic": {"name": "revised salicylic acid", "n_atoms": 16, "n_configs": 100_000},
    "malonaldehyde": {"name": "revised malonaldehyde", "n_atoms": 9, "n_configs": 100_000},
    "paracetamol": {"name": "revised paracetamol", "n_atoms": 20, "n_configs": 100_000},
}


class RMD17Splits(NamedTuple):
    """Container for train / val / test DataLoaders + metadata."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    z: torch.Tensor           # atomic numbers [N_atoms]
    n_atoms: int
    mean_energy: float
    std_energy: float


# ── Public API ───────────────────────────────────────────────────────

def get_rmd17(
    molecule: str = "ethanol",
    *,
    root: str | Path | None = None,
    n_train: int = 1_000,
    n_val: int = 1_000,
    cutoff: float = 5.0,
    batch_size: int = 32,
    seed: int = 42,
) -> RMD17Splits:
    """
    Load an rMD17 molecule and return train / val / test DataLoaders.

    Parameters
    ----------
    molecule : str
        Key in :data:`RMD17_MOLECULES` (e.g. ``"ethanol"``, ``"aspirin"``).
    root : path, optional
        Download / cache directory.  Defaults to ``<repo>/data/``.
    n_train, n_val : int
        Number of samples for training and validation.
    cutoff : float
        Radius (Å) for edge construction.
    batch_size : int
        Mini-batch size for all loaders.
    seed : int
        Random seed for the split.

    Returns
    -------
    RMD17Splits
        Named tuple with ``train_loader``, ``val_loader``, ``test_loader``,
        ``z`` (atomic numbers), ``n_atoms``, ``mean_energy``, ``std_energy``.
    """
    from torch_geometric.datasets import MD17

    if molecule not in RMD17_MOLECULES:
        raise ValueError(
            f"Unknown molecule {molecule!r}. "
            f"Choose from: {list(RMD17_MOLECULES)}"
        )

    root = Path(root) if root else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    info = RMD17_MOLECULES[molecule]
    dataset = MD17(root=str(root), name=info["name"])

    # ── Edge construction ────────────────────────────────────────────
    transform = RadiusGraph(r=cutoff, loop=False, max_num_neighbors=64)
    processed: list[Data] = []
    for d in dataset:
        # Ensure positions track gradients later (for force computation)
        d.pos = d.pos.float()
        d.y = d.energy.float()
        d.force = d.force.float()
        processed.append(transform(d))

    # ── Shuffled split ───────────────────────────────────────────────
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(processed))

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    train_data = [processed[i] for i in train_idx]
    val_data   = [processed[i] for i in val_idx]
    test_data  = [processed[i] for i in test_idx]

    # ── Energy normalisation stats (from training set only) ──────────
    energies = torch.tensor([d.y.item() for d in train_data])
    mean_e = energies.mean().item()
    std_e  = energies.std().item()

    # ── DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    z = processed[0].z

    return RMD17Splits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        z=z,
        n_atoms=z.shape[0],
        mean_energy=mean_e,
        std_energy=std_e,
    )
