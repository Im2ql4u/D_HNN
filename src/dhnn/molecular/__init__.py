"""
dhnn.molecular — Molecular dynamics extensions
===============================================

Extends D-HNN to atom-scale systems using rMD17 data,
SchNet force-field backbones, and normal-mode projections.
"""

from dhnn.molecular.datasets import get_rmd17, RMD17_MOLECULES
from dhnn.molecular.schnet_hamiltonian import SchNetHamiltonian
from dhnn.molecular.copresheaf import CopresheafHamiltonian
from dhnn.molecular.training_mol import train_schnet
from dhnn.molecular.normal_modes import (
    compute_hessian,
    normal_mode_analysis,
    project_to_modes,
    project_from_modes,
)

__all__ = [
    "get_rmd17",
    "RMD17_MOLECULES",
    "SchNetHamiltonian",
    "CopresheafHamiltonian",
    "train_schnet",
    "compute_hessian",
    "normal_mode_analysis",
    "project_to_modes",
    "project_from_modes",
]
