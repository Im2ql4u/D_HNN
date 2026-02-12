"""
dhnn — Dissipative Hamiltonian Neural Networks
================================================

A PyTorch library implementing HNN, D-HNN, and baseline models
for learning dynamical systems with physical structure.
"""

from dhnn.models import BaselineNN, HNN, DHNN
from dhnn.integrators import rk4_step, leapfrog_step, integrate_rk4, integrate_leapfrog, rollout
from dhnn.data import generate_data, generate_data_multi_rho, spring_ode, pendulum_ode, damped_spring_ode
from dhnn.training import train_model, train_dhnn, loss_baseline, loss_hnn, loss_dhnn
from dhnn.plotting import thesis_viridial, thesis_viridial_r, C
from dhnn.animation import create_dhnn_animation

__all__ = [
    "BaselineNN", "HNN", "DHNN",
    "rk4_step", "leapfrog_step", "integrate_rk4", "integrate_leapfrog", "rollout",
    "generate_data", "spring_ode", "pendulum_ode", "damped_spring_ode",
    "train_model", "train_dhnn", "loss_baseline", "loss_hnn", "loss_dhnn",
    "thesis_viridial", "thesis_viridial_r", "C",
    "create_dhnn_animation",
]
