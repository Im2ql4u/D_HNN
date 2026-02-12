"""
Numerical integrators for Hamiltonian systems.

- RK4         (standard, non-symplectic)
- Leapfrog    (Störmer–Verlet, symplectic)
- rollout()   (RK4 with a learned dynamics function)
"""

from __future__ import annotations

import numpy as np
import torch


# ── RK4 ─────────────────────────────────────────────────────────────

def rk4_step(f, y: torch.Tensor, dt: float) -> torch.Tensor:
    """One step of the classic 4th-order Runge–Kutta method."""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rk4(f, y0: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
    """Integrate *f* for *n_steps* using RK4.  Returns (n_steps+1, *y.shape)."""
    traj = [y0]
    y = y0.clone()
    for _ in range(n_steps):
        y = rk4_step(f, y, dt)
        traj.append(y.clone())
    return torch.stack(traj)


# ── Leapfrog / Störmer–Verlet ────────────────────────────────────────

def leapfrog_step(dH_dq, dH_dp, q: torch.Tensor, p: torch.Tensor,
                  dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the Störmer–Verlet (leapfrog) method."""
    p_half = p - 0.5 * dt * dH_dq(q, p)
    q_new  = q + dt * dH_dp(q, p_half)
    p_new  = p_half - 0.5 * dt * dH_dq(q_new, p_half)
    return q_new, p_new


def integrate_leapfrog(dH_dq, dH_dp, q0: torch.Tensor, p0: torch.Tensor,
                       dt: float, n_steps: int):
    """Integrate using leapfrog.  Returns (qs, ps) each of shape (n_steps+1,)."""
    qs, ps = [q0], [p0]
    q, p = q0.clone(), p0.clone()
    for _ in range(n_steps):
        q, p = leapfrog_step(dH_dq, dH_dp, q, p, dt)
        qs.append(q.clone())
        ps.append(p.clone())
    return torch.stack(qs), torch.stack(ps)


# ── Learned-dynamics rollout ─────────────────────────────────────────

def rollout(dynamics_fn, y0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """
    Integrate a learned dynamics function using RK4.

    Parameters
    ----------
    dynamics_fn : callable
        Maps a (2,) tensor → (2,) tensor of time derivatives.
    y0 : ndarray of shape (2,)
        Initial condition [q0, p0].
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.

    Returns
    -------
    traj : ndarray of shape (n_steps+1, 2)
    """
    y = torch.tensor(y0, dtype=torch.float32)
    traj = [y.detach().numpy().copy()]
    for _ in range(n_steps):
        with torch.no_grad():
            k1 = dynamics_fn(y)
            k2 = dynamics_fn(y + 0.5 * dt * k1)
            k3 = dynamics_fn(y + 0.5 * dt * k2)
            k4 = dynamics_fn(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(y.detach().numpy().copy())
    return np.array(traj)
