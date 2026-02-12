"""
Data generation for Hamiltonian / dissipative systems.

Each system is defined by:
    1. An ODE right-hand side  f(y) → dy/dt
    2. A Hamiltonian function  H(q, p) → scalar energy
    3. A sampler for random initial conditions
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# =====================================================================
#  ODE right-hand sides  (numpy, used by solve_ivp)
# =====================================================================

def spring_ode(y: np.ndarray) -> np.ndarray:
    """Simple harmonic oscillator: dq=p, dp=-q  (k=m=1)."""
    return np.array([y[1], -y[0]])


def pendulum_ode(y: np.ndarray, g: float = 3.0) -> np.ndarray:
    """Ideal pendulum: dq=p, dp=-g·sin(q)."""
    return np.array([y[1], -g * np.sin(y[0])])


def damped_spring_ode(y: np.ndarray, rho: float = 0.5) -> np.ndarray:
    """Damped harmonic oscillator: dq=p, dp=-q-ρp."""
    return np.array([y[1], -y[0] - rho * y[1]])


def damped_pendulum_ode(y: np.ndarray, g: float = 3.0,
                        rho: float = 0.5) -> np.ndarray:
    """Damped pendulum: dq=p, dp=-g·sin(q)-ρp."""
    return np.array([y[1], -g * np.sin(y[0]) - rho * y[1]])


# =====================================================================
#  Hamiltonians (for energy evaluation)
# =====================================================================

def H_spring(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * q**2 + 0.5 * p**2


def H_pendulum(q: np.ndarray, p: np.ndarray, g: float = 3.0) -> np.ndarray:
    return g * (1 - np.cos(q)) + 0.5 * p**2


# =====================================================================
#  Initial-condition samplers
# =====================================================================

def sample_spring() -> tuple[float, float]:
    """Random IC on a ring in phase space (r ∈ [0.4, 1.5])."""
    r = np.random.uniform(0.4, 1.5)
    phi = np.random.uniform(0, 2 * np.pi)
    return r * np.cos(phi), r * np.sin(phi)


def sample_pendulum() -> tuple[float, float]:
    """Random IC on a Hamiltonian level set (H ∈ [1.3, 2.3])."""
    while True:
        H_target = np.random.uniform(1.3, 2.3)
        q = np.random.uniform(-np.pi, np.pi)
        p_sq = 2 * (H_target - 3 * (1 - np.cos(q)))
        if p_sq >= 0:
            return q, np.sqrt(p_sq) * np.random.choice([-1, 1])


def sample_damped_spring() -> tuple[float, float]:
    """Same as spring sampler (damping is in the ODE, not the IC)."""
    return sample_spring()


# =====================================================================
#  Dataset generation
# =====================================================================

def generate_data(
    ode_fn,
    n_sims: int,
    n_steps: int,
    t_end: float,
    sampler,
    sigma: float = 0.01,
    rtol: float = 1e-10,
    atol: float = 1e-10,
):
    """
    Integrate an ODE from random ICs to produce labelled data.

    Parameters
    ----------
    ode_fn  : callable  y → dy/dt  (numpy arrays of shape (2,))
    n_sims  : int       number of trajectories
    n_steps : int       time-steps per trajectory
    t_end   : float     final time
    sampler : callable  () → (q0, p0)
    sigma   : float     Gaussian noise level on labels

    Returns
    -------
    t_eval  : (n_steps,) ndarray
    x_all   : (n_sims, n_steps, 2) ndarray   — states  [q, p]
    y_all   : (n_sims, n_steps, 2) ndarray   — derivatives [dq, dp]
    """
    t_eval = np.linspace(0, t_end, n_steps)
    x_list, y_list = [], []

    for _ in range(n_sims):
        q0, p0 = sampler()
        sol = solve_ivp(
            lambda _t, y: ode_fn(y),
            [0, t_end],
            [q0, p0],
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        x = sol.y.T                                       # (n_steps, 2)
        dx = np.array([ode_fn(x[i]) for i in range(len(x))])
        dx += sigma * np.random.randn(*dx.shape)
        x_list.append(x)
        y_list.append(dx)

    return t_eval, np.array(x_list), np.array(y_list)


def generate_data_multi_rho(
    ode_factory,
    n_sims_per_rho: int,
    n_steps: int,
    t_end: float,
    sampler,
    rho_values: list[float] | np.ndarray,
    sigma: float = 0.01,
    rtol: float = 1e-10,
    atol: float = 1e-10,
):
    """
    Generate training data across multiple damping coefficients.

    Each trajectory is labelled with its ρ so the loss can use
    the correct dissipation strength per sample.

    Parameters
    ----------
    ode_factory : callable(rho) → callable(y) → dy/dt
    rho_values  : array of ρ values to sample from

    Returns
    -------
    t_eval : (n_steps,)
    x_all  : (N, n_steps, 2)  states
    y_all  : (N, n_steps, 2)  derivatives
    rho_all: (N,)              per-trajectory ρ
    """
    t_eval = np.linspace(0, t_end, n_steps)
    x_list, y_list, rho_list = [], [], []

    for rho in rho_values:
        ode_fn = ode_factory(rho)
        for _ in range(n_sims_per_rho):
            q0, p0 = sampler()
            sol = solve_ivp(
                lambda _t, y: ode_fn(y),
                [0, t_end], [q0, p0],
                t_eval=t_eval, rtol=rtol, atol=atol,
            )
            x = sol.y.T
            dx = np.array([ode_fn(x[i]) for i in range(len(x))])
            dx += sigma * np.random.randn(*dx.shape)
            x_list.append(x)
            y_list.append(dx)
            rho_list.append(rho)

    return t_eval, np.array(x_list), np.array(y_list), np.array(rho_list)


def generate_trajectory(ode_fn, y0: np.ndarray, t_end: float,
                        n_steps: int, **ivp_kw) -> tuple[np.ndarray, np.ndarray]:
    """
    Single deterministic trajectory (no noise, no random IC).

    Returns
    -------
    t : (n_steps,)
    y : (n_steps, 2)
    """
    t_eval = np.linspace(0, t_end, n_steps)
    sol = solve_ivp(
        lambda _t, y: ode_fn(y), [0, t_end], y0,
        t_eval=t_eval, rtol=1e-12, atol=1e-12, **ivp_kw,
    )
    return sol.t, sol.y.T


# =====================================================================
#  Mesh-grid helper (for vector-field visualisation)
# =====================================================================

def meshgrid(n: int = 20, lim: float = 2.0):
    """
    Return a regular (n×n) grid of (q, p) points.

    Returns
    -------
    Q, P : (n, n) ndarrays
    coords : (n*n, 2) ndarray  — flattened
    """
    q = np.linspace(-lim, lim, n)
    p = np.linspace(-lim, lim, n)
    Q, P = np.meshgrid(q, p)
    coords = np.stack([Q.ravel(), P.ravel()], axis=-1)
    return Q, P, coords
