"""
Publication-quality plotting helpers.

All functions expect ``plt.style.use("Thesis_style.mplstyle")`` to have
been called beforehand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def _save(fig, name: str | None):
    if name is not None:
        path = FIGURE_DIR / name
        fig.savefig(path, bbox_inches="tight")
        print(f"  → saved {path}")


# ── Energy surface ───────────────────────────────────────────────────

def plot_energy_surfaces(H_funcs, titles, q_range=(-np.pi, np.pi),
                         p_range=(-3, 3), n=200, save_as=None):
    """Contour plot of one or more Hamiltonians."""
    q = np.linspace(*q_range, n)
    p = np.linspace(*p_range, n)
    Q, P = np.meshgrid(q, p)

    fig, axes = plt.subplots(1, len(H_funcs), figsize=(14 * len(H_funcs), 12))
    if len(H_funcs) == 1:
        axes = [axes]

    for ax, H, title in zip(axes, H_funcs, titles):
        Hv = H(Q, P)
        cs = ax.contourf(Q, P, Hv, levels=30, cmap="coolwarm")
        ax.set_xlabel("$q$")
        ax.set_ylabel("$p$")
        ax.set_title(title)
        plt.colorbar(cs, ax=ax, label="$\\mathcal{H}$")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Training curves ──────────────────────────────────────────────────

def plot_training_curves(curves: dict[str, tuple[list, list]],
                         save_as=None):
    """
    Parameters
    ----------
    curves : {label: (train_losses, test_losses)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    for label, (tr, te) in curves.items():
        axes[0].semilogy(tr, label=label)
        axes[1].semilogy(te, label=label)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Test Loss");  axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Comparison panel (trajectory + phase + energy + MSE) ─────────────

def plot_comparison(t, ground_truth, predictions: dict[str, np.ndarray],
                    H_func=None, system_name="System", save_as=None):
    """
    Four-panel comparison figure.

    Parameters
    ----------
    t            : (n_steps,) time array
    ground_truth : (n_steps, 2)
    predictions  : {label: (n_steps, 2)}
    H_func       : callable(q, p) returning energy (optional)
    """
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # 1 ── Time series
    ax = axes[0]
    ax.plot(t, ground_truth[:, 0], "k", alpha=0.3, lw=4, label="$q$ (true)")
    ax.plot(t, ground_truth[:, 1], "k--", alpha=0.3, lw=4, label="$p$ (true)")
    for label, pred in predictions.items():
        ax.plot(t[:len(pred)], pred[:, 0], label=f"$q$ {label}")
        ax.plot(t[:len(pred)], pred[:, 1], ls="--", label=f"$p$ {label}")
    ax.set_xlabel("$t$"); ax.set_ylabel("state")
    ax.set_title(f"{system_name} — Time Series"); ax.legend(fontsize=16)

    # 2 ── Phase portrait
    ax = axes[1]
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], "k", alpha=0.3, lw=4, label="True")
    for label, pred in predictions.items():
        ax.plot(pred[:, 0], pred[:, 1], label=label)
    ax.set_xlabel("$q$"); ax.set_ylabel("$p$")
    ax.set_title("Phase Portrait"); ax.legend(fontsize=16)

    # 3 ── Energy
    ax = axes[2]
    if H_func is not None:
        E_true = H_func(ground_truth[:, 0], ground_truth[:, 1])
        ax.plot(t, E_true, "k", alpha=0.3, lw=4, label="True")
        for label, pred in predictions.items():
            E = H_func(pred[:, 0], pred[:, 1])
            ax.plot(t[:len(E)], E, label=label)
    ax.set_xlabel("$t$"); ax.set_ylabel("$\\mathcal{H}$")
    ax.set_title("Energy"); ax.legend(fontsize=16)

    # 4 ── MSE accumulation
    ax = axes[3]
    for label, pred in predictions.items():
        L = min(len(pred), len(ground_truth))
        mse = np.mean((pred[:L] - ground_truth[:L]) ** 2, axis=1)
        ax.plot(t[:L], mse, label=label)
    ax.set_xlabel("$t$"); ax.set_ylabel("MSE")
    ax.set_title("Coordinate MSE"); ax.legend(fontsize=16)

    fig.suptitle(system_name, fontsize=40, y=1.02)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Vector field ─────────────────────────────────────────────────────

def plot_vector_field(Q, P, U, V, title="Vector Field", overlay_points=None,
                      save_as=None):
    """Quiver plot of a 2-D vector field."""
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.quiver(Q, P, U * 0.1, V * 0.1, color="grey", alpha=0.7)
    if overlay_points is not None:
        ax.scatter(overlay_points[:, 0], overlay_points[:, 1],
                   c="C0", zorder=5, s=60, label="ICs")
        ax.legend()
    ax.set_xlabel("$q$"); ax.set_ylabel("$p$")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Integrator comparison ────────────────────────────────────────────

def plot_integrator_comparison(t, E_rk4, E_lf, traj_rk4_q, traj_rk4_p,
                               traj_lf_q, traj_lf_p, save_as=None):
    """Three-panel comparison of RK4 vs leapfrog."""
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))

    axes[0].plot(traj_rk4_q, traj_rk4_p, label="RK4")
    axes[0].plot(traj_lf_q, traj_lf_p, ls="--", label="Leapfrog")
    axes[0].set_xlabel("$q$"); axes[0].set_ylabel("$p$")
    axes[0].set_title("Phase Portrait"); axes[0].legend()

    axes[1].plot(t, E_rk4 - E_rk4[0], label="RK4")
    axes[1].plot(t, E_lf - E_lf[0], ls="--", label="Leapfrog")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("$\\Delta \\mathcal{H}$")
    axes[1].set_title("Energy Drift"); axes[1].legend()

    axes[2].semilogy(t, np.abs(E_rk4 - E_rk4[0]) + 1e-18, label="RK4")
    axes[2].semilogy(t, np.abs(E_lf - E_lf[0]) + 1e-18, ls="--", label="Leapfrog")
    axes[2].set_xlabel("$t$"); axes[2].set_ylabel("$|\\Delta \\mathcal{H}|$")
    axes[2].set_title("Energy Error (log)"); axes[2].legend()

    fig.tight_layout()
    _save(fig, save_as)
    return fig
