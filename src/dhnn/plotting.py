"""
Publication-quality plotting helpers with thesis_viridial colormap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# ── Custom thesis colormap ───────────────────────────────────────────
thesis_viridial = LinearSegmentedColormap.from_list("thesis_viridial", [
    "#2e003e",   # deep violet
    "#2f7a7a",   # dusty teal
    "#e8d9bf",   # ivory white
], N=256)

# Reversed version (ivory → teal → violet)
thesis_viridial_r = LinearSegmentedColormap.from_list("thesis_viridial_r", [
    "#e8d9bf", "#2f7a7a", "#2e003e",
], N=256)

# ── Discrete line colours ────────────────────────────────────────────
C = {
    "gt":       "#888888",   # neutral grey — ground truth
    "hnn":      "#2e003e",   # deep violet
    "dhnn":     "#2f7a7a",   # dusty teal
    "baseline": "#cc5500",   # warm orange (complement)
}


def _save(fig, name: str | None):
    if name is not None:
        path = FIGURE_DIR / name
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  → saved {path}")


# ── Energy surface ───────────────────────────────────────────────────

def plot_energy_surfaces(H_funcs, titles, q_range=(-np.pi, np.pi),
                         p_range=(-3, 3), n=200, save_as=None):
    """Contour plot of one or more Hamiltonians."""
    q = np.linspace(*q_range, n)
    p = np.linspace(*p_range, n)
    Q, P = np.meshgrid(q, p)

    fig, axes = plt.subplots(1, len(H_funcs), figsize=(16 * len(H_funcs), 14))
    if len(H_funcs) == 1:
        axes = [axes]

    for ax, H, title in zip(axes, H_funcs, titles):
        Hv = H(Q, P)
        cs = ax.contourf(Q, P, Hv, levels=40, cmap=thesis_viridial_r, alpha=0.92)
        ax.contour(Q, P, Hv, levels=15, colors="k", linewidths=0.4, alpha=0.25)
        ax.set_xlabel("$q$"); ax.set_ylabel("$p$"); ax.set_title(title)
        cb = plt.colorbar(cs, ax=ax, label="$\\mathcal{H}$", shrink=0.85)
        cb.ax.tick_params(labelsize=22)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Training curves ──────────────────────────────────────────────────

def plot_training_curves(curves: dict[str, tuple[list, list]],
                         save_as=None):
    _color_map = {"Baseline": C["baseline"], "HNN": C["hnn"], "D-HNN": C["dhnn"]}
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    for label, (tr, te) in curves.items():
        c = _color_map.get(label, None)
        axes[0].semilogy(tr, label=label, alpha=0.85, color=c)
        axes[1].semilogy(te, label=label, alpha=0.85, color=c)
    for ax, title in zip(axes, ["Train Loss", "Test Loss"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=24, framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Comparison panel (trajectory + phase + energy + MSE) ─────────────

def plot_comparison(t, ground_truth, predictions: dict[str, np.ndarray],
                    H_func=None, system_name="System", save_as=None):
    _cm = {"HNN": C["hnn"], "Baseline": C["baseline"], "D-HNN": C["dhnn"]}
    fig, axes = plt.subplots(2, 2, figsize=(28, 22))

    # 1 — Time series
    ax = axes[0, 0]
    ax.plot(t, ground_truth[:, 0], color=C["gt"], alpha=0.4, lw=5, label="$q$ (true)")
    ax.plot(t, ground_truth[:, 1], color=C["gt"], ls="--", alpha=0.4, lw=5, label="$p$ (true)")
    for label, pred in predictions.items():
        c = _cm.get(label, None)
        ax.plot(t[:len(pred)], pred[:, 0], label=f"$q$ — {label}", alpha=0.9, color=c)
        ax.plot(t[:len(pred)], pred[:, 1], ls="--", label=f"$p$ — {label}", alpha=0.7, color=c)
    ax.set_xlabel("$t$"); ax.set_ylabel("State")
    ax.set_title("Time Series"); ax.legend(fontsize=18, loc="best", framealpha=0.9)

    # 2 — Phase portrait
    ax = axes[0, 1]
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], color=C["gt"],
            alpha=0.35, lw=6, label="True", zorder=1)
    for label, pred in predictions.items():
        c = _cm.get(label, None)
        ax.plot(pred[:, 0], pred[:, 1], label=label, lw=2.5, alpha=0.9, zorder=2, color=c)
    ax.scatter([ground_truth[0, 0]], [ground_truth[0, 1]],
               c="red", s=150, zorder=10, edgecolors="k", linewidths=1.5, label="IC")
    ax.set_xlabel("$q$"); ax.set_ylabel("$p$")
    ax.set_title("Phase Portrait"); ax.legend(fontsize=18, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    # 3 — Energy
    ax = axes[1, 0]
    if H_func is not None:
        E_true = H_func(ground_truth[:, 0], ground_truth[:, 1])
        ax.plot(t, E_true, color=C["gt"], alpha=0.4, lw=5, label="True")
        for label, pred in predictions.items():
            c = _cm.get(label, None)
            E = H_func(pred[:, 0], pred[:, 1])
            ax.plot(t[:len(E)], E, label=label, alpha=0.9, color=c)
        ax.axhline(E_true[0], ls=":", color="grey", alpha=0.4, lw=1.5)
    ax.set_xlabel("$t$"); ax.set_ylabel("$\\mathcal{H}$")
    ax.set_title("Energy"); ax.legend(fontsize=18, framealpha=0.9)

    # 4 — MSE
    ax = axes[1, 1]
    for label, pred in predictions.items():
        c = _cm.get(label, None)
        L = min(len(pred), len(ground_truth))
        mse = np.mean((pred[:L] - ground_truth[:L]) ** 2, axis=1)
        ax.semilogy(t[:L], mse + 1e-18, label=label, alpha=0.9, color=c)
    ax.set_xlabel("$t$"); ax.set_ylabel("MSE (log)")
    ax.set_title("Coordinate MSE"); ax.legend(fontsize=18, framealpha=0.9)

    fig.suptitle(system_name, fontsize=42, y=1.01)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Vector field ─────────────────────────────────────────────────────

def plot_vector_field(Q, P, U, V, title="Vector Field", overlay_points=None,
                      save_as=None):
    fig, ax = plt.subplots(figsize=(16, 14))
    speed = np.sqrt(U**2 + V**2)
    ax.quiver(Q, P, U * 0.1, V * 0.1, speed, cmap=thesis_viridial, alpha=0.8)
    if overlay_points is not None:
        ax.scatter(overlay_points[:, 0], overlay_points[:, 1],
                   c=C["dhnn"], zorder=5, s=80, edgecolors="k", linewidths=0.8, label="ICs")
        ax.legend(fontsize=20)
    ax.set_xlabel("$q$"); ax.set_ylabel("$p$"); ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ── Integrator comparison ────────────────────────────────────────────

def plot_integrator_comparison(t, E_rk4, E_lf, traj_rk4_q, traj_rk4_p,
                               traj_lf_q, traj_lf_p, save_as=None):
    fig, axes = plt.subplots(1, 3, figsize=(28, 12))

    axes[0].plot(traj_rk4_q, traj_rk4_p, label="RK4", lw=2.5, alpha=0.85, color=C["hnn"])
    axes[0].plot(traj_lf_q, traj_lf_p, ls="--", label="Leapfrog", lw=2.5, alpha=0.85, color=C["dhnn"])
    axes[0].scatter([traj_rk4_q[0]], [traj_rk4_p[0]], c="red", s=150,
                    zorder=10, edgecolors="k", linewidths=1.5, label="IC")
    axes[0].set_xlabel("$q$"); axes[0].set_ylabel("$p$")
    axes[0].set_title("Phase Portrait"); axes[0].legend(fontsize=20)
    axes[0].set_aspect("equal", adjustable="datalim")

    axes[1].plot(t, E_rk4 - E_rk4[0], label="RK4", alpha=0.85, color=C["hnn"])
    axes[1].plot(t, E_lf - E_lf[0], ls="--", label="Leapfrog", alpha=0.85, color=C["dhnn"])
    axes[1].axhline(0, ls=":", color="grey", alpha=0.4, lw=1.5)
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("$\\Delta \\mathcal{H}$")
    axes[1].set_title("Energy Drift"); axes[1].legend(fontsize=20)

    axes[2].semilogy(t, np.abs(E_rk4 - E_rk4[0]) + 1e-18, label="RK4", alpha=0.85, color=C["hnn"])
    axes[2].semilogy(t, np.abs(E_lf - E_lf[0]) + 1e-18, ls="--", label="Leapfrog", alpha=0.85, color=C["dhnn"])
    axes[2].set_xlabel("$t$"); axes[2].set_ylabel("$|\\Delta \\mathcal{H}|$")
    axes[2].set_title("Energy Error (log)"); axes[2].legend(fontsize=20)

    fig.tight_layout()
    _save(fig, save_as)
    return fig
