"""
Animated D-HNN trajectory comparison for varying damping ρ.

Creates an MP4 video showing GT and D-HNN paths being drawn
simultaneously across a grid of damping coefficients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from dhnn.integrators import rollout
from dhnn.data import generate_trajectory

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────
thesis_viridial = LinearSegmentedColormap.from_list("thesis_viridial", [
    "#2e003e", "#2f7a7a", "#e8d9bf",
], N=256)

GT_COLOR  = "#888888"
DHNN_COLOR = "#2e003e"
TRAIL_ALPHA_MIN = 0.15


def create_dhnn_animation(
    dhnn_model,
    sampler,
    rho_values: list[float],
    *,
    training_rho: float = 0.5,
    t_end: float = 15.0,
    n_steps: int = 300,
    fps: int = 30,
    save_as: str = "dhnn_animation.mp4",
    style_path: str | None = None,
    fixed_y0: np.ndarray | None = None,
) -> Path:
    """
    Animate D-HNN vs ground truth for multiple damping coefficients.

    Parameters
    ----------
    dhnn_model : DHNN model (already trained)
    sampler    : callable () → (q0, p0)
    rho_values : list of ρ values to animate
    training_rho : ρ used during training (for correct scaling)
    fixed_y0   : if given, use the same IC for all panels
    """
    if style_path:
        plt.style.use(style_path)

    n_rho = len(rho_values)
    ncols = min(3, n_rho)
    nrows = (n_rho + ncols - 1) // ncols

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(9 * ncols, 8 * nrows))
    if nrows == 1 and ncols == 1:
        axes_grid = np.array([[axes_grid]])
    elif nrows == 1:
        axes_grid = axes_grid[np.newaxis, :]
    elif ncols == 1:
        axes_grid = axes_grid[:, np.newaxis]

    # Use same IC for fair comparison
    if fixed_y0 is not None:
        y0_shared = fixed_y0
    else:
        q0, p0 = sampler()
        y0_shared = np.array([q0, p0])

    # ── Generate all trajectories ────────────────────────────────────
    data = []
    dhnn_model.eval()

    for rho in rho_values:
        y0 = y0_shared.copy()
        ode_fn = lambda y, _rho=rho: np.array([y[1], -y[0] - _rho * y[1]])
        t_eval, gt = generate_trajectory(ode_fn, y0, t_end, n_steps)
        dt = t_eval[1] - t_eval[0]

        pred = rollout(
            lambda x, _rho=rho: dhnn_model.time_derivative(x, rho=_rho),
            y0, dt, n_steps - 1,
        )
        data.append((t_eval, gt, pred, y0))

    dhnn_model.train()

    # ── Set up plot elements ─────────────────────────────────────────
    artists = []  # (line_gt, line_pred, dot_gt, dot_pred, time_txt)

    for idx, (rho, (t_eval, gt, pred, y0)) in enumerate(zip(rho_values, data)):
        r, c = idx // ncols, idx % ncols
        ax = axes_grid[r, c]

        all_q = np.concatenate([gt[:, 0], pred[:, 0]])
        all_p = np.concatenate([gt[:, 1], pred[:, 1]])
        margin = max(0.3, 0.15 * (all_q.max() - all_q.min()))
        ax.set_xlim(all_q.min() - margin, all_q.max() + margin)
        ax.set_ylim(all_p.min() - margin, all_p.max() + margin)

        ax.set_title(f"$\\rho = {rho:.2f}$", fontsize=22, pad=12)
        ax.set_xlabel("$q$", fontsize=16)
        ax.set_ylabel("$p$", fontsize=16)
        ax.tick_params(labelsize=13)

        # IC marker
        ax.scatter([y0[0]], [y0[1]], c="red", s=120, zorder=10,
                   edgecolors="k", linewidths=1.5)

        # Faint full GT path as reference
        ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, lw=1.0, alpha=0.15, zorder=0)

        line_gt, = ax.plot([], [], color=GT_COLOR, lw=3.0, alpha=0.7,
                           label="Ground Truth", zorder=2)
        line_pred, = ax.plot([], [], color=DHNN_COLOR, lw=2.5,
                             label="D-HNN", zorder=3)
        dot_gt, = ax.plot([], [], 'o', color=GT_COLOR, ms=9, zorder=5,
                          markeredgecolor="k", markeredgewidth=0.8)
        dot_pred, = ax.plot([], [], 'o', color=DHNN_COLOR, ms=9, zorder=5,
                            markeredgecolor="k", markeredgewidth=0.8)
        time_txt = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                           fontsize=14, va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="white", alpha=0.8))

        ax.legend(fontsize=13, loc="upper right", framealpha=0.85)
        artists.append((line_gt, line_pred, dot_gt, dot_pred, time_txt))

    # Hide unused panels
    for idx in range(n_rho, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes_grid[r, c].set_visible(False)

    fig.tight_layout(pad=2.0)

    # ── Animation ────────────────────────────────────────────────────
    n_frames = n_steps

    def update(frame):
        out = []
        for idx, (t_eval, gt, pred, y0) in enumerate(data):
            lg, lp, dg, dp, tt = artists[idx]

            f = min(frame + 1, len(gt))
            lg.set_data(gt[:f, 0], gt[:f, 1])

            fp = min(frame + 1, len(pred))
            lp.set_data(pred[:fp, 0], pred[:fp, 1])

            if frame < len(gt):
                dg.set_data([gt[min(frame, len(gt)-1), 0]],
                            [gt[min(frame, len(gt)-1), 1]])
            if frame < len(pred):
                dp.set_data([pred[min(frame, len(pred)-1), 0]],
                            [pred[min(frame, len(pred)-1), 1]])

            tt.set_text(f"$t = {t_eval[min(frame, len(t_eval)-1)]:.1f}$")

            out.extend([lg, lp, dg, dp, tt])
        return out

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=True,
    )

    path = FIGURE_DIR / save_as
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(path), writer=writer)
    print(f"  → saved animation {path}")
    plt.close(fig)

    return path
