"""
Animated D-HNN trajectory comparison for varying damping ρ.

Creates an MP4 video on a *single* plot: GT and D-HNN paths are drawn
simultaneously for one ρ, then the plot transitions to the next ρ,
cycling through the full list.  A quiver field coloured with the
thesis_viridial colormap is shown in the background.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from dhnn.integrators import rollout
from dhnn.data import generate_trajectory, meshgrid

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# ── Colourmap ────────────────────────────────────────────────────────
thesis_viridial = LinearSegmentedColormap.from_list("thesis_viridial", [
    "#2e003e", "#2f7a7a", "#e8d9bf",
], N=256)

# Frames per segment  (fast & punchy)
_DRAW_FRAMES  = 90           # frames spent drawing one trajectory
_PAUSE_FRAMES = 18           # short hold before transition
_FADE_FRAMES  = 10           # fade-out to next ρ


def _cycle_colors() -> tuple[str, str]:
    """Return the first two colours from the active prop_cycle."""
    cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", [])
    c0 = cycle[0] if len(cycle) > 0 else "#1f77b4"
    c1 = cycle[1] if len(cycle) > 1 else "#a13333"
    return c0, c1


def create_dhnn_animation(
    dhnn_model,
    sampler,
    rho_values: list[float],
    *,
    t_end: float = 15.0,
    n_steps: int = 300,
    fps: int = 30,
    save_as: str = "dhnn_animation.mp4",
    style_path: str | None = None,
    fixed_y0: np.ndarray | None = None,
    quiver_n: int = 22,
    quiver_lim: float = 2.0,
) -> Path:
    """
    Single-axis animation: draw GT + D-HNN, then switch to next ρ.

    Parameters
    ----------
    dhnn_model  : trained DHNN
    sampler     : () → (q0, p0)
    rho_values  : ρ values to cycle through (shown sequentially)
    quiver_n    : grid density for the background vector field
    quiver_lim  : axis limit for the quiver grid
    """
    if style_path:
        plt.style.use(style_path)

    # Pull colours from the active mplstyle prop_cycle
    COLOR_GT, COLOR_DHNN = _cycle_colors()

    # ── Use a shared IC ──────────────────────────────────────────────
    if fixed_y0 is not None:
        y0 = fixed_y0.copy()
    else:
        q0, p0 = sampler()
        y0 = np.array([q0, p0])

    # ── Pre-compute trajectories for every ρ ─────────────────────────
    dhnn_model.eval()
    segments: list[dict] = []

    for rho in rho_values:
        ode_fn = lambda y, _r=rho: np.array([y[1], -y[0] - _r * y[1]])
        t_eval, gt = generate_trajectory(ode_fn, y0, t_end, n_steps)
        dt = t_eval[1] - t_eval[0]
        pred = rollout(
            lambda x, _r=rho: dhnn_model.time_derivative(x, rho=_r),
            y0, dt, n_steps - 1,
        )
        segments.append(dict(rho=rho, t=t_eval, gt=gt, pred=pred))

    dhnn_model.train()

    # ── Compute global axis limits from all trajectories ─────────────
    all_q = np.concatenate([s["gt"][:, 0] for s in segments] +
                           [s["pred"][:, 0] for s in segments])
    all_p = np.concatenate([s["gt"][:, 1] for s in segments] +
                           [s["pred"][:, 1] for s in segments])
    margin = 0.25
    qlim = (min(all_q.min(), -quiver_lim) - margin,
            max(all_q.max(),  quiver_lim) + margin)
    plim = (min(all_p.min(), -quiver_lim) - margin,
            max(all_p.max(),  quiver_lim) + margin)

    # ── Background quiver field (conservative spring) ────────────────
    Q_bg, P_bg, coords_bg = meshgrid(quiver_n, quiver_lim)
    vf_bg = np.array([[c[1], -c[0]] for c in coords_bg])   # spring only
    speed_bg = np.sqrt(vf_bg[:, 0]**2 + vf_bg[:, 1]**2)
    speed_bg_norm = speed_bg / (speed_bg.max() + 1e-12)     # 0-1 for cmap

    # ── Create figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(*qlim)
    ax.set_ylim(*plim)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$")

    # Quiver background — visible arrows coloured by speed
    U = vf_bg[:, 0].reshape(Q_bg.shape)
    V = vf_bg[:, 1].reshape(Q_bg.shape)
    mag = np.sqrt(U**2 + V**2)
    mag_safe = np.where(mag == 0, 1.0, mag)
    U_norm = U / mag_safe
    V_norm = V / mag_safe
    ax.quiver(
        Q_bg, P_bg, U_norm, V_norm,
        speed_bg_norm.reshape(Q_bg.shape),
        cmap=thesis_viridial, alpha=0.40,
        scale=28, width=0.004, headwidth=3.5, headlength=4,
        zorder=0, clim=(0, 1),
    )

    # IC marker
    ax.scatter([y0[0]], [y0[1]], c="red", s=200, zorder=12,
               edgecolors="k", linewidths=2.0, label="IC")

    # Animated artists — use prop_cycle colours
    line_gt,   = ax.plot([], [], color=COLOR_GT,   lw=4.0, alpha=0.55,
                         label="Ground Truth", zorder=2)
    line_pred, = ax.plot([], [], color=COLOR_DHNN,  lw=3.0,
                         label="D-HNN", zorder=3)
    dot_gt,    = ax.plot([], [], 'o', color=COLOR_GT,   ms=11, zorder=6,
                         markeredgecolor="k", markeredgewidth=1.2)
    dot_pred,  = ax.plot([], [], 'o', color=COLOR_DHNN,  ms=11, zorder=6,
                         markeredgecolor="k", markeredgewidth=1.2)

    # Text overlays
    bg_face = mpl.rcParams.get("axes.facecolor", "white")
    rho_txt  = ax.text(
        0.03, 0.97, "", transform=ax.transAxes,
        fontsize=32, va="top", ha="left", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_face,
                  edgecolor="#aaaaaa", alpha=0.90),
    )
    time_txt = ax.text(
        0.03, 0.88, "", transform=ax.transAxes,
        fontsize=22, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_face,
                  edgecolor="#aaaaaa", alpha=0.80),
    )
    seg_txt  = ax.text(
        0.97, 0.03, "", transform=ax.transAxes,
        fontsize=18, va="bottom", ha="right", color="#777777",
    )

    ax.legend(fontsize=20, loc="upper right", framealpha=0.90,
              edgecolor="#aaaaaa", fancybox=True)
    fig.suptitle("D-HNN — Damped Harmonic Oscillator", fontsize=36,
                 fontweight="semibold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Frame mapping ────────────────────────────────────────────────
    frames_per_segment = _DRAW_FRAMES + _PAUSE_FRAMES + _FADE_FRAMES
    total_frames = frames_per_segment * len(segments)

    # Stride: map DRAW frames → trajectory indices
    step = max(1, n_steps // _DRAW_FRAMES)

    def update(frame):
        seg_idx = frame // frames_per_segment
        local   = frame %  frames_per_segment

        if seg_idx >= len(segments):
            seg_idx = len(segments) - 1
            local = frames_per_segment - 1

        s = segments[seg_idx]
        gt, pred, t_arr = s["gt"], s["pred"], s["t"]
        rho = s["rho"]

        rho_txt.set_text(f"$\\rho = {rho:.2f}$")
        seg_txt.set_text(f"{seg_idx + 1} / {len(segments)}")

        if local < _DRAW_FRAMES:
            # Drawing phase
            k = min((local + 1) * step, len(gt))
            line_gt.set_data(gt[:k, 0], gt[:k, 1])
            line_gt.set_alpha(0.55)
            kp = min((local + 1) * step, len(pred))
            line_pred.set_data(pred[:kp, 0], pred[:kp, 1])
            line_pred.set_alpha(1.0)

            ig = min(k - 1, len(gt) - 1)
            ip = min(kp - 1, len(pred) - 1)
            dot_gt.set_data([gt[ig, 0]], [gt[ig, 1]])
            dot_pred.set_data([pred[ip, 0]], [pred[ip, 1]])
            dot_gt.set_alpha(1.0)
            dot_pred.set_alpha(1.0)
            time_txt.set_text(f"$t = {t_arr[ig]:.1f}$")

        elif local < _DRAW_FRAMES + _PAUSE_FRAMES:
            # Full path visible, just hold
            line_gt.set_data(gt[:, 0], gt[:, 1])
            line_pred.set_data(pred[:, 0], pred[:, 1])
            dot_gt.set_data([gt[-1, 0]], [gt[-1, 1]])
            dot_pred.set_data([pred[-1, 0]], [pred[-1, 1]])
            time_txt.set_text(f"$t = {t_arr[-1]:.1f}$")

        else:
            # Fade-out
            fade_local = local - _DRAW_FRAMES - _PAUSE_FRAMES
            alpha = max(0.0, 1.0 - fade_local / _FADE_FRAMES)
            line_gt.set_alpha(alpha * 0.55)
            line_pred.set_alpha(alpha)
            dot_gt.set_alpha(alpha)
            dot_pred.set_alpha(alpha)

        return [line_gt, line_pred, dot_gt, dot_pred,
                rho_txt, time_txt, seg_txt]

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=True,
    )

    path = FIGURE_DIR / save_as
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(path), writer=writer)
    print(f"  → saved animation {path}")
    plt.close(fig)

    return path
