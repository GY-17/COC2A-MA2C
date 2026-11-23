"""Draw the COC2A-MA2C framework figure used in the manuscript.

The output is vector-first: PDF and SVG are emitted alongside a high-resolution
PNG preview so the figure can be imported into AxGlyph or inserted directly in
LaTeX.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent


def rounded(ax, x, y, w, h, text="", fc="#ffffff", ec="#1f2937", lw=1.2,
            radius=0.12, ls="-", fs=9, weight="normal", color="#18212b",
            ha="center", va="center", z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fs,
                fontweight=weight, color=color, zorder=z + 1)
    return patch


def arrow(ax, x1, y1, x2, y2, text=None, color="#1f4e79", lw=1.4,
          style="-|>", ls="-", rad=0.0, fs=8, text_offset=(0, 0), z=3):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text((x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1],
                text, fontsize=fs, color=color, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85),
                zorder=z + 1)
    return patch


def label(ax, x, y, text, fs=9, weight="normal", color="#18212b", ha="center",
          va="center", z=4):
    ax.text(x, y, text, fontsize=fs, fontweight=weight, color=color,
            ha=ha, va=va, zorder=z)


def draw_truck(ax, x, y, scale=1.0, color="#3977a9"):
    ax.add_patch(Rectangle((x, y), 0.46 * scale, 0.24 * scale,
                           facecolor=color, edgecolor="#17324d", lw=0.8, zorder=5))
    ax.add_patch(Rectangle((x + 0.36 * scale, y + 0.04 * scale), 0.18 * scale,
                           0.20 * scale, facecolor="#83a9c6", edgecolor="#17324d", lw=0.8, zorder=5))
    ax.add_patch(Circle((x + 0.13 * scale, y - 0.01 * scale), 0.055 * scale,
                        fc="#18212b", ec="white", lw=0.7, zorder=6))
    ax.add_patch(Circle((x + 0.44 * scale, y - 0.01 * scale), 0.055 * scale,
                        fc="#18212b", ec="white", lw=0.7, zorder=6))


def draw_drone(ax, x, y, scale=1.0, color="#9d4c9f"):
    ax.add_patch(Circle((x, y), 0.10 * scale, fc=color, ec="#4b2350", lw=0.8, zorder=5))
    for dx, dy in [(-0.22, 0), (0.22, 0), (0, 0.19), (0, -0.19)]:
        ax.add_patch(Circle((x + dx * scale, y + dy * scale), 0.07 * scale,
                            fc="#d67bd5", ec="#4b2350", lw=0.7, zorder=5))
        ax.plot([x, x + dx * scale], [y, y + dy * scale], color="#4b2350",
                lw=0.7, zorder=5)


def draw_request(ax, x, y, color="#f28e2b", text=""):
    ax.add_patch(FancyBboxPatch((x - 0.13, y - 0.13), 0.26, 0.26,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=color, ec="#9b4d00", lw=0.8, zorder=5))
    label(ax, x, y, text or "i", fs=7, weight="bold", color="white", z=6)


def draw_pipeline(ax, x, y, labels, colors, width, h=0.42, fs=8):
    gap = 0.10
    cur = x
    for txt, col in zip(labels, colors):
        w = width / len(labels) - gap * (len(labels) - 1) / len(labels)
        rounded(ax, cur, y, w, h, txt, fc=col, ec="#52606d", lw=0.9,
                radius=0.06, fs=fs, weight="bold")
        cur += w + gap


def main():
    fig, ax = plt.subplots(figsize=(20, 11), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")

    navy = "#17324d"
    blue = "#2f6fa3"
    green = "#4e8b57"
    orange = "#e58b3d"
    purple = "#7a4d99"

    # Environment strip.
    rounded(ax, 0.45, 8.65, 19.1, 1.95, fc="#fbfcfd", ec="#58616b",
            lw=1.2, radius=0.05, ls="--", z=1)
    label(ax, 10.0, 10.34, "Environment", fs=15, weight="bold")
    draw_truck(ax, 0.85, 9.78, 0.78)
    label(ax, 1.75, 9.90, "Truck", fs=8, ha="left")
    draw_drone(ax, 2.95, 9.91, 0.42)
    label(ax, 3.45, 9.90, "Drone", fs=8, ha="left")
    draw_truck(ax, 4.55, 9.78, 0.78, color="#507f65")
    draw_drone(ax, 4.98, 10.08, 0.27)
    label(ax, 5.45, 9.90, "Truck with drone", fs=8, ha="left")
    ax.plot([0.85, 6.1], [9.55, 9.55], color="#c8cdd2", lw=0.8)
    ax.add_patch(Rectangle((0.86, 8.94), 0.26, 0.22, fc="#b85b50", ec="#75372f", lw=0.8))
    label(ax, 1.37, 9.05, "Depot", fs=8, ha="left")
    draw_request(ax, 3.04, 9.06, text="+")
    label(ax, 3.45, 9.05, "Request (truck/UAV)", fs=8, ha="left")
    draw_request(ax, 5.20, 9.06, color="#7b4a9a", text="+")
    label(ax, 5.62, 9.05, "Request (UAV only)", fs=8, ha="left")

    # Small route sketch in the environment.
    route_nodes = [(8.0, 9.07), (9.4, 9.77), (11.0, 9.20), (12.7, 9.86),
                   (14.1, 9.20), (15.7, 9.74), (17.4, 9.15), (18.6, 9.75)]
    for i, (x, y) in enumerate(route_nodes):
        draw_request(ax, x, y, color="#f28e2b" if i % 2 == 0 else "#7b4a9a", text="+")
        if i:
            arrow(ax, route_nodes[i - 1][0] + 0.14, route_nodes[i - 1][1],
                  x - 0.14, y, color=navy, lw=1.1, style="-|>", ls="--")
    draw_truck(ax, 10.3, 9.51, 0.50)
    draw_truck(ax, 14.85, 9.46, 0.50, color="#507f65")
    draw_drone(ax, 12.0, 9.95, 0.24)
    draw_drone(ax, 16.7, 9.94, 0.24)

    # Decentralized actors.
    rounded(ax, 0.45, 4.55, 12.20, 3.70, fc="#f8fbff", ec="#1f2937",
            lw=1.4, radius=0.22, ls="--", z=1)
    label(ax, 6.55, 8.00, "Decentralized Actors", fs=14, weight="bold")

    # Truck actor.
    rounded(ax, 0.83, 6.40, 11.35, 1.22, fc="#dbeaf7", ec="#487ca7", lw=1.0, radius=0.16)
    label(ax, 1.20, 7.34, "Truck Actor", fs=10, weight="bold", ha="left")
    label(ax, 0.65, 6.98, r"$s^{TR}_{t,k}$", fs=9, color=navy)
    arrow(ax, 0.82, 7.00, 1.35, 7.00, color=navy, lw=1.1)
    draw_pipeline(ax, 1.40, 6.82, ["FC", "FC", "LSTM"], ["#ffd5b8", "#ffd5b8", "#f6dc55"], 2.20, h=0.38, fs=7)
    arrow(ax, 3.70, 7.01, 4.20, 7.01, color=navy, lw=1.1)
    rounded(ax, 4.22, 6.78, 2.28, 0.44, "Routing head", fc="#ffffff", ec="#829ab0", lw=0.8, radius=0.06, fs=8)
    rounded(ax, 4.22, 6.30, 2.28, 0.38, "Mode head", fc="#ffffff", ec="#829ab0", lw=0.8, radius=0.06, fs=8)
    arrow(ax, 6.50, 6.99, 7.03, 6.99, color=navy, lw=1.1)
    arrow(ax, 6.50, 6.49, 7.03, 6.49, color=navy, lw=1.1)
    rounded(ax, 7.05, 6.78, 1.35, 0.44, "Softmax", fc="#fff4bc", ec="#a28e34", lw=0.8, radius=0.06, fs=8)
    rounded(ax, 7.05, 6.30, 1.35, 0.38, "Softmax", fc="#fff4bc", ec="#a28e34", lw=0.8, radius=0.06, fs=8)
    label(ax, 5.15, 6.13, "logits + feasibility mask", fs=7, color="#5b6770")
    arrow(ax, 8.43, 6.98, 8.92, 6.98, color=navy, lw=1.1)
    rounded(ax, 8.94, 6.74, 1.02, 0.48, "sample", fc="#d9e86f", ec="#82912c", lw=0.9, radius=0.06, fs=8, weight="bold")
    arrow(ax, 9.96, 6.98, 11.00, 6.98, text=r"$a^{TR}_{t,k}$", color=navy, lw=1.1, fs=8, text_offset=(0, 0.18))
    arrow(ax, 10.00, 6.60, 11.00, 6.30, text=r"$\pi^{TR}_{t,k}$", color=navy, lw=1.0, fs=8, text_offset=(0.05, 0.14))

    # Drone actor.
    rounded(ax, 0.83, 4.88, 11.35, 1.22, fc="#e8e2f4", ec="#8067a2", lw=1.0, radius=0.16)
    label(ax, 1.20, 5.82, "UAV Actor", fs=10, weight="bold", ha="left")
    label(ax, 0.65, 5.40, r"$s^{DR}_{t,\ell}$", fs=9, color=navy)
    arrow(ax, 0.82, 5.42, 1.35, 5.42, color=navy, lw=1.1)
    draw_pipeline(ax, 1.40, 5.23, ["FC", "FC", "LSTM"], ["#ffd5b8", "#ffd5b8", "#f6dc55"], 2.20, h=0.38, fs=7)
    arrow(ax, 3.70, 5.42, 4.20, 5.42, color=navy, lw=1.1)
    rounded(ax, 4.22, 5.60, 2.02, 0.35, "Service head", fc="#ffffff", ec="#9883b0", lw=0.8, radius=0.06, fs=7.5)
    rounded(ax, 4.22, 5.18, 2.02, 0.35, "Rendezvous head", fc="#ffffff", ec="#9883b0", lw=0.8, radius=0.06, fs=7.5)
    rounded(ax, 4.22, 4.76, 2.02, 0.35, "Mode head", fc="#ffffff", ec="#9883b0", lw=0.8, radius=0.06, fs=7.5)
    for yy in [5.77, 5.35, 4.93]:
        arrow(ax, 6.24, yy, 6.72, yy, color=navy, lw=1.0)
        rounded(ax, 6.74, yy - 0.17, 1.22, 0.34, "Softmax", fc="#fff4bc", ec="#a28e34", lw=0.8, radius=0.06, fs=7)
    label(ax, 5.28, 4.62, "logits + feasibility mask", fs=7, color="#5b6770")
    arrow(ax, 7.98, 5.42, 8.52, 5.42, color=navy, lw=1.1)
    rounded(ax, 8.54, 5.18, 1.02, 0.48, "sample", fc="#d9e86f", ec="#82912c", lw=0.9, radius=0.06, fs=8, weight="bold")
    arrow(ax, 9.56, 5.42, 11.00, 5.42, text=r"$a^{DR}_{t,\ell}$", color=navy, lw=1.1, fs=8, text_offset=(0, 0.18))
    arrow(ax, 9.56, 5.06, 11.00, 4.88, text=r"$\pi^{DR}_{t,\ell}$", color=navy, lw=1.0, fs=8, text_offset=(0.05, 0.14))

    # Replay buffer.
    rounded(ax, 13.05, 4.70, 6.48, 3.25, fc="#fbfcfd", ec="#1f2937",
            lw=1.4, radius=0.22, ls="--", z=1)
    label(ax, 16.30, 7.60, "On-policy Rollout Buffer", fs=12, weight="bold")
    label(ax, 16.30, 7.25, r"$\mathcal{B}=\{S_t,o_t,a_t,\pi_t,r_t,m_t\}$", fs=9, color=navy)
    rounded(ax, 13.65, 5.12, 5.25, 1.63, fc="#eaf5e4", ec="#6aa05b", lw=1.0, radius=0.18)
    label(ax, 16.28, 6.45, "Stored transition", fs=9, weight="bold", color="#365c36")
    rounded(ax, 14.10, 5.82, 1.46, 0.38, r"one-hot $a_t$", fc="#fff4bc", ec="#d0ad2d", lw=0.8, radius=0.05, fs=8)
    rounded(ax, 15.78, 5.82, 1.46, 0.38, r"joint $\pi_t$", fc="#cfe5f6", ec="#5f8cad", lw=0.8, radius=0.05, fs=8)
    rounded(ax, 17.46, 5.82, 1.05, 0.38, r"$r_t,m_t$", fc="#f9dfd6", ec="#bb6b57", lw=0.8, radius=0.05, fs=8)
    rounded(ax, 14.10, 5.28, 4.40, 0.38, r"counterfactual $c_{i,t}=(a_{-i,t},\pi_{i,t})$", fc="#e1d8f1", ec="#8067a2", lw=0.8, radius=0.05, fs=8)

    # Actor-to-buffer arrows.
    arrow(ax, 11.02, 6.98, 13.02, 6.98, text=r"$a_t,\pi_t$", color=navy, lw=1.1, fs=8, text_offset=(0, 0.20))
    arrow(ax, 11.02, 5.42, 13.02, 5.90, color=navy, lw=1.1)
    arrow(ax, 10.10, 8.10, 13.02, 7.45, text=r"$r_t$", color=green, lw=1.1, fs=8, text_offset=(0, 0.20))

    # Centralized critic outer frame.
    rounded(ax, 0.45, 0.45, 19.10, 3.65, fc="#fbfcfd", ec="#1f2937",
            lw=1.4, radius=0.22, ls="--", z=1)
    label(ax, 10.0, 3.83, "Centralized Critic + COC2A", fs=14, weight="bold")

    # Global observation and Transformer encoder.
    rounded(ax, 0.80, 1.00, 3.40, 2.43, fc="#eef5fb", ec="#6e9bc0", lw=1.0, radius=0.14)
    label(ax, 2.50, 3.20, "Global observation", fs=10, weight="bold")
    label(ax, 2.50, 2.88, r"$S_t=\{o_{1,t},\ldots,o_{N,t}\}$", fs=9, color=navy)
    for j, col in enumerate(["#4b91c4", "#72a8cf", "#9eb9d6", "#c8bddf"]):
        for i in range(3):
            ax.add_patch(Rectangle((1.20 + j * 0.35, 1.52 + i * 0.23), 0.22, 0.18,
                                   fc=col, ec="#4b6174", lw=0.5, zorder=3))
    label(ax, 2.48, 1.15, r"$(B,N,G_{\rm dim})$", fs=8, color="#53616e")
    rounded(ax, 4.45, 1.00, 2.15, 2.43, fc="#fff1e6", ec="#d38d55", lw=1.0, radius=0.14)
    label(ax, 5.52, 3.20, "Embedding", fs=9, weight="bold")
    draw_pipeline(ax, 4.80, 2.18, ["Input", "Dense"], ["#ffd6b2", "#f4b183"], 1.45, h=0.38, fs=7)
    rounded(ax, 4.80, 1.54, 1.45, 0.38, "MHSA x L", fc="#d6e8f5", ec="#6695b9", lw=0.8, radius=0.05, fs=7)
    label(ax, 5.52, 1.20, "Transformer encoder", fs=8, color="#53616e")
    arrow(ax, 4.20, 2.20, 4.43, 2.20, color=navy, lw=1.2)
    arrow(ax, 6.62, 2.20, 7.00, 2.20, color=navy, lw=1.2)

    rounded(ax, 7.02, 1.00, 1.75, 2.43, fc="#f0e9f8", ec="#9676b3", lw=1.0, radius=0.14)
    label(ax, 7.90, 3.20, "Mean pooling", fs=9, weight="bold")
    for i in range(4):
        ax.add_patch(Rectangle((7.32, 1.58 + i * 0.29), 0.95, 0.17,
                               fc="#b7a0ca", ec="#76558c", lw=0.5, zorder=3))
    arrow(ax, 8.78, 2.20, 9.15, 2.20, color=navy, lw=1.2)
    rounded(ax, 9.17, 1.65, 1.05, 1.10, "$z_s$\nGlobal\nembedding", fc="#e1d8f1", ec="#8067a2", lw=1.0, radius=0.10, fs=9, weight="bold")

    # Coordination state side branch.
    rounded(ax, 8.95, 0.64, 2.00, 0.58, r"coordination state $e_{i,t}^{q}$", fc="#e6f2e2", ec="#6aa05b", lw=1.0, radius=0.10, fs=8)
    arrow(ax, 8.60, 1.02, 8.95, 0.93, color=green, lw=1.1, ls="--")
    label(ax, 7.45, 0.73, "mode, payload, energy, host/flying", fs=7, color=green, ha="left")
    arrow(ax, 10.05, 1.22, 11.15, 1.22, text=r"$z_s$", color=navy, lw=1.1, fs=8, text_offset=(0, 0.18))
    arrow(ax, 10.95, 0.93, 11.15, 1.00, text=r"$e_{i,t}^{q}$", color=green, lw=1.0, fs=8, text_offset=(0, 0.16))

    # COC2A module.
    rounded(ax, 11.18, 0.67, 7.95, 2.78, fc="#f8dddd", ec="#b85f5f", lw=1.4, radius=0.20)
    label(ax, 15.15, 3.16, "COC2A: coordination-aware counterfactual credit assignment", fs=10, weight="bold", color="#7c3030")
    rounded(ax, 11.52, 2.13, 2.55, 0.76, fc="#fff3df", ec="#cf9861", lw=1.0, radius=0.10)
    label(ax, 12.80, 2.63, r"$g_{\omega}^{q}(z_s,e_{i,t}^{q})$", fs=9, weight="bold")
    label(ax, 12.80, 2.34, "MLP weight generator", fs=7.5, color="#6c5a4b")
    rounded(ax, 14.25, 2.13, 1.72, 0.76, "Softmax", fc="#fff4bc", ec="#a28e34", lw=1.0, radius=0.10, fs=9, weight="bold")
    label(ax, 16.92, 2.51, r"$[\psi^{ind},\psi^{joint}]$", fs=9, color="#7c3030")
    arrow(ax, 14.08, 2.51, 14.22, 2.51, color=navy, lw=1.1)
    arrow(ax, 15.98, 2.51, 16.30, 2.51, color=navy, lw=1.1)
    label(ax, 16.75, 2.17, "optimized by CMA-ES", fs=7, color="#7c3030")

    # Q decomposition and counterfactual baselines.
    rounded(ax, 11.52, 0.98, 2.30, 0.85, fc="#fce9c9", ec="#cf9861", lw=1.0, radius=0.10)
    label(ax, 12.67, 1.56, r"$Q_{\phi}(z_s,x)=V(z_s)+w(z_s)^\top x$", fs=8.5, weight="bold")
    label(ax, 12.67, 1.23, "value + action-weight heads", fs=7.3, color="#6c5a4b")
    arrow(ax, 13.82, 1.40, 14.15, 1.40, color=navy, lw=1.1)
    rounded(ax, 14.18, 1.02, 1.72, 0.72, "$Q(z_s,a_t)$\nrealized", fc="#d8e9f5", ec="#6590b0", lw=0.9, radius=0.08, fs=8)
    rounded(ax, 16.08, 1.72, 1.63, 0.58, r"$b^{joint}(z_s,\pi_t)$", fc="#d8e9f5", ec="#6590b0", lw=0.9, radius=0.08, fs=8)
    rounded(ax, 16.08, 0.94, 1.63, 0.58, r"$b^{ind}(z_s,c_{i,t})$", fc="#e1d8f1", ec="#8067a2", lw=0.9, radius=0.08, fs=8)
    arrow(ax, 15.90, 1.42, 16.05, 1.95, color=navy, lw=1.0)
    arrow(ax, 15.90, 1.38, 16.05, 1.23, color=navy, lw=1.0)
    rounded(ax, 17.98, 1.15, 0.82, 0.74, r"$A_{i,t}$", fc="#d9e86f", ec="#82912c", lw=1.0, radius=0.10, fs=11, weight="bold")
    arrow(ax, 17.74, 1.97, 18.00, 1.74, color=navy, lw=1.0, text="+", fs=9, text_offset=(0, 0.03))
    arrow(ax, 17.74, 1.23, 18.00, 1.40, color=navy, lw=1.0, text="-", fs=9, text_offset=(0, 0.03))
    label(ax, 18.40, 2.13, "weighted", fs=7, color="#7c3030")
    label(ax, 18.40, 0.92, "advantage", fs=7, color="#7c3030")

    # Buffer feeds counterfactual inputs.
    arrow(ax, 16.30, 4.67, 14.92, 3.48, text=r"$a_t,\pi_t,c_{i,t}$", color=purple,
          lw=1.2, fs=8, text_offset=(0.25, 0.10))
    # Loss/update paths.
    arrow(ax, 18.82, 1.50, 19.15, 1.50, color=green, lw=1.2)
    arrow(ax, 19.15, 1.50, 19.15, 0.27, color=green, lw=1.2)
    arrow(ax, 19.15, 0.27, 10.80, 0.27, text=r"actor loss $\nabla_\theta J$", color=green,
          lw=1.2, fs=8, text_offset=(0, 0.18))
    arrow(ax, 6.20, 0.52, 6.20, 0.27, color=blue, lw=1.2)
    arrow(ax, 6.20, 0.27, 2.20, 0.27, text=r"critic loss $\nabla_\phi L$", color=blue,
          lw=1.2, fs=8, text_offset=(0, 0.18))
    arrow(ax, 2.20, 0.27, 2.20, 0.92, color=blue, lw=1.2)

    # Coordination input comes from actor observations as well.
    arrow(ax, 10.95, 5.00, 10.95, 4.15, color=green, lw=1.0, ls="--")
    arrow(ax, 10.95, 4.15, 9.80, 3.40, color=green, lw=1.0, ls="--")
    label(ax, 10.12, 4.12, r"$e_{i,t}^{q}$", fs=8, color=green)

    # A small explanatory legend inside the bottom panel.
    label(ax, 2.55, 0.60, "CTDE training path", fs=7.5, color="#52606d")
    ax.plot([1.00, 1.55], [0.60, 0.60], color=blue, lw=1.7)
    label(ax, 4.10, 0.60, "policy-gradient path", fs=7.5, color="#52606d")
    ax.plot([3.00, 3.55], [0.60, 0.60], color=green, lw=1.7)

    # Save vector and raster variants.
    fig.savefig(OUT / "framework_COC2A.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / "framework_COC2A.svg", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / "framework_COC2A.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
