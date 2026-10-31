#!/usr/bin/env python3
"""Generate TDK thesis figures from table data (+ optional measurement plots).

Outputs PDF (vector) and PNG into docs/Picture/.
Numbers match docs/TDK.tex experimental tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Polygon, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent / "Picture"
OUT.mkdir(parents=True, exist_ok=True)

# ---- data from TDK.tex ----
TAU = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])
KEPT = {
    "Szöveges VLM – frekvencia": np.array([35, 52, 74, 111, 177, 265, 383]),
    "Szöveges VLM – lefedőhalmaz": np.array([16, 18, 32, 55, 107, 199, 346]),
    "Multimodális VLM – frekvencia": np.array([19, 27, 40, 68, 118, 221, 371]),
    "Multimodális VLM – lefedőhalmaz": np.array([7, 10, 15, 23, 51, 141, 329]),
}
RMSE_VY = {
    "Szöveges VLM – frekvencia": np.array([0.02426, 0.02251, 0.15630, 0.01743, 0.02251, 0.01654, 0.01555]),
    "Szöveges VLM – lefedőhalmaz": np.array([0.05403, 0.04635, 0.03576, 0.02673, 0.01875, 0.02562, 0.01719]),
    "Multimodális VLM – frekvencia": np.array([0.07876, 0.04037, 0.03886, 0.03261, 0.02829, 0.02058, 0.01919]),
    "Multimodális VLM – lefedőhalmaz": np.array([0.04136, 0.05342, 0.04917, 0.04656, 0.03580, 0.03051, 0.02139]),
}
RMSE_YAW = {
    "Szöveges VLM – frekvencia": np.array([0.004134, 0.004415, 0.019870, 0.004102, 0.005467, 0.004423, 0.003753]),
    "Szöveges VLM – lefedőhalmaz": np.array([0.009368, 0.008619, 0.008570, 0.006542, 0.005161, 0.002433, 0.004655]),
    "Multimodális VLM – frekvencia": np.array([0.012490, 0.008055, 0.007765, 0.003305, 0.006864, 0.005296, 0.004964]),
    "Multimodális VLM – lefedőhalmaz": np.array([0.006462, 0.009667, 0.009465, 0.009231, 0.006194, 0.006481, 0.004882]),
}

# Relative Δe (%) vs szöveges VLM τ=0.95 (table 4.3)
DELTA = [
    ("PCA min-max + frekvencia", 28.90, 15.94),
    ("PCA z-score + frekvencia", 12.50, 8.93),
    ("PCA min-max + k-means", 80.17, 45.08),
    ("PCA min-max + k-medoids", 79.44, 61.11),
    ("PCA z-score + k-means", 57.38, 46.57),
    ("PCA z-score + k-medoids", 43.59, 38.98),
    ("Szöveges VLM, τ=0,95", 0.00, 0.00),
    ("Szöveges VLM, τ=0,93", 5.38, -7.26),
    ("Szöveges VLM, τ=0,96", -5.99, -15.15),
]

COLORS = {
    "Szöveges VLM – frekvencia": "#1f4e79",
    "Szöveges VLM – lefedőhalmaz": "#5b9bd5",
    "Multimodális VLM – frekvencia": "#c45911",
    "Multimodális VLM – lefedőhalmaz": "#ed7d31",
}
MARKERS = {
    "Szöveges VLM – frekvencia": "o",
    "Szöveges VLM – lefedőhalmaz": "s",
    "Multimodális VLM – frekvencia": "^",
    "Multimodális VLM – lefedőhalmaz": "D",
}
STYLES = {
    "Szöveges VLM – frekvencia": "-",
    "Szöveges VLM – lefedőhalmaz": "--",
    "Multimodális VLM – frekvencia": "-",
    "Multimodális VLM – lefedőhalmaz": "--",
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=10)


def save(fig, stem: str):
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf.name} / {png.name}")


def fig_kept_vs_threshold():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, y in KEPT.items():
        ax.plot(
            TAU,
            y,
            color=COLORS[name],
            linestyle=STYLES[name],
            marker=MARKERS[name],
            markersize=6,
            linewidth=2.0,
            label=name,
        )
    ax.axvline(0.95, color="#666666", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(0.951, 40, "τ = 0,95", fontsize=9, color="#555555", rotation=90, va="bottom")
    ax.set_xlabel(r"Hasonlósági küszöbérték $\tau$", fontsize=11)
    ax.set_ylabel("Megtartott manőverek száma", fontsize=11)
    ax.set_xticks(TAU)
    ax.set_xticklabels([f"{t:.2f}".replace(".", ",") for t in TAU])
    ax.set_ylim(0, 420)
    style_axes(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "tdk_kept_vs_threshold")


def fig_rmse_vs_threshold():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)
    for ax, data, ylabel, title in [
        (axes[0], RMSE_VY, r"RMSE $v_y$", "Oldalirányú sebesség"),
        (axes[1], RMSE_YAW, r"RMSE $\dot{\psi}$", "Perdületi sebesség"),
    ]:
        for name, y in data.items():
            ax.plot(
                TAU,
                y,
                color=COLORS[name],
                linestyle=STYLES[name],
                marker=MARKERS[name],
                markersize=5.5,
                linewidth=1.8,
                label=name,
            )
        ax.axvline(0.95, color="#666666", linestyle=":", linewidth=1.1, alpha=0.9)
        ax.set_xlabel(r"$\tau$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(TAU)
        ax.set_xticklabels([f"{t:.2f}".replace(".", ",") for t in TAU], fontsize=8)
        style_axes(ax)
    # clip outlier annotation on left
    axes[0].annotate(
        "0,156",
        xy=(0.92, 0.15630),
        xytext=(0.905, 0.12),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
        color="#555555",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, "tdk_rmse_vs_threshold")


def fig_tradeoff_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, data, ylabel, title in [
        (axes[0], RMSE_VY, r"RMSE $v_y$", "Oldalirányú sebesség"),
        (axes[1], RMSE_YAW, r"RMSE $\dot{\psi}$", "Perdületi sebesség"),
    ]:
        for name in KEPT:
            x = KEPT[name]
            y = data[name]
            ax.scatter(
                x,
                y,
                c=COLORS[name],
                marker=MARKERS[name],
                s=55,
                label=name,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
            # highlight τ=0.95
            i95 = 5
            ax.scatter(
                [x[i95]],
                [y[i95]],
                facecolors="none",
                edgecolors="black",
                s=140,
                linewidths=1.4,
                zorder=4,
            )
        ax.set_xlabel("Megtartott manőverek száma", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        style_axes(ax)
    # shared legend + note
    handles, labels = axes[0].get_legend_handles_labels()
    ring = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor="k",
        markersize=10,
        linestyle="None",
        label=r"$\tau = 0{,}95$",
    )
    handles.append(ring)
    labels.append(r"$\tau = 0{,}95$")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    # hide extreme outlier on vy for readability? Keep but set ylim with note
    axes[0].set_ylim(0.01, 0.09)
    axes[0].text(0.02, 0.97, "Megjegyzés: a 0,156 RMSE pont\n(τ=0,92, szöveges+frekv.)\na tengelyen kívül esik.", transform=axes[0].transAxes, fontsize=7.5, va="top", color="#555555")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    save(fig, "tdk_tradeoff_scatter")


def fig_pca_vlm_delta():
    labels = [d[0] for d in DELTA]
    dvy = np.array([d[1] for d in DELTA])
    dyaw = np.array([d[2] for d in DELTA])
    y = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=False)
    for ax, vals, title in [
        (axes[0], dvy, r"Relatív eltérés $\Delta e_{v_y}$ (\%)"),
        (axes[1], dyaw, r"Relatív eltérés $\Delta e_{\dot{\psi}}$ (\%)"),
    ]:
        colors = []
        for lab, v in zip(labels, vals):
            if lab.startswith("Szöveges VLM, τ=0,95"):
                colors.append("#1f4e79")
            elif lab.startswith("Szöveges VLM"):
                colors.append("#5b9bd5")
            elif "z-score + frekvencia" in lab:
                colors.append("#c45911")  # best PCA highlight
            else:
                colors.append("#9e9e9e")
        ax.barh(y, vals, color=colors, height=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(title, fontsize=10)
        style_axes(ax)
        ax.grid(True, axis="x", alpha=0.3, linewidth=0.6)
        # annotate best PCA
        for i, (lab, v) in enumerate(zip(labels, vals)):
            if "z-score + frekvencia" in lab:
                ax.text(v + 1.5, i, f"+{v:.2f}".replace(".", ",") + r"\,\%", va="center", fontsize=8, color="#c45911", fontweight="bold")
            elif abs(v) > 0.01 and not lab.endswith("0,95"):
                ax.text(
                    v + (1.2 if v >= 0 else -1.2),
                    i,
                    f"{v:+.2f}".replace(".", ",").replace("+", "+") + r"\,\%",
                    va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7.5,
                    color="#444444",
                )
    axes[0].set_title(r"Referencia: szöveges VLM + frekvencia, $\tau=0{,}95$ ($\Delta e=0$)", fontsize=10)
    fig.tight_layout()
    save(fig, "tdk_pca_vlm_delta")


def fig_algorithm_schematic():
    """Toy graph illustration of frequency vs greedy set cover."""
    # 6 nodes in a circle-ish layout
    pos = {
        0: (0.0, 0.6),
        1: (0.7, 1.0),
        2: (1.4, 0.6),
        3: (1.4, 0.0),
        4: (0.7, -0.4),
        5: (0.0, 0.0),
    }
    # edges above τ
    edges = [(0, 1), (1, 2), (0, 5), (1, 5), (2, 3), (3, 4), (4, 5), (1, 4)]

    def draw_graph(ax, title, kept, removed, show_edges=True, highlight_remove=None, highlight_keep=None):
        ax.set_xlim(-0.45, 1.85)
        ax.set_ylim(-0.75, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=6)
        if show_edges:
            for i, j in edges:
                x = [pos[i][0], pos[j][0]]
                y = [pos[i][1], pos[j][1]]
                ax.plot(x, y, color="#bbbbbb", linewidth=1.6, zorder=1)
        for i, (x, y) in pos.items():
            if i in kept:
                fc, ec = "#1f4e79", "#0d2c4a"
            else:
                fc, ec = "#d9d9d9", "#888888"
            if highlight_remove is not None and i == highlight_remove:
                fc, ec = "#c45911", "#843c0c"
            if highlight_keep is not None and i == highlight_keep:
                fc, ec = "#2e7d32", "#1b5e20"
            circ = Circle((x, y), 0.14, facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2)
            ax.add_patch(circ)
            ax.text(x, y, str(i + 1), ha="center", va="center", color="white" if i in kept or i == highlight_remove or i == highlight_keep else "#333333", fontsize=9, fontweight="bold", zorder=3)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.8))

    # Frequency row
    draw_graph(axes[0, 0], "Frekvencia — kiindulás\n(redundáns élek)", kept=set(range(6)), removed=set())
    draw_graph(
        axes[0, 1],
        "Frekvencia — lépés\n(legnagyobb fokú: 2)",
        kept={0, 1, 2, 3, 4, 5},
        removed=set(),
        highlight_remove=1,
    )
    # after removing high-degree nodes iteratively → independent set e.g. {0,2,4}
    draw_graph(
        axes[0, 2],
        "Frekvencia — eredmény\nmegtartott: 1, 3, 5",
        kept={0, 2, 4},
        removed={1, 3, 5},
        show_edges=False,
    )
    # draw remaining as no τ-edges among kept
    for i, j in [(0, 2), (2, 4), (0, 4)]:
        pass  # no edges among independent set

    # Greedy set cover row — same start, select cover centers
    draw_graph(axes[1, 0], "Lefedőhalmaz — kiindulás\n(lefedő szomszédságok)", kept=set(range(6)), removed=set())
    draw_graph(
        axes[1, 1],
        "Lefedőhalmaz — lépés\n(legnagyobb lefedés: 2)",
        kept={0, 1, 2, 3, 4, 5},
        removed=set(),
        highlight_keep=1,
    )
    draw_graph(
        axes[1, 2],
        "Lefedőhalmaz — eredmény\nmegtartott: 2, 4",
        kept={1, 3},
        removed={0, 2, 4, 5},
        show_edges=False,
    )
    # show cover links dashed from kept to removed
    for src, dsts in [(1, [0, 2, 4]), (3, [2, 4, 5])]:
        for d in dsts:
            if d not in (1, 3):
                axes[1, 2].annotate(
                    "",
                    xy=pos[d],
                    xytext=pos[src],
                    arrowprops=dict(arrowstyle="-", color="#2e7d32", lw=1.0, linestyle="--", alpha=0.7),
                    zorder=1,
                )
                axes[1, 2].plot([pos[src][0], pos[d][0]], [pos[src][1], pos[d][1]], color="#2e7d32", linewidth=1.0, linestyle="--", alpha=0.55, zorder=1)

    # rebuild kept circles on top
    for i in (1, 3):
        x, y = pos[i]
        circ = Circle((x, y), 0.14, facecolor="#1f4e79", edgecolor="#0d2c4a", linewidth=1.4, zorder=2)
        axes[1, 2].add_patch(circ)
        axes[1, 2].text(x, y, str(i + 1), ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=3)
    for i in (0, 2, 4, 5):
        x, y = pos[i]
        circ = Circle((x, y), 0.14, facecolor="#d9d9d9", edgecolor="#888888", linewidth=1.4, zorder=2)
        axes[1, 2].add_patch(circ)
        axes[1, 2].text(x, y, str(i + 1), ha="center", va="center", color="#333333", fontsize=9, fontweight="bold", zorder=3)

    legend_elems = [
        mpatches.Patch(facecolor="#1f4e79", edgecolor="#0d2c4a", label="Megtartott"),
        mpatches.Patch(facecolor="#d9d9d9", edgecolor="#888888", label="Eltávolított / lefedett"),
        mpatches.Patch(facecolor="#c45911", edgecolor="#843c0c", label="Következő eltávolítás"),
        mpatches.Patch(facecolor="#2e7d32", edgecolor="#1b5e20", label="Következő megtartás"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Szemléltető példa (nem a dolgozat mérési mátrixából)", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    save(fig, "tdk_redundancy_algorithms")


def fig_bicycle_model():
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.8, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # body
    body = FancyBboxPatch((1.5, -0.35), 4.5, 0.7, boxstyle="round,pad=0.02,rounding_size=0.15", facecolor="#e8eef5", edgecolor="#1f4e79", linewidth=1.8)
    ax.add_patch(body)

    # CoG
    cog = (3.7, 0.0)
    ax.plot(*cog, "o", color="#1f4e79", markersize=8, zorder=5)
    ax.text(cog[0], 0.55, "CoG", ha="center", fontsize=10, fontweight="bold", color="#1f4e79")

    # axles
    ax.plot([2.0, 2.0], [-0.55, 0.55], color="#333", lw=2.5)
    ax.plot([6.0, 6.0], [-0.55, 0.55], color="#333", lw=2.5)
    # wheels as rectangles
    for x, ang, label in [(2.0, 0, "hátsó"), (6.0, 18, "első")]:
        wh = Rectangle((-0.35, -0.12), 0.7, 0.24, facecolor="#555", edgecolor="k", lw=0.8)
        t = plt.matplotlib.transforms.Affine2D().rotate_deg(ang).translate(x, 0) + ax.transData
        wh.set_transform(t)
        ax.add_patch(wh)

    # lf, lr
    ax.annotate("", xy=(6.0, -1.0), xytext=(cog[0], -1.0), arrowprops=dict(arrowstyle="<->", color="#c45911", lw=1.4))
    ax.text((6.0 + cog[0]) / 2, -1.25, r"$\ell_f$", ha="center", color="#c45911", fontsize=12)
    ax.annotate("", xy=(2.0, -1.0), xytext=(cog[0], -1.0), arrowprops=dict(arrowstyle="<->", color="#c45911", lw=1.4))
    ax.text((2.0 + cog[0]) / 2, -1.25, r"$\ell_r$", ha="center", color="#c45911", fontsize=12)

    # velocities at CoG
    ax.annotate("", xy=(cog[0] + 1.6, 0), xytext=cog, arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.8))
    ax.text(cog[0] + 1.7, 0.15, r"$v_x$", color="#2e7d32", fontsize=11)
    ax.annotate("", xy=(cog[0], 1.2), xytext=cog, arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.8))
    ax.text(cog[0] + 0.15, 1.25, r"$v_y$", color="#2e7d32", fontsize=11)

    # yaw rate arc
    arc = Arc(cog, 1.1, 1.1, angle=0, theta1=20, theta2=110, color="#6a1b9a", lw=1.6)
    ax.add_patch(arc)
    ax.annotate("", xy=(cog[0] - 0.15, 0.55), xytext=(cog[0] + 0.05, 0.52), arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1.2))
    ax.text(cog[0] - 0.85, 0.85, r"$\dot{\psi}$", color="#6a1b9a", fontsize=12)

    # steering angle delta
    ax.annotate("", xy=(6.55, 0.55), xytext=(6.0, 0.0), arrowprops=dict(arrowstyle="->", color="#c45911", lw=1.5))
    ax.text(6.7, 0.65, r"$\delta$", color="#c45911", fontsize=12)

    # lateral forces
    ax.annotate("", xy=(6.0, 1.5), xytext=(6.0, 0.55), arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.8))
    ax.text(6.15, 1.55, r"$F_{y,f}$", color="#1f4e79", fontsize=11)
    ax.annotate("", xy=(2.0, 1.5), xytext=(2.0, 0.55), arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.8))
    ax.text(0.7, 1.55, r"$F_{y,r}$", color="#1f4e79", fontsize=11)

    # slip angles note
    ax.text(
        4.0,
        2.45,
        r"Egynyomvonalú modell: $F_y(\alpha)=D\sin(C\,\arctan(B\alpha))$",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    ax.text(4.0, -1.65, "Elöl: kormányzott tengely; hátul: nem kormányzott tengely", ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout()
    save(fig, "tdk_bicycle_model")


def fig_representation_flowchart():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eef5", ec="#1f4e79"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3))

    # shared start
    box(0.3, 3.0, 2.0, 1.0, "Mérési idősorok\n10805×40", fc="#f5f5f5", ec="#555555")
    box(2.8, 3.0, 2.0, 1.0, "Strukturált\ndiagramok +\nmetaadatok", fc="#f5f5f5", ec="#555555")
    arrow(2.3, 3.5, 2.8, 3.5)

    # branch labels
    ax.text(5.8, 6.55, "Szöveges VLM-reprezentáció", ha="center", fontsize=10, fontweight="bold", color="#1f4e79")
    ax.text(5.8, 2.55, "Multimodális VLM-reprezentáció", ha="center", fontsize=10, fontweight="bold", color="#c45911")

    # upper branch (text)
    box(5.0, 5.2, 2.0, 1.0, "14 diagram\n4 VLM-feladat\n(Qwen3-VL-8B)", fc="#d6e3f0", ec="#1f4e79")
    box(7.4, 5.2, 2.0, 1.0, "Fusion:\negyesített\nszakmai leírás", fc="#d6e3f0", ec="#1f4e79")
    box(9.8, 5.2, 1.9, 1.0, "Qwen3-\nEmbedding-8B\n4096-d", fc="#d6e3f0", ec="#1f4e79")
    arrow(4.8, 3.9, 5.5, 5.2)
    arrow(7.0, 5.7, 7.4, 5.7)
    arrow(9.4, 5.7, 9.8, 5.7)

    # lower branch (multimodal)
    box(5.0, 1.1, 2.0, 1.0, "10 laterális\ndiagram\n4 VLM-feladat", fc="#fce4d6", ec="#c45911")
    box(7.4, 1.1, 2.0, 1.0, "Nincs fusion:\n4× technical\nsummary", fc="#fce4d6", ec="#c45911")
    box(9.8, 1.1, 1.9, 1.0, "Qwen3-VL-\nEmbedding-8B\nszöveg+10 kép", fc="#fce4d6", ec="#c45911")
    arrow(4.8, 3.1, 5.5, 2.1)
    arrow(7.0, 1.6, 7.4, 1.6)
    arrow(9.4, 1.6, 9.8, 1.6)

    # merge to redundancy
    box(5.2, 3.15, 2.4, 0.7, "Csoporton belüli\nkoszinusz-hasonlóság", fc="#eeeeee", ec="#555555")
    arrow(10.75, 5.2, 7.0, 3.85)
    arrow(10.75, 2.1, 7.0, 3.15)
    box(8.2, 3.15, 2.4, 0.7, "Redundanciaszűrés\n(frekvencia / lefedőhalmaz)", fc="#eeeeee", ec="#555555")
    arrow(7.6, 3.5, 8.2, 3.5)

    ax.text(6.0, 0.35, "Mindkét ág: manőverenként egy 4096 dimenziós vektor → azonos szűrési pipeline", ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, "tdk_representation_flowchart")


def fig_maneuver_excitations():
    """Plot steering inputs for DLC / sinus / chirp from data_tesla."""
    try:
        import scipy.io as sio
    except ImportError:
        print("scipy missing — skip maneuver excitation figure")
        return

    root = Path(__file__).resolve().parents[1] / "data_tesla"
    files = {
        "Kettős sávváltás": root / "allando_v_savvaltas_kozepes_v80.mat",
        "Szinuszos kormányzás": root / "allando_v_sin_a8_f3_v80.mat",
        "Csirpszerű kormányzás": root / "allando_v_chirp_a3_v80.mat",
    }
    if not all(p.exists() for p in files.values()):
        # fallback any matching
        files = {
            "Kettős sávváltás": next(root.glob("allando_v_savvaltas_kozepes_*.mat")),
            "Szinuszos kormányzás": next(root.glob("allando_v_sin_a8_f3_*.mat")),
            "Csirpszerű kormányzás": next(root.glob("allando_v_chirp_a3_*.mat")),
        }

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.8), sharex=False)
    for ax, (title, path) in zip(axes, files.items()):
        d = sio.loadmat(path)
        steer = d["steeringangel"].ravel()
        t = np.arange(len(steer)) * 0.01
        # drop first 25 s run-up for readability where long
        i0 = 2500 if len(steer) > 4000 else 0
        # window
        if "sávváltás" in title:
            i1 = i0 + 800
        elif "Szinusz" in title:
            i1 = i0 + 2000
        else:
            i1 = i0 + 4000
        i1 = min(i1, len(steer))
        tt = t[i0:i1] - t[i0]
        ss = np.rad2deg(steer[i0:i1]) if np.nanmax(np.abs(steer[i0:i1])) < 2 else steer[i0:i1]
        # CarMaker often stores steering in rad; if max < 2 treat as rad
        ax.plot(tt, ss, color="#1f4e79", linewidth=1.3)
        ax.set_ylabel(r"$\delta$ [$^\circ$]", fontsize=10)
        ax.set_title(title, fontsize=10, loc="left")
        style_axes(ax)
    axes[-1].set_xlabel("Idő [s]", fontsize=11)
    fig.tight_layout()
    save(fig, "tdk_maneuver_excitations")


def fig_tsne_improved():
    """Replot t-SNE with short Hungarian group labels from Chroma v1."""
    try:
        import chromadb
        from sklearn.manifold import TSNE
    except ImportError as e:
        print(f"skip t-SNE: {e}")
        return

    chroma_path = Path(__file__).resolve().parents[1] / "chroma_db" / "tesla_vlm"
    if not chroma_path.exists():
        print("skip t-SNE: chroma missing")
        return

    short = {
        "allando_v_savvaltas": "Áll. sávváltás",
        "allando_v_chirp": "Áll. csirip",
        "allando_v_sin": "Áll. szinusz",
        "valtozo_v_savvaltas_gas": "Vált. sávv., gáz",
        "valtozo_v_savvaltas_fek": "Vált. sávv., fék",
        "valtozo_v_sin_gas": "Vált. szinusz, gáz",
        "valtozo_v_sin_fek": "Vált. szinusz, fék",
    }
    prefixes = list(short.keys())

    client = chromadb.PersistentClient(path=str(chroma_path))
    # try common collection names
    coll = None
    for name in ("tesla_vlm_descriptions", "tesla_vlm", "text_vlm_tesla"):
        try:
            coll = client.get_collection(name)
            break
        except Exception:
            continue
    if coll is None:
        print("skip t-SNE: collection not found", client.list_collections())
        return

    res = coll.get(include=["embeddings", "metadatas"])
    ids = list(res.get("ids") or [])
    embs = np.asarray(res["embeddings"], dtype=np.float64)
    groups = []
    for mid in ids:
        g = None
        for p in prefixes:
            if mid.startswith(p):
                g = p
                break
        groups.append(g)
    groups = np.array(groups)
    mask = groups != None  # noqa: E711
    embs = embs[mask]
    groups = groups[mask]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    xy = tsne.fit_transform(embs)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(prefixes)))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for i, p in enumerate(prefixes):
        m = groups == p
        if not np.any(m):
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=28, c=[colors[i]], edgecolors="none", alpha=0.85, label=f"{short[p]} ({m.sum()})")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False, markerscale=1.4, handletextpad=0.4)
    fig.tight_layout()
    save(fig, "ALL_groups_tsne_2d")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig_kept_vs_threshold()
    fig_rmse_vs_threshold()
    fig_tradeoff_scatter()
    fig_pca_vlm_delta()
    fig_algorithm_schematic()
    fig_bicycle_model()
    fig_representation_flowchart()
    fig_maneuver_excitations()
    fig_tsne_improved()
    print(f"Done. Output: {OUT}")


if __name__ == "__main__":
    main()
