"""
Plot a cosine/correlation similarity matrix from JSON data.

Outputs PNG and EPS under Jurnal_plots/readable/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT_DIR = Path("Jurnal_plots/readable")

LABEL_FONTSIZE = 24
TICK_FONTSIZE = 21
ANNOT_FONTSIZE = 19
CBAR_TICK_FONTSIZE = 19

MATRIX_DATA = {
    "manoeuvres": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "correlation_matrix": [
        [1.00, 0.68, 0.80, 0.70, 0.56, 0.81, 0.70, 0.75, 0.35],
    [0.68, 1.00, 0.61, 0.96, 0.20, 0.77, 0.87, 0.69, 0.64],
    [0.80, 0.61, 1.00, 0.68, 0.84, 0.59, 0.81, 0.64, 0.57],
    [0.70, 0.96, 0.68, 1.00, 0.31, 0.84, 0.95, 0.80, 0.69],
    [0.56, 0.20, 0.84, 0.31, 1.00, 0.32, 0.55, 0.47, 0.36],
    [0.81, 0.77, 0.59, 0.84, 0.32, 1.00, 0.81, 0.87, 0.27],
    [0.70, 0.87, 0.81, 0.95, 0.55, 0.81, 1.00, 0.86, 0.71],
    [0.75, 0.69, 0.64, 0.80, 0.47, 0.87, 0.86, 1.00, 0.54],
    [0.35, 0.64, 0.57, 0.69, 0.36, 0.27, 0.71, 0.54, 1.00]
    ],
}


def load_matrix_data(path: Path | None = None) -> dict:
    if path is None:
        return MATRIX_DATA
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_correlation_matrix(
    data: dict,
    out_stem: str = "cosine_similarity_matrix_9x9",
) -> None:
    manoeuvres = [str(m) for m in data["manoeuvres"]]
    matrix = np.array(data["correlation_matrix"], dtype=float)

    plt.figure(figsize=(13, 11))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        annot_kws={"size": ANNOT_FONTSIZE},
        xticklabels=manoeuvres,
        yticklabels=manoeuvres,
        cmap="coolwarm",
        vmin=0.2,
        vmax=1.0,
        square=True,
        cbar_kws={"label": ""},
    )
    ax.set_xlabel("Manoeuvres", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Manoeuvres", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    eps_path = OUT_DIR / f"{out_stem}.eps"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {eps_path}")


def main() -> None:
    plot_correlation_matrix(MATRIX_DATA)


if __name__ == "__main__":
    main()
