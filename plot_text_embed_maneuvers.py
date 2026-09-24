#!/usr/bin/env python3
"""
2D/3D PCA and t-SNE plots of averaged text-embedding maneuver vectors.

One figure set per maneuver group; each maneuver gets its own color.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

GROUP_PREFIXES = [
    "allando_v_savvaltas",
    "allando_v_chirp",
    "allando_v_sin",
    "valtozo_v_savvaltas_gas",
    "valtozo_v_savvaltas_fek",
    "valtozo_v_sin_gas",
    "valtozo_v_sin_fek",
]


def distinct_colors(n: int) -> list[tuple[float, float, float]]:
    """n visually distinct RGB colors (HSV wheel)."""
    if n <= 0:
        return []
    return [colorsys.hsv_to_rgb(i / n, 0.75, 0.95) for i in range(n)]


def short_label(maneuver: str, group: str) -> str:
    prefix = group + "_"
    if maneuver.startswith(prefix):
        return maneuver[len(prefix) :]
    return maneuver


def load_grouped_vectors(average_dir: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    """Return {group: (names, vectors[N,D])} for non-empty groups."""
    groups: dict[str, list[str]] = {p: [] for p in GROUP_PREFIXES}
    for path in sorted(average_dir.glob("*.npy")):
        name = path.stem
        for prefix in GROUP_PREFIXES:
            if name.startswith(prefix + "_") or name == prefix:
                groups[prefix].append(name)
                break

    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for prefix, names in groups.items():
        if not names:
            continue
        vectors = np.stack([np.load(average_dir / f"{n}.npy") for n in names])
        out[prefix] = (names, vectors)
    return out


def embed_2d_3d(
    vectors: np.ndarray,
    method: str,
    random_state: int,
    perplexity: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (coords_2d[N,2], coords_3d[N,3], subtitle with explained variance if PCA)."""
    n = vectors.shape[0]
    if method == "pca":
        pca2 = PCA(n_components=2, random_state=random_state)
        pca3 = PCA(n_components=min(3, vectors.shape[1]), random_state=random_state)
        xy = pca2.fit_transform(vectors)
        xyz = pca3.fit_transform(vectors)
        if xyz.shape[1] == 2:
            xyz = np.column_stack([xyz, np.zeros(n)])
        var2 = 100.0 * pca2.explained_variance_ratio_.sum()
        var3 = 100.0 * pca3.explained_variance_ratio_.sum()
        subtitle = f"PCA  var2D={var2:.1f}%  var3D={var3:.1f}%"
        return xy, xyz, subtitle

    # t-SNE: perplexity must be < n
    perp = min(perplexity, max(2.0, (n - 1) / 3.0))
    tsne2 = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    tsne3 = TSNE(
        n_components=3,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    xy = tsne2.fit_transform(vectors)
    xyz = tsne3.fit_transform(vectors)
    subtitle = f"t-SNE  perplexity={perp:.1f}"
    return xy, xyz, subtitle


def plot_group(
    group: str,
    names: list[str],
    vectors: np.ndarray,
    out_dir: Path,
    methods: list[str],
    random_state: int,
    perplexity: float,
    annotate: bool,
) -> None:
    colors = distinct_colors(len(names))
    labels = [short_label(n, group) for n in names]

    for method in methods:
        xy, xyz, subtitle = embed_2d_3d(
            vectors, method=method, random_state=random_state, perplexity=perplexity
        )

        # --- 2D ---
        fig, ax = plt.subplots(figsize=(11, 9))
        for i, (lab, color) in enumerate(zip(labels, colors)):
            ax.scatter(
                xy[i, 0],
                xy[i, 1],
                c=[color],
                s=55,
                edgecolors="k",
                linewidths=0.3,
                label=lab,
                zorder=3,
            )
            if annotate:
                ax.annotate(
                    lab,
                    (xy[i, 0], xy[i, 1]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6,
                    alpha=0.85,
                )
        ax.set_title(f"{group} — {subtitle} (2D)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.25)
        if len(names) <= 40:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=6,
                frameon=False,
                ncol=1 if len(names) <= 20 else 2,
            )
            fig.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig.tight_layout()
        out_2d = out_dir / f"{group}_{method}_2d.png"
        fig.savefig(out_2d, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_2d}")

        # --- 3D ---
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        for i, (lab, color) in enumerate(zip(labels, colors)):
            ax.scatter(
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                c=[color],
                s=55,
                edgecolors="k",
                linewidths=0.3,
                label=lab,
                depthshade=True,
            )
            if annotate:
                ax.text(xyz[i, 0], xyz[i, 1], xyz[i, 2], lab, fontsize=5, alpha=0.8)
        ax.set_title(f"{group} — {subtitle} (3D)")
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        ax.set_zlabel("C3")
        if len(names) <= 30:
            ax.legend(loc="upper left", fontsize=5, frameon=False, ncol=2)
        fig.tight_layout()
        out_3d = out_dir / f"{group}_{method}_3d.png"
        fig.savefig(out_3d, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_3d}")


def plot_all_groups_overview(
    grouped: dict[str, tuple[list[str], np.ndarray]],
    out_dir: Path,
    methods: list[str],
    random_state: int,
    perplexity: float,
) -> None:
    """One scatter per method: color = maneuver group (7 colors)."""
    names: list[str] = []
    group_ids: list[int] = []
    vectors_list: list[np.ndarray] = []
    group_list = list(grouped.keys())
    for gi, group in enumerate(group_list):
        gnames, gvecs = grouped[group]
        names.extend(gnames)
        group_ids.extend([gi] * len(gnames))
        vectors_list.append(gvecs)
    vectors = np.vstack(vectors_list)
    group_ids_arr = np.asarray(group_ids)
    group_colors = distinct_colors(len(group_list))

    for method in methods:
        xy, xyz, subtitle = embed_2d_3d(
            vectors, method=method, random_state=random_state, perplexity=perplexity
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        for gi, group in enumerate(group_list):
            mask = group_ids_arr == gi
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=[group_colors[gi]],
                s=40,
                edgecolors="k",
                linewidths=0.25,
                label=f"{group} (n={mask.sum()})",
                alpha=0.9,
            )
        ax.set_title(f"All Tesla groups — {subtitle} (2D)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        out_2d = out_dir / f"ALL_groups_{method}_2d.png"
        fig.savefig(out_2d, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_2d}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for gi, group in enumerate(group_list):
            mask = group_ids_arr == gi
            ax.scatter(
                xyz[mask, 0],
                xyz[mask, 1],
                xyz[mask, 2],
                c=[group_colors[gi]],
                s=40,
                edgecolors="k",
                linewidths=0.25,
                label=f"{group} (n={mask.sum()})",
                depthshade=True,
            )
        ax.set_title(f"All Tesla groups — {subtitle} (3D)")
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        ax.set_zlabel("C3")
        ax.legend(loc="upper left", fontsize=7, frameon=True)
        fig.tight_layout()
        out_3d = out_dir / f"ALL_groups_{method}_3d.png"
        fig.savefig(out_3d, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_3d}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCA / t-SNE plots of averaged text-embedding maneuvers."
    )
    parser.add_argument(
        "--average-dir",
        default="data_bottleneck/text_embed_tesla/averaged_manoeuvres",
        help="Directory with {maneuver}.npy averages.",
    )
    parser.add_argument(
        "--out-dir",
        default="Results/text_embed_tesla_viz",
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "tsne"],
        choices=["pca", "tsne"],
        help="Projection methods (default: pca tsne).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (auto-capped per group size).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for PCA/t-SNE.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not draw short labels next to points.",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Skip the all-groups overview plots.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Only these group prefixes (default: all 7).",
    )
    parser.add_argument("--root", default=".", help="Project root.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    average_dir = Path(args.average_dir)
    if not average_dir.is_absolute():
        average_dir = root / average_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_grouped_vectors(average_dir)
    if not grouped:
        raise SystemExit(f"No averaged .npy files found under {average_dir}")

    if args.groups:
        wanted = set(args.groups)
        grouped = {k: v for k, v in grouped.items() if k in wanted}
        if not grouped:
            raise SystemExit(f"No matching groups for {args.groups}")

    print(f"Loaded groups from {average_dir}:")
    for g, (names, vecs) in grouped.items():
        print(f"  {g}: n={len(names)}, dim={vecs.shape[1]}")

    for group, (names, vectors) in grouped.items():
        plot_group(
            group=group,
            names=names,
            vectors=vectors,
            out_dir=out_dir,
            methods=list(args.methods),
            random_state=args.random_state,
            perplexity=args.perplexity,
            annotate=not args.no_annotate,
        )

    if not args.no_overview:
        plot_all_groups_overview(
            grouped=grouped,
            out_dir=out_dir,
            methods=list(args.methods),
            random_state=args.random_state,
            perplexity=args.perplexity,
        )

    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
