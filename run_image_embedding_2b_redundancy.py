#!/usr/bin/env python3
"""
Redundancy filter + kept table + latent-space plots for image_embedding_2b.

Steps:
  1. Load 2048-d vectors from chroma_db/image_embedding_2b
  2. Within-group cosine matrices + greedy removal at thresholds 90..99
  3. Write kept-count table (CSV + PNG)
  4. PCA / t-SNE 2D+3D overview (one color per maneuver group)

  source .venv/bin/activate
  python run_image_embedding_2b_redundancy.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from cosine_similarity import CosineSimilarity
from pca_baseline import GROUP_PREFIXES, THRESHOLDS
from plot_text_embed_maneuvers import (
    load_grouped_vectors_from_chroma,
    plot_all_groups_overview,
)

MODEL_NAME = "image_embedding_2b"
DEFAULT_CHROMA = Path("chroma_db/image_embedding_2b")
DEFAULT_COLLECTION = "image_embedding_2b"
DEFAULT_MATRICES = Path("cosine_similarity_matrices/image_embedding_2b")
DEFAULT_VIZ = Path("Results/image_embedding_2b_viz")
DEFAULT_TABLE_CSV = Path("Results/image_embedding_2b_kept_vs_threshold.csv")
DEFAULT_TABLE_PNG = Path("Results/image_embedding_2b_kept_vs_threshold.png")


def maneuver_group(maneuver: str) -> str:
    for prefix in GROUP_PREFIXES:
        if maneuver.startswith(prefix + "_") or maneuver == prefix:
            return prefix
    return "unknown"


def build_groups(maneuvers: list[str]) -> list[list[str]]:
    buckets: dict[str, list[str]] = {p: [] for p in GROUP_PREFIXES}
    for m in maneuvers:
        g = maneuver_group(m)
        if g in buckets:
            buckets[g].append(m)
    return [buckets[p] for p in GROUP_PREFIXES]


def load_all_embeddings(
    chroma_path: Path, collection_name: str
) -> dict[str, np.ndarray]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    result = collection.get(include=["embeddings"])
    ids = list(result.get("ids") or [])
    embeddings = result.get("embeddings")
    if embeddings is None or not ids:
        raise SystemExit(f"No embeddings in {chroma_path} / {collection_name}")
    return {
        str(i): np.asarray(e, dtype=np.float64)
        for i, e in zip(ids, embeddings)
    }


def run_cosine(
    id_to_emb: dict[str, np.ndarray],
    matrices_dir: Path,
    *,
    plot: bool,
    thresholds: range | list[int],
) -> list[dict]:
    present = sorted(id_to_emb)
    groups = build_groups(present)
    similarity_matrices: dict[int, tuple[list[str], np.ndarray]] = {}

    for idx, group in enumerate(groups, start=1):
        if len(group) < 2:
            print(f"cosine group {idx} ({GROUP_PREFIXES[idx-1]}): skip n={len(group)}")
            continue
        names = sorted(group)
        vectors = np.stack([id_to_emb[m] for m in names], axis=0)
        sim = cosine_similarity(vectors)
        similarity_matrices[idx] = (names, sim)
        print(f"cosine group {idx} ({GROUP_PREFIXES[idx-1]}): n={len(names)}")

    matrices_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        plotter = CosineSimilarity(
            "", str(matrices_dir), threshold=90, model_name=MODEL_NAME
        )
        for _idx, (names, matrix) in similarity_matrices.items():
            plotter.plot_confusion_matrix(names, matrix)

    n_universe = len(present)
    rows: list[dict] = []
    for threshold in thresholds:
        cos_sim = CosineSimilarity(
            "", str(matrices_dir), threshold=int(threshold), model_name=MODEL_NAME
        )
        cos_sim.similarity_matrices = similarity_matrices
        redundant = cos_sim.detect_redundancy()
        removed = cos_sim.remove_redundancy(redundant)
        n_removed = sum(len(v) for v in removed.values())
        n_kept = n_universe - n_removed
        row = {
            "threshold": int(threshold),
            "n_universe": n_universe,
            "n_removed": n_removed,
            "n_kept": n_kept,
            "groups_with_pairs": len(redundant),
        }
        rows.append(row)
        print(
            f"  threshold {threshold}%: removed={n_removed}  kept={n_kept}  "
            f"groups_with_pairs={len(redundant)}"
        )
    return rows


def write_kept_table(
    rows: list[dict],
    csv_path: Path,
    png_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "n_universe",
                "n_removed",
                "n_kept",
                "groups_with_pairs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    thresholds = [r["threshold"] for r in rows]
    kept = [r["n_kept"] for r in rows]
    removed = [r["n_removed"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, kept, "o-", color="#1f77b4", label="kept", linewidth=2)
    ax.plot(thresholds, removed, "s--", color="#d62728", label="removed", linewidth=1.5)
    for t, k in zip(thresholds, kept):
        ax.annotate(
            str(k),
            (t, k),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(thresholds)
    ax.set_xlabel("Cosine threshold (%)")
    ax.set_ylabel("Maneuver count")
    ax.set_title(
        f"{MODEL_NAME}: kept / removed vs threshold  "
        f"(universe={rows[0]['n_universe']})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Compact table image
    table_png = png_path.with_name(png_path.stem + "_table.png")
    cell_text = [
        [str(r["threshold"]), str(r["n_kept"]), str(r["n_removed"])] for r in rows
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["threshold %", "kept", "removed"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)
    ax.set_title(f"{MODEL_NAME} redundancy filter", pad=12)
    fig.tight_layout()
    fig.savefig(table_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {table_png}")


def run_viz(
    chroma_path: Path,
    collection: str,
    viz_dir: Path,
    methods: list[str],
) -> None:
    grouped = load_grouped_vectors_from_chroma(chroma_path, collection)
    viz_dir.mkdir(parents=True, exist_ok=True)
    print("Groups for latent plots:")
    for g, (names, vecs) in grouped.items():
        print(f"  {g}: n={len(names)} dim={vecs.shape[1]}")
    plot_all_groups_overview(
        grouped=grouped,
        out_dir=viz_dir,
        methods=methods,
        random_state=42,
        perplexity=30.0,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--matrices-dir", type=Path, default=DEFAULT_MATRICES)
    p.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ)
    p.add_argument("--table-csv", type=Path, default=DEFAULT_TABLE_CSV)
    p.add_argument("--table-png", type=Path, default=DEFAULT_TABLE_PNG)
    p.add_argument("--no-plot-matrices", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "tsne"],
        choices=["pca", "tsne"],
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=list(THRESHOLDS),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading {args.chroma_path} / {args.collection}")
    id_to_emb = load_all_embeddings(args.chroma_path, args.collection)
    print(f"n={len(id_to_emb)}  dim={next(iter(id_to_emb.values())).shape[0]}")

    rows = run_cosine(
        id_to_emb,
        args.matrices_dir,
        plot=not args.no_plot_matrices,
        thresholds=args.thresholds,
    )
    write_kept_table(rows, args.table_csv, args.table_png)

    summary = {
        "model": MODEL_NAME,
        "chroma_path": str(args.chroma_path),
        "collection": args.collection,
        "rows": rows,
    }
    summary_path = args.matrices_dir / "kept_summary.json"
    args.matrices_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")

    if not args.no_viz:
        run_viz(args.chroma_path, args.collection, args.viz_dir, list(args.methods))

    print("\nKept table:")
    print(f"{'thr%':>5}  {'kept':>5}  {'removed':>7}")
    for r in rows:
        print(f"{r['threshold']:>5}  {r['n_kept']:>5}  {r['n_removed']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
