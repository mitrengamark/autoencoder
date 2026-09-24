#!/usr/bin/env python3
"""
Within-group cosine similarity + redundancy removal for text-embedding averages.

Uses data_bottleneck/text_embed_tesla/averaged_manoeuvres (skip-2500 means)
and the same CosineSimilarity pipeline as the AE/PCA baselines (thresholds 90-99).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from clustering_baseline import build_manoeuvre_groups_from_averaged
from cosine_similarity import CosineSimilarity
from pca_baseline import THRESHOLDS


def run_cosine_pipeline(
    average_dir: str,
    save_dir: str,
    model_name: str,
    thresholds: range | list[int],
    plot: bool,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    manoeuvre_groups = build_manoeuvre_groups_from_averaged(average_dir)
    sizes = [len(g) for g in manoeuvre_groups]
    print(f"Groups: {sizes} (total {sum(sizes)}) from {average_dir}")

    for i, threshold in enumerate(thresholds):
        cos_sim = CosineSimilarity(
            average_dir,
            save_dir,
            threshold=threshold,
            model_name=model_name,
        )
        cos_sim.compute_cosine_similarity_within_groups(
            manoeuvre_groups, plot=(plot and i == 0)
        )
        redundant_pairs = cos_sim.detect_redundancy()
        removed = cos_sim.remove_redundancy(redundant_pairs)
        n_removed = sum(len(v) for v in removed.values())
        print(
            f"Threshold {threshold}%: "
            f"groups_with_pairs={len(redundant_pairs)}, removed={n_removed}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cosine similarity redundancy filter on text-embed maneuver averages."
    )
    p.add_argument(
        "--average-dir",
        default="data_bottleneck/text_embed_tesla/averaged_manoeuvres",
        help="Directory with {maneuver}.npy averages.",
    )
    p.add_argument(
        "--save-dir",
        default="cosine_similarity_matrices/text_embed_tesla",
        help="Output dir for heatmaps + manoeuvres_for_removing_*.json.",
    )
    p.add_argument(
        "--model-name",
        default="text_embed_tesla",
        help="Name used in output JSON filenames.",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=list(THRESHOLDS),
        help="Similarity thresholds in percent (default: 90..99).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip heatmap PNGs (still writes JSON for all thresholds).",
    )
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    average_dir = Path(args.average_dir)
    save_dir = Path(args.save_dir)
    if not average_dir.is_absolute():
        average_dir = root / average_dir
    if not save_dir.is_absolute():
        save_dir = root / save_dir

    if not average_dir.is_dir():
        raise SystemExit(f"Average dir not found: {average_dir}")

    run_cosine_pipeline(
        average_dir=str(average_dir),
        save_dir=str(save_dir),
        model_name=args.model_name,
        thresholds=args.thresholds,
        plot=not args.no_plot,
    )
    print(f"Done. Outputs under {save_dir}")


if __name__ == "__main__":
    main()
