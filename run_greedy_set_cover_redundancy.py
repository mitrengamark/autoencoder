#!/usr/bin/env python3
"""
Run Greedy Set Cover redundancy filtering on precomputed cosine matrices.

Does not recompute embeddings or modify the existing frequency-based filter.

Example:
  python run_greedy_set_cover_redundancy.py \\
    --matrices-dir cosine_similarity_matrices/text_vlm_tesla \\
    --model-name text_vlm_tesla \\
    --results-dir redundancy_results/greedy_set_cover \\
    --mat-source data_tesla \\
    --mat-out-root data_filtered_mat \\
    --export-thresholds 90 95 98
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_filtered_mat_from_removing_json import copy_or_link, index_source_mats
from greedy_set_cover_redundancy import (
    DEFAULT_THRESHOLDS,
    EXPORT_THRESHOLDS,
    coverage_metrics,
    filter_all_groups,
    kept_from_old_removed,
    load_group_matrices,
    load_old_removed,
    write_threshold_outputs,
)


def export_mats(
    *,
    kept: list[str],
    removed: list[str],
    threshold: int,
    source_mats: dict[str, Path],
    out_dir: Path,
    mode: str,
    params: dict,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        **params,
        "threshold": threshold,
        "tau": threshold / 100.0,
        "n_kept": len(kept),
        "n_removed": len(removed),
        "kept": kept,
        "removed": removed,
    }
    (out_dir / "filter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    copied = 0
    for name in kept:
        src = source_mats.get(name)
        if src is None:
            continue
        copy_or_link(src, out_dir / src.name, mode)
        copied += 1
    return copied


def write_comparison(
    *,
    groups,
    results_by_thr: dict[int, object],
    old_kept_by_thr: dict[int, set[str]],
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = out_dir / "comparison_kept_counts.csv"
    metrics_path = out_dir / "comparison_coverage_metrics.csv"
    plot_path = out_dir / "comparison_kept_vs_threshold.png"
    set_diff_path = out_dir / "set_differences.json"

    thresholds = sorted(results_by_thr)
    count_rows = []
    metric_rows = []
    set_diffs = {}

    old_kept_counts = []
    new_kept_counts = []

    for thr in thresholds:
        result = results_by_thr[thr]
        tau = thr / 100.0
        new_kept = set(result.kept)
        old_kept = old_kept_by_thr[thr]
        original = result.original_count

        count_rows.append(
            {
                "Threshold": f"{tau:.2f}",
                "Original count": original,
                "Old method kept": len(old_kept),
                "New method kept": len(new_kept),
                "Difference": len(new_kept) - len(old_kept),
            }
        )
        old_kept_counts.append(len(old_kept))
        new_kept_counts.append(len(new_kept))

        for method_name, kept in (("old_frequency", old_kept), ("greedy_set_cover", new_kept)):
            m = coverage_metrics(groups, kept, tau)
            metric_rows.append(
                {
                    "Threshold": f"{tau:.2f}",
                    "Method": method_name,
                    "Kept count": int(m["kept_count"]),
                    "Reduction pct": round(m["reduction_pct"], 4),
                    "Coverage pct": round(m["coverage_pct"], 4),
                    "Mean assignment similarity": (
                        ""
                        if np.isnan(m["mean_assignment_similarity"])
                        else round(m["mean_assignment_similarity"], 6)
                    ),
                    "Min assignment similarity": (
                        ""
                        if np.isnan(m["min_assignment_similarity"])
                        else round(m["min_assignment_similarity"], 6)
                    ),
                }
            )

        only_old = sorted(old_kept - new_kept)
        only_new = sorted(new_kept - old_kept)
        set_diffs[str(thr)] = {
            "intersection_size": len(old_kept & new_kept),
            "only_old_method": only_old,
            "only_new_method": only_new,
            "jaccard": (
                len(old_kept & new_kept) / len(old_kept | new_kept)
                if (old_kept | new_kept)
                else 1.0
            ),
        }

    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Threshold",
                "Original count",
                "Old method kept",
                "New method kept",
                "Difference",
            ],
        )
        writer.writeheader()
        writer.writerows(count_rows)

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Threshold",
                "Method",
                "Kept count",
                "Reduction pct",
                "Coverage pct",
                "Mean assignment similarity",
                "Min assignment similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    set_diff_path.write_text(
        json.dumps(set_diffs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    xs = [thr / 100.0 for thr in thresholds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, old_kept_counts, marker="o", label="Frequency removal (current)")
    ax.plot(xs, new_kept_counts, marker="s", label="Greedy set cover (new)")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Kept maneuvers")
    ax.set_title("Kept maneuvers vs cosine threshold")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return counts_path, metrics_path, plot_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--matrices-dir",
        default="cosine_similarity_matrices/text_vlm_tesla",
        help="Directory with precomputed cosine matrices + old removal JSONs.",
    )
    p.add_argument("--model-name", default="text_vlm_tesla")
    p.add_argument(
        "--results-dir",
        default="redundancy_results/greedy_set_cover",
        help="Output directory for new-method results (does not overwrite old).",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=list(DEFAULT_THRESHOLDS),
    )
    p.add_argument("--mat-source", default="data_tesla")
    p.add_argument("--mat-out-root", default="data_filtered_mat")
    p.add_argument(
        "--export-thresholds",
        nargs="+",
        type=int,
        default=list(EXPORT_THRESHOLDS),
    )
    p.add_argument(
        "--mat-link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    p.add_argument(
        "--skip-mat-export",
        action="store_true",
        help="Only compute filtering + comparison tables.",
    )
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    matrices_dir = (root / args.matrices_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    mat_source = (root / args.mat_source).resolve()
    mat_out_root = (root / args.mat_out_root).resolve()

    if not matrices_dir.is_dir():
        raise SystemExit(f"Missing matrices dir: {matrices_dir}")

    print(f"Loading matrices from {matrices_dir}")
    groups = load_group_matrices(matrices_dir, args.model_name)
    n = sum(len(g.manoeuvres) for g in groups)
    print(f"Loaded {len(groups)} groups, {n} maneuvers")

    results_by_thr = {}
    old_kept_by_thr = {}

    for thr in args.thresholds:
        print(f"Filtering threshold={thr} (tau={thr/100:.2f})")
        result = filter_all_groups(groups, thr)
        thr_dir = results_dir / f"threshold_{thr}"
        write_threshold_outputs(result, thr_dir)
        results_by_thr[thr] = result

        old_removed = load_old_removed(matrices_dir, args.model_name, thr)
        old_kept_by_thr[thr] = kept_from_old_removed(groups, old_removed)
        print(
            f"  new kept={len(result.kept)}  "
            f"old kept={len(old_kept_by_thr[thr])}  "
            f"Δ={len(result.kept) - len(old_kept_by_thr[thr])}"
        )

    counts_path, metrics_path, plot_path = write_comparison(
        groups=groups,
        results_by_thr=results_by_thr,
        old_kept_by_thr=old_kept_by_thr,
        out_dir=results_dir,
    )
    print(f"Wrote {counts_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {plot_path}")

    if not args.skip_mat_export:
        source_mats = index_source_mats(mat_source)
        if not source_mats:
            raise SystemExit(f"No .mat files in {mat_source}")
        for thr in args.export_thresholds:
            if thr not in results_by_thr:
                print(f"Skip mat export for {thr}: not in --thresholds")
                continue
            result = results_by_thr[thr]
            out_dir = mat_out_root / f"greedy_set_cover_{thr}"
            copied = export_mats(
                kept=result.kept,
                removed=result.removed,
                threshold=thr,
                source_mats=source_mats,
                out_dir=out_dir,
                mode=args.mat_link_mode,
                params={
                    "method": "greedy_set_cover",
                    "source_model": args.model_name,
                    "matrices_dir": str(matrices_dir),
                    "algorithm": "greedy_set_cover",
                    "selection": "argmax |C_i ∩ U| with mean-similarity then id tie-break",
                },
            )
            print(
                f"mat export {thr}: kept={len(result.kept)} copied={copied} → {out_dir}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
