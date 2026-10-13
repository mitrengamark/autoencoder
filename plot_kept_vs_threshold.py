#!/usr/bin/env python3
"""
Kept-maneuver counts vs cosine threshold with 0.01%% steps (90.00 .. 99.99).

Loads saved within-group similarity matrices, then sweeps thresholds with a
*nested / monotonic* greedy removal:

  edge set at thr_high  ⊆  edge set at thr_low   (when thr_high > thr_low)

Independent greedy at each threshold is NOT monotone (smaller graphs can make
greedy remove more vertices). This script therefore sweeps from high→low,
keeps a cumulative removed set, and only grows it — guaranteeing that kept(thr)
is non-decreasing in thr.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METHODS = [
    "text_vlm_tesla",
    "text_embed_tesla",
    "text_embed_tesla_dtw",
    "text_embed_tesla_pairwise",
    "text_embed_tesla_concat",
]


def count_universe(mat_source: Path, fallback: int = 458) -> int:
    if mat_source.is_dir():
        n = len(list(mat_source.glob("*.mat")))
        if n:
            return n
    return fallback


def load_group_matrices(
    method_dir: Path, method: str
) -> list[tuple[list[str], np.ndarray]]:
    """Load all *_cosine_similarity_matrix_*_labels.json + matching .npy."""
    groups: list[tuple[list[str], np.ndarray]] = []
    for labels_path in sorted(
        method_dir.glob(f"{method}_cosine_similarity_matrix_*_labels.json")
    ):
        stem = labels_path.name[: -len("_labels.json")]
        npy_path = method_dir / f"{stem}.npy"
        if not npy_path.is_file():
            print(f"  warning: missing matrix for {labels_path.name}, skip")
            continue
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        names = list(meta.get("manoeuvres") or [])
        matrix = np.load(npy_path).astype(np.float64)
        if len(names) != matrix.shape[0]:
            raise ValueError(
                f"{labels_path.name}: {len(names)} labels vs matrix {matrix.shape}"
            )
        groups.append((names, matrix))
    return groups


def universe_size_from_groups(groups: list[tuple[list[str], np.ndarray]]) -> int:
    names: set[str] = set()
    for gnames, _ in groups:
        names.update(gnames)
    return len(names)


def build_edge_lists(
    groups: list[tuple[list[str], np.ndarray]],
) -> list[list[tuple[float, str, str]]]:
    """Per group: list of (similarity, name_i, name_j) for i<j."""
    edge_lists: list[list[tuple[float, str, str]]] = []
    for names, matrix in groups:
        edges: list[tuple[float, str, str]] = []
        n = len(names)
        for i in range(n):
            row = matrix[i]
            ni = names[i]
            for j in range(i + 1, n):
                edges.append((float(row[j]), ni, names[j]))
        # High similarity first — helps when activating edges as thr drops
        edges.sort(key=lambda e: e[0], reverse=True)
        edge_lists.append(edges)
    return edge_lists


def pick_most_common(counter: Counter[str]) -> str:
    """Deterministic tie-break: highest count, then lexicographically smallest name."""
    best_name = ""
    best_count = -1
    for name, count in counter.items():
        if count > best_count or (count == best_count and name < best_name):
            best_name = name
            best_count = count
    return best_name


def greedy_clear_edges(
    pairs: list[tuple[str, str]],
    removed: set[str],
) -> None:
    """Grow `removed` until no pair remains among survivors (CosineSimilarity greedy)."""
    active = [
        (a, b) for a, b in pairs if a not in removed and b not in removed
    ]
    while active:
        counter: Counter[str] = Counter()
        for a, b in active:
            counter[a] += 1
            counter[b] += 1
        victim = pick_most_common(counter)
        removed.add(victim)
        active = [(a, b) for a, b in active if victim not in (a, b)]


def sweep_kept_monotonic(
    groups: list[tuple[list[str], np.ndarray]],
    total: int,
    thr_start: float,
    thr_stop: float,
    thr_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monotone kept(thr): sweep thr from high→low, accumulate removals.

    Because E(thr_high) ⊆ E(thr_low), we only ever add edges / removals as thr
    decreases, so kept is non-decreasing as thr increases.
    """
    n_steps = int(round((thr_stop - thr_start) / thr_step))
    thresholds = thr_start + thr_step * np.arange(n_steps + 1)
    thresholds = np.round(thresholds, 2)
    thresholds = thresholds[thresholds <= thr_stop + 1e-9]

    edge_lists = build_edge_lists(groups)
    # Per group: cumulative removed names; pointer into sorted edge list
    removed_sets: list[set[str]] = [set() for _ in edge_lists]
    edge_idx = [0 for _ in edge_lists]

    # Map thr -> removed count while sweeping high to low
    removed_at: dict[float, int] = {}
    thr_desc = thresholds[::-1]

    for k, thr in enumerate(thr_desc):
        thr_frac = float(thr) / 100.0
        for g, edges in enumerate(edge_lists):
            # Activate newly included edges (sim >= thr_frac)
            i = edge_idx[g]
            while i < len(edges) and edges[i][0] >= thr_frac:
                i += 1
            if i == edge_idx[g]:
                # No new edges at this threshold step
                continue
            edge_idx[g] = i
            # All edges with sim >= thr among survivors; grow removed only.
            active_pairs = [
                (a, b)
                for (_s, a, b) in edges[: edge_idx[g]]
                if a not in removed_sets[g] and b not in removed_sets[g]
            ]
            if active_pairs:
                greedy_clear_edges(active_pairs, removed_sets[g])

        removed_total = sum(len(s) for s in removed_sets)
        removed_at[float(thr)] = removed_total
        if (k + 1) % 100 == 0 or k == 0 or k == len(thr_desc) - 1:
            print(
                f"    thr={thr:.2f}%  removed={removed_total}  "
                f"kept={total - removed_total}",
                flush=True,
            )

    kept = np.array(
        [total - removed_at[float(t)] for t in thresholds], dtype=np.int32
    )

    # Sanity: non-decreasing kept
    diffs = np.diff(kept)
    if np.any(diffs < 0):
        bad = int(np.argmin(diffs))
        raise RuntimeError(
            f"Internal error: kept not monotone near "
            f"{thresholds[bad]:.2f} → {thresholds[bad+1]:.2f} "
            f"({kept[bad]} → {kept[bad+1]})"
        )
    return thresholds, kept


def write_csv(
    path: Path,
    methods: list[str],
    thresholds: np.ndarray,
    series: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold_pct", *methods])
        for i, thr in enumerate(thresholds):
            writer.writerow(
                [f"{thr:.2f}", *[int(series[m][i]) for m in methods]]
            )
    print(f"Wrote {path}")


def plot_lines(
    path: Path,
    methods: list[str],
    thresholds: np.ndarray,
    series: dict[str, np.ndarray],
    total: int,
    thr_step: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in methods:
        ax.plot(thresholds, series[method], linewidth=1.8, label=method)
    ax.set_xlabel("Cosine similarity threshold (%)")
    ax.set_ylabel("Kept maneuvers")
    ax.set_title(
        f"Kept maneuvers vs threshold "
        f"(step={thr_step:.2f}%, monotone nested greedy, universe={total})"
    )
    ax.set_xlim(float(thresholds[0]), float(thresholds[-1]))
    ax.set_ylim(0, total * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fine-grained (0.01%%) kept-vs-threshold curves from saved "
            "cosine similarity matrices."
        )
    )
    p.add_argument(
        "--matrices-root",
        default="cosine_similarity_matrices",
    )
    p.add_argument(
        "--mat-source",
        default="data_tesla",
        help="Universe size from .mat count (fallback 458).",
    )
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--thr-start", type=float, default=90.0)
    p.add_argument("--thr-stop", type=float, default=99.99)
    p.add_argument(
        "--thr-step",
        type=float,
        default=0.01,
        help="Threshold step in percent points (default: 0.01 = századonként).",
    )
    p.add_argument("--out-png", default="Results/kept_vs_threshold.png")
    p.add_argument("--out-csv", default="Results/kept_vs_threshold.csv")
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    matrices_root = resolve(args.matrices_root)
    total_fallback = count_universe(resolve(args.mat_source))

    series: dict[str, np.ndarray] = {}
    thresholds_ref: np.ndarray | None = None
    methods_ok: list[str] = []

    for method in args.methods:
        method_dir = matrices_root / method
        print(f"=== {method} ===", flush=True)
        if not method_dir.is_dir():
            print(f"  missing dir {method_dir}, skip")
            continue
        groups = load_group_matrices(method_dir, method)
        if not groups:
            print("  no similarity matrices found, skip")
            continue
        n_in_matrices = universe_size_from_groups(groups)
        # Prefer global mat universe; warn if matrices cover fewer ids
        total = total_fallback
        if n_in_matrices != total:
            print(
                f"  note: matrices cover {n_in_matrices} maneuvers, "
                f"universe={total} (kept = universe - removed)"
            )
        print(
            f"  groups={len(groups)}  sweep "
            f"{args.thr_start:.2f}..{args.thr_stop:.2f} "
            f"step={args.thr_step:.2f}",
            flush=True,
        )
        thresholds, kept = sweep_kept_monotonic(
            groups,
            total=total,
            thr_start=args.thr_start,
            thr_stop=args.thr_stop,
            thr_step=args.thr_step,
        )
        if thresholds_ref is None:
            thresholds_ref = thresholds
        elif not np.allclose(thresholds_ref, thresholds):
            raise RuntimeError(f"Threshold grid mismatch for {method}")
        series[method] = kept
        methods_ok.append(method)
        print(
            f"  done: kept@90.00={kept[0]}  "
            f"kept@95.00={kept[np.argmin(np.abs(thresholds - 95.0))]}  "
            f"kept@end={kept[-1]}",
            flush=True,
        )

    if not methods_ok or thresholds_ref is None:
        raise SystemExit("No methods produced data.")

    write_csv(resolve(args.out_csv), methods_ok, thresholds_ref, series)
    plot_lines(
        resolve(args.out_png),
        methods_ok,
        thresholds_ref,
        series,
        total_fallback,
        args.thr_step,
    )


if __name__ == "__main__":
    main()
