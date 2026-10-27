#!/usr/bin/env python3
"""
Greedy Set Cover / Dominating Set redundancy filtering.

Works from precomputed within-group cosine similarity matrices. Does not modify
the existing frequency-based removal in cosine_similarity.CosineSimilarity.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from pca_baseline import GROUP_PREFIXES

DEFAULT_THRESHOLDS = tuple(range(90, 100))
EXPORT_THRESHOLDS = (90, 95, 98)


@dataclass
class GroupData:
    name: str
    manoeuvres: list[str]
    matrix: np.ndarray  # float64, square


@dataclass
class GroupFilterResult:
    group: str
    kept: list[str]
    removed: list[str]
    covers: dict[str, list[str]]  # representative -> covered originals (incl. self)
    assignments: dict[str, dict[str, float]]  # removed -> {representative, similarity}
    selection_order: list[str] = field(default_factory=list)


@dataclass
class ThresholdFilterResult:
    threshold: int
    tau: float
    groups: list[GroupFilterResult]
    original_count: int

    @property
    def kept(self) -> list[str]:
        names: list[str] = []
        for g in self.groups:
            names.extend(g.kept)
        return sorted(names)

    @property
    def removed(self) -> list[str]:
        names: list[str] = []
        for g in self.groups:
            names.extend(g.removed)
        return sorted(names)

    def kept_by_group(self) -> dict[str, list[str]]:
        return {g.group: list(g.kept) for g in self.groups}

    def removed_by_group_indexed(self) -> dict[str, list[str]]:
        """1-based group index keys, matching manoeuvres_for_removing_*.json."""
        out: dict[str, list[str]] = {}
        for idx, g in enumerate(self.groups, start=1):
            out[str(idx)] = list(g.removed)
        return out


def load_group_matrices(method_dir: Path, model_name: str) -> list[GroupData]:
    """Load within-group matrices; order follows GROUP_PREFIXES when possible."""
    by_group: dict[str, GroupData] = {}
    for labels_path in sorted(
        method_dir.glob(f"{model_name}_cosine_similarity_matrix_*_labels.json")
    ):
        stem = labels_path.name[: -len("_labels.json")]
        npy_path = method_dir / f"{stem}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"Missing matrix for {labels_path.name}: {npy_path}")
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        group = str(meta.get("group") or "")
        names = list(meta.get("manoeuvres") or [])
        matrix = np.load(npy_path).astype(np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{npy_path}: expected square matrix, got {matrix.shape}")
        if len(names) != matrix.shape[0]:
            raise ValueError(
                f"{labels_path.name}: {len(names)} labels vs matrix {matrix.shape}"
            )
        by_group[group] = GroupData(name=group, manoeuvres=names, matrix=matrix)

    ordered: list[GroupData] = []
    for prefix in GROUP_PREFIXES:
        if prefix in by_group:
            ordered.append(by_group.pop(prefix))
    # Any unexpected leftover groups (should be empty for Tesla workflow).
    for group in sorted(by_group):
        ordered.append(by_group[group])
    return ordered


def load_old_removed(
    method_dir: Path, model_name: str, threshold: int
) -> dict[str, list[str]]:
    path = method_dir / f"manoeuvres_for_removing_{threshold}_{model_name}.json"
    if not path.is_file():
        matches = sorted(method_dir.glob(f"manoeuvres_for_removing_{threshold}_*.json"))
        if not matches:
            raise FileNotFoundError(f"No old-method removal JSON for threshold {threshold}")
        path = matches[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): list(v) for k, v in data.items()}


def old_method_remove_from_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    """Replicate CosineSimilarity.remove_redundancy frequency greedy on pairs."""
    redundant = [list(p) for p in pairs]
    removed: list[str] = []
    while redundant:
        counter = Counter(maneuver for pair in redundant for maneuver in pair)
        most_common, _ = counter.most_common(1)[0]
        removed.append(most_common)
        redundant = [pair for pair in redundant if most_common not in pair]
    return removed


def _cover_sets(matrix: np.ndarray, tau: float) -> list[set[int]]:
    n = matrix.shape[0]
    covers: list[set[int]] = []
    for i in range(n):
        covered = {i}
        row = matrix[i]
        for j in range(n):
            if j != i and row[j] >= tau:
                covered.add(j)
        covers.append(covered)
    return covers


def greedy_set_cover_indices(
    matrix: np.ndarray,
    tau: float,
    labels: list[str] | None = None,
) -> tuple[list[int], list[set[int]]]:
    """
    Classic greedy set cover on redundancy neighborhoods.

    Candidates are all i not yet in S (including already-covered non-representatives).
    Score = |C_i ∩ U|; ties: higher mean similarity to newly covered others, then label.
    """
    n = matrix.shape[0]
    if n == 0:
        return [], []
    if labels is None:
        labels = [f"{i:08d}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match matrix size")

    covers = _cover_sets(matrix, tau)
    uncovered: set[int] = set(range(n))
    selected: list[int] = []
    selected_set: set[int] = set()

    while uncovered:
        best_i = -1
        best_new: set[int] = set()
        best_mean = -1.0
        best_label = ""

        for i in range(n):
            if i in selected_set:
                continue
            newly = covers[i] & uncovered
            if not newly:
                continue
            score = len(newly)
            others = newly - {i}
            if others:
                mean_sim = float(np.mean([matrix[i, j] for j in others]))
            else:
                mean_sim = -1.0  # self-only cover ranks below any positive edge cover on ties
            label = labels[i]
            better = False
            if best_i < 0:
                better = True
            elif score > len(best_new):
                better = True
            elif score == len(best_new):
                if mean_sim > best_mean + 1e-15:
                    better = True
                elif abs(mean_sim - best_mean) <= 1e-15 and label < best_label:
                    better = True
            if better:
                best_i = i
                best_new = newly
                best_mean = mean_sim
                best_label = label

        if best_i < 0:
            # Isolated leftover (should not happen: every i covers itself).
            leftover = min(uncovered)
            best_i = leftover
            best_new = {leftover}

        selected.append(best_i)
        selected_set.add(best_i)
        uncovered -= covers[best_i]

    return selected, covers


def assign_removed_to_representatives(
    names: list[str],
    matrix: np.ndarray,
    kept_idx: set[int],
    removed_idx: Iterable[int],
    tau: float,
) -> dict[str, dict[str, float]]:
    """Map each removed maneuver to the most similar kept representative (≥ tau)."""
    assignments: dict[str, dict[str, float]] = {}
    kept_list = sorted(kept_idx)
    for j in removed_idx:
        best_i = None
        best_sim = -np.inf
        for i in kept_list:
            sim = float(matrix[j, i])
            if sim >= tau and sim > best_sim:
                best_sim = sim
                best_i = i
            elif sim >= tau and abs(sim - best_sim) <= 1e-15 and best_i is not None:
                if names[i] < names[best_i]:
                    best_i = i
                    best_sim = sim
        if best_i is None:
            raise ValueError(
                f"Removed maneuver {names[j]} has no kept neighbor at tau={tau}"
            )
        assignments[names[j]] = {
            "representative": names[best_i],
            "similarity": best_sim,
        }
    return assignments


def filter_group(group: GroupData, tau: float) -> GroupFilterResult:
    names = group.manoeuvres
    matrix = group.matrix
    selected_idx, covers = greedy_set_cover_indices(matrix, tau, labels=names)
    kept_idx = set(selected_idx)
    all_idx = set(range(len(names)))
    removed_idx = sorted(all_idx - kept_idx)

    cover_names: dict[str, list[str]] = {}
    for i in selected_idx:
        covered = sorted(names[j] for j in covers[i])
        cover_names[names[i]] = covered

    assignments = assign_removed_to_representatives(
        names, matrix, kept_idx, removed_idx, tau
    )

    return GroupFilterResult(
        group=group.name,
        kept=sorted(names[i] for i in kept_idx),
        removed=[names[i] for i in removed_idx],
        covers=cover_names,
        assignments=assignments,
        selection_order=[names[i] for i in selected_idx],
    )


def filter_all_groups(
    groups: list[GroupData], threshold: int
) -> ThresholdFilterResult:
    tau = threshold / 100.0
    results = [filter_group(g, tau) for g in groups]
    original = sum(len(g.manoeuvres) for g in groups)
    return ThresholdFilterResult(
        threshold=threshold,
        tau=tau,
        groups=results,
        original_count=original,
    )


def coverage_metrics(
    groups: list[GroupData],
    kept: set[str],
    tau: float,
) -> dict[str, float]:
    """
    Shared coverage definition for old and new methods.

    A maneuver is covered if it is kept, or there exists a kept maneuver in the
    same group with cosine similarity >= tau.
    """
    original = 0
    covered = 0
    assignment_sims: list[float] = []
    removed_count = 0

    for g in groups:
        name_to_idx = {n: i for i, n in enumerate(g.manoeuvres)}
        kept_in_group = [n for n in g.manoeuvres if n in kept]
        kept_idx = [name_to_idx[n] for n in kept_in_group]
        for j, name in enumerate(g.manoeuvres):
            original += 1
            if name in kept:
                covered += 1
                continue
            removed_count += 1
            best = -np.inf
            for i in kept_idx:
                sim = float(g.matrix[j, i])
                if sim > best:
                    best = sim
            if best >= tau:
                covered += 1
                assignment_sims.append(best)

    reduction_pct = (
        100.0 * (original - len(kept)) / original if original else 0.0
    )
    coverage_pct = 100.0 * covered / original if original else 100.0
    avg_sim = float(np.mean(assignment_sims)) if assignment_sims else float("nan")
    min_sim = float(np.min(assignment_sims)) if assignment_sims else float("nan")

    return {
        "original_count": float(original),
        "kept_count": float(len(kept)),
        "removed_count": float(removed_count),
        "reduction_pct": reduction_pct,
        "coverage_pct": coverage_pct,
        "mean_assignment_similarity": avg_sim,
        "min_assignment_similarity": min_sim,
        "n_assigned_removed": float(len(assignment_sims)),
    }


def kept_from_old_removed(
    groups: list[GroupData], removed_by_index: dict[str, list[str]]
) -> set[str]:
    universe = {n for g in groups for n in g.manoeuvres}
    removed: set[str] = set()
    for vals in removed_by_index.values():
        removed.update(vals)
    return universe - removed


def result_to_serializable(result: ThresholdFilterResult) -> dict:
    groups_out = []
    for g in result.groups:
        groups_out.append(
            {
                "group": g.group,
                "n_original": len(g.kept) + len(g.removed),
                "n_kept": len(g.kept),
                "n_removed": len(g.removed),
                "kept": g.kept,
                "removed": g.removed,
                "selection_order": g.selection_order,
                "covers": g.covers,
                "assignments": g.assignments,
            }
        )
    kept = result.kept
    removed = result.removed
    return {
        "method": "greedy_set_cover",
        "threshold": result.threshold,
        "tau": result.tau,
        "original_count": result.original_count,
        "kept_count": len(kept),
        "removed_count": len(removed),
        "reduction_pct": 100.0 * len(removed) / result.original_count
        if result.original_count
        else 0.0,
        "kept": kept,
        "removed": removed,
        "groups": groups_out,
    }


def write_threshold_outputs(result: ThresholdFilterResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result_to_serializable(result)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "kept.json").write_text(
        json.dumps(
            {
                "threshold": result.threshold,
                "count": len(result.kept),
                "by_group": result.kept_by_group(),
                "all": result.kept,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "removed.json").write_text(
        json.dumps(
            {
                "threshold": result.threshold,
                "count": len(result.removed),
                "by_group_index": result.removed_by_group_indexed(),
                "assignments": {
                    name: asg
                    for g in result.groups
                    for name, asg in g.assignments.items()
                },
                "all": result.removed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    # Compatibility with existing mat-export tooling.
    (out_dir / f"manoeuvres_for_removing_{result.threshold}_greedy_set_cover.json").write_text(
        json.dumps(result.removed_by_group_indexed(), indent=4) + "\n",
        encoding="utf-8",
    )
