#!/usr/bin/env python3
"""
Within-group trajectory similarity for text embeddings (vs single mean-vector cosine).

Two methods (both compared within the 7 maneuver groups):

1) pairwise — exact mean of all row-vs-row cosines between two maneuvers.
   For unit-normalized row embeddings, mean_{i,j} cos(e_i, f_j) = μ_A · μ_B
   where μ = mean of L2-normalized rows. Computed by streaming Chroma (no T×T matrix).

2) dtw — DTW on a time-subsampled embedding path with local cost (1 - cos).
   Similarity = mean cosine along the optimal warping path (1 - DTW/path_length).

Outputs match the CosineSimilarity layout (heatmap + .npy/.csv/_labels.json +
manoeuvres_for_removing_{threshold}_{model}.json for 90..99).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from cosine_similarity import CosineSimilarity
from pca_baseline import GROUP_PREFIXES, THRESHOLDS


def build_groups_from_names(names: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {p: [] for p in GROUP_PREFIXES}
    for manoeuvre in sorted(names):
        for prefix in GROUP_PREFIXES:
            if manoeuvre.startswith(prefix + "_") or manoeuvre == prefix:
                groups[prefix].append(manoeuvre)
                break
    return [groups[p] for p in GROUP_PREFIXES]


def load_completed_maneuvers(progress_path: Path) -> list[str]:
    with progress_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    maneuvers = []
    for name in sorted(data.get("completed_files", [])):
        if name.endswith("_combined.txt"):
            maneuvers.append(name.replace("_combined.txt", ""))
    return maneuvers


def count_text_rows(texts_root: Path, maneuver: str) -> int:
    txt_path = texts_root / f"{maneuver}_combined.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing text file for row count: {txt_path}")
    with txt_path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def subsample_indices(n_rows: int, skip_rows: int, n_points: int) -> np.ndarray:
    """Evenly spaced row indices in [skip_rows, n_rows)."""
    if n_rows <= skip_rows:
        raise ValueError(f"n_rows={n_rows} <= skip_rows={skip_rows}")
    usable = n_rows - skip_rows
    if n_points >= usable:
        return np.arange(skip_rows, n_rows, dtype=np.int64)
    # linspace over usable span, inclusive of both ends
    offsets = np.linspace(0, usable - 1, n_points)
    return (skip_rows + np.round(offsets).astype(np.int64)).astype(np.int64)


def l2_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def stream_maneuver_features(
    collection,
    maneuver: str,
    n_rows: int,
    skip_rows: int,
    batch_size: int,
    subsample_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Stream row embeddings after skip_rows.

    Returns:
        mu: mean of L2-normalized rows (for exact average pairwise cosine), shape (D,)
        traj: subsampled raw embeddings in time order, shape (L, D)
        n_used: number of rows that entered mu
    """
    want = {int(i): k for k, i in enumerate(subsample_idx.tolist())}
    traj = np.empty((len(subsample_idx), 0), dtype=np.float32)  # filled after dim known
    traj_filled = np.zeros(len(subsample_idx), dtype=bool)

    dim: int | None = None
    unit_sum: np.ndarray | None = None
    n_used = 0
    missing = 0

    for batch_start in range(skip_rows, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)
        ids = [f"{maneuver}__{i}" for i in range(batch_start, batch_end)]
        result = collection.get(ids=ids, include=["embeddings"])
        got_ids = result.get("ids") or []
        raw_embeddings = result.get("embeddings")
        if raw_embeddings is None:
            continue
        embeddings = list(raw_embeddings)
        if not embeddings:
            continue

        if len(got_ids) != len(ids):
            missing += len(ids) - len(got_ids)

        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{maneuver}: unexpected embedding shape {arr.shape}")

        if dim is None:
            dim = arr.shape[1]
            unit_sum = np.zeros(dim, dtype=np.float64)
            traj = np.zeros((len(subsample_idx), dim), dtype=np.float32)

        # Map returned ids back to row indices (order follows request when all present)
        id_to_row = {}
        for gid in got_ids:
            # "{maneuver}__{idx}"
            idx_str = gid.rsplit("__", 1)[-1]
            id_to_row[gid] = int(idx_str)

        # Rebuild array aligned to got_ids order
        for local_i, gid in enumerate(got_ids):
            row_idx = id_to_row[gid]
            vec = arr[local_i]
            # unit contribution to mu
            nrm = float(np.linalg.norm(vec))
            if nrm > 0:
                unit_sum += vec / nrm
                n_used += 1
            if row_idx in want:
                traj[want[row_idx]] = vec.astype(np.float32)
                traj_filled[want[row_idx]] = True

    if n_used == 0 or unit_sum is None:
        raise RuntimeError(
            f"{maneuver}: no embeddings found for rows [{skip_rows}, {n_rows})"
        )

    if missing:
        print(
            f"  warning: {maneuver}: {missing} missing ids "
            f"(used {n_used} for pairwise mu)",
            file=sys.stderr,
        )

    if not traj_filled.all():
        n_miss = int((~traj_filled).sum())
        # Drop missing subsample slots (keep time order of filled ones)
        print(
            f"  warning: {maneuver}: {n_miss}/{len(subsample_idx)} "
            f"subsample points missing — dropping them",
            file=sys.stderr,
        )
        traj = traj[traj_filled]

    mu = (unit_sum / n_used).astype(np.float64)
    return mu, traj, n_used


def pairwise_similarity_matrix(mus: list[np.ndarray]) -> np.ndarray:
    """S[i,j] = μ_i · μ_j = exact mean row-vs-row cosine; diagonal forced to 1."""
    M = np.stack(mus, axis=0)  # [N, D]
    S = M @ M.T
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float64)


@njit(cache=True)
def _dtw_mean_cosine(sim: np.ndarray) -> float:
    """
    DTW on cost = 1 - sim; return mean cosine along optimal path.
    sim: [n, m] cosine matrix between trajectory points.
    """
    n, m = sim.shape
    # DP of cumulative cost; also track path length
    # Use flat arrays for numba simplicity
    inf = 1.0e30
    dp = np.empty((n + 1) * (m + 1), dtype=np.float64)
    pl = np.empty((n + 1) * (m + 1), dtype=np.int64)
    for k in range(dp.size):
        dp[k] = inf
        pl[k] = 0
    dp[0] = 0.0
    pl[0] = 0

    def idx(i, j):
        return i * (m + 1) + j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 1.0 - sim[i - 1, j - 1]
            # predecessors: (i-1,j), (i,j-1), (i-1,j-1)
            best = dp[idx(i - 1, j)]
            best_len = pl[idx(i - 1, j)]
            left = dp[idx(i, j - 1)]
            if left < best:
                best = left
                best_len = pl[idx(i, j - 1)]
            diag = dp[idx(i - 1, j - 1)]
            if diag < best:
                best = diag
                best_len = pl[idx(i - 1, j - 1)]
            dp[idx(i, j)] = best + c
            pl[idx(i, j)] = best_len + 1

    total_cost = dp[idx(n, m)]
    path_len = pl[idx(n, m)]
    if path_len <= 0:
        return 0.0
    # mean cost along path = 1 - mean cosine
    return 1.0 - (total_cost / path_len)


def dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-DTW similarity in [~0, 1] between trajectories [T,D]."""
    a_u = l2_normalize_rows(a.astype(np.float64))
    b_u = l2_normalize_rows(b.astype(np.float64))
    sim = a_u @ b_u.T
    # clip tiny numerical drift
    sim = np.clip(sim, -1.0, 1.0)
    return float(_dtw_mean_cosine(sim))


def dtw_similarity_matrix(trajectories: list[np.ndarray]) -> np.ndarray:
    n = len(trajectories)
    S = np.eye(n, dtype=np.float64)
    total_pairs = n * (n - 1) // 2
    done = 0
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            s = dtw_similarity(trajectories[i], trajectories[j])
            S[i, j] = s
            S[j, i] = s
            done += 1
            if done % 500 == 0 or done == total_pairs:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (total_pairs - done) / max(rate, 1e-6)
                print(
                    f"    DTW pairs {done}/{total_pairs} "
                    f"({rate:.1f}/s, ETA {eta:.0f}s)",
                    flush=True,
                )
    return S


def group_name_from_first(manoeuvre: str) -> str:
    for prefix in GROUP_PREFIXES:
        if manoeuvre.startswith(prefix + "_") or manoeuvre == prefix:
            return prefix
    return "ismeretlen_csoport"


def save_and_filter(
    similarity_matrices: dict[int, tuple[list[str], np.ndarray]],
    save_dir: str,
    model_name: str,
    thresholds: list[int],
    plot: bool,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # Plot once using CosineSimilarity helpers
    if plot:
        plotter = CosineSimilarity(
            directory="",
            save_dir=save_dir,
            threshold=thresholds[0],
            model_name=model_name,
        )
        for _idx, (names, matrix) in similarity_matrices.items():
            plotter.plot_confusion_matrix(names, matrix)

    for threshold in thresholds:
        cos_sim = CosineSimilarity(
            directory="",
            save_dir=save_dir,
            threshold=threshold,
            model_name=model_name,
        )
        cos_sim.similarity_matrices = similarity_matrices
        redundant = cos_sim.detect_redundancy()
        removed = cos_sim.remove_redundancy(redundant)
        n_removed = sum(len(v) for v in removed.values())
        print(
            f"  threshold {threshold}%: "
            f"groups_with_pairs={len(redundant)}, removed={n_removed}"
        )


def summarize_offdiag(names: list[str], S: np.ndarray) -> str:
    n = S.shape[0]
    if n < 2:
        return "n<2"
    mask = ~np.eye(n, dtype=bool)
    vals = S[mask]
    return (
        f"n={n} offdiag mean={vals.mean():.4f} "
        f"min={vals.min():.4f} max={vals.max():.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Within-group pairwise-average and/or DTW similarity "
            "on text-embedding trajectories."
        )
    )
    p.add_argument("--chroma-path", default="chroma_db/tesla")
    p.add_argument("--collection", default="vehicle_states")
    p.add_argument("--texts-root", default="texts/data3")
    p.add_argument(
        "--average-dir",
        default="data_bottleneck/text_embed_tesla/averaged_manoeuvres",
        help="Used only to discover which maneuvers exist / group membership.",
    )
    p.add_argument(
        "--cache-dir",
        default="data_bottleneck/text_embed_tesla/trajectory_cache",
        help="Cache for μ vectors + subsampled trajectories.",
    )
    p.add_argument(
        "--pairwise-save-dir",
        default="cosine_similarity_matrices/text_embed_tesla_pairwise",
    )
    p.add_argument(
        "--dtw-save-dir",
        default="cosine_similarity_matrices/text_embed_tesla_dtw",
    )
    p.add_argument("--pairwise-model-name", default="text_embed_tesla_pairwise")
    p.add_argument("--dtw-model-name", default="text_embed_tesla_dtw")
    p.add_argument("--skip-rows", type=int, default=2500)
    p.add_argument(
        "--subsample",
        type=int,
        default=64,
        help="Number of evenly spaced points kept for DTW (default: 64).",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["pairwise", "dtw", "both"],
        default=["both"],
    )
    p.add_argument("--thresholds", nargs="+", type=int, default=list(THRESHOLDS))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--limit", type=int, help="Smoke-test: first N maneuvers only.")
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Re-stream from Chroma even if cache files exist.",
    )
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit(
            "Missing chromadb. Use .venv-embeddings / "
            "pip install -r requirements-embeddings.txt"
        ) from exc

    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    def resolve(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    chroma_path = resolve(args.chroma_path)
    texts_root = resolve(args.texts_root)
    average_dir = resolve(args.average_dir)
    cache_dir = resolve(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    methods = set(args.methods)
    if "both" in methods:
        methods = {"pairwise", "dtw"}

    progress_path = chroma_path / ".embed_progress.json"
    if progress_path.is_file():
        maneuvers = load_completed_maneuvers(progress_path)
    elif average_dir.is_dir():
        maneuvers = sorted(
            f[:-4] for f in os.listdir(average_dir) if f.endswith(".npy")
        )
    else:
        raise SystemExit("Need chroma progress file or average-dir with .npy files.")

    if args.limit is not None:
        maneuvers = maneuvers[: args.limit]

    groups = build_groups_from_names(maneuvers)
    print(
        "Groups:",
        [len(g) for g in groups],
        f"(total {sum(len(g) for g in groups)})",
    )
    print(f"Methods: {sorted(methods)} | subsample={args.subsample} | skip={args.skip_rows}")

    # --- load or build cache ---
    mu_by_name: dict[str, np.ndarray] = {}
    traj_by_name: dict[str, np.ndarray] = {}

    need_stream = []
    for m in maneuvers:
        mu_path = cache_dir / f"{m}__mu.npy"
        traj_path = cache_dir / f"{m}__traj{args.subsample}.npy"
        have_mu = mu_path.is_file() and not args.rebuild_cache
        have_traj = traj_path.is_file() and not args.rebuild_cache
        if "pairwise" in methods and have_mu:
            mu_by_name[m] = np.load(mu_path)
        if "dtw" in methods and have_traj:
            traj_by_name[m] = np.load(traj_path)
        need_mu = "pairwise" in methods and m not in mu_by_name
        need_traj = "dtw" in methods and m not in traj_by_name
        # If pairwise needs stream, also refresh traj when dtw requested (same pass)
        if need_mu or need_traj:
            need_stream.append(m)

    if need_stream:
        print(
            f"Streaming {len(need_stream)} maneuvers from Chroma "
            f"(cache hits: mu={len(mu_by_name)}, traj={len(traj_by_name)}) ..."
        )
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(args.collection)
        t0 = time.time()
        for idx, maneuver in enumerate(need_stream, start=1):
            n_rows = count_text_rows(texts_root, maneuver)
            sub_idx = subsample_indices(n_rows, args.skip_rows, args.subsample)
            file_t0 = time.time()
            mu, traj, n_used = stream_maneuver_features(
                collection,
                maneuver,
                n_rows=n_rows,
                skip_rows=args.skip_rows,
                batch_size=args.batch_size,
                subsample_idx=sub_idx,
            )
            np.save(cache_dir / f"{maneuver}__mu.npy", mu.astype(np.float32))
            np.save(
                cache_dir / f"{maneuver}__traj{args.subsample}.npy",
                traj.astype(np.float32),
            )
            if "pairwise" in methods:
                mu_by_name[maneuver] = mu
            if "dtw" in methods:
                traj_by_name[maneuver] = traj
            print(
                f"  [{idx}/{len(need_stream)}] {maneuver}: "
                f"rows={n_rows}, used={n_used}, traj={traj.shape[0]}, "
                f"{time.time() - file_t0:.1f}s",
                flush=True,
            )
        print(f"Cache build done in {time.time() - t0:.1f}s under {cache_dir}")
    else:
        print(f"Using full cache under {cache_dir}")

    # Warm up numba before timed DTW loops
    if "dtw" in methods and traj_by_name:
        sample = next(iter(traj_by_name.values()))
        _ = dtw_similarity(sample[:8], sample[:8])

    # --- pairwise ---
    if "pairwise" in methods:
        print("\n=== pairwise (mean row-vs-row cosine) ===")
        sim_mats: dict[int, tuple[list[str], np.ndarray]] = {}
        for g_idx, group in enumerate(groups, start=1):
            valid = [m for m in group if m in mu_by_name]
            if len(valid) < 2:
                print(f"  group {g_idx} ({group_name_from_first(group[0]) if group else '?'}): skip")
                continue
            mus = [mu_by_name[m] for m in valid]
            S = pairwise_similarity_matrix(mus)
            sim_mats[g_idx] = (valid, S)
            gname = group_name_from_first(valid[0])
            print(f"  {gname}: {summarize_offdiag(valid, S)}")

        save_and_filter(
            sim_mats,
            save_dir=str(resolve(args.pairwise_save_dir)),
            model_name=args.pairwise_model_name,
            thresholds=args.thresholds,
            plot=not args.no_plot,
        )
        print(f"Pairwise outputs -> {resolve(args.pairwise_save_dir)}")

    # --- dtw ---
    if "dtw" in methods:
        print("\n=== dtw (subsampled trajectory) ===")
        sim_mats = {}
        for g_idx, group in enumerate(groups, start=1):
            valid = [m for m in group if m in traj_by_name]
            if len(valid) < 2:
                print(f"  group {g_idx}: skip")
                continue
            gname = group_name_from_first(valid[0])
            print(f"  {gname}: computing DTW for {len(valid)} maneuvers ...", flush=True)
            trajs = [traj_by_name[m] for m in valid]
            t0 = time.time()
            S = dtw_similarity_matrix(trajs)
            print(f"  {gname}: done in {time.time() - t0:.1f}s | {summarize_offdiag(valid, S)}")
            sim_mats[g_idx] = (valid, S)

        save_and_filter(
            sim_mats,
            save_dir=str(resolve(args.dtw_save_dir)),
            model_name=args.dtw_model_name,
            thresholds=args.thresholds,
            plot=not args.no_plot,
        )
        print(f"DTW outputs -> {resolve(args.dtw_save_dir)}")

    print("Done.")


if __name__ == "__main__":
    main()
