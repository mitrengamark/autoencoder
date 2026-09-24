#!/usr/bin/env python3
"""
Within-group cosine similarity on *concatenated* row embeddings.

Conceptually each maneuver is one vector of length T*D
(T = n_rows - skip_rows ≈ 8305, D = 4096): rows laid end-to-end.

We never materialize the T*D vector in RAM. For aligned trajectories:

    concat_cos(A,B) = sum_t <e_t, f_t>  /  (||A||_2 * ||B||_2)
    ||A||_2^2       = sum_t ||e_t||_2^2

Pipeline (memory-conscious):
  1) Stream each maneuver from Chroma once → float16 memmap on disk (~68 MiB each)
  2) Close Chroma (releases the huge HNSW/embedding cache)
  3) Per group, accumulate Gram + norms over small time-batches read from memmaps

Memory protection:
  - system MemAvailable watchdog (stricter during compute; looser only while Chroma
    must stay open for export — Chroma alone maps tens of GiB)
  - oom_score_adj prefers killing this job over SSH
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from cosine_similarity import CosineSimilarity
from embed_texts_to_chroma import (
    DEFAULT_MEM_POLL_SEC,
    apply_oom_victim_score,
    read_mem_available_bytes,
)
from pca_baseline import GROUP_PREFIXES, THRESHOLDS

# Mutable floor for the single system-memory watchdog thread.
_stop_requested = threading.Event()
_watchdog_min_bytes: list[int] = [0]  # 0 = disabled


def start_mutable_memory_watchdog(poll_sec: float = DEFAULT_MEM_POLL_SEC) -> None:
    """One daemon thread; threshold updated via set_watchdog_min_avail_gb()."""

    def _on_sigterm(signum, frame) -> None:  # noqa: ARG001
        _stop_requested.set()

    signal.signal(signal.SIGTERM, _on_sigterm)

    def _loop() -> None:
        while not _stop_requested.is_set():
            floor = _watchdog_min_bytes[0]
            if floor > 0:
                avail = read_mem_available_bytes()
                if avail is not None and avail < floor:
                    msg = (
                        f"WATCHDOG: system MemAvailable={avail / (1024**3):.2f} GiB "
                        f"< {floor / (1024**3):.2f} GiB — stopping to protect SSH/machine"
                    )
                    print(msg, flush=True)
                    print(msg, file=sys.stderr, flush=True)
                    _stop_requested.set()
                    try:
                        os.kill(os.getpid(), signal.SIGTERM)
                    except OSError:
                        os._exit(2)
                    return
            time.sleep(poll_sec)

    threading.Thread(target=_loop, name="system-mem-watchdog", daemon=True).start()


def set_watchdog_min_avail_gb(min_avail_gb: float) -> None:
    if min_avail_gb <= 0:
        _watchdog_min_bytes[0] = 0
        print("Memory watchdog disabled", flush=True)
        return
    _watchdog_min_bytes[0] = int(min_avail_gb * (1024**3))
    print(
        f"Memory watchdog floor: MemAvailable < {min_avail_gb:.2f} GiB",
        flush=True,
    )


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
    out = []
    for name in sorted(data.get("completed_files", [])):
        if name.endswith("_combined.txt"):
            out.append(name.replace("_combined.txt", ""))
    return out


def count_text_rows(texts_root: Path, maneuver: str) -> int:
    txt_path = texts_root / f"{maneuver}_combined.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing text file: {txt_path}")
    with txt_path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def group_name(manoeuvre: str) -> str:
    for prefix in GROUP_PREFIXES:
        if manoeuvre.startswith(prefix + "_") or manoeuvre == prefix:
            return prefix
    return "ismeretlen_csoport"


def mem_avail_gb() -> float | None:
    avail = read_mem_available_bytes()
    return None if avail is None else avail / (1024**3)


def cache_path(cache_dir: Path, maneuver: str, t_len: int, dim: int) -> Path:
    return cache_dir / f"{maneuver}__T{t_len}_D{dim}.f16.npy"


def export_maneuver_f16(
    collection,
    maneuver: str,
    n_rows: int,
    skip_rows: int,
    batch_size: int,
    out_path: Path,
) -> tuple[int, int]:
    """Stream rows [skip_rows, n_rows) from Chroma into a float16 .npy memmap."""
    t_len = n_rows - skip_rows
    dim: int | None = None
    mm: np.memmap | None = None
    written = 0
    missing = 0

    for batch_start in range(skip_rows, n_rows, batch_size):
        if _stop_requested.is_set():
            if mm is not None:
                mm.flush()
                del mm
            if out_path.is_file():
                out_path.unlink(missing_ok=True)
            raise SystemExit(
                f"Stopped by memory watchdog while exporting {maneuver} "
                f"(wrote {written}/{t_len} rows)"
            )

        batch_end = min(batch_start + batch_size, n_rows)
        ids = [f"{maneuver}__{i}" for i in range(batch_start, batch_end)]
        result = collection.get(ids=ids, include=["embeddings"])
        got_ids = result.get("ids") or []
        raw = result.get("embeddings")
        if raw is None or not got_ids:
            missing += len(ids)
            continue

        emb = np.asarray(list(raw), dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"{maneuver}: bad emb shape {emb.shape}")

        if dim is None:
            dim = int(emb.shape[1])
            # allocate full file up front
            mm = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float16, shape=(t_len, dim)
            )

        # Align to requested ids (Chroma may drop missing)
        id_to_vec = {gid: emb[i] for i, gid in enumerate(got_ids)}
        for req_id in ids:
            dest = written
            if dest >= t_len:
                break
            vec = id_to_vec.get(req_id)
            if vec is None:
                missing += 1
                mm[dest] = 0  # type: ignore[index]
            else:
                mm[dest] = vec.astype(np.float16)  # type: ignore[index]
            written += 1

    if mm is None or dim is None or written == 0:
        raise RuntimeError(f"{maneuver}: no embeddings exported")

    mm.flush()
    del mm
    if written != t_len:
        print(
            f"  warning: {maneuver}: wrote {written}/{t_len} "
            f"(missing_ids≈{missing})",
            file=sys.stderr,
            flush=True,
        )
    return written, dim


def ensure_trajectory_cache(
    collection,
    maneuvers: list[str],
    texts_root: Path,
    cache_dir: Path,
    skip_rows: int,
    batch_size: int,
    rebuild: bool,
) -> dict[str, Path]:
    """Export missing float16 trajectories; return maneuver -> path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    # Probe dim once
    probe_m = maneuvers[0]
    n0 = count_text_rows(texts_root, probe_m)
    probe = collection.get(ids=[f"{probe_m}__{skip_rows}"], include=["embeddings"])
    raw = probe.get("embeddings")
    if raw is None:
        raise RuntimeError(f"Cannot probe dim via {probe_m}__{skip_rows}")
    raw_embs = list(raw)
    if not raw_embs:
        raise RuntimeError(f"Cannot probe dim via {probe_m}__{skip_rows}")
    dim = int(np.asarray(raw_embs[0]).shape[0])

    need: list[str] = []
    for m in maneuvers:
        n_rows = count_text_rows(texts_root, m)
        t_len = n_rows - skip_rows
        if t_len <= 0:
            raise ValueError(f"{m}: n_rows={n_rows} <= skip_rows={skip_rows}")
        path = cache_path(cache_dir, m, t_len, dim)
        if path.is_file() and not rebuild:
            # quick shape check
            arr = np.load(path, mmap_mode="r")
            if arr.shape == (t_len, dim):
                paths[m] = path
                continue
        need.append(m)

    print(
        f"Trajectory cache: have={len(paths)}, need_export={len(need)}, "
        f"dim={dim}, skip={skip_rows}, MemAvail={mem_avail_gb():.1f} GiB",
        flush=True,
    )

    t0 = time.time()
    for idx, m in enumerate(need, start=1):
        n_rows = count_text_rows(texts_root, m)
        t_len = n_rows - skip_rows
        path = cache_path(cache_dir, m, t_len, dim)
        if path.is_file():
            path.unlink()
        file_t0 = time.time()
        written, got_dim = export_maneuver_f16(
            collection, m, n_rows, skip_rows, batch_size, path
        )
        if got_dim != dim:
            raise RuntimeError(f"{m}: dim {got_dim} != expected {dim}")
        paths[m] = path
        print(
            f"  [{idx}/{len(need)}] {m}: T={written}, "
            f"{time.time() - file_t0:.1f}s -> {path.name} "
            f"(MemAvail={mem_avail_gb():.1f} GiB)",
            flush=True,
        )

    print(f"Cache ready in {time.time() - t0:.1f}s ({len(paths)} files)", flush=True)
    return paths


def choose_time_batch(n_maneuvers: int, dim: int, max_sheet_mb: float) -> int:
    max_bytes = max_sheet_mb * (1024**2)
    return max(1, int(max_bytes // (max(n_maneuvers, 1) * dim * 4)))


def concat_cosine_from_memmaps(
    paths: list[Path],
    max_sheet_mb: float,
    progress_every: int = 100,
) -> np.ndarray:
    """Accumulate concat-cosine Gram from float16 memmaps (one sheet at a time)."""
    maps = [np.load(p, mmap_mode="r") for p in paths]
    n = len(maps)
    t_len, dim = maps[0].shape
    for i, mm in enumerate(maps):
        if mm.shape != (t_len, dim):
            raise ValueError(
                f"Shape mismatch: {paths[i].name} {mm.shape} vs {(t_len, dim)}"
            )

    B = choose_time_batch(n, dim, max_sheet_mb)
    # Cap B so float32 sheet stays manageable
    B = min(B, t_len)
    G = np.zeros((n, n), dtype=np.float64)
    norm_sq = np.zeros(n, dtype=np.float64)

    print(
        f"  concat from memmap: n={n}, T={t_len}, D={dim}, "
        f"time_batch={B} (~{n * B * dim * 4 / (1024**2):.1f} MiB/sheet), "
        f"MemAvail={mem_avail_gb():.1f} GiB",
        flush=True,
    )

    t0 = time.time()
    rows_done = 0
    batch_i = 0
    for t in range(0, t_len, B):
        if _stop_requested.is_set():
            raise SystemExit(
                f"Stopped by memory watchdog during concat "
                f"(rows_done={rows_done}/{t_len})"
            )
        t_end = min(t + B, t_len)
        b = t_end - t
        # Sheet in float32 for stable gemm
        E = np.empty((n, b, dim), dtype=np.float32)
        for i, mm in enumerate(maps):
            E[i] = np.asarray(mm[t:t_end], dtype=np.float32)

        flat = E.reshape(n, b * dim)
        G += (flat @ flat.T).astype(np.float64)
        norm_sq += np.einsum("nd,nd->n", flat, flat, dtype=np.float64)
        rows_done += b
        batch_i += 1
        del E, flat

        if batch_i % progress_every == 0 or rows_done >= t_len:
            elapsed = time.time() - t0
            rate = rows_done / max(elapsed, 1e-6)
            eta = (t_len - rows_done) / max(rate, 1e-6)
            print(
                f"    rows {rows_done}/{t_len} "
                f"({rate:.0f} rows/s, ETA {eta:.0f}s) "
                f"MemAvail={mem_avail_gb():.1f} GiB",
                flush=True,
            )

    # release memmaps
    del maps

    norms = np.sqrt(np.maximum(norm_sq, 1e-30))
    S = G / np.outer(norms, norms)
    np.fill_diagonal(S, 1.0)
    np.clip(S, -1.0, 1.0, out=S)
    return S


def summarize_offdiag(S: np.ndarray) -> str:
    n = S.shape[0]
    if n < 2:
        return "n<2"
    mask = ~np.eye(n, dtype=bool)
    v = S[mask]
    return f"n={n} offdiag mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}"


def save_and_filter(
    similarity_matrices: dict[int, tuple[list[str], np.ndarray]],
    save_dir: str,
    model_name: str,
    thresholds: list[int],
    plot: bool,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if plot:
        plotter = CosineSimilarity(
            "", save_dir, threshold=thresholds[0], model_name=model_name
        )
        for _idx, (names, matrix) in similarity_matrices.items():
            plotter.plot_confusion_matrix(names, matrix)

    for threshold in thresholds:
        cos_sim = CosineSimilarity(
            "", save_dir, threshold=threshold, model_name=model_name
        )
        cos_sim.similarity_matrices = similarity_matrices
        redundant = cos_sim.detect_redundancy()
        removed = cos_sim.remove_redundancy(redundant)
        n_removed = sum(len(v) for v in removed.values())
        print(
            f"  threshold {threshold}%: "
            f"groups_with_pairs={len(redundant)}, removed={n_removed}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Within-group cosine on concatenated row embeddings "
            "(float16 disk cache + streaming Gram; Chroma closed before compute)."
        )
    )
    p.add_argument("--chroma-path", default="chroma_db/tesla")
    p.add_argument("--collection", default="vehicle_states")
    p.add_argument("--texts-root", default="texts/data3")
    p.add_argument(
        "--average-dir",
        default="data_bottleneck/text_embed_tesla/averaged_manoeuvres",
    )
    p.add_argument(
        "--cache-dir",
        default="data_bottleneck/text_embed_tesla/concat_f16_cache",
        help="Per-maneuver float16 trajectory memmaps.",
    )
    p.add_argument(
        "--save-dir",
        default="cosine_similarity_matrices/text_embed_tesla_concat",
    )
    p.add_argument("--model-name", default="text_embed_tesla_concat")
    p.add_argument("--skip-rows", type=int, default=2500)
    p.add_argument("--export-batch-size", type=int, default=512)
    p.add_argument(
        "--max-sheet-mb",
        type=float,
        default=256.0,
        help="Max RAM for one float32 sheet during Gram accumulate.",
    )
    p.add_argument(
        "--export-min-avail-mem-gb",
        type=float,
        default=0.5,
        help=(
            "During Chroma export: stop if MemAvailable < this (GiB). "
            "Must be low — Chroma's embedding cache alone uses tens of GiB. "
            "Watchdog starts only AFTER Chroma is open. Default: 0.5."
        ),
    )
    p.add_argument(
        "--compute-min-avail-mem-gb",
        type=float,
        default=5.0,
        help=(
            "After Chroma is closed: stop if MemAvailable < this (GiB) "
            "during Gram/cosine compute. Default: 5.0."
        ),
    )
    p.add_argument("--mem-poll-sec", type=float, default=DEFAULT_MEM_POLL_SEC)
    p.add_argument("--thresholds", nargs="+", type=int, default=list(THRESHOLDS))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--limit", type=int, help="Smoke test: first N maneuvers.")
    p.add_argument("--groups", nargs="+", help="Optional group prefix filter.")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Do not open Chroma; require a complete float16 cache.",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Only build the float16 cache from Chroma, then exit (frees RAM).",
    )
    p.add_argument("--root", default=".")
    return p


def resolve_maneuver_list(args, root: Path) -> list[str]:
    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    chroma_path = resolve(args.chroma_path)
    average_dir = resolve(args.average_dir)
    progress_path = chroma_path / ".embed_progress.json"
    if progress_path.is_file():
        maneuvers = load_completed_maneuvers(progress_path)
    elif average_dir.is_dir():
        maneuvers = sorted(
            f[:-4] for f in os.listdir(average_dir) if f.endswith(".npy")
        )
    else:
        raise SystemExit("Need chroma progress or average-dir.")
    if args.limit is not None:
        maneuvers = maneuvers[: args.limit]
    return maneuvers


def run_export_subprocess(argv_extra: list[str]) -> None:
    """Run export in a child process so Chroma RAM is returned to the OS on exit."""
    import subprocess

    cmd = [sys.executable, str(Path(__file__).resolve()), "--export-only", *argv_extra]
    print(f"Spawning export subprocess:\n  {' '.join(cmd)}", flush=True)
    print(f"  MemAvail before export child: {mem_avail_gb():.1f} GiB", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Export subprocess failed with code {proc.returncode}")

    # Wait for OS to reclaim Chroma's RSS
    for _ in range(30):
        avail = mem_avail_gb()
        if avail is not None and avail >= 8.0:
            break
        time.sleep(1.0)
    print(f"  MemAvail after export child: {mem_avail_gb():.1f} GiB", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    apply_oom_victim_score()
    start_mutable_memory_watchdog(poll_sec=args.mem_poll_sec)
    set_watchdog_min_avail_gb(0)

    chroma_path = resolve(args.chroma_path)
    texts_root = resolve(args.texts_root)
    cache_dir = resolve(args.cache_dir)
    save_dir = resolve(args.save_dir)

    maneuvers = resolve_maneuver_list(args, root)
    groups = build_groups_from_names(maneuvers)
    if args.groups:
        wanted = set(args.groups)
        groups = [
            (g if prefix in wanted else [])
            for g, prefix in zip(groups, GROUP_PREFIXES)
        ]

    print(
        "Groups:",
        [len(g) for g in groups],
        f"(total {sum(len(g) for g in groups)})",
        flush=True,
    )

    # Default path: export in a child (so Chroma RSS dies with it), then compute here.
    if not args.skip_export and not args.export_only:
        extra: list[str] = [
            "--root",
            str(root),
            "--chroma-path",
            str(chroma_path),
            "--texts-root",
            str(texts_root),
            "--average-dir",
            str(resolve(args.average_dir)),
            "--cache-dir",
            str(cache_dir),
            "--skip-rows",
            str(args.skip_rows),
            "--export-batch-size",
            str(args.export_batch_size),
            "--export-min-avail-mem-gb",
            str(args.export_min_avail_mem_gb),
            "--mem-poll-sec",
            str(args.mem_poll_sec),
        ]
        if args.limit is not None:
            extra += ["--limit", str(args.limit)]
        if args.rebuild_cache:
            extra.append("--rebuild-cache")
        if args.groups:
            extra += ["--groups", *args.groups]
        run_export_subprocess(extra)
        args.skip_export = True

    path_by_name: dict[str, Path] = {}

    if args.export_only or not args.skip_export:
        try:
            import chromadb
        except ImportError as exc:
            raise SystemExit("Missing chromadb — use .venv-embeddings") from exc

        print(f"Opening Chroma at {chroma_path} ...", flush=True)
        print(f"  MemAvail before open: {mem_avail_gb():.1f} GiB", flush=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(args.collection)
        # First real get pulls embeddings into RSS
        _ = collection.get(
            ids=[f"{maneuvers[0]}__{args.skip_rows}"], include=["embeddings"]
        )
        avail_after = mem_avail_gb()
        print(f"  MemAvail after open+probe: {avail_after:.1f} GiB", flush=True)

        set_watchdog_min_avail_gb(args.export_min_avail_mem_gb)
        if avail_after is not None and avail_after < args.export_min_avail_mem_gb:
            raise SystemExit(
                f"MemAvailable after Chroma open ({avail_after:.2f} GiB) "
                f"already below export floor ({args.export_min_avail_mem_gb} GiB)."
            )

        path_by_name = ensure_trajectory_cache(
            collection,
            maneuvers,
            texts_root=texts_root,
            cache_dir=cache_dir,
            skip_rows=args.skip_rows,
            batch_size=args.export_batch_size,
            rebuild=args.rebuild_cache,
        )
        print("Export-only done — exiting so OS reclaims Chroma RAM.", flush=True)
        return

    # -------- Compute from memmaps (no Chroma) --------
    print("Loading cache paths (no Chroma) ...", flush=True)
    for m in maneuvers:
        n_rows = count_text_rows(texts_root, m)
        t_len = n_rows - args.skip_rows
        matches = list(cache_dir.glob(f"{m}__T{t_len}_D*.f16.npy"))
        if not matches:
            raise SystemExit(f"Missing cache for {m} under {cache_dir}")
        path_by_name[m] = matches[0]
    print(f"  {len(path_by_name)} trajectories, MemAvail={mem_avail_gb():.1f} GiB", flush=True)

    set_watchdog_min_avail_gb(args.compute_min_avail_mem_gb)
    if (mem_avail_gb() or 0) < args.compute_min_avail_mem_gb:
        raise SystemExit(
            f"MemAvailable {mem_avail_gb():.1f} GiB < compute floor "
            f"{args.compute_min_avail_mem_gb} GiB — aborting before Gram accumulate."
        )

    sim_mats: dict[int, tuple[list[str], np.ndarray]] = {}
    for g_idx, group in enumerate(groups, start=1):
        if len(group) < 2:
            continue
        if _stop_requested.is_set():
            raise SystemExit("Stopped by memory watchdog before next group")

        gname = group_name(group[0])
        missing = [m for m in group if m not in path_by_name]
        if missing:
            raise SystemExit(f"{gname}: missing cache for {missing[:3]}...")

        print(f"\n=== {gname} ({len(group)} maneuvers) ===", flush=True)
        t0 = time.time()
        S = concat_cosine_from_memmaps(
            [path_by_name[m] for m in group],
            max_sheet_mb=args.max_sheet_mb,
        )
        print(
            f"  done in {time.time() - t0:.1f}s | {summarize_offdiag(S)}",
            flush=True,
        )
        sim_mats[g_idx] = (list(group), S)
        del S
        gc.collect()

    if not sim_mats:
        raise SystemExit("No groups computed.")

    print("\nSaving matrices + redundancy filters ...", flush=True)
    save_and_filter(
        sim_mats,
        save_dir=str(save_dir),
        model_name=args.model_name,
        thresholds=args.thresholds,
        plot=not args.no_plot,
    )
    print(f"Done. Outputs under {save_dir}", flush=True)


if __name__ == "__main__":
    main()
