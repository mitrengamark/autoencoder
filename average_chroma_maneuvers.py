#!/usr/bin/env python3
"""
Average row-level Chroma embeddings per maneuver (Tesla), matching the AE pipeline:

    vectors = embeddings[skip_rows:]   # default skip first 2500 of 10805
    mean = vectors.mean(axis=0)
    np.save(out_dir / f"{maneuver}.npy", mean)

Reads IDs as ``{maneuver}__{row_idx}`` from chroma_db/tesla.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_completed_maneuvers(progress_path: Path) -> list[str]:
    with progress_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    files = sorted(data.get("completed_files", []))
    maneuvers = []
    for name in files:
        if not name.endswith("_combined.txt"):
            continue
        maneuvers.append(name.replace("_combined.txt", ""))
    return maneuvers


def count_text_rows(texts_root: Path, maneuver: str) -> int:
    txt_path = texts_root / f"{maneuver}_combined.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing text file for row count: {txt_path}")
    with txt_path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def average_maneuver(
    collection,
    maneuver: str,
    n_rows: int,
    skip_rows: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Return (mean_vector, n_used). Raises if nothing to average."""
    if n_rows <= skip_rows:
        raise ValueError(
            f"{maneuver}: n_rows={n_rows} <= skip_rows={skip_rows}, nothing to average"
        )

    start = skip_rows
    dim: int | None = None
    total = None
    n_used = 0
    missing = 0

    for batch_start in range(start, n_rows, batch_size):
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
            total = np.zeros(dim, dtype=np.float64)
        elif arr.shape[1] != dim:
            raise ValueError(
                f"{maneuver}: dim mismatch {arr.shape[1]} vs expected {dim}"
            )

        total += arr.sum(axis=0)
        n_used += arr.shape[0]

    if n_used == 0 or total is None:
        raise RuntimeError(f"{maneuver}: no embeddings found for rows [{start}, {n_rows})")

    if missing:
        print(
            f"  warning: {missing} missing ids in [{start}, {n_rows}) "
            f"(used {n_used})",
            file=sys.stderr,
        )

    return (total / n_used).astype(np.float32), n_used


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average Tesla Chroma embeddings per maneuver (skip first N rows)."
    )
    parser.add_argument(
        "--chroma-path",
        default="chroma_db/tesla",
        help="Chroma persist directory (default: chroma_db/tesla).",
    )
    parser.add_argument(
        "--collection",
        default="vehicle_states",
        help="Chroma collection name (default: vehicle_states).",
    )
    parser.add_argument(
        "--texts-root",
        default="texts/data3",
        help="Text dir used only for per-maneuver row counts (default: texts/data3).",
    )
    parser.add_argument(
        "--out-dir",
        default="data_bottleneck/text_embed_tesla/averaged_manoeuvres",
        help="Output directory for {maneuver}.npy averages.",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=2500,
        help="Drop the first N row embeddings before averaging (default: 2500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Chroma get() batch size (default: 512).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Only process the first N maneuvers (smoke test).",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root for relative paths (default: .).",
    )
    return parser


def main() -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit(
            "Missing chromadb. Install with: pip install -r requirements-embeddings.txt"
        ) from exc

    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    chroma_path = Path(args.chroma_path)
    if not chroma_path.is_absolute():
        chroma_path = root / chroma_path
    texts_root = Path(args.texts_root)
    if not texts_root.is_absolute():
        texts_root = root / texts_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    progress_path = chroma_path / ".embed_progress.json"
    if not progress_path.is_file():
        raise SystemExit(f"Progress file not found: {progress_path}")

    maneuvers = load_completed_maneuvers(progress_path)
    if args.limit is not None:
        maneuvers = maneuvers[: args.limit]

    if not maneuvers:
        raise SystemExit("No completed maneuvers in progress file.")

    out_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(args.collection)

    print(
        f"Averaging {len(maneuvers)} maneuvers from {chroma_path} "
        f"(skip_rows={args.skip_rows}) -> {out_dir}"
    )

    t0 = time.time()
    ok = 0
    skipped_existing = 0
    for idx, maneuver in enumerate(maneuvers, start=1):
        out_path = out_dir / f"{maneuver}.npy"
        if out_path.is_file():
            skipped_existing += 1
            print(f"[{idx}/{len(maneuvers)}] skip existing: {out_path.name}")
            continue

        n_rows = count_text_rows(texts_root, maneuver)
        file_t0 = time.time()
        mean_vec, n_used = average_maneuver(
            collection,
            maneuver,
            n_rows=n_rows,
            skip_rows=args.skip_rows,
            batch_size=args.batch_size,
        )
        np.save(out_path, mean_vec)
        ok += 1
        elapsed = time.time() - file_t0
        print(
            f"[{idx}/{len(maneuvers)}] {maneuver}: "
            f"rows={n_rows}, used={n_used}, dim={mean_vec.shape[0]}, "
            f"{elapsed:.1f}s -> {out_path.name}"
        )

    print(
        f"Done. Wrote {ok} new averages "
        f"(skipped_existing={skipped_existing}) in {time.time() - t0:.1f}s under {out_dir}"
    )


if __name__ == "__main__":
    main()
