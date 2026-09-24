#!/usr/bin/env python3
"""
Per-maneuver PCA/t-SNE of ALL row embeddings from Chroma (no skip).

Points are open circles colored by time order (row_idx gradient).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_maneuvers(progress_path: Path) -> list[str]:
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    out = []
    for name in sorted(data.get("completed_files", [])):
        if name.endswith("_combined.txt"):
            out.append(name.replace("_combined.txt", ""))
    return out


def count_text_rows(texts_root: Path, maneuver: str) -> int:
    path = texts_root / f"{maneuver}_combined.txt"
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fetch_all_embeddings(
    collection,
    maneuver: str,
    n_rows: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings[N,dim], row_idx[N]) sorted by time. Prefer where= filter."""
    del batch_size  # kept for CLI compatibility; where-fetch loads the whole maneuver

    result = collection.get(
        where={"maneuver": maneuver},
        include=["embeddings", "metadatas"],
    )
    ids = result.get("ids") or []
    raw = result.get("embeddings")
    metas = result.get("metadatas") or []
    if raw is None or not ids:
        raise RuntimeError(f"{maneuver}: no embeddings found")

    emb = np.asarray(list(raw), dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"{maneuver}: bad embedding shape {emb.shape}")

    times = np.zeros(len(ids), dtype=np.int32)
    for i, meta in enumerate(metas):
        if meta and "row_idx" in meta:
            times[i] = int(meta["row_idx"])
        else:
            # fallback: parse from id ``maneuver__row``
            times[i] = int(str(ids[i]).rsplit("__", 1)[-1])

    order = np.argsort(times)
    emb = emb[order]
    times = times[order]

    if len(times) != n_rows:
        print(
            f"  warning: expected {n_rows} rows from text, got {len(times)} from Chroma",
            file=sys.stderr,
        )
    return emb, times


def project(
    vectors: np.ndarray,
    method: str,
    dims: int,
    random_state: int,
    perplexity: float,
) -> tuple[np.ndarray, str]:
    n = vectors.shape[0]
    if method == "pca":
        n_comp = min(dims, vectors.shape[1], n)
        pca = PCA(n_components=n_comp, random_state=random_state)
        coords = pca.fit_transform(vectors)
        if coords.shape[1] < dims:
            pad = np.zeros((n, dims - coords.shape[1]), dtype=coords.dtype)
            coords = np.hstack([coords, pad])
        var = 100.0 * float(pca.explained_variance_ratio_.sum())
        return coords, f"PCA  var={var:.1f}%"

    perp = min(perplexity, max(2.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=dims,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(vectors)
    return coords, f"t-SNE  perplexity={perp:.1f}"


def plot_open_circles_2d(
    coords: np.ndarray,
    times: np.ndarray,
    title: str,
    out_path: Path,
    cmap_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    norm = Normalize(vmin=float(times.min()), vmax=float(times.max()))
    cmap = cm.get_cmap(cmap_name) if hasattr(cm, "get_cmap") else plt.colormaps[cmap_name]
    edgecolors = cmap(norm(times))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=14,
        facecolors="none",
        edgecolors=edgecolors,
        linewidths=0.7,
        alpha=0.85,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row_idx (time order)")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_open_circles_3d(
    coords: np.ndarray,
    times: np.ndarray,
    title: str,
    out_path: Path,
    cmap_name: str,
) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(times.min()), vmax=float(times.max()))
    cmap = cm.get_cmap(cmap_name) if hasattr(cm, "get_cmap") else plt.colormaps[cmap_name]
    edgecolors = cmap(norm(times))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=12,
        facecolors="none",
        edgecolors=edgecolors,
        linewidths=0.6,
        alpha=0.8,
        depthshade=False,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("row_idx (time order)")
    ax.set_title(title)
    ax.set_xlabel("C1")
    ax.set_ylabel("C2")
    ax.set_zlabel("C3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-maneuver PCA/t-SNE of all Chroma row embeddings (time-colored open circles)."
    )
    p.add_argument("--chroma-path", default="chroma_db/tesla")
    p.add_argument("--collection", default="vehicle_states")
    p.add_argument("--texts-root", default="texts/data3")
    p.add_argument(
        "--out-dir",
        default="Results/text_embed_tesla_viz/per_maneuver",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pca"],
        choices=["pca", "tsne"],
        help="Default: pca only (tsne is slow for ~10k points × 458 maneuvers).",
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 3],
        choices=[2, 3],
        help="2 and/or 3 (default: both).",
    )
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--perplexity", type=float, default=40.0)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--limit", type=int, help="Only first N maneuvers.")
    p.add_argument(
        "--maneuvers",
        nargs="+",
        help="Only these maneuver names (without _combined).",
    )
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit("pip install chromadb") from exc

    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    def resolve(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    chroma_path = resolve(args.chroma_path)
    texts_root = resolve(args.texts_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = chroma_path / ".embed_progress.json"
    maneuvers = load_maneuvers(progress_path)
    if args.maneuvers:
        wanted = set(args.maneuvers)
        maneuvers = [m for m in maneuvers if m in wanted]
    if args.limit is not None:
        maneuvers = maneuvers[: args.limit]
    if not maneuvers:
        raise SystemExit("No maneuvers to plot.")

    print(f"Opening Chroma at {chroma_path} ...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(args.collection)
    print(f"Plotting {len(maneuvers)} maneuvers -> {out_dir}")

    t0 = time.time()
    done = 0
    skipped = 0
    for idx, maneuver in enumerate(maneuvers, start=1):
        # Skip if all requested outputs already exist
        needed = []
        for method in args.methods:
            for d in args.dims:
                needed.append(out_dir / f"{maneuver}_{method}_{d}d.png")
        if all(p.is_file() for p in needed):
            skipped += 1
            print(f"[{idx}/{len(maneuvers)}] skip existing: {maneuver}")
            continue

        n_rows = count_text_rows(texts_root, maneuver)
        file_t0 = time.time()
        vectors, times = fetch_all_embeddings(
            collection, maneuver, n_rows=n_rows, batch_size=args.batch_size
        )
        print(
            f"[{idx}/{len(maneuvers)}] {maneuver}: loaded {vectors.shape[0]}×{vectors.shape[1]} "
            f"in {time.time() - file_t0:.1f}s"
        )

        for method in args.methods:
            for d in args.dims:
                out_path = out_dir / f"{maneuver}_{method}_{d}d.png"
                if out_path.is_file():
                    continue
                coords, subtitle = project(
                    vectors,
                    method=method,
                    dims=d,
                    random_state=args.random_state,
                    perplexity=args.perplexity,
                )
                title = f"{maneuver} — {subtitle} ({d}D), n={len(times)}, time gradient"
                if d == 2:
                    plot_open_circles_2d(coords, times, title, out_path, args.cmap)
                else:
                    plot_open_circles_3d(coords, times, title, out_path, args.cmap)
                print(f"  wrote {out_path.name}")

        done += 1

    print(
        f"Done. new={done} skipped_existing={skipped} "
        f"in {time.time() - t0:.1f}s -> {out_dir}"
    )


if __name__ == "__main__":
    main()
