#!/usr/bin/env python3
"""
Build filtered .mat folders from redundancy-removal JSONs.

For each method × threshold (default 90/95/98):
  kept = all .mat stems in --source-dir  minus  names listed in
         cosine_similarity_matrices/<method>/manoeuvres_for_removing_<thr>_*.json

Copies (or hardlinks) kept files into:
  <out-root>/<method>_<threshold>/

Source mats are expected under data_tesla/ (may be empty until you add them).
Stem matching also accepts ``{name}_combined.mat``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

THRESHOLDS_DEFAULT = (90, 95, 98)

# Text-embedding redundancy methods only.
DEFAULT_METHODS = (
    "text_embed_tesla",
    "text_embed_tesla_pairwise",
    "text_embed_tesla_dtw",
    "text_embed_tesla_concat",
)


def load_removed(json_path: Path) -> set[str]:
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    removed: set[str] = set()
    for group in data.values():
        removed.update(group)
    return removed


def find_removal_json(method_dir: Path, threshold: int, method: str) -> Path | None:
    """Prefer exact model_name match, else any manoeuvres_for_removing_{thr}_*.json."""
    exact = method_dir / f"manoeuvres_for_removing_{threshold}_{method}.json"
    if exact.is_file():
        return exact
    matches = sorted(method_dir.glob(f"manoeuvres_for_removing_{threshold}_*.json"))
    return matches[0] if matches else None


def index_source_mats(source_dir: Path) -> dict[str, Path]:
    """
    Map maneuver basename -> .mat path.

    Accepts both ``name.mat`` and ``name_combined.mat`` (prefers exact name.mat).
    """
    by_name: dict[str, Path] = {}
    for path in sorted(source_dir.glob("*.mat")):
        stem = path.stem
        keys = [stem]
        if stem.endswith("_combined"):
            keys.append(stem[: -len("_combined")])
        for key in keys:
            # Prefer non-_combined file if both exist
            if key not in by_name or (
                by_name[key].stem.endswith("_combined") and not stem.endswith("_combined")
            ):
                by_name[key] = path
    return by_name


def fallback_universe(root: Path) -> set[str]:
    """Maneuver names when data_tesla is still empty (for manifests)."""
    candidates = [
        root / "data_bottleneck" / "text_embed_tesla" / "averaged_manoeuvres",
        root / "data_bottleneck" / "OG_remake" / "averaged_manoeuvres",
        root / "data3",
    ]
    names: set[str] = set()
    for folder in candidates:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.suffix in {".npy", ".csv", ".mat"}:
                stem = path.stem
                if stem.endswith("_combined"):
                    stem = stem[: -len("_combined")]
                names.add(stem)
        if names:
            return names
    return names


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def process_one(
    method: str,
    threshold: int,
    matrices_dir: Path,
    source_mats: dict[str, Path],
    universe: set[str],
    out_root: Path,
    mode: str,
    dry_run: bool,
    write_manifest: bool,
) -> dict:
    method_dir = matrices_dir / method
    if not method_dir.is_dir():
        return {"method": method, "threshold": threshold, "status": "missing_method_dir"}

    json_path = find_removal_json(method_dir, threshold, method)
    if json_path is None:
        return {"method": method, "threshold": threshold, "status": "missing_json"}

    removed = load_removed(json_path)
    kept_names = sorted(universe - removed)

    out_dir = out_root / f"{method}_{threshold}"
    result = {
        "method": method,
        "threshold": threshold,
        "status": "ok",
        "json": str(json_path),
        "out_dir": str(out_dir),
        "n_universe": len(universe),
        "n_source": len(source_mats),
        "n_removed_listed": len(removed),
        "n_kept": len(kept_names),
        "n_copied": 0,
        "kept_missing_mat": sorted(set(kept_names) - set(source_mats)),
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_manifest:
        manifest = {
            "method": method,
            "threshold": threshold,
            "source_json": str(json_path),
            "n_universe": len(universe),
            "n_source_mats": len(source_mats),
            "removed": sorted(removed),
            "kept": kept_names,
            "kept_missing_mat": result["kept_missing_mat"],
        }
        (out_dir / "filter_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    if not source_mats:
        result["status"] = "no_source_mats"
        return result

    copied = 0
    for name in kept_names:
        src = source_mats.get(name)
        if src is None:
            continue
        copy_or_link(src, out_dir / src.name, mode)
        copied += 1
    result["n_copied"] = copied
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Copy non-redundant .mat files from data_tesla into "
            "per-method_threshold folders using manoeuvres_for_removing JSONs."
        )
    )
    p.add_argument(
        "--source-dir",
        default="data_tesla",
        help="Directory with source .mat files (default: data_tesla).",
    )
    p.add_argument(
        "--matrices-dir",
        default="cosine_similarity_matrices",
        help="Root of manoeuvres_for_removing_*.json folders.",
    )
    p.add_argument(
        "--out-root",
        default="data_filtered_mat",
        help="Output root for <method>_<threshold>/ folders.",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=list(THRESHOLDS_DEFAULT),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Method folder names under matrices-dir (default: Tesla full-pipeline set).",
    )
    p.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place files into output folders (default: copy).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done.",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write filter_manifest.json into each output folder.",
    )
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    source_dir = resolve(args.source_dir)
    matrices_dir = resolve(args.matrices_dir)
    out_root = resolve(args.out_root)

    if not matrices_dir.is_dir():
        raise SystemExit(f"Matrices dir not found: {matrices_dir}")

    source_dir.mkdir(parents=True, exist_ok=True)
    source_mats = index_source_mats(source_dir)
    if source_mats:
        universe = set(source_mats)
        universe_src = "data_tesla stems"
    else:
        universe = fallback_universe(root)
        universe_src = "fallback (averaged_manoeuvres / data3)"
        if not universe:
            raise SystemExit(
                "No .mat in data_tesla and no fallback maneuver list found."
            )

    print(
        f"Source mats: {source_dir} ({len(source_mats)} files) | "
        f"universe={len(universe)} via {universe_src}",
        flush=True,
    )
    if not source_mats:
        print(
            "WARNING: no .mat files yet — writing manifests with kept/removed lists; "
            "re-run after placing .mat files in data_tesla/.",
            file=sys.stderr,
            flush=True,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Methods: {len(args.methods)} | thresholds: {args.thresholds} | "
        f"mode={args.mode} | out={out_root}",
        flush=True,
    )

    results = []
    for method in args.methods:
        for thr in args.thresholds:
            r = process_one(
                method=method,
                threshold=thr,
                matrices_dir=matrices_dir,
                source_mats=source_mats,
                universe=universe,
                out_root=out_root,
                mode=args.mode,
                dry_run=args.dry_run,
                write_manifest=not args.no_manifest,
            )
            results.append(r)
            status = r["status"]
            if status in {"ok", "no_source_mats", "dry_run"}:
                missing = len(r.get("kept_missing_mat") or [])
                extra = (
                    f"kept={r['n_kept']}/{r['n_universe']} "
                    f"removed={r['n_removed_listed']} "
                    f"copied={r['n_copied']}"
                )
                if missing and status == "ok":
                    extra += f" missing_mat={missing}"
                if status == "no_source_mats":
                    extra += " (manifest only)"
            else:
                extra = status
            print(f"  {method}_{thr}: {extra}", flush=True)

    ok = sum(1 for r in results if r["status"] in {"ok", "no_source_mats", "dry_run"})
    bad = [r for r in results if r["status"] not in {"ok", "no_source_mats", "dry_run"}]
    print(f"\nDone. ok={ok}/{len(results)} under {out_root}", flush=True)
    if bad:
        print("Issues:", flush=True)
        for r in bad:
            print(f"  {r['method']}_{r['threshold']}: {r['status']}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
