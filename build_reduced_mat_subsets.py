"""
Build filtered .mat subsets from reduced_manoeuvres_lists txt files.

For each list (threshold 90/95/98, methods containing pca/kmeans/kmedoid),
copy all source .mat files except those marked for removal in the txt.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

LISTS_DIR = Path("reduced_manoeuvres_lists")
MAT_DIRS = {"bmw": Path("mat/data_bmw"), "tesla": Path("mat/data_tesla")}
OUT_ROOT = Path("data_filtered_mat")
THRESHOLDS = {"90", "95", "98"}
METHOD_TOKENS = ("pca", "kmeans", "kmedoid")

LIST_NAME_RE = re.compile(r"^manoeuvres_for_removing_(\d+)_(.+)\.txt$")


def parse_removed_manoeuvres(txt_path: Path) -> set[str]:
    """Parse removed manoeuvre names from a one-line selected_manoeuvres txt."""
    content = txt_path.read_text(encoding="utf-8").strip()
    if "=" not in content:
        raise ValueError(f"Unexpected format (no '='): {txt_path}")
    _, names_part = content.split("=", 1)
    return {name.strip() for name in names_part.split(",") if name.strip()}


def matches_method_filter(method: str) -> bool:
    method_lower = method.lower()
    return any(token in method_lower for token in METHOD_TOKENS)


def dataset_for_method(method: str) -> str:
    return "bmw" if "bmw" in method.lower() else "tesla"


def dimred_and_select(dataset: str, method: str) -> tuple[str, str | None]:
    """Split a raw method token into (dim-reduction label, selection label).

    Dim-reduction is either "VAE" (the OG_remake bottleneck) or "pca_minmax" /
    "pca_zscore". Selection is None for the cosine-similarity baseline, or
    "kmeans" / "kmedoids" when that clustering method replaced the cosine
    threshold step.
    """
    remainder = method
    select: str | None = None
    for token in ("kmeans", "kmedoids"):
        prefix = f"{token}_"
        if remainder.startswith(prefix):
            select = token
            remainder = remainder[len(prefix) :]
            break

    if dataset == "tesla":
        remainder = remainder.replace("data3_", "")
        remainder = remainder.replace("OG_remake", "VAE")
    else:
        remainder = remainder.replace("data_bmw_cutted_", "")
        remainder = remainder.replace("bmw_OG_remake", "VAE")

    dimred = re.sub(r"_+", "_", remainder).strip("_")
    return dimred, select


def output_dir_name(dataset: str, method: str, threshold: str) -> str:
    """Folder name reflecting the full workflow: <dataset>_<dimred>(_<select>)_<threshold>.

    e.g. tesla_pca_minmax_90, tesla_VAE_kmeans_90, tesla_pca_minmax_kmedoids_90.
    """
    dimred, select = dimred_and_select(dataset, method)
    parts = [dataset, dimred]
    if select:
        parts.append(select)
    parts.append(threshold)
    return "_".join(parts)


def list_source_mats(source_dir: Path) -> dict[str, Path]:
    """Map manoeuvre basename (no extension) -> full .mat path."""
    mats: dict[str, Path] = {}
    for path in sorted(source_dir.glob("*.mat")):
        mats[path.stem] = path
    return mats


def process_list(txt_path: Path) -> dict:
    match = LIST_NAME_RE.match(txt_path.name)
    if not match:
        return {"skipped": True, "reason": "name pattern"}

    threshold, method = match.group(1), match.group(2)
    if threshold not in THRESHOLDS:
        return {"skipped": True, "reason": "threshold"}
    if not matches_method_filter(method):
        return {"skipped": True, "reason": "method filter"}

    dataset = dataset_for_method(method)
    source_dir = MAT_DIRS[dataset]
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source mat directory: {source_dir}")

    removed = parse_removed_manoeuvres(txt_path)
    source_mats = list_source_mats(source_dir)
    missing_removed = sorted(removed - set(source_mats))

    out_dir = OUT_ROOT / output_dir_name(dataset, method, threshold)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name, src_path in source_mats.items():
        if name in removed:
            continue
        shutil.copy2(src_path, out_dir / src_path.name)
        copied += 1

    return {
        "skipped": False,
        "method": method,
        "threshold": threshold,
        "dataset": dataset,
        "removed": len(removed),
        "copied": copied,
        "total": len(source_mats),
        "missing_removed": missing_removed,
        "out_dir": out_dir,
    }


def main() -> None:
    if not LISTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing lists directory: {LISTS_DIR}")

    results: list[dict] = []
    skipped = 0

    for txt_path in sorted(LISTS_DIR.glob("manoeuvres_for_removing_*.txt")):
        result = process_list(txt_path)
        if result.get("skipped"):
            skipped += 1
            continue

        results.append(result)
        print(
            f"{output_dir_name(result['dataset'], result['method'], result['threshold'])}: "
            f"removed={result['removed']}, copied={result['copied']}/{result['total']}"
        )
        if result["missing_removed"]:
            print(
                f"  warning: {len(result['missing_removed'])} removed names not in source mats"
            )

    print()
    print(f"Processed lists: {len(results)}")
    print(f"Skipped lists: {skipped}")
    print(f"Output root: {OUT_ROOT.resolve()}")

    if results:
        total_missing = sum(len(r["missing_removed"]) for r in results)
        if total_missing:
            print(f"Total missing removed names across lists: {total_missing}")


if __name__ == "__main__":
    main()
