"""
Build filtered .mat subsets from reduced_manoeuvres_lists txt files.

Each txt is produced by get_reduced_manoeuvres_list.py and contains the
*kept* (non-redundant) maneuvers as:

    selected_manoeuvres = name1, name2, ...

For each list (threshold 90/95/98, methods containing pca/kmeans/kmedoid),
copy only those selected source .mat files into the output folder.
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

TIER1_SUFFIXES = (
    "pca_minmax_90",
    "pca_minmax_95",
    "pca_minmax_98",
    "VAE_kmeans_90",
    "VAE_kmeans_95",
    "VAE_kmeans_98",
    "VAE_kmedoids_90",
    "VAE_kmedoids_95",
    "VAE_kmedoids_98",
)


def parse_selected_manoeuvres(txt_path: Path) -> set[str]:
    """Parse kept manoeuvre names from a one-line selected_manoeuvres txt."""
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

    selected = parse_selected_manoeuvres(txt_path)
    source_mats = list_source_mats(source_dir)
    missing_selected = sorted(selected - set(source_mats))

    out_dir = OUT_ROOT / output_dir_name(dataset, method, threshold)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in sorted(selected):
        src_path = source_mats.get(name)
        if src_path is None:
            continue
        shutil.copy2(src_path, out_dir / src_path.name)
        copied += 1

    return {
        "skipped": False,
        "method": method,
        "threshold": threshold,
        "dataset": dataset,
        "selected": len(selected),
        "copied": copied,
        "total": len(source_mats),
        "removed": len(source_mats) - copied,
        "missing_selected": missing_selected,
        "out_dir": out_dir,
    }


def organize_into_tiers() -> None:
    """Move generated folders into tier1 / tier2 for the identification handoff."""
    tier1 = OUT_ROOT / "tier1"
    tier2 = OUT_ROOT / "tier2"
    tier1.mkdir(parents=True, exist_ok=True)
    tier2.mkdir(parents=True, exist_ok=True)

    for path in sorted(OUT_ROOT.iterdir()):
        if not path.is_dir() or path.name in {"tier1", "tier2"}:
            continue
        is_tier1 = any(
            path.name == f"{ds}_{suffix}"
            for ds in ("bmw", "tesla")
            for suffix in TIER1_SUFFIXES
        )
        dest_root = tier1 if is_tier1 else tier2
        dest = dest_root / path.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(path), str(dest))


def main() -> None:
    if not LISTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing lists directory: {LISTS_DIR}")

    # Clear previous (possibly inverted) outputs, including tier folders.
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

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
            f"selected={result['selected']}, copied={result['copied']}/{result['total']} "
            f"(removed={result['removed']})"
        )
        if result["missing_selected"]:
            print(
                f"  warning: {len(result['missing_selected'])} selected names not in source mats"
            )

    organize_into_tiers()

    print()
    print(f"Processed lists: {len(results)}")
    print(f"Skipped lists: {skipped}")
    print(f"Output root: {OUT_ROOT.resolve()}")
    print(f"tier1 folders: {len(list((OUT_ROOT / 'tier1').iterdir()))}")
    print(f"tier2 folders: {len(list((OUT_ROOT / 'tier2').iterdir()))}")

    if results:
        total_missing = sum(len(r["missing_selected"]) for r in results)
        if total_missing:
            print(f"Total missing selected names across lists: {total_missing}")


if __name__ == "__main__":
    main()
