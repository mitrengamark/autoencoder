"""
Compute average redundancy per method and threshold (Table 3 style).

Redundancy = fraction of maneuvers removed at a given cosine similarity threshold.
Outputs per-dataset and averaged CSVs to Results/.
"""

import json
import os

import pandas as pd

THRESHOLDS = range(90, 100)
ROOT = "cosine_similarity_matrices"
BOTTLENECK_ROOT = "data_bottleneck"
OUTPUT_DIR = "Results"

# Standard methods: folder under cosine_similarity_matrices/<folder>/
METHODS = {
    "legacy_og": {
        # Original OG evaluation in Redundant_manoeuvres/ (single dataset)
        "legacy": {
            "json_dir": "Redundant_manoeuvres",
            "json_pattern": "manoeuvres_for_removing_{threshold}.json",
            "total": 458,
        },
    },
    # VAE cosine — per-dataset and averaged (Table 3 "proposed")
    "og_remake": {
        "data3": "OG_remake",
    },
    "bmw_og_remake": {
        "bmw": "bmw_OG_remake",
    },
    "proposed_vae_cosine": {
        "data3": "OG_remake",
        "bmw": "bmw_OG_remake",
    },
    "pca_minmax_cosine": {
        "data3": "pca_data3_minmax",
        "bmw": "pca_data_bmw_cutted_minmax",
    },
    "pca_zscore_cosine": {
        "data3": "pca_data3_zscore",
        "bmw": "pca_data_bmw_cutted_zscore",
    },
    "kmeans_vae": {
        "data3": "kmeans_OG_remake",
        "bmw": "kmeans_bmw_OG_remake",
    },
    "kmedoids_vae": {
        "data3": "kmedoids_OG_remake",
        "bmw": "kmedoids_bmw_OG_remake",
    },
    "kmeans_pca_minmax": {
        "data3": "kmeans_pca_data3_minmax",
        "bmw": "kmeans_pca_data_bmw_cutted_minmax",
    },
    "kmedoids_pca_minmax": {
        "data3": "kmedoids_pca_data3_minmax",
        "bmw": "kmedoids_pca_data_bmw_cutted_minmax",
    },
    "kmeans_pca_zscore": {
        "data3": "kmeans_pca_data3_zscore",
        "bmw": "kmeans_pca_data_bmw_cutted_zscore",
    },
    "kmedoids_pca_zscore": {
        "data3": "kmedoids_pca_data3_zscore",
        "bmw": "kmedoids_pca_data_bmw_cutted_zscore",
    },
}


def representation_folder(folder):
    """Map removal JSON folder to data_bottleneck representation folder."""
    if folder.startswith("kmeans_"):
        return folder[len("kmeans_") :]
    if folder.startswith("kmedoids_"):
        return folder[len("kmedoids_") :]
    return folder


def count_total(folder):
    """Total maneuver count from averaged_manoeuvres .npy files."""
    repr_folder = representation_folder(folder)
    average_dir = os.path.join(BOTTLENECK_ROOT, repr_folder, "averaged_manoeuvres")
    if os.path.isdir(average_dir):
        return len([f for f in os.listdir(average_dir) if f.endswith(".npy")])
    return 458


def redundancy_from_json(json_path, total):
    """Return (removed, total, redundancy_pct) from a removal JSON path."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing removal JSON: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    removed_set = set()
    for group in data.values():
        removed_set.update(group)

    removed = len(removed_set)
    pct = removed / total * 100 if total else 0.0
    return removed, total, pct


def redundancy(folder_or_spec, threshold):
    """
    Return (removed, total, redundancy_pct).

    folder_or_spec:
      - str: folder under cosine_similarity_matrices/
      - dict: {"json_dir", "json_pattern", "total"} for custom locations
    """
    if isinstance(folder_or_spec, dict):
        json_path = os.path.join(
            folder_or_spec["json_dir"],
            folder_or_spec["json_pattern"].format(threshold=threshold),
        )
        total = folder_or_spec.get("total", 458)
        return redundancy_from_json(json_path, total)

    folder = folder_or_spec
    json_path = os.path.join(
        ROOT, folder, f"manoeuvres_for_removing_{threshold}_{folder}.json"
    )
    total = count_total(folder)
    return redundancy_from_json(json_path, total)


def collect_results():
    """Collect per-dataset and averaged redundancy rows."""
    per_dataset_rows = []
    average_rows = []

    for method, folders in METHODS.items():
        for threshold in THRESHOLDS:
            dataset_pcts = {}

            for dataset, folder_or_spec in folders.items():
                removed, total, pct = redundancy(folder_or_spec, threshold)
                per_dataset_rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "threshold": threshold / 100,
                        "removed": removed,
                        "total": total,
                        "redundancy_pct": round(pct, 1),
                    }
                )
                dataset_pcts[dataset] = pct

            data3_pct = dataset_pcts.get("data3")
            bmw_pct = dataset_pcts.get("bmw")
            legacy_pct = dataset_pcts.get("legacy")

            # Average across available datasets (data3+bmw, or single legacy)
            available = [v for v in (data3_pct, bmw_pct, legacy_pct) if v is not None]
            avg_pct = sum(available) / len(available) if available else 0.0

            average_rows.append(
                {
                    "method": method,
                    "threshold": threshold / 100,
                    "redundancy_data3_pct": (
                        round(data3_pct, 1) if data3_pct is not None else None
                    ),
                    "redundancy_bmw_pct": (
                        round(bmw_pct, 1) if bmw_pct is not None else None
                    ),
                    "redundancy_legacy_pct": (
                        round(legacy_pct, 1) if legacy_pct is not None else None
                    ),
                    "avg_redundancy_pct": round(avg_pct, 1),
                }
            )

    return pd.DataFrame(per_dataset_rows), pd.DataFrame(average_rows)


def write_outputs(df_per_dataset, df_average):
    """Write CSV outputs and print Table 3-style pivot."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_dataset_path = os.path.join(OUTPUT_DIR, "redundancy_per_dataset.csv")
    average_path = os.path.join(OUTPUT_DIR, "redundancy_average.csv")
    table3_path = os.path.join(OUTPUT_DIR, "redundancy_table3.csv")

    df_per_dataset.to_csv(per_dataset_path, index=False)
    df_average.to_csv(average_path, index=False)

    pivot = df_average.pivot(
        index="threshold", columns="method", values="avg_redundancy_pct"
    )
    pivot = pivot.sort_index(ascending=True)
    pivot.to_csv(table3_path)

    print("=" * 80)
    print("Table 3 style: Average redundancy (%) by threshold and method")
    print("=" * 80)
    print(pivot.round(0).astype(int).to_string())
    print()
    print(f"Saved: {per_dataset_path}")
    print(f"Saved: {average_path}")
    print(f"Saved: {table3_path}")


def main():
    df_per_dataset, df_average = collect_results()
    write_outputs(df_per_dataset, df_average)


if __name__ == "__main__":
    main()
