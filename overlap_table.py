"""
Same-domain overlap of redundant maneuvers across EV (data3) and ICE (BMW).

Table 5-style summary: for each method and threshold, compare removed-maneuver
sets from cosine_similarity_matrices JSONs on both datasets.
"""

import json
import os

import pandas as pd

ROOT = "cosine_similarity_matrices"
OUTPUT_DIR = "Results"
THRESHOLDS = [90, 95, 98]

# EV (data3) folder -> ICE (BMW) folder
METHOD_PAIRS = {
    "proposed_vae": {
        "ev": "OG_remake",
        "ice": "bmw_OG_remake",
    },
    "pca_minmax": {
        "ev": "pca_data3_minmax",
        "ice": "pca_data_bmw_cutted_minmax",
    },
    "pca_zscore": {
        "ev": "pca_data3_zscore",
        "ice": "pca_data_bmw_cutted_zscore",
    },
    "kmeans_vae": {
        "ev": "kmeans_OG_remake",
        "ice": "kmeans_bmw_OG_remake",
    },
    "kmedoids_vae": {
        "ev": "kmedoids_OG_remake",
        "ice": "kmedoids_bmw_OG_remake",
    },
    "kmeans_pca_minmax": {
        "ev": "kmeans_pca_data3_minmax",
        "ice": "kmeans_pca_data_bmw_cutted_minmax",
    },
    "kmedoids_pca_minmax": {
        "ev": "kmedoids_pca_data3_minmax",
        "ice": "kmedoids_pca_data_bmw_cutted_minmax",
    },
    "kmeans_pca_zscore": {
        "ev": "kmeans_pca_data3_zscore",
        "ice": "kmeans_pca_data_bmw_cutted_zscore",
    },
    "kmedoids_pca_zscore": {
        "ev": "kmedoids_pca_data3_zscore",
        "ice": "kmedoids_pca_data_bmw_cutted_zscore",
    },
}


def json_path(folder, threshold):
    return os.path.join(
        ROOT, folder, f"manoeuvres_for_removing_{threshold}_{folder}.json"
    )


def load_removed_set(folder, threshold):
    path = json_path(folder, threshold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing removal JSON: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return {item for sublist in data.values() for item in sublist}


def overlap_metrics(set_ev, set_ice):
    common = set_ev & set_ice
    union = set_ev | set_ice
    jaccard = len(common) / len(union) if union else 0.0
    overlap = (
        len(common) / min(len(set_ev), len(set_ice))
        if min(len(set_ev), len(set_ice))
        else 0.0
    )
    return {
        "red_ev": len(set_ev),
        "red_ice": len(set_ice),
        "common": len(common),
        "jaccard": round(jaccard, 4),
        "overlap": round(overlap, 4),
    }


def collect_results():
    """One summary row per method × threshold, plus long rows for Table 5 layout."""
    summary_rows = []
    long_rows = []

    for method, folders in METHOD_PAIRS.items():
        ev_folder = folders["ev"]
        ice_folder = folders["ice"]

        for threshold in THRESHOLDS:
            set_ev = load_removed_set(ev_folder, threshold)
            set_ice = load_removed_set(ice_folder, threshold)
            metrics = overlap_metrics(set_ev, set_ice)

            summary_rows.append(
                {
                    "threshold": threshold / 100,
                    "method": method,
                    "ev_folder": ev_folder,
                    "ice_folder": ice_folder,
                    **metrics,
                }
            )

            for filt, red in (("EV", metrics["red_ev"]), ("ICE", metrics["red_ice"])):
                long_rows.append(
                    {
                        "threshold": threshold / 100,
                        "method": method,
                        "filt": filt,
                        "red": red,
                        "common": metrics["common"],
                        "jaccard": metrics["jaccard"],
                        "overlap": metrics["overlap"],
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def build_table5_style(df_long):
    """Pivot to printable Table 5-style layout (two Filt rows per method block)."""
    return df_long[
        ["threshold", "method", "filt", "red", "common", "jaccard", "overlap"]
    ].copy()


def print_table(df_long):
    print("=" * 72)
    print("Table 5 style: Same-domain overlap of redundant maneuvers (EV vs ICE)")
    print("=" * 72)

    for threshold in sorted(df_long["threshold"].unique()):
        print(f"\nThr. {threshold:.2f}")
        sub = df_long[df_long["threshold"] == threshold]
        for method in sub["method"].unique():
            block = sub[sub["method"] == method]
            print(f"  {method}")
            for _, row in block.iterrows():
                print(
                    f"    Filt {row['filt']:>3}  Red {int(row['red']):>3}  "
                    f"Common {int(row['common']):>3}  "
                    f"Jacc {row['jaccard']:.3f}  Overlap {row['overlap']:.3f}"
                )


def write_outputs(df_summary, df_long, df_table5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(OUTPUT_DIR, "overlap_summary.csv")
    per_method_path = os.path.join(OUTPUT_DIR, "overlap_per_method.csv")
    table5_path = os.path.join(OUTPUT_DIR, "overlap_table5_style.csv")

    df_summary.to_csv(summary_path, index=False)
    df_long.to_csv(per_method_path, index=False)
    df_table5.to_csv(table5_path, index=False)

    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_method_path}")
    print(f"Saved: {table5_path}")


def main():
    df_summary, df_long = collect_results()
    df_table5 = build_table5_style(df_long)
    print_table(df_long)
    write_outputs(df_summary, df_long, df_table5)


if __name__ == "__main__":
    main()
