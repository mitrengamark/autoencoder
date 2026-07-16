"""
PCA baseline for maneuver redundancy reduction (Reviewer #2).

Mirrors the VAE workflow: PCA(8) -> per-maneuver time averaging -> within-group
cosine similarity -> redundant maneuver removal.
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosine_similarity import CosineSimilarity
from data_preprocess import load_and_average_manoeuvres

GROUP_PREFIXES = [
    "allando_v_savvaltas",
    "allando_v_chirp",
    "allando_v_sin",
    "valtozo_v_savvaltas_gas",
    "valtozo_v_savvaltas_fek",
    "valtozo_v_sin_gas",
    "valtozo_v_sin_fek",
]

DATA_FOLDERS = ["data3", "data_bmw_cutted"]
NORMALIZATIONS = ["minmax", "zscore"]
LATENT_DIM = 8
THRESHOLDS = range(90, 100)


def build_manoeuvre_groups(data_dir):
    """Build the 7 maneuver groups by filename prefix (same membership as VAE workflow)."""
    groups = {prefix: [] for prefix in GROUP_PREFIXES}
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith("_combined.csv"):
            continue
        manoeuvre = filename.replace("_combined.csv", "")
        for prefix in GROUP_PREFIXES:
            if manoeuvre.startswith(prefix + "_") or manoeuvre == prefix:
                groups[prefix].append(manoeuvre)
                break
    return [groups[p] for p in GROUP_PREFIXES]


def load_all_manoeuvres(data_dir):
    """Load all maneuver CSVs with alphabetically sorted columns."""
    manoeuvres = {}
    column_names = None

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith("_combined.csv"):
            continue
        manoeuvre = filename.replace("_combined.csv", "")
        df = pd.read_csv(os.path.join(data_dir, filename))
        df = df[sorted(df.columns)]
        if column_names is None:
            column_names = list(df.columns)
        elif list(df.columns) != column_names:
            raise ValueError(f"Column mismatch in {filename}")
        manoeuvres[manoeuvre] = df.values.astype(np.float32)

    return manoeuvres, column_names


def normalize_global(data, norm_type):
    """Global normalization across all stacked timesteps."""
    if norm_type == "minmax":
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        return (data - data_min) / (data_max - data_min + 1e-8)
    if norm_type == "zscore":
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    raise ValueError(f"Unsupported normalization: {norm_type}")


def fit_pca_and_save(manoeuvres, norm_type, model_name):
    """Fit global PCA(8), transform each maneuver, save (T, 8) .npy files."""
    manoeuvre_names = sorted(manoeuvres.keys())
    stacked = np.vstack([manoeuvres[name] for name in manoeuvre_names])
    print(f"  Stacked shape: {stacked.shape}")

    normalized = normalize_global(stacked, norm_type)
    pca = PCA(n_components=LATENT_DIM)
    pca.fit(normalized)
    print(f"  PCA explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.4f}")

    single_dir = os.path.join("data_bottleneck", model_name, "single_manoeuvres")
    os.makedirs(single_dir, exist_ok=True)

    offset = 0
    for name in manoeuvre_names:
        n_rows = manoeuvres[name].shape[0]
        transformed = pca.transform(normalized[offset : offset + n_rows])
        np.save(os.path.join(single_dir, f"{name}.npy"), transformed.astype(np.float32))
        offset += n_rows

    print(f"  Saved {len(manoeuvre_names)} files to {single_dir}")
    return single_dir


def run_cosine_similarity_pipeline(average_dir, save_dir, manoeuvre_groups):
    """Run CosineSimilarity for thresholds 90-99; plot matrices once."""
    os.makedirs(save_dir, exist_ok=True)
    for i, threshold in enumerate(THRESHOLDS):
        cos_sim = CosineSimilarity(average_dir, save_dir, threshold=threshold)
        cos_sim.compute_cosine_similarity_within_groups(
            manoeuvre_groups, plot=(i == 0)
        )
        redundant_pairs = cos_sim.detect_redundancy()
        cos_sim.remove_redundancy(redundant_pairs)
        print(f"  Threshold {threshold}%: redundant groups = {len(redundant_pairs)}")


def process_variant(data_dir, norm_type):
    """Full PCA baseline pipeline for one data folder + normalization variant."""
    folder_slug = os.path.basename(data_dir)
    model_name = f"pca_{folder_slug}_{norm_type}"
    print(f"\n=== {model_name} ({data_dir}, {norm_type}) ===")

    manoeuvre_groups = build_manoeuvre_groups(data_dir)
    group_sizes = [len(g) for g in manoeuvre_groups]
    print(f"  Maneuver groups: {group_sizes} (total {sum(group_sizes)})")

    manoeuvres, _ = load_all_manoeuvres(data_dir)
    print(f"  Loaded {len(manoeuvres)} maneuvers")

    single_dir = fit_pca_and_save(manoeuvres, norm_type, model_name)

    average_dir = os.path.join("data_bottleneck", model_name, "averaged_manoeuvres")
    print(f"  Averaging: {single_dir} -> {average_dir}")
    load_and_average_manoeuvres(single_dir, average_dir)

    save_dir = os.path.join("cosine_similarity_matrices", model_name)
    print(f"  Cosine similarity -> {save_dir}")
    run_cosine_similarity_pipeline(average_dir, save_dir, manoeuvre_groups)


def main():
    for data_dir in DATA_FOLDERS:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        for norm_type in NORMALIZATIONS:
            process_variant(data_dir, norm_type)

    print("\nDone. Outputs under data_bottleneck/pca_* and cosine_similarity_matrices/pca_*")


if __name__ == "__main__":
    main()
