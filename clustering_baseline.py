"""
Clustering-based selection baseline.

Reuses averaged maneuver vectors and matches the cosine method's per-group
reduction budget (thresholds 90-99). Outputs the same JSON format as
cosine_similarity.py for downstream comparison.
"""

import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

from pca_baseline import GROUP_PREFIXES, THRESHOLDS
from cosine_similarity import CosineSimilarity

SOURCE_MODELS = [
    "bmw_OG_remake",
    "OG_remake",
    "pca_data3_minmax",
    "pca_data3_zscore",
    "pca_data_bmw_cutted_minmax",
    "pca_data_bmw_cutted_zscore",
]

ALGORITHMS = ["kmeans", "kmedoids"]
RANDOM_STATE = 42


def build_manoeuvre_groups_from_averaged(average_dir):
    """Build 7 maneuver groups from averaged_manoeuvres/*.npy filenames."""
    groups = {prefix: [] for prefix in GROUP_PREFIXES}
    for filename in sorted(os.listdir(average_dir)):
        if not filename.endswith(".npy"):
            continue
        manoeuvre = filename[:-4]
        for prefix in GROUP_PREFIXES:
            if manoeuvre.startswith(prefix + "_") or manoeuvre == prefix:
                groups[prefix].append(manoeuvre)
                break
    return [groups[p] for p in GROUP_PREFIXES]


def load_vectors(average_dir, manoeuvres):
    """Load averaged (8,) vectors for the given maneuver names."""
    vectors = []
    for name in manoeuvres:
        path = os.path.join(average_dir, name + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing averaged vector: {path}")
        vectors.append(np.load(path))
    return np.stack(vectors)


def kept_count_per_group(manoeuvre_groups, removed_set):
    """Number of maneuvers to keep per group (same budget as cosine method)."""
    kept = []
    for members in manoeuvre_groups:
        removed_in_group = sum(1 for m in members if m in removed_set)
        k = len(members) - removed_in_group
        k = max(1, min(k, len(members)))
        kept.append(k)
    return kept


def load_cosine_removed_set(source_model, threshold):
    """Load removed maneuvers from the cosine baseline JSON for this source."""
    json_path = os.path.join(
        "cosine_similarity_matrices",
        source_model,
        f"manoeuvres_for_removing_{threshold}_{source_model}.json",
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing cosine JSON: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    removed = set()
    for group in data.values():
        removed.update(group)
    return removed


def kmedoids_pam(distance_matrix, k, random_state=RANDOM_STATE):
    """Partitioning around medoids on a precomputed distance matrix."""
    n = distance_matrix.shape[0]
    if k >= n:
        return list(range(n))

    rng = np.random.RandomState(random_state)
    medoid_indices = [int(rng.randint(n))]

    for _ in range(1, k):
        d_to_nearest = np.min(distance_matrix[:, medoid_indices], axis=1)
        probs = d_to_nearest ** 2
        total = probs.sum()
        if total <= 0:
            remaining = [i for i in range(n) if i not in medoid_indices]
            medoid_indices.append(remaining[0] if remaining else 0)
            continue
        probs /= total
        medoid_indices.append(int(rng.choice(n, p=probs)))

    while True:
        assignments = np.argmin(distance_matrix[:, medoid_indices], axis=1)
        new_medoid_indices = []

        for cluster_id in range(k):
            cluster_points = np.where(assignments == cluster_id)[0]
            if len(cluster_points) == 0:
                new_medoid_indices.append(medoid_indices[cluster_id])
                continue
            sub_dist = distance_matrix[np.ix_(cluster_points, cluster_points)]
            new_medoid_indices.append(int(cluster_points[np.argmin(sub_dist.sum(axis=1))]))

        if new_medoid_indices == medoid_indices:
            break
        medoid_indices = new_medoid_indices

    return medoid_indices


def select_kmeans_representatives(vectors, names, k):
    """k-means clustering; keep the maneuver closest to each centroid."""
    n = len(names)
    if k >= n:
        return set(names)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(vectors)

    representatives = set()
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = km.cluster_centers_[cluster_id]
        cluster_vectors = vectors[cluster_indices]
        closest = cluster_indices[np.argmin(np.linalg.norm(cluster_vectors - centroid, axis=1))]
        representatives.add(names[closest])
    return representatives


def select_kmedoids_representatives(vectors, names, k):
    """k-medoids (PAM) on cosine distance; medoids are actual maneuvers."""
    n = len(names)
    if k >= n:
        return set(names)

    dist_matrix = cosine_distances(vectors)
    medoid_indices = kmedoids_pam(dist_matrix, k, random_state=RANDOM_STATE)
    return {names[i] for i in medoid_indices}


def select_representatives(vectors, names, k, algorithm):
    if algorithm == "kmeans":
        return select_kmeans_representatives(vectors, names, k)
    if algorithm == "kmedoids":
        return select_kmedoids_representatives(vectors, names, k)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def build_removed_by_group(manoeuvre_groups, kept_counts, average_dir, algorithm):
    """Run clustering per group; return removed lists keyed by group index."""
    removed_by_group = {}

    for group_idx, (members, k) in enumerate(zip(manoeuvre_groups, kept_counts), start=1):
        if len(members) == 0:
            removed_by_group[str(group_idx)] = []
            continue

        vectors = load_vectors(average_dir, members)
        representatives = select_representatives(vectors, members, k, algorithm)
        removed = [m for m in members if m not in representatives]
        removed_by_group[str(group_idx)] = removed

    return removed_by_group


def process_source_algorithm(source_model, algorithm):
    """Generate clustering removal JSONs for one source model and algorithm."""
    average_dir = os.path.join("data_bottleneck", source_model, "averaged_manoeuvres")
    if not os.path.isdir(average_dir):
        print(f"SKIP missing averaged dir: {average_dir}")
        return

    output_dir_name = f"{algorithm}_{source_model}"
    output_dir = os.path.join("cosine_similarity_matrices", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    manoeuvre_groups = build_manoeuvre_groups_from_averaged(average_dir)
    print(f"\n=== {output_dir_name} ===")

    for threshold in THRESHOLDS:
        removed_set = load_cosine_removed_set(source_model, threshold)
        kept_counts = kept_count_per_group(manoeuvre_groups, removed_set)
        removed_by_group = build_removed_by_group(
            manoeuvre_groups, kept_counts, average_dir, algorithm
        )

        total_members = sum(len(g) for g in manoeuvre_groups)
        total_removed = sum(len(v) for v in removed_by_group.values())
        expected_removed = len(removed_set)

        json_path = os.path.join(
            output_dir,
            f"manoeuvres_for_removing_{threshold}_{output_dir_name}.json",
        )
        with open(json_path, "w") as f:
            json.dump(removed_by_group, f, indent=4)

        print(
            f"  threshold {threshold}%: removed {total_removed}/{total_members} "
            f"(cosine budget: {expected_removed}) -> {json_path}"
        )


def plot_similarity_matrices(source_model, algorithm):
    """Generate cosine similarity heatmaps for a clustering output folder."""
    average_dir = os.path.join("data_bottleneck", source_model, "averaged_manoeuvres")
    output_dir_name = f"{algorithm}_{source_model}"
    save_dir = os.path.join("cosine_similarity_matrices", output_dir_name)

    if not os.path.isdir(average_dir):
        print(f"SKIP missing averaged dir: {average_dir}")
        return

    manoeuvre_groups = build_manoeuvre_groups_from_averaged(average_dir)
    cos_sim = CosineSimilarity(
        average_dir,
        save_dir,
        threshold=99,
        model_name=output_dir_name,
    )
    cos_sim.compute_cosine_similarity_within_groups(manoeuvre_groups, plot=True)
    png_count = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    print(f"  matrices: {png_count} PNGs -> {save_dir}")


def plot_all_similarity_matrices():
    """Plot cosine similarity matrices for all clustering variants."""
    for source_model in SOURCE_MODELS:
        for algorithm in ALGORITHMS:
            print(f"\nPlotting matrices for {algorithm}_{source_model} ...")
            plot_similarity_matrices(source_model, algorithm)


def main():
    for source_model in SOURCE_MODELS:
        for algorithm in ALGORITHMS:
            process_source_algorithm(source_model, algorithm)

    plot_all_similarity_matrices()

    print("\nDone. Outputs under cosine_similarity_matrices/kmeans_* and kmedoids_*")


if __name__ == "__main__":
    main()
