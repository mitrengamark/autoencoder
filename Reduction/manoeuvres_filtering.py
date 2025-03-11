import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist

# from random_data import (
#     generate_clustered_data,
#     generate_advanced_sinusoidal_spiral_data,
#     plot_clusters,
# )
# import sys
# import os

# # Hozzáadjuk a projekt gyökérkönyvtárát a Python elérési útvonalához
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Config.load_config import (
    eps,
    min_samples,
    n_clusters,
    seed,
    num_workers,
    parallel,
    distance_metric,
)


class ManoeuvresFiltering:
    def __init__(
        self,
        reduced_data=None,
        bottleneck_data=None,
        labels=None,
        label_mapping=None,
        eps=eps,
        min_samples=min_samples,
        n_clusters=n_clusters,
    ):
        self.reduced_data = reduced_data
        self.bottleneck_data = bottleneck_data
        self.labels = labels
        self.label_mapping = label_mapping
        self.boundary_manoeuvres = set()
        self.boundary_indices = None
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def filter_manoeuvres(self):
        # self.dbscan_clustering()
        # self.plot_dbscan_clusters()

        if self.bottleneck_data.shape[1] == 2:
            cluster_info = self.kmeans_clustering()
            self.plot_kmeans_clusters()
            self.plot_cluster_info(cluster_info)

        # self.find_uniformly_distributed_manoeuvres()
        filtered_reduced_data, filtered_labels = self.remove_redundant_manoeuvres()
        return filtered_reduced_data, filtered_labels

    # -----------------------------------clustering-----------------------------------

    def dbscan_clustering(self):
        """
        DBSCAN klaszterezés végrehajtása az eredeti magasabb dimenziós térben.
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = dbscan.fit_predict(self.bottleneck_data)

        clusters_info = defaultdict(lambda: defaultdict(int))

        for idx, cluster in enumerate(self.cluster_labels):
            if cluster == -1:
                continue  # Zajadatok kihagyása
            manoeuvre_idx = self.labels[idx]
            manoeuvre_name = next(
                (
                    name
                    for name, value in self.label_mapping.items()
                    if value == manoeuvre_idx
                ),
                f"Unknown_{manoeuvre_idx}",
            )
            clusters_info[cluster][manoeuvre_name] += 1

        print("DBSCAN klaszterek és tartalmuk:")
        for cluster, manoeuvres in clusters_info.items():
            print(f"Klaszter {cluster}: {dict(manoeuvres)}")

        return clusters_info

    def kmeans_clustering(self):
        """
        K-Means klaszterezés Cosine Similarity alapján.
        Az adatokat először normalizáljuk, hogy az Euklideszi távolság megfeleljen a Cosine távolságnak.
        """
        normalized_data = normalize(self.bottleneck_data, norm="l2")

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed)
        self.kmeans_labels = kmeans.fit_predict(normalized_data)

        clusters_info = defaultdict(lambda: defaultdict(int))

        for idx, cluster in enumerate(self.kmeans_labels):
            manoeuvre_idx = self.labels[idx]
            manoeuvre_name = next(
                (
                    name
                    for name, value in self.label_mapping.items()
                    if value == manoeuvre_idx
                ),
                f"Unknown_{manoeuvre_idx}",
            )
            clusters_info[cluster][manoeuvre_name] += 1

        print("K-Means klaszterek és tartalmuk:")
        for cluster, manoeuvres in clusters_info.items():
            print(f"Klaszter {cluster}: {dict(manoeuvres)}")

        return clusters_info

    # -----------------------------------plotting-----------------------------------

    def plot_dbscan_clusters(self):
        """
        A DBSCAN klaszterek vizualizációja.
        """
        plt.figure(figsize=(10, 6))
        unique_clusters = np.unique(self.cluster_labels)
        colors = plt.cm.get_cmap("tab10", len(unique_clusters))

        for cluster in unique_clusters:
            mask = self.cluster_labels == cluster
            if cluster == -1:
                plt.scatter(
                    self.reduced_data[mask, 0],
                    self.reduced_data[mask, 1],
                    label="Zajadatok",
                    color="gray",
                    alpha=0.5,
                    s=10,
                )
            else:
                plt.scatter(
                    self.reduced_data[mask, 0],
                    self.reduced_data[mask, 1],
                    label=f"Klaszter {cluster}",
                    color=colors(cluster % 10),
                    alpha=0.7,
                    s=10,
                )

        plt.title("DBSCAN Klaszterek")
        plt.xlabel("T-SNE Komponens 1")
        plt.ylabel("T-SNE Komponens 2")
        plt.legend(loc="best", fontsize="small", markerscale=0.7)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_kmeans_clusters(self):
        plt.figure(figsize=(10, 6))
        unique_clusters = np.unique(self.kmeans_labels)
        colors = plt.cm.get_cmap("tab10", len(unique_clusters))

        for cluster in unique_clusters:
            mask = self.kmeans_labels == cluster
            plt.scatter(
                self.reduced_data[mask, 0],
                self.reduced_data[mask, 1],
                label=f"Klaszter {cluster}",
                color=colors(cluster % 10),
                alpha=0.7,
                s=10,
            )

        plt.title("K-Means Klaszterek")
        plt.xlabel("T-SNE Komponens 1")
        plt.ylabel("T-SNE Komponens 2")
        plt.legend(loc="best", fontsize="small", markerscale=0.7)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_cluster_info(self, clusters_info):
        """
        K-Means klaszterek oszlopdiagramjának létrehozása, amely megmutatja, hogy egyes klaszterekben
        mely manőverek hány adatponttal szerepelnek.

        :param clusters_info: Dictionary, amely tartalmazza a klasztereket és azokhoz tartozó manőverek számosságát.
        """
        num_clusters = len(clusters_info)
        fig, axes = plt.subplots(
            num_clusters, 1, figsize=(8, 4 * num_clusters), constrained_layout=True
        )

        if num_clusters == 1:
            axes = [axes]

        for i, (cluster, manoeuvres) in enumerate(clusters_info.items()):
            manoeuvre_names = list(manoeuvres.keys())
            counts = list(manoeuvres.values())

            axes[i].barh(manoeuvre_names, counts, color="skyblue")
            axes[i].set_title(f"Klaszter {cluster}")
            axes[i].set_xlabel("Adatpontok száma")
            axes[i].set_ylabel("Manőverek")
            axes[i].grid(axis="x", linestyle="--", alpha=0.7)

    plt.show()

    # -----------------------------------redundant_manoeuvres-----------------------------------

    def filter_redundant_manoeuvres_pearson(self, threshold=0.7):
        """
        Kiszűri a redundáns manővereket a Pearson-korreláció alapján többdimenziós adatok esetén.
        threshold: A minimális korrelációs érték, amely felett két manőver redundánsnak számít.
        """
        print("Redundáns manőverek keresése Pearson-korrelációval...")

        # Manőverenként átlagolunk minden dimenzióban
        unique_labels = np.unique(self.labels)
        mean_vectors = {
            label: np.mean(self.reduced_data[self.labels == label], axis=0)
            for label in unique_labels
        }

        # Keresünk redundáns manővereket
        redundant_pairs = set()
        label_list = list(mean_vectors.keys())

        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                label1, label2 = label_list[i], label_list[j]
                corr, _ = pearsonr(mean_vectors[label1], mean_vectors[label2])

                if abs(corr) > threshold:
                    redundant_pairs.add((label1, label2))

        print(f"Redundáns manőver párok: {redundant_pairs}")

        return redundant_pairs

    def filter_redundant_manoeuvres_spearman(self, threshold=0.7):
        """
        Kiszűri a redundáns manővereket a Spearman-korreláció alapján többdimenziós adatok esetén.
        threshold: A minimális korrelációs érték, amely felett két manőver redundánsnak számít.
        """
        print("Spearman-alapú redundáns manőverek keresése...")

        # Manőverenként átlagolunk minden dimenzióban
        unique_labels = np.unique(self.labels)
        mean_vectors = {
            label: np.mean(self.reduced_data[self.labels == label], axis=0)
            for label in unique_labels
        }

        # Keresünk redundáns manővereket
        redundant_pairs = set()
        label_list = list(mean_vectors.keys())

        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                label1, label2 = label_list[i], label_list[j]
                rho, _ = spearmanr(mean_vectors[label1], mean_vectors[label2])

                if abs(rho) > threshold:
                    redundant_pairs.add((label1, label2))

        print(f"Redundáns manőver párok (Spearman): {redundant_pairs}")

        return redundant_pairs

    def filter_redundant_manoeuvres_kendall(self, threshold=0.7):
        """
        Kiszűri a redundáns manővereket a Kendall-tau korreláció alapján többdimenziós adatok esetén.
        threshold: A minimális korrelációs érték, amely felett két manőver redundánsnak számít.
        """
        print("Kendall-alapú redundáns manőverek keresése...")

        # Manőverenként átlagolunk minden dimenzióban
        unique_labels = np.unique(self.labels)
        mean_vectors = {
            label: np.mean(self.reduced_data[self.labels == label], axis=0)
            for label in unique_labels
        }

        # Keresünk redundáns manővereket
        redundant_pairs = set()
        label_list = list(mean_vectors.keys())

        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                label1, label2 = label_list[i], label_list[j]
                tau, _ = kendalltau(mean_vectors[label1], mean_vectors[label2])

                if abs(tau) > threshold:
                    redundant_pairs.add((label1, label2))

        print(f"Redundáns manőver párok (Kendall): {redundant_pairs}")

        return redundant_pairs

    def check_cluster_dominance(self):
        """
        Klaszteren belüli dominancia ellenőrzése. Ha egy manőver túlságosan dominál egy klaszterben,
        akkor lehet, hogy redundáns vagy rosszul definiált.
        """
        print("Klaszter dominancia vizsgálata...")

        cluster_counts = defaultdict(lambda: defaultdict(int))

        for idx, cluster in enumerate(self.kmeans_labels):
            manoeuvre_idx = self.labels[idx]
            cluster_counts[cluster][manoeuvre_idx] += 1

        for cluster, manoeuvres in cluster_counts.items():
            total = sum(manoeuvres.values())
            for manoeuvre, count in manoeuvres.items():
                ratio = count / total
                if ratio > 0.8:  # Ha egy manőver >80%-ot fed le egy klaszteren belül
                    print(
                        f"Klaszter {cluster}: A(z) {manoeuvre} túlságosan dominál ({ratio:.2%})"
                    )

        return cluster_counts

    def find_uniformly_distributed_manoeuvres(self, threshold=0.05):
        """
        Megvizsgálja, hogy van-e olyan manőver, amely minden klaszterben nagyjából egyforma arányban oszlik meg.
        - threshold: A maximális szórás, amely felett a manővert nem tekintjük egyenletesen eloszlónak.
        """

        print("Egyenletesen eloszló manőverek keresése...")

        cluster_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(int)

        # Adatok összegyűjtése: klaszterenként megszámoljuk a manővereket
        for idx, cluster in enumerate(self.kmeans_labels):
            manoeuvre_idx = self.labels[idx]
            cluster_counts[manoeuvre_idx][cluster] += 1
            total_counts[manoeuvre_idx] += 1

        uniformly_distributed_manoeuvres = []

        for manoeuvre, cluster_distribution in cluster_counts.items():
            total = total_counts[manoeuvre]
            cluster_ratios = []

            # Számoljuk ki az egyes klaszterekben lévő arányokat
            for cluster in range(self.n_clusters):
                ratio = cluster_distribution[cluster] / total if total > 0 else 0
                cluster_ratios.append(ratio)

            # Számoljuk ki a szórást
            std_dev = np.std(cluster_ratios)
            print(f"A(z) {manoeuvre} szórása: {std_dev:.4f}")

            if std_dev < threshold:  # Ha az eloszlás szórása alacsony, akkor egyenletes
                manoeuvre_name = self.reverse_label_mapping.get(
                    manoeuvre, f"Unknown_{manoeuvre}"
                )
                uniformly_distributed_manoeuvres.append(manoeuvre_name)
                print(
                    f"A(z) {manoeuvre_name} egyenletesen oszlik el minden klaszterben (szórás: {std_dev:.4f})"
                )

        return uniformly_distributed_manoeuvres

    def filter_by_distance(self, threshold=0.5, batch_size=100):
        """
        Cosine Similarity vagy Euklideszi távolság alapján kiszűri a redundáns manővereket.

        threshold: A hasonlósági küszöb, amely felett két manőver redundánsnak számít.
        batch_size: A párhuzamos számításokhoz a batch méret.
        """
        print("Távolsági redundancia szűrés Cosine Similarity használatával...")
        print(
            f"Adatok statisztikája normalizálás előtt: min={np.min(self.bottleneck_data)}, max={np.max(self.bottleneck_data)}, mean={np.mean(self.bottleneck_data)}, std={np.std(self.bottleneck_data)}"
        )

        if isinstance(self.bottleneck_data, list):
            self.bottleneck_data = np.vstack(self.bottleneck_data)

        if isinstance(self.labels, list):
            self.labels = np.concatenate(self.labels)

        # Normalizálás csak akkor, ha az adott metrikához szükséges
        if distance_metric == "cosine":
            print("Cosine Similarity: L2 normalizálás aktiválva.")
            self.bottleneck_data = normalize(self.bottleneck_data, axis=1)

        elif distance_metric == "euclidean":
            print("Euklideszi távolság: StandardScaler aktiválva.")
            scaler = StandardScaler()
            self.bottleneck_data = scaler.fit_transform(self.bottleneck_data)

        print(
            f"Adatok statisztikája skálázás után: min={np.min(self.bottleneck_data)}, max={np.max(self.bottleneck_data)}, mean={np.mean(self.bottleneck_data)}, std={np.std(self.bottleneck_data)}"
        )
        print(f"Első 5 minta adata normalizálás után:\n{self.bottleneck_data[:5]}")

        n = self.bottleneck_data.shape[0]
        redundant_pairs = set()
        total_batches = (n + batch_size - 1) // batch_size  # Összes batch száma

        similarity_function = compute_similarities if distance_metric == "cosine" else compute_euclidean_similarities

        if parallel:
            # Párhuzamos számítás elindítása
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(0, n, batch_size):
                    batch_indices = list(range(i, min(i + batch_size, n)))
                    futures.append(
                        executor.submit(
                            similarity_function,
                            batch_indices,
                            self.bottleneck_data,
                            threshold,
                            self.labels,
                        )
                    )

                # Eredmények összegyűjtése
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    print(f"Eredmény feldolgozása: {completed}/{len(futures)}")
                    redundant_pairs.update(future.result())
        else:
            # **Soros (egyszálú) számítás**
            for batch_idx, i in enumerate(range(0, n, batch_size), start=1):
                print(f"Feldolgozás: {batch_idx}/{total_batches} batch...")
                batch_indices = list(range(i, min(i + batch_size, n)))
                redundant_pairs.update(
                    similarity_function(
                        batch_indices, self.bottleneck_data, threshold, self.labels
                    )
                )

        # Redundáns manőverek kiírása
        redundant_manoeuvre_names = {
            (
                self.reverse_label_mapping.get(pair[0], f"Unknown_{pair[0]}"),
                self.reverse_label_mapping.get(pair[1], f"Unknown_{pair[1]}"),
            )
            for pair in redundant_pairs
        }

        print(f"Redundáns manőver párok ({distance_metric}): {redundant_manoeuvre_names}")

        return redundant_pairs

    # -----------------------------------Remove_redundant_manoeuvers-----------------------------------

    def remove_redundant_manoeuvres(self):
        """
        Kiszűri a redundáns manővereket a következő módszerekkel:
        - Pearson korreláció
        - Spearman korreláció
        - Kendall korreláció
        - Cosine Similarity

        **Működés:**
        1. Az összes redundáns pár összegyűjtése.
        2. A legtöbbször szereplő címke kiválasztása.
        3. A kiválasztott címkét hozzáadjuk a redundáns manőverek listájához.
        4. Az ehhez a címkéhez tartozó összes párt eltávolítjuk a redundáns párok listájából.
        5. Ismételjük, amíg vannak redundáns párok.
        """
        print(f"Redundáns manőverek keresése több módszerrel...")

        # Összegyűjtjük az összes redundáns párt
        redundant_pairs = set()
        # redundant_pairs.update(self.filter_redundant_manoeuvres_pearson())
        # redundant_pairs.update(self.filter_redundant_manoeuvres_spearman())
        # redundant_pairs.update(self.filter_redundant_manoeuvres_kendall())
        redundant_pairs.update(self.filter_by_distance())

        print(f"Összesített redundáns párok: {redundant_pairs}")

        # Ha nincs redundáns pár, nincs mit szűrni
        if not redundant_pairs:
            print("Nincsenek redundáns manőverek.")
            return set()

        # Lépésenként eltávolítandó manőverek
        redundant_manoeuvres = set()

        while redundant_pairs:
            # Számláljuk, hogy melyik címke hányszor szerepel a redundáns párokban
            counter = Counter()
            for a, b in redundant_pairs:
                counter[a] += 1
                counter[b] += 1

            # Kiválasztjuk azt a címkét, amelyik a legtöbbször szerepel
            most_common_label, _ = counter.most_common(1)[0]
            redundant_manoeuvres.add(most_common_label)

            # Töröljük az összes párt, ahol ez a címke szerepelt
            redundant_pairs = {
                pair for pair in redundant_pairs if most_common_label not in pair
            }

        print(f"Végleges redundáns manőverek: {redundant_manoeuvres}")

        # Ellenőrizzük, hogy a redundant_manoeuvres megfelelő típusú-e
        redundant_manoeuvres = np.array(list(redundant_manoeuvres))

        # **Eltávolítjuk a redundáns manőverekhez tartozó adatokat**
        mask = np.isin(
            self.labels, redundant_manoeuvres, invert=True
        )  # True, ha NEM redundáns
        filtered_reduced_data = self.reduced_data[
            mask
        ]  # Csak a nem-redundáns adatokat tartjuk meg
        filtered_labels = self.labels[mask]  # Frissítjük a címkéket is

        print(
            f"Redundáns manőverek eltávolítva, új adatméret: {filtered_reduced_data.shape}"
        )

        return filtered_reduced_data, filtered_labels  # A címkéket is visszaadjuk


def compute_similarities(batch_indices, full_data, threshold, labels):
    """
    Egy batch cosine similarity mátrixát számolja ki és visszaadja a redundáns párokat.
    """
    batch_data = full_data[batch_indices]  # Kiválasztjuk a megfelelő adatokat
    similarities = cosine_similarity(batch_data, full_data)
    redundant_pairs = set()

    # print(f"Cosine Similarity első 5 sor első 5 oszlopa:\n{similarities[:5, :5]}")

    for row_idx, row in enumerate(similarities):
        for j in range(full_data.shape[0]):
            if batch_indices[row_idx] == j:  # Önmaga ellenőrzésének kizárása
                continue
            if row[j] > threshold and labels[batch_indices[row_idx]] != labels[j]:
                redundant_pairs.add(
                    tuple(sorted((labels[batch_indices[row_idx]], labels[j])))
                )

    return redundant_pairs

def compute_euclidean_similarities(batch_indices, full_data, threshold, labels):
    """
    Egy batch euklideszi távolságát számolja ki és visszaadja a redundáns párokat.
    """
    batch_data = full_data[batch_indices]
    distances = cdist(batch_data, full_data, metric='euclidean')
    
    redundant_pairs = set()

    for row_idx, row in enumerate(distances):
        for j in range(full_data.shape[0]):
            if batch_indices[row_idx] == j:  # Önmaga ellenőrzésének kizárása
                continue
            if row[j] < threshold and labels[batch_indices[row_idx]] != labels[j]:
                redundant_pairs.add(
                    tuple(sorted((labels[batch_indices[row_idx]], labels[j])))
                )

    return redundant_pairs


# bottleneck_data, labels, label_mapping = generate_clustered_data(
#     n_samples=151270, n_clusters=14, n_features=8
# )

# mf = ManoeuvresFiltering(
#     bottleneck_data=bottleneck_data,
#     labels=labels,
#     label_mapping=label_mapping,
# )
# # Plotoljuk a generált adatokat
# plot_clusters(bottleneck_data, labels)
# redundant_pairs = mf.filter_by_distance()

# print("Redundáns párok:", redundant_pairs)
