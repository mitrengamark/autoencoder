import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from Config.load_config import method, seed

# from random_data import (
#     generate_clustered_data,
#     generate_advanced_sinusoidal_spiral_data,
# )
# import sys
# import os

# # Hozzáadjuk a projekt gyökérkönyvtárát a Python elérési útvonalához
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Config.load_config import eps, min_samples, n_clusters


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
        cluster_info = self.kmeans_clustering()

        if self.bottleneck_data.shape[1] == 2:
            self.plot_kmeans_clusters()
            self.plot_cluster_info(cluster_info)

        self.find_uniformly_distributed_manoeuvres()
        filtered_reduced_data = self.remove_redundant_manoeuvres()
        return filtered_reduced_data

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
        mean_vectors = {label: np.mean(self.reduced_data[self.labels == label], axis=0) for label in unique_labels}

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
        Kiszűri a redundáns manővereket a Spearman-korreláció alapján.
        threshold: A minimális korrelációs érték, amely felett két manőver redundánsnak számít.
        """
        print("Spearman-alapú redundáns manőverek keresése...")

        # Az adatok pandas DataFrame-be alakítása
        df = pd.DataFrame(
            self.reduced_data,
            columns=[f"comp_{i}" for i in range(self.reduced_data.shape[1])],
        )
        df["labels"] = self.labels

        # Kiszámítjuk a Spearman-korrelációs mátrixot
        num_columns = df.drop("labels", axis=1).shape[1]
        correlation_matrix = pd.DataFrame(
            index=df.columns[:-1], columns=df.columns[:-1]
        )

        for i in range(num_columns):
            for j in range(i, num_columns):
                col1, col2 = df.columns[i], df.columns[j]
                rho, _ = spearmanr(df[col1], df[col2])
                correlation_matrix.loc[col1, col2] = rho
                correlation_matrix.loc[col2, col1] = rho  # Mivel szimmetrikus

        # Keresünk redundáns párokat
        redundant_pairs = set()
        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    redundant_pairs.add(
                        (correlation_matrix.index[i], correlation_matrix.columns[j])
                    )

        print(f"Redundáns manőver párok (Spearman): {redundant_pairs}")

        return redundant_pairs

    def filter_redundant_manoeuvres_kendall(self, threshold=0.7):
        """
        Kiszűri a redundáns manővereket a Kendall-tau korreláció alapján.
        threshold: A minimális korrelációs érték, amely felett két manőver redundánsnak számít.
        """
        print("Kendall-alapú redundáns manőverek keresése...")

        # Az adatok pandas DataFrame-be alakítása
        df = pd.DataFrame(
            self.reduced_data,
            columns=[f"comp_{i}" for i in range(self.reduced_data.shape[1])],
        )
        df["labels"] = self.labels

        # Kiszámítjuk a Kendall-tau korrelációs mátrixot
        num_columns = df.drop("labels", axis=1).shape[1]
        correlation_matrix = pd.DataFrame(
            index=df.columns[:-1], columns=df.columns[:-1]
        )

        for i in range(num_columns):
            for j in range(i, num_columns):
                col1, col2 = df.columns[i], df.columns[j]
                tau, _ = kendalltau(df[col1], df[col2])
                correlation_matrix.loc[col1, col2] = tau
                correlation_matrix.loc[col2, col1] = tau  # Mivel szimmetrikus

        # Keresünk redundáns párokat
        redundant_pairs = set()
        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    redundant_pairs.add(
                        (correlation_matrix.index[i], correlation_matrix.columns[j])
                    )

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

    def filter_by_distance(self, threshold=0.1):
        """
        Az euklideszi távolságok alapján kiszűri a redundáns manővereket.
        Ha két manőver közötti távolság nagyon kicsi, az egyik elhagyható.
        """
        print("Távolsági redundancia szűrés...")

        distances = euclidean_distances(self.reduced_data)
        redundant_pairs = set()

        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):  # Csak az egyik irányba vizsgáljuk
                if distances[i, j] < threshold and self.labels[i] != self.labels[j]:
                    redundant_pairs.add(tuple(sorted((self.labels[i], self.labels[j]))))

        redundant_manoeuvre_names = {
            (
                self.reverse_label_mapping.get(pair[0], f"Unknown_{pair[0]}"),
                self.reverse_label_mapping.get(pair[1], f"Unknown_{pair[1]}"),
            )
            for pair in redundant_pairs
        }

        print(f"Redundáns manőver párok (távolság alapú): {redundant_manoeuvre_names}")
        return redundant_pairs

    # -----------------------------------Remove_redundant_manoeuvers-----------------------------------

    def remove_redundant_manoeuvres(self):
        """
        Kiszűri a redundáns manővereket a boundary manőverek figyelembevételével.
        - Ha egy redundáns pár mindkét eleme boundary, akkor csak az egyik marad meg.
        - Ha egy redundáns pár egyik tagja boundary, akkor a boundary nem marad meg.
        - Ha egyik sem boundary, akkor csak az egyik marad meg.
        - Ahány redundáns pár van, annyi redundáns manőver maradjon a végső listában.
        """
        redundant_manoeuvres = set()  # Halmaz a redundáns manőverek tárolására
        boundary_manoeuvres = {
            int(m) for m in self.boundary_manoeuvres
        }  # Határon lévő manőverek

        print(f"Redundáns manőverek keresése {method} módszerrel...")

        # Választott módszer szerint redundáns párok keresése
        if method == "pearson":
            redundant_pairs = self.filter_redundant_manoeuvres_pearson()
        elif method == "spearman":
            redundant_pairs = self.filter_redundant_manoeuvres_spearman()
        elif method == "kendall":
            redundant_pairs = self.filter_redundant_manoeuvres_kendall()
        else:
            raise ValueError(
                "Érvénytelen módszer! Használható értékek: 'pearson', 'spearman', 'kendall'"
            )

        redundant_pairs.update(self.filter_by_distance())  # Mindkét módszer eredményei
        for pair in redundant_pairs:
            a, b = int(pair[0]), int(pair[1])  # Kicsomagoljuk a párt

            if a in boundary_manoeuvres and b in boundary_manoeuvres:
                redundant_manoeuvres.add(
                    a
                )  # Ha mindkettő boundary, csak az elsőt tartjuk meg.
            elif a in boundary_manoeuvres:
                redundant_manoeuvres.add(b)  # Ha az 'a' boundary, akkor 'b' marad meg.
            elif b in boundary_manoeuvres:
                redundant_manoeuvres.add(a)  # Ha a 'b' boundary, akkor 'a' marad meg.
            else:
                redundant_manoeuvres.add(
                    a
                )  # Ha egyik sem boundary, az elsőt tartjuk meg.

        redundant_manoeuvres = list({int(m) for m in redundant_manoeuvres})
        redundant_manoeuvre_names = {
            self.reverse_label_mapping.get(m, f"Unknown_{m}")
            for m in redundant_manoeuvres
        }
        print("Boundary manőverek:", boundary_manoeuvres)
        print("Redundáns párok:", redundant_pairs)
        print("Végleges redundáns manőverek:", redundant_manoeuvres)
        print("Végleges redundáns manőverek nevei:", redundant_manoeuvre_names)

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

        return filtered_reduced_data


# np.random.seed(42)
# num_samples = 50000
# num_clusters = 20
# reduced_data, labels, label_mapping = generate_clustered_data(
#     n_samples=num_samples, n_clusters=num_clusters
# )
# # reduced_data, labels = generate_advanced_sinusoidal_spiral_data(n_samples=num_samples, n_clusters=num_clusters)

# # ManoeuvresFiltering osztály inicializálása és filter_manoeuvres meghívása
# filtering = ManoeuvresFiltering(reduced_data=reduced_data, labels=labels, label_mapping=label_mapping)
# filtering.filter_manoeuvres()
