import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict

# from random_data import (
#     generate_clustered_data,
#     generate_advanced_sinusoidal_spiral_data,
# )
# import sys
# import os

# # Hozzáadjuk a projekt gyökérkönyvtárát a Python elérési útvonalához
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from load_config import eps, min_samples, n_clusters


class ManoeuvresFiltering:
    def __init__(
        self,
        reduced_data=None,
        labels=None,
        label_mapping=None,
        eps=eps,
        min_samples=min_samples,
        n_clusters=n_clusters,
    ):
        self.reduced_data = reduced_data
        self.labels = labels
        self.label_mapping = label_mapping
        self.boundary_manoeuvres = set()
        self.boundary_indices = None
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters

    def filter_manoeuvres(self):
        self.find_boundary_manoeuvres()
        self.plot_boundary()
        # self.dbscan_clustering()
        # self.plot_dbscan_clusters()
        cluster_info = self.kmeans_clustering()
        self.plot_kmeans_clusters()
        self.plot_cluster_info(cluster_info)
        redundant_manoeuvres = self.filter_redundant_manoeuvres()
        self.check_cluster_dominance()
        self.filter_by_distance()

    def find_boundary_manoeuvres(self):
        """
        Meghatározza a határon lévő manővereket a konvex burok segítségével.

        :param reduced_data: A látenstér 2D-s adatai (numpy array, shape: [n_samples, 2])
        :param labels: A pontokhoz tartozó manőverek címkéi
        :return: Egy halmaz a határon lévő manőverek címkéivel
        """
        hull = ConvexHull(self.reduced_data)
        self.boundary_indices = hull.vertices  # A konvex burok indexei

        self.boundary_manoeuvres = {
            int(label) for label in self.labels[self.boundary_indices]
        }
        reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        boundary_manoeuvre_names = {
            reverse_label_mapping.get(label, f"Unknown_{label}")
            for label in self.boundary_manoeuvres
        }
        print("Határon lévő manőverek:", boundary_manoeuvre_names)

    # -----------------------------------clustering-----------------------------------

    def dbscan_clustering(self):
        """
        DBSCAN klaszterezés végrehajtása, majd klaszterenként meghatározza,
        hogy mely manőverek találhatók benne és hány adatpontjuk van.
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = dbscan.fit_predict(self.reduced_data)

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
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans_labels = kmeans.fit_predict(self.reduced_data)

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

    def plot_boundary(self):
        """
        Kirajzolja a konvex burkot és a határon lévő manővereket.

        :param reduced_data: A látenstér 2D-s adatai
        :param labels: A pontokhoz tartozó manőverek címkéi
        :param boundary_indices: A konvex burok által meghatározott indexek
        """
        plt.figure(figsize=(10, 6))

        # Egyedi színek a címkékhez
        unique_labels = np.unique(self.labels)
        color_map = plt.get_cmap("tab20", len(unique_labels))
        label_colors = {label: color_map(i) for i, label in enumerate(unique_labels)}

        # Minden manőver pontjai
        for label in unique_labels:
            mask = self.labels == label
            plt.scatter(
                self.reduced_data[mask, 0],
                self.reduced_data[mask, 1],
                color=label_colors[label],
                label=label,
                alpha=0.5,
                s=10,
            )

        # Határon lévő pontok kiemelése
        plt.scatter(
            self.reduced_data[self.boundary_indices, 0],
            self.reduced_data[self.boundary_indices, 1],
            color="black",
            edgecolors="white",
            s=40,
            label="Határ",
        )

        # Konvex burok kirajzolása
        hull = ConvexHull(self.reduced_data)
        for simplex in hull.simplices:
            plt.plot(self.reduced_data[simplex, 0], self.reduced_data[simplex, 1], "k-")

        plt.title("Határon lévő manőverek azonosítása")
        plt.xlabel("T-SNE Komponens 1")
        plt.ylabel("T-SNE Komponens 2")
        plt.legend(loc="best", fontsize="small", markerscale=0.7)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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

    def filter_redundant_manoeuvres(self, threshold=0.9):
        """
        Kiszűri a redundáns manővereket a Pearson-korreláció alapján.
        threshold: A minimális korrelációs érték, amely felett két manőver redundánsnak számít.
        """
        print("Redundáns manőverek keresése...")

        # Az adatok pandas DataFrame-be alakítása
        df = pd.DataFrame(
            self.reduced_data,
            columns=[f"comp_{i}" for i in range(self.reduced_data.shape[1])],
        )
        df["labels"] = self.labels

        # Kiszámítjuk a korrelációs mátrixot
        correlation_matrix = df.drop("labels", axis=1).corr()

        # Keresünk redundáns párokat
        redundant_pairs = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    redundant_pairs.add(
                        (correlation_matrix.columns[i], correlation_matrix.columns[j])
                    )

        print(f"Redundáns manőver párok: {redundant_pairs}")

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
                    # Hozzáadjuk a párokat rendezett formában, hogy ne legyen (3,5) és (5,3) külön
                    redundant_pairs.add(tuple(sorted((self.labels[i], self.labels[j]))))

        print(f"Redundáns manőver párok (távolság alapú): {redundant_pairs}")
        return redundant_pairs


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
