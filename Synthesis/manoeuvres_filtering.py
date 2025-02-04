import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from collections import defaultdict


class ManoeuvresFiltering:
    def __init__(self, reduced_data=None, labels=None, eps=10, min_samples=100):
        self.reduced_data = reduced_data
        self.labels = labels
        self.boundary_manoeuvres = set()
        self.boundary_indices = None
        self.eps = eps
        self.min_samples = min_samples

    def filter_manoeuvres(self):
        self.find_boundary_manoeuvres()
        self.plot_boundary()
        self.dbscan_clustering()
        self.plot_dbscan_clusters()

    def find_boundary_manoeuvres(self):
        """
        Meghatározza a határon lévő manővereket a konvex burok segítségével.

        :param reduced_data: A látenstér 2D-s adatai (numpy array, shape: [n_samples, 2])
        :param labels: A pontokhoz tartozó manőverek címkéi
        :return: Egy halmaz a határon lévő manőverek címkéivel
        """
        hull = ConvexHull(self.reduced_data)
        self.boundary_indices = hull.vertices  # A konvex burok indexei

        self.boundary_manoeuvres = set(
            self.labels[self.boundary_indices]
        )  # A határon lévő manőverek címkéi
        print("Határon lévő manőverek:", self.boundary_manoeuvres)

    def remove_redundant_manoeuvres(self):
        """
        Ha két manőver közel azonos térrészt fed le, akkor azt amelyik nem tartozik a határon lévő manőverek közé, el kell dobni.
        Ha egyik sem tartozik a határon lévő manőverek közé, akkor mindegy melyiket dobjuk el.
        """
        unique_labels = np.unique(self.labels)
        remaining_labels = set(unique_labels)  # Kezdetben az összes manőver megmarad
        to_remove = set()

        # Két manőver közötti távolságok számítása
        for i, label_1 in enumerate(unique_labels):
            if label_1 in to_remove:  # Ha már el lett távolítva, nem vizsgáljuk
                continue
            mask_1 = self.labels == label_1
            points_1 = self.reduced_data[mask_1]

            for j, label_2 in enumerate(unique_labels):
                if i >= j or label_2 in to_remove:  # Ne nézzük kétszer ugyanazt a párt
                    continue
                mask_2 = self.labels == label_2
                points_2 = self.reduced_data[mask_2]

                # Két manőver közötti távolság kiszámítása
                dist_matrix = euclidean_distances(points_1, points_2)
                mean_distance = np.mean(dist_matrix)
                mean_distances = []
                mean_distances.append(mean_distance)

                # Ha az átlagos távolság nagyon kicsi, akkor ezek lefedik ugyanazt a térrészt
                if mean_distance < 60:  # A küszöb állítható
                    if label_1 in self.boundary_manoeuvres:
                        to_remove.add(label_2)
                    elif label_2 in self.boundary_manoeuvres:
                        to_remove.add(label_1)
                    else:
                        to_remove.add(
                            label_2
                        )  # Ha egyik sem határon lévő, akkor egyet eltávolítunk

            print(f"Manőverek közötti átlagos távolságok: {mean_distances}")

        remaining_labels -= to_remove
        print(f"Eltávolított redundáns manőverek: {to_remove}")
        print(f"Megmaradt manőverek: {remaining_labels}")

        # Frissítjük az adatokat a megmaradt címkék szerint
        mask_remaining = np.isin(self.labels, list(remaining_labels))
        self.reduced_data = self.reduced_data[mask_remaining]
        self.labels = self.labels[mask_remaining]

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
            manoeuvre = self.labels[idx]
            clusters_info[cluster][manoeuvre] += 1

        print("DBSCAN klaszterek és tartalmuk:")
        for cluster, manoeuvres in clusters_info.items():
            print(f"Klaszter {cluster}: {dict(manoeuvres)}")

        return clusters_info

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
