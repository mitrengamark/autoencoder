import os
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from Picture_saving.name_pictures import fig_names
from Picture_saving.save_fig import save_figure
from Config.load_config import (
    tsneplot,
    dimension,
    num_manoeuvres,
    n_clusters,
    use_cosine_similarity,
    coloring,
    step,
    save_fig,
    latent_dim,
    removing_steps,
)


class Visualise:
    def __init__(
        self,
        bottleneck_outputs=None,
        labels=None,
        model_name=None,
        label_mapping=None,
        sign_change_indices=None,
    ):
        self.bottleneck_outputs = bottleneck_outputs
        self.labels = labels
        self.model_name = model_name
        self.label_mapping = label_mapping
        self.sign_change_indices = sign_change_indices
        self.max_iter = 100
        self.tol = 1e-4
        self.tsne_cache_path = "Analyse/tsne_cache.pkl"
        self.plot = tsneplot

    def compute_data_hash(self):
        """Egyedi hash generálása a bemenetek alapján, hogy észleljük a változásokat"""
        hasher = hashlib.sha256()
        hasher.update(self.bottleneck_outputs.tobytes())
        hasher.update(self.labels.tobytes())
        return hasher.hexdigest()

    def load_cached_tsne(self):
        """Betölti a korábban kiszámított T-SNE eredményeket, ha a bemenet változatlan"""
        if os.path.exists(self.tsne_cache_path):
            with open(self.tsne_cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache["hash"] == self.compute_data_hash():
                print("Korábban kiszámított T-SNE eredmények betöltése...")
                return cache["tsne_data"]
        return None

    def visualize_bottleneck(self):
        """
        Vizualizálja a bottleneck kimeneteket PCA vagy T-SNE segítségével.

        :param bottleneck_outputs: A bottleneck által generált adatok (numpy array).
        :param labels: Az egyes mintákhoz tartozó címkék (list vagy numpy array).
        :param model_name: A modell neve (pl. "VAE" vagy "MAE").
        :param label_mapping: A címkékhez tartozó szöveges leírások (dict).
        """
        assert len(self.bottleneck_outputs) == len(
            self.labels
        ), f"A bottleneck kimenetek ({len(self.bottleneck_outputs)}) és a címkék ({len(self.labels)}) mérete nem egyezik!"

        self.tsne_title = f"T-SNE - {self.model_name} Bottleneck"

        print("T-SNE Visualization:")
        self.visualize_with_tsne()

        return self.reduced_data

    def calculate_tsne(self, data):
        perplexity = min(50, max(5, data.shape[0] // 10))

        # Ellenőrizzük, van-e cache-elt eredmény
        cached_tsne = self.load_cached_tsne()
        if cached_tsne is not None:
            reduced_data = cached_tsne
        else:
            print("Új T-SNE számítás indítása...")
            tsne = TSNE(
                n_components=dimension,
                perplexity=perplexity,
                learning_rate=200,
                max_iter=1000,
                random_state=42,
            )
            reduced_data = tsne.fit_transform(data)
            self.save_tsne_results(reduced_data)  # Mentjük az új eredményt

        self.reduced_data = reduced_data

    def visualize_with_tsne(self):
        """
        Adatok vizualizálása T-SNE használatával.

        :param data: A bemeneti adatok (numpy array).
        :param labels: Az egyes mintákhoz tartozó címkék (list).
        :param title: A grafikon címe.
        """
        if latent_dim == 2:
            self.reduced_data = self.bottleneck_outputs
        else:
            self.calculate_tsne(self.bottleneck_outputs)

        print(f"Reduced data shape: {self.reduced_data.shape}")

        if self.plot == 1:
            fig = plt.figure(figsize=(16, 8))
            unique_labels = np.unique(self.labels)
            num_labels = len(unique_labels)

            if num_labels <= 10:
                colors = cm.get_cmap("tab10", num_labels)
                color_list = [colors(i) for i in range(num_labels)]
            elif num_labels <= 20:
                colors = cm.get_cmap("tab20", num_labels)
                color_list = [colors(i) for i in range(num_labels)]
            else:
                color_list = cm.get_cmap("nipy_spectral", num_labels)(
                    np.linspace(0, 1, num_labels)
                )

            if dimension == 3:
                ax = fig.add_subplot(111, projection="3d")
            elif dimension == 2:
                ax = plt.gca()
            else:
                raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")

            sc = None
            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                label_data = self.reduced_data[mask]

                label_data = np.delete(
                    label_data, np.arange(0, len(label_data), removing_steps), axis=0
                )

                description = next(
                    (
                        key
                        for key, value in self.label_mapping.items()
                        if value == label
                    ),
                    f"Manoeuvre {label}",
                )
                description = description.replace("_combined", "")
                if num_manoeuvres == 1:
                    if coloring == 1:
                        indices = (
                            self.sign_change_indices[description]
                            if self.sign_change_indices
                            and description in self.sign_change_indices
                            else []
                        )
                        current_color = "blue"
                        for j in range(label_data.shape[0]):
                            # Színezés előjelváltás alapján
                            if j in indices:
                                current_color = (
                                    "red" if current_color == "blue" else "blue"
                                )

                            if dimension == 3:
                                ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label_data[j, 2],
                                    label=description if j == 0 else "",
                                    color=current_color,
                                    alpha=1,
                                    facecolors="none",
                                )
                            elif dimension == 2:
                                ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label=description if j == 0 else "",
                                    color=current_color,
                                    alpha=1,
                                    facecolors="none",
                                )
                            else:
                                raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")
                    else:
                        cmap = cm.get_cmap(
                            "turbo", label_data.shape[0] // step + 1
                        )  # Lépésenként színváltás
                        for j in range(label_data.shape[0]):
                            color_index = j // step  # Minden `step` elem után más szín
                            color = cmap(color_index)

                            if dimension == 3:
                                sc = ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label_data[j, 2],
                                    label=description if j == 0 else "",
                                    color=color,
                                    alpha=1,
                                    facecolors="none",
                                )
                            elif dimension == 2:
                                sc = ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label=description if j == 0 else "",
                                    color=color,
                                    alpha=1,
                                    facecolors="none",
                                )
                            else:
                                raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")
                else:
                    for j in range(label_data.shape[0]):
                        if dimension == 3:
                            ax.scatter(
                                label_data[j, 0],
                                label_data[j, 1],
                                label_data[j, 2],
                                label=description if j == 0 else "",
                                color=color_list[i],
                                alpha=1,
                                facecolors="none",
                            )
                        elif dimension == 2:
                            ax.scatter(
                                label_data[j, 0],
                                label_data[j, 1],
                                label=description if j == 0 else "",
                                color=color_list[i],
                                alpha=1,
                                facecolors="none",
                            )
                        else:
                            raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")

            # handles, _ = ax.get_legend_handles_labels()
            # for handle in handles:
            #     handle.set_alpha(1.0)  # Legendben alpha érték kikapcsolása

            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax)  # Színskála hozzáadása
                cbar.set_label("Idősorrend (lépések)")
                cbar.mappable.set_cmap("turbo")

            ax.set_title(self.tsne_title)
            ax.set_xlabel("T-SNE Komponens 1")
            ax.set_ylabel("T-SNE Komponens 2")

            if dimension == 3:
                ax.set_zlabel("T-SNE Komponens 3")

            ax.legend(title="Manőverek", loc="best", fontsize="small", markerscale=0.8)
            ax.grid(True)

            plt.tight_layout()

            if save_fig == 1:
                save_path = fig_names(description)
                save_figure(fig, save_path)
            else:
                plt.show()

    def save_tsne_results(self, tsne_data):
        """Elmenti a kiszámított T-SNE eredményeket fájlba"""
        cache = {"hash": self.compute_data_hash(), "tsne_data": tsne_data}
        with open(self.tsne_cache_path, "wb") as f:
            pickle.dump(cache, f)

    def kmeans_clustering(self):
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            # Az adott osztályhoz tartozó adatok kiszűrése
            mask = self.labels == label
            class_data = self.bottleneck_outputs[mask]
            reduced_class_data = self.reduced_data[mask]

            if use_cosine_similarity:
                class_clusters, _ = self.cosine_kmeans(class_data)
                technique = "Cosine Similarity"
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                class_clusters = kmeans.fit_predict(class_data)
                technique = "Euclidean Distance"

            description = next(
                (
                    key.replace("_combined", "")
                    for key, value in self.label_mapping.items()
                    if value == label
                ),
                f"Manoeuvre {label}",
            )

            # Vizualizáció az osztályon belüli klaszterekkel
            plt.figure(figsize=(8, 6))
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
            for cluster_idx in range(n_clusters):
                cluster_mask = class_clusters == cluster_idx
                plt.scatter(
                    reduced_class_data[cluster_mask, 0],
                    reduced_class_data[cluster_mask, 1],
                    label=f"Klaszter {cluster_idx + 1}",
                    color=colors[cluster_idx],
                )

            plt.title(f"{description} klaszterezése (K-Means - {technique})")
            plt.xlabel("Főkomponens 1")
            plt.ylabel("Főkomponens 2")
            plt.legend(title="Klaszterek", loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def cosine_kmeans(self, data):
        # data = normalize(data, norm='l2')

        # Véletlenszerű centroid inicializálás
        n_samples = data.shape[0]
        random_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = data[random_indices]

        print(f"max_iter értéke: {self.max_iter}, típusa: {type(self.max_iter)}")
        for _ in range(self.max_iter):
            # Cosine distance számítása
            distances = cosine_distances(data, centroids)

            # Hozzárendelés a legközelebbi centroidhoz
            labels = np.argmin(distances, axis=1)

            # Új centroidok számítása
            new_centroids = []
            for i in range(n_clusters):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    centroid = centroid / np.linalg.norm(
                        centroid
                    )  # Unit norm normalizáció
                    new_centroids.append(centroid)
                else:
                    # Üres klaszter esetén random újra inicializálás
                    new_centroids.append(data[np.random.choice(n_samples)])

            new_centroids = np.array(new_centroids)

            # Konvergencia ellenőrzése
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break
            centroids = new_centroids

        return labels, centroids
