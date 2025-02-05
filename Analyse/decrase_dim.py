import os
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from Config.load_config import tsneplot, dimension


class Visualise:
    def __init__(
        self,
        bottleneck_outputs=None,
        labels=None,
        model_name=None,
        label_mapping=None,
        sign_change_indices=None,
        num_manoeuvres=None,
        n_clusters=None,
        use_cosine_similarity=None,
    ):
        self.bottleneck_outputs = bottleneck_outputs
        self.labels = labels
        self.model_name = model_name
        self.label_mapping = label_mapping
        self.sign_change_indices = sign_change_indices
        self.num_manoeuvres = num_manoeuvres
        self.n_clusters = n_clusters
        self.use_cosine_similarity = use_cosine_similarity
        self.max_iter = 100
        self.tol = 1e-4
        self.tsne_cache_path = "Analyse/tsne_cache.pkl"
        self.plot = tsneplot
        self.dimension = dimension

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
        print(
            f"Data shape: {self.bottleneck_outputs.shape}, Labels shape: {self.labels.shape}"
        )
        assert len(self.bottleneck_outputs) == len(
            self.labels
        ), f"A bottleneck kimenetek ({len(self.bottleneck_outputs)}) és a címkék ({len(self.labels)}) mérete nem egyezik!"

        # Címképzés dinamikusan
        if self.model_name == "VAE":
            self.pca_title = f"PCA - {self.model_name} Bottleneck"
            self.lda_title = f"LDA - {self.model_name} Bottleneck"
            self.tsne_title = f"T-SNE - {self.model_name} Bottleneck"
        else:  # MAE esetén egyszerűsített cím
            self.pca_title = f"PCA - {self.model_name} Bottleneck"
            self.lda_title = f"LDA - {self.model_name} Bottleneck"
            self.tsne_title = f"T-SNE - {self.model_name} Bottleneck"

        # print("PCA Visualization:")
        # self.visualize_with_pca()

        # print("LDA Visualization:")
        # self.visualize_with_lda()

        print("T-SNE Visualization:")
        self.visualize_with_tsne()

        return self.reduced_data

    def visualize_with_pca(self):
        """
        Adatok vizualizálása PCA használatával.

        :param data: A bemeneti adatok (numpy array).
        :param labels: Az egyes mintákhoz tartozó címkék (list vagy numpy array).
        :param title: A grafikon címe.
        :param label_mapping: A címkékhez tartozó szöveges leírások (dict).
        """
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.bottleneck_outputs)

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(self.labels)
        colors = cm.get_cmap("tab20", len(unique_labels))

        # Címkék szerinti szétválasztás
        for i, label in enumerate(unique_labels):
            mask = self.labels == label  # Boolean maszk
            label_data = reduced_data[mask]
            description = next(
                (
                    key.replace("_combined", "")
                    for key, value in self.label_mapping.items()
                    if value == label
                ),
                f"Manoeuvre {label}",
            )
            alphas = np.linspace(
                0.1, 1.0, len(label_data)
            )  # Alpha értékek lineárisan növekednek
            for j in range(len(label_data)):
                plt.scatter(
                    label_data[j, 0],
                    label_data[j, 1],
                    label=description if j == 0 else "",
                    color=colors(i),
                    alpha=alphas[j],
                )

        handles, _ = plt.gca().get_legend_handles_labels()
        for handle in handles:
            handle.set_alpha(1.0)  # Legendben alpha érték kikapcsolása

        plt.title(self.pca_title)
        plt.xlabel("Főkomponens 1")
        plt.ylabel("Főkomponens 2")
        plt.legend(title="Címkék", loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_with_lda(self):
        """
        Adatok vizualizálása LDA használatával, külön színezéssel és folyamatosan növekvő alpha értékekkel.

        :param data: A bemeneti adatok (numpy array).
        :param labels: Az egyes mintákhoz tartozó címkék (list vagy numpy array).
        :param title: A grafikon címe.
        """
        n_features = self.bottleneck_outputs.shape[1]
        n_classes = len(np.unique(self.labels))
        max_components = min(n_features, n_classes - 1)  # LDA maximális dimenziószáma

        if max_components < 1:
            print(
                f"[INFO] LDA nem alkalmazható: túl kevés osztály ({n_classes}) vagy feature ({n_features})."
            )
            return

        lda = LDA(n_components=max_components)
        reduced_data = lda.fit_transform(self.bottleneck_outputs, self.labels)

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(self.labels)
        colors = cm.get_cmap("tab20", len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            label_data = reduced_data[mask]
            description = next(
                (
                    key.replace("_combined", "")
                    for key, value in self.label_mapping.items()
                    if value == label
                ),
                f"Manoeuvre {label}",
            )
            alphas = np.linspace(
                0.1, 1.0, len(label_data)
            )  # Alpha értékek lineárisan növekvő
            for j in range(len(label_data)):
                # Ha csak egy dimenzió van, a második koordináta konstans
                if max_components == 1:
                    plt.scatter(
                        label_data[j, 0],
                        0,
                        label=description if j == 0 else "",
                        color=colors(i),
                        alpha=alphas[j],
                    )

                else:
                    plt.scatter(
                        label_data[j, 0],
                        label_data[j, 1],
                        label=description if j == 0 else "",
                        color=colors(i),
                        alpha=alphas[j],
                    )

        # Legend alpha értékének kikapcsolása
        handles, _ = plt.gca().get_legend_handles_labels()
        for handle in handles:
            handle.set_alpha(1.0)

        plt.title(self.lda_title)
        plt.xlabel("LDA Komponens 1")
        if max_components > 1:
            plt.ylabel("LDA Komponens 2")
        plt.legend(title="Címkék", loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_with_tsne(self):
        """
        Adatok vizualizálása T-SNE használatával.

        :param data: A bemeneti adatok (numpy array).
        :param labels: Az egyes mintákhoz tartozó címkék (list).
        :param title: A grafikon címe.
        """
        perplexity = min(50, max(5, self.bottleneck_outputs.shape[0] // 10))

        # Ellenőrizzük, van-e cache-elt eredmény
        cached_tsne = self.load_cached_tsne()
        if cached_tsne is not None:
            reduced_data = cached_tsne
        else:
            print("Új T-SNE számítás indítása...")
            tsne = TSNE(
                n_components=self.dimension,
                perplexity=perplexity,
                learning_rate=200,
                max_iter=1000,
                random_state=42,
            )
            reduced_data = tsne.fit_transform(self.bottleneck_outputs)
            self.save_tsne_results(reduced_data)  # Mentjük az új eredményt

        self.reduced_data = reduced_data

        if self.plot == 1:
            plt.figure(figsize=(10, 7))
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
                markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]  # Marker lista
                marker_cycle = markers * (
                    num_labels // len(markers) + 1
                )  # Marker ciklus

            if self.dimension == 3:
                ax = plt.figure(figsize=(10, 7)).add_subplot(111, projection="3d")
            elif self.dimension == 2:
                ax = plt.gca()
            else:
                raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")

            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                label_data = reduced_data[mask]
                description = next(
                    (
                        key
                        for key, value in self.label_mapping.items()
                        if value == label
                    ),
                    f"Manoeuvre {label}",
                )
                indices = (
                    self.sign_change_indices[description]
                    if self.sign_change_indices
                    and description in self.sign_change_indices
                    else []
                )
                description = description.replace("_combined", "")
                alphas = np.linspace(
                    0.1, 1.0, label_data.shape[0]
                )  # Alpha értékek lineárisan növekednek
                if self.num_manoeuvres == 1:
                    current_color = "blue"
                    for j in range(label_data.shape[0]):
                        # Színezés előjelváltás alapján
                        if j in indices:
                            current_color = "red" if current_color == "blue" else "blue"

                        if self.dimension == 3:
                            ax.scatter(
                                label_data[j, 0],
                                label_data[j, 1],
                                label_data[j, 2],
                                label=description if j == 0 else "",
                                color=current_color,
                                alpha=alphas[j],
                            )
                        elif self.dimension == 2:
                            ax.scatter(
                                label_data[j, 0],
                                label_data[j, 1],
                                label=description if j == 0 else "",
                                color=current_color,
                                alpha=alphas[j],
                            )
                        else:
                            raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")
                else:
                    for j in range(label_data.shape[0]):
                        if num_labels > 20:
                            if self.dimension == 3:
                                ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label_data[j, 2],
                                    label=description if j == 0 else "",
                                    color=color_list[i],
                                    marker=marker_cycle[i],
                                    alpha=alphas[j],
                                )
                            elif self.dimension == 2:
                                ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label=description if j == 0 else "",
                                    color=color_list[i],
                                    marker=marker_cycle[i],
                                    alpha=alphas[j],
                                )
                            else:
                                raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")
                        else:
                            if self.dimension == 3:
                                ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label_data[j, 2],
                                    label=description if j == 0 else "",
                                    color=color_list[i],
                                    alpha=alphas[j],
                                )
                            elif self.dimension == 2:
                                ax.scatter(
                                    label_data[j, 0],
                                    label_data[j, 1],
                                    label=description if j == 0 else "",
                                    color=color_list[i],
                                    alpha=alphas[j],
                                )
                            else:
                                raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")

            handles, _ = ax.get_legend_handles_labels()
            for handle in handles:
                handle.set_alpha(1.0)  # Legendben alpha érték kikapcsolása

            ax.set_title(self.tsne_title)
            ax.set_xlabel("T-SNE Komponens 1")
            ax.set_ylabel("T-SNE Komponens 2")

            if self.dimension == 3:
                ax.set_zlabel("T-SNE Komponens 3")

            ax.legend(title="Manőverek", loc="best", fontsize="small", markerscale=0.8)
            ax.grid(True)

            plt.tight_layout()
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

            if self.use_cosine_similarity:
                class_clusters, _ = self.cosine_kmeans(class_data)
                technique = "Cosine Similarity"
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
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
            colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))
            for cluster_idx in range(self.n_clusters):
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
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = data[random_indices]

        print(f"max_iter értéke: {self.max_iter}, típusa: {type(self.max_iter)}")
        for _ in range(self.max_iter):
            # Cosine distance számítása
            distances = cosine_distances(data, centroids)

            # Hozzárendelés a legközelebbi centroidhoz
            labels = np.argmin(distances, axis=1)

            # Új centroidok számítása
            new_centroids = []
            for i in range(self.n_clusters):
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
