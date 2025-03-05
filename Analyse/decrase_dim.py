import os
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from Config.load_config import (
    tsneplot,
    dimension,
    num_manoeuvres,
    coloring,
    step,
    save_fig,
    latent_dim,
    folder_name,
    selected_manoeuvres,
    tsne_dir,
    remove_start,
    manoeuvers_tsne_dir,
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

    def compute_data_hash(self):
        """Egyedi hash generálása a bemenetek alapján, hogy észleljük a változásokat"""
        hasher = hashlib.sha256()
        hasher.update(self.bottleneck_outputs.tobytes())
        hasher.update(self.labels.tobytes())
        return hasher.hexdigest()

    def find_existing_tsne(self, data_hash):
        """Megkeresi, hogy van-e már elmentett T-SNE fájl az adott hash alapján"""
        search_dir = manoeuvers_tsne_dir if num_manoeuvres == 1 else tsne_dir
        for filename in os.listdir(search_dir):
            file_path = os.path.join(search_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    cache = pickle.load(f)
                if cache["hash"] == data_hash:
                    print(f"Megtalált korábban elmentett T-SNE fájl: {file_path}")
                    return cache["tsne_data"]
            except Exception as e:
                print(f"Hiba történt a(z) {file_path} fájl olvasásakor: {e}")
        return None
    
    def save_tsne_results(self, tsne_data, data_hash):
        """Elmenti a kiszámított T-SNE eredményeket egy egyedi nevű fájlba"""
        save_dir = manoeuvers_tsne_dir if num_manoeuvres == 1 else tsne_dir
        label = selected_manoeuvres[0] if num_manoeuvres == 1 else "multiple"
        filename = f"{save_dir}/tsne_{label}_{data_hash}.pkl"

        cache = {"hash": data_hash, "tsne_data": tsne_data}
        with open(filename, "wb") as f:
            pickle.dump(cache, f)

        print(f"T-SNE adatok mentve: {filename}")

    def calculate_tsne(self, data):
        """T-SNE számítása és mentése, ha szükséges"""
        data_hash = self.compute_data_hash()

        # Ellenőrizzük, van-e cache-elt eredmény
        cached_tsne = self.find_existing_tsne(data_hash)
        if cached_tsne is not None:
            reduced_data = cached_tsne
        else:
            print("Új T-SNE számítás indítása...")
            perplexity = min(50, max(5, data.shape[0] // 10))
            tsne = TSNE(
                n_components=dimension,
                perplexity=perplexity,
                learning_rate=200,
                max_iter=1000,
                random_state=42,
            )
            reduced_data = tsne.fit_transform(data)
            self.save_tsne_results(reduced_data, data_hash)  # Mentjük az új eredményt

        self.reduced_data = reduced_data

        # **ELSŐ 2500 ADAT ELTÁVOLÍTÁSA MINDEN MANŐVERBŐL**
        if remove_start == 1 and num_manoeuvres > 1:
            new_data = []
            new_labels = []
            unique_labels = np.unique(self.labels)

            for label in unique_labels:
                mask = self.labels == label
                label_data = self.reduced_data[mask]

                if len(label_data) > 2500:
                    new_data.append(label_data[2500:])  # Első X törlése
                    new_labels.append(self.labels[mask][2500:])

            self.reduced_data = np.vstack(new_data)
            self.labels = np.concatenate(new_labels)

    def visualize_with_tsne(self, plot=tsneplot):
        """
        Adatok vizualizálása T-SNE használatával.

        :param data: A bemeneti adatok (numpy array).
        :param labels: Az egyes mintákhoz tartozó címkék (list).
        :param title: A grafikon címe.
        """
        assert len(self.bottleneck_outputs) == len(
            self.labels
        ), f"A bottleneck kimenetek ({len(self.bottleneck_outputs)}) és a címkék ({len(self.labels)}) mérete nem egyezik!"

        self.tsne_title = f"T-SNE - {self.model_name} Bottleneck"

        if plot == 1:
            print("T-SNE Visualization:")

        if latent_dim == 2:
            self.reduced_data = self.bottleneck_outputs
        else:
            self.calculate_tsne(self.bottleneck_outputs)

        print(f"Reduced data shape: {self.reduced_data.shape}")

        if plot == 1:
            fig = plt.figure(figsize=(16, 8))
            unique_labels = np.unique(self.labels)
            num_labels = len(unique_labels)

            base_colors = cm.get_cmap("tab20", 20)  # 20 alap szín
            markers = [
                "o",
                "D",
                "X",
                "+",
                "s",
                "^",
                "P",
                "*",
                "v",
                "<",
                ">",
                "p",
                "h",
                "H",
                "d",
                "1",
                "|",
                "x",
                "8",
                "_",
            ]  # 20 alakzat

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

                description = next(
                    (
                        key
                        for key, value in self.label_mapping.items()
                        if value == label
                    ),
                    f"Manoeuvre {label}",
                )
                description = description.replace("_combined", "")

                color_index = i % 20
                marker_index = i // 20

                color = base_colors(color_index)
                marker = markers[marker_index % len(markers)]

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
                    if dimension == 3:
                        ax.scatter(
                            label_data[j, 0],
                            label_data[j, 1],
                            label_data[j, 2],
                            label=description if j == 0 else "",
                            color=color,
                            alpha=1,
                            facecolors="none",
                            marker=marker,
                        )
                    elif dimension == 2:
                        ax.scatter(
                            label_data[:, 0],
                            label_data[:, 1],
                            label=description,
                            color=color,
                            alpha=1,
                            facecolors="none",
                            marker=marker,
                        )
                    else:
                        raise ValueError("A dimenziószám csak 2 vagy 3 lehet!")

            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax)  # Színskála hozzáadása
                cbar.set_label("Idősorrend (lépések)")
                cbar.mappable.set_cmap("turbo")

            ax.set_title(self.tsne_title)
            ax.set_xlabel("T-SNE Komponens 1")
            ax.set_ylabel("T-SNE Komponens 2")

            if dimension == 3:
                ax.set_zlabel("T-SNE Komponens 3")

            if num_labels < 30:
                ax.legend(
                    title="Manőverek", loc="best", fontsize="small", markerscale=0.8
                )
            else:
                ax.legend(
                    title=folder_name, loc="best", fontsize="small", markerscale=0.8
                )

            ax.grid(True)

            plt.tight_layout()

            if save_fig:
                if num_manoeuvres == 1:
                    filename = f"Results/{folder_name}/{selected_manoeuvres[0]}.png"
                else:
                    filename = f"Results/more_manoeuvres/{folder_name}.png"

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)

            # plt.show()
        if num_manoeuvres > 1:
            return self.reduced_data, self.labels
        else:
            return self.reduced_data, folder_name
