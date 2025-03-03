import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
from Config.load_config import folder_name


def plot_all_tsne_data(all_tsne_data, all_labels):
    """
    Kiplotolja az összegyűjtött T-SNE adatokat, minden egyes manővert külön színnel.

    :param all_tsne_data: Lista, amely tartalmazza az összes manőver T-SNE adatait (numpy array-ek listája).
    :param all_labels: Lista, amely tartalmazza az egyes adatokhoz tartozó címkéket (azonos sorrendben az adatokkal).
    """
    if len(all_tsne_data) == 0 or len(all_labels) == 0:
        print("Nincsenek TSNE adatok a plotoláshoz!")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Az all_labels lista átalakítása számlistává, ha stringeket tartalmaz
    unique_labels = sorted(set(label for label in all_labels))
    label_to_number = {label: idx for idx, label in enumerate(unique_labels)}

    # Ellenőrizzük, hogy megfelelő méretű-e a címkék és az adatok halmaza
    all_tsne_data = np.vstack(all_tsne_data)  # Az összes TSNE adat egyesítése
    numeric_labels = np.concatenate([[label_to_number[label]] * tsne.shape[0] for tsne, label in zip(all_tsne_data, all_labels)])

    unique_numeric_labels = np.unique(numeric_labels)

    # Színek és alakzatok beállítása
    colors = cm.get_cmap("tab10", len(unique_numeric_labels))  # Színek száma a címkékhez igazítva
    markers = ["o", "s", "D", "X", "P", "^", "v", "<", ">", "*"]

    for i, label in enumerate(unique_numeric_labels):
        mask = numeric_labels == label
        label_data = all_tsne_data[mask]

        color_idx = i % len(colors.colors)  # Körbeforgatjuk a színeket
        marker_idx = i % len(markers)  # Körbeforgatjuk az alakzatokat

        ax.scatter(
            label_data[:, 0],
            label_data[:, 1],
            label=unique_labels[label],  # Eredeti szöveges címke megjelenítése
            color=colors(color_idx),
            marker=markers[marker_idx],
            alpha=0.7,
            edgecolors="black",
        )

    ax.set_title("T-SNE Vizualizáció - Több manőver", fontsize=14)
    ax.set_xlabel("T-SNE Komponens 1")
    ax.set_ylabel("T-SNE Komponens 2")
    ax.legend(title="Manőverek", loc="best", fontsize="small", markerscale=0.8)
    ax.grid(True)

    plt.tight_layout()

    save_path = f"Results/more_manoeuvres/{folder_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"A plot elmentve: {save_path}")

    plt.show()
