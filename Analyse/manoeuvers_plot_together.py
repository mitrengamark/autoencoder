import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


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

    unique_labels = list(set(np.concatenate(all_labels)))  # Összes egyedi címke
    num_labels = len(unique_labels)

    # Színek és alakzatok beállítása
    colors = cm.get_cmap("tab20", num_labels)  # Maximum 20 szín
    markers = [
        "o",
        "s",
        "D",
        "X",
        "P",
        "^",
        "v",
        "<",
        ">",
        "*",
        "h",
        "H",
        "p",
        "1",
        "+",
        "x",
        "|",
        "_",
        "d",
        "4",
    ]

    for i, (tsne_data, labels) in enumerate(zip(all_tsne_data, all_labels)):
        for label in np.unique(labels):
            mask = labels == label
            label_data = tsne_data[mask]

            color_idx = unique_labels.index(label) % 20  # Körbeforgatjuk a 20 színt
            marker_idx = (
                unique_labels.index(label) // 20
            )  # Ha több mint 20 címke van, új marker

            ax.scatter(
                label_data[:, 0],
                label_data[:, 1],
                label=label,
                color=colors(color_idx),
                marker=markers[marker_idx % len(markers)],
                alpha=0.7,
                edgecolors="black",
            )

    ax.set_title("T-SNE Vizualizáció - Több manőver", fontsize=14)
    ax.set_xlabel("T-SNE Komponens 1")
    ax.set_ylabel("T-SNE Komponens 2")
    ax.legend(title="Manőverek", loc="best", fontsize="small", markerscale=0.8)
    ax.grid(True)

    plt.tight_layout()
    plt.show()
