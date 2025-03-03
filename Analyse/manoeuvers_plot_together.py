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

    print("all_labels tartalma:", all_labels)
    print("Típusok:", [type(label) for label in all_labels])

    # Konvertáljuk az all_labels elemeit stringgé, ha ndarray típusúak
    all_labels = [l.item() if isinstance(l, np.ndarray) else l for l in all_labels]

    # Egyedi címkék és a számokhoz való hozzárendelés
    unique_labels = list(set(all_labels))
    label_to_number = {label: i for i, label in enumerate(unique_labels)}

    # Ellenőrizzük a TSNE adatok struktúráját
    print("TSNE adatok méretei:", [tsne.shape for tsne in all_tsne_data])

    # Még az összefűzés előtt generáljuk a numeric_labels tömböt!
    numeric_labels_list = [
        np.full((tsne.shape[0],), label_to_number[label])
        for tsne, label in zip(all_tsne_data, all_labels)
    ]

    # Most fűzzük össze a numeric_labels tömböt
    numeric_labels = np.concatenate(numeric_labels_list)

    # Most fűzzük össze a TSNE adatokat
    all_tsne_data = np.vstack(all_tsne_data)

    print("Összefűzött adatok alakja:", all_tsne_data.shape)
    print("Címkék alakja:", numeric_labels.shape)
    print("Címkék:", numeric_labels)

    unique_numeric_labels = np.unique(numeric_labels)

    # Színek és alakzatok beállítása
    colors = cm.get_cmap(
        "tab20", len(unique_numeric_labels)
    )  # Színek száma a címkékhez igazítva
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

    for i, label in enumerate(unique_numeric_labels):
        mask = numeric_labels == label
        label_data = all_tsne_data[mask]

        color_idx = i % len(colors.colors)  # Körbeforgatjuk a színeket
        marker_idx = (i // 20) % len(markers)  # Minden 20. label után változik a marker

        ax.scatter(
            label_data[:, 0],
            label_data[:, 1],
            label=unique_labels[label],  # Eredeti szöveges címke megjelenítése
            color=colors(color_idx),
            marker=markers[marker_idx],
            alpha=0.7,
            facecolors="none",
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
