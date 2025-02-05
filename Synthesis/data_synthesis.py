import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from Config.load_config import grid, max_sample


def remove_redundant_data(latent_data, grid_size=grid, max_sample=max_sample):
    """
    Eltávolítja a túl sűrűn előforduló adatokat a látenstérben.

    :param latent_data: Látenstér adatok (N, 2) alakban.
    :param grid_size: Hány cellára osztjuk a teret.
    :param max_sample: Ha egy cellában több mint max_sample adatpont van, ritkítunk.
    :return: Szűrt látenstér adatok.
    """
    # A látenstér minimum és maximum értékei
    x_min, y_min = np.min(latent_data, axis=0)
    x_max, y_max = np.max(latent_data, axis=0)

    # Rács létrehozása
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)

    # Hány adat van az egyes rács cellákban?
    grid_counts, _, _ = np.histogram2d(
        latent_data[:, 0], latent_data[:, 1], bins=[x_bins, y_bins]
    )

    # Magas sűrűségű területek azonosítása
    dense_cells = np.argwhere(grid_counts > max_sample)

    # Távolság-alapú szűrés
    filtered_latent_data = []
    # kd_tree = cKDTree(latent_data)
    for point in latent_data:
        x_idx = np.searchsorted(x_bins, point[0]) - 1
        y_idx = np.searchsorted(y_bins, point[1]) - 1

        if [x_idx, y_idx] in dense_cells.tolist():
            # Ha túl sűrű, csak véletlenszerűen veszünk ki belőle egy kis részt
            if np.random.rand() < 0.3:  # Csak 30%-át hagyjuk meg
                filtered_latent_data.append(point)
        else:
            filtered_latent_data.append(point)

    filtered_latent_data = np.array(filtered_latent_data)
    plot_removed_data(latent_data, filtered_latent_data)

    return filtered_latent_data


def plot_removed_data(latent_data, filtered_latent_data):
    # Vizualizáció az eredeti és szűrt adatokkal
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(latent_data[:, 0], latent_data[:, 1], s=5, alpha=1)
    plt.title("Eredeti Látenstér")
    plt.xlabel("T-SNE Komponens 1")
    plt.ylabel("T-SNE Komponens 2")

    plt.subplot(1, 2, 2)
    plt.scatter(filtered_latent_data[:, 0], filtered_latent_data[:, 1], s=5, alpha=1)
    plt.title("Szűrt Látenstér (Redundáns Adatok Eltávolítva)")
    plt.xlabel("T-SNE Komponens 1")
    plt.ylabel("T-SNE Komponens 2")

    plt.tight_layout()
    plt.show()

    return filtered_latent_data
