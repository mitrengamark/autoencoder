import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def find_boundary_manoeuvres(reduced_data, labels):
    """
    Meghatározza a határon lévő manővereket a konvex burok segítségével.

    :param reduced_data: A látenstér 2D-s adatai (numpy array, shape: [n_samples, 2])
    :param labels: A pontokhoz tartozó manőverek címkéi
    :return: Egy halmaz a határon lévő manőverek címkéivel
    """
    hull = ConvexHull(reduced_data)
    boundary_indices = hull.vertices  # A konvex burok indexei

    boundary_manoeuvres = set(labels[boundary_indices])  # A határon lévő manőverek címkéi
    return boundary_manoeuvres, boundary_indices


def plot_boundary(reduced_data, labels, boundary_indices):
    """
    Kirajzolja a konvex burkot és a határon lévő manővereket.

    :param reduced_data: A látenstér 2D-s adatai
    :param labels: A pontokhoz tartozó manőverek címkéi
    :param boundary_indices: A konvex burok által meghatározott indexek
    """
    plt.figure(figsize=(10, 6))
    
    # Egyedi színek a címkékhez
    unique_labels = np.unique(labels)
    color_map = plt.get_cmap("tab20", len(unique_labels))
    label_colors = {label: color_map(i) for i, label in enumerate(unique_labels)}

    # Minden manőver pontjai
    for label in unique_labels:
        mask = labels == label
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                    color=label_colors[label], label=label, alpha=0.5, s=10)

    # Határon lévő pontok kiemelése
    plt.scatter(reduced_data[boundary_indices, 0], reduced_data[boundary_indices, 1], 
                color='black', edgecolors='white', s=40, label="Határ")

    # Konvex burok kirajzolása
    hull = ConvexHull(reduced_data)
    for simplex in hull.simplices:
        plt.plot(reduced_data[simplex, 0], reduced_data[simplex, 1], 'k-')

    plt.title("Határon lévő manőverek azonosítása")
    plt.xlabel("T-SNE Komponens 1")
    plt.ylabel("T-SNE Komponens 2")
    plt.legend(loc="best", fontsize="small", markerscale=0.7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Ha közvetlenül futtatod a fájlt:
# if __name__ == "__main__":
#     # Példa bemenetek (valódi adatokat kell behelyettesíteni!)
#     np.random.seed(42)
#     sample_data = np.random.rand(200, 2) * 100  # 200 véletlen pont a [0,100] tartományban
#     sample_labels = np.random.choice(["A", "B", "C", "D", "E"], size=200)  # Véletlen címkék

#     # Határon lévő manőverek azonosítása
#     boundary_manoeuvres, boundary_indices = find_boundary_manoeuvres(sample_data, sample_labels)
#     print("Határon lévő manőverek:", boundary_manoeuvres)

#     # Vizualizáció
#     plot_boundary(sample_data, sample_labels, boundary_indices)
