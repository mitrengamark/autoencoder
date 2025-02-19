import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from Config.load_config import inconsistent_points_distance, time_difference


def filter_inconsistent_points(data, labels):
    """
    Azokat az adatpontokat töröljük, amelyek közel helyezkednek el egymáshoz térben,
    de nagy az időrendi különbségük. A nagyobb mértékben eltérő pontot töröljük.

    :param data: Az adatpontok (N x D dimenziós numpy tömb)
    :param labels: Az időrendi címkék (N hosszú numpy tömb)
    :return: A megtisztított adat és címke tömb, valamint a törölt pontok indexei
    """
    print(f"Idő különbségek keresése...")

    # Távolságmátrix számítása (Euklidészi távolság)
    dist_matrix = distance.cdist(data, data, metric="euclidean")

    # Indexek a törléshez
    to_remove = set()

    # Minden adatpár vizsgálata
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if (
                dist_matrix[i, j] < inconsistent_points_distance
            ):  # Ha közel vannak egymáshoz térben
                if (
                    abs(labels[i] - labels[j]) > time_difference
                ):  # Ha időben nagy az eltérés
                    if labels[i] < labels[j]:  # A kisebb időértékűt töröljük
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

    # Maszk létrehozása a törlendő indexek alapján
    mask = np.array([i not in to_remove for i in range(len(data))])

    # Szűrt adatok
    filtered_data = data[mask]
    filtered_labels = labels[mask]

    # Törölt (outlier) adatok
    outlier_data = data[list(to_remove)]

    print(f"Törölt adatok száma: {len(to_remove)}")

    # Új adathalmaz vizualizálása
    plt.figure(figsize=(8, 6))

    # Megmaradt pontok (egyszínű)
    plt.scatter(
        filtered_data[:, 0],
        filtered_data[:, 1],
        color="blue",
        alpha=0.5,
        label="Megmaradt adatok",
    )

    # Törölt outlier pontok (piros)
    plt.scatter(
        outlier_data[:, 0],
        outlier_data[:, 1],
        color="red",
        alpha=0.7,
        label="Kiszűrt outlierek",
    )

    plt.legend()
    plt.title("Szűrt adatok - időrendi alapú tisztítás")
    plt.show()

    return filtered_data, filtered_labels


def filter_outliers_by_grid(data, labels, grid_size=10, threshold=4000):
    """
    Egy adott méretű rácsban (grid) eltávolítja az átlagtól abszolút értékben nagyban eltérő adatokat.

    :param data: Az adatpontok (N x D dimenziós numpy tömb).
    :param labels: Az időrendi címkék (N hosszú numpy tömb).
    :param grid_size: A rács felbontása (minél nagyobb, annál több cella lesz).
    :param threshold: Az abszolút eltérés küszöbértéke.
    :return: A megtisztított adat és címke tömb, valamint a törölt pontok indexei.
    """

    print(f"Grid-alapú outlier szűrés ({grid_size}x{grid_size})...")

    # A rács (grid) határainak meghatározása
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    x_step = (x_max - x_min) / grid_size
    y_step = (y_max - y_min) / grid_size

    # Minden grid cellához tartozó indexeket tároljuk
    grid_cells = {}

    for i in range(len(data)):
        x_idx = int((data[i, 0] - x_min) / x_step)
        y_idx = int((data[i, 1] - y_min) / y_step)
        cell = (x_idx, y_idx)

        if cell not in grid_cells:
            grid_cells[cell] = []

        grid_cells[cell].append(i)

    # Indexek a törléshez
    to_remove = set()

    for cell, indices in grid_cells.items():
        if len(indices) > 1:  # Csak olyan cellákat vizsgálunk, ahol van több adatpont
            local_labels = labels[indices]
            avg_label = np.mean(local_labels)

            for i in indices:
                if abs(labels[i] - avg_label) > threshold:  # Ha abszolút eltérés nagy
                    to_remove.add(i)

    # Maszk létrehozása a törlendő indexek alapján
    mask = np.array([i not in to_remove for i in range(len(data))])

    # Szűrt adatok
    filtered_data = data[mask]
    filtered_labels = labels[mask]

    # Törölt (outlier) adatok
    outlier_data = data[list(to_remove)]

    print(f"Törölt adatok száma: {len(to_remove)}")

    # Új adathalmaz vizualizálása
    plt.figure(figsize=(8, 6))

    # Megmaradt pontok (egyszínű)
    plt.scatter(
        filtered_data[:, 0],
        filtered_data[:, 1],
        color="blue",
        alpha=0.5,
        label="Megmaradt adatok",
    )

    # Törölt outlier pontok (piros)
    plt.scatter(
        outlier_data[:, 0],
        outlier_data[:, 1],
        color="red",
        alpha=0.7,
        label="Kiszűrt outlierek",
    )

    plt.legend()
    plt.title("Grid-alapú időrendi outlier szűrés")
    plt.show()

    return filtered_data, filtered_labels
