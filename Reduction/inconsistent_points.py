import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def filter_inconsistent_points(data, labels, threshold=0.1):
    """
    Azokat az adatpontokat töröljük, amelyek közel helyezkednek el egymáshoz térben,
    de nagy az időrendi különbségük. A kisebb címkéjű pontot töröljük.

    :param data: Az adatpontok (N x D dimenziós numpy tömb)
    :param labels: Az időrendi címkék (N hosszú numpy tömb)
    :param threshold: A maximális megengedett térbeli távolság (pl. 0.2)
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
            if dist_matrix[i, j] < threshold:  # Ha közel vannak egymáshoz térben
                if abs(labels[i] - labels[j]) > 1000:  # Ha időben nagy az eltérés
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
    plt.scatter(filtered_data[:, 0], filtered_data[:, 1], color="blue", alpha=0.5, label="Megmaradt adatok")

    # Törölt outlier pontok (piros)
    plt.scatter(outlier_data[:, 0], outlier_data[:, 1], color="red", alpha=0.7, label="Kiszűrt outlierek")

    plt.legend()
    plt.title("Szűrt adatok - időrendi alapú tisztítás")
    plt.show()

    return filtered_data, filtered_labels
