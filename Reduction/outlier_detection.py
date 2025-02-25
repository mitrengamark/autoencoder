import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import OPTICS
from sklearn.neighbors import LocalOutlierFactor
from Config.load_config import save_fig, selected_manoeuvres, folder_name


def detect_outliers(data):
    """
    Az OPTICS és LOF kombinációjával detektálja az outliereket.

    :param data: A bemeneti adathalmaz (numpy array)
    :return: A végső outlierek indexei
    """
    print(f"Outlierek detektálása...")

    if "chirp" in selected_manoeuvres:
        n_neighbors = 416
        contamination = 0.1
    elif "savvaaltas" in selected_manoeuvres:
        n_neighbors = 416
        contamination = 0.3
    else:
        n_neighbors = 416
        contamination = 0.15


    optics_outlier_indices = optics_clustering(data)
    lof_outlier_indices = lof_outlier_detection(data, n_neighbors, contamination)

    # Két módszer kombinálása: Csak azok az outlierek maradnak, amelyeket mindkét módszer annak lát
    final_outlier_indices = np.intersect1d(optics_outlier_indices, lof_outlier_indices)

    print(f"Törölt adatok száma: {len(final_outlier_indices)}")

    # Vizualizáció
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c="purple", label="Klaszterezett adatok")
    plt.scatter(
        data[final_outlier_indices, 0],
        data[final_outlier_indices, 1],
        c="red",
        label="OPTICS Outlierek",
    )
    plt.legend()
    plt.title("OPTICS + LOF - Finomhangolt Outlier Detekció")

    if save_fig:
        filename = f"Results/{folder_name}/{selected_manoeuvres[0]}/optics.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.show()

    return final_outlier_indices


def optics_clustering(data):
    """
    OPTICS klaszterezést hajt végre és visszaadja az outlierek indexeit.

    :param data: A bemeneti adathalmaz (numpy array)
    :return: Az OPTICS által detektált outlierek indexei
    """
    optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
    labels = optics.fit_predict(data)
    optics_outliers = labels == -1  # OPTICS által megjelölt outlierek
    outlier_indices = np.where(optics_outliers)[0]  # Az outlierek sorindexei

    return outlier_indices


def lof_outlier_detection(data, n_neighbors, contamination):
    """
    LOF (Local Outlier Factor) alapú anomália detekció.

    :param data: A bemeneti adathalmaz (numpy array)
    :return: A LOF által detektált outlierek indexei
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_outliers = lof.fit_predict(data) == -1  # LOF által megjelölt outlierek
    outlier_indices = np.where(lof_outliers)[0]

    return outlier_indices
