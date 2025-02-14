import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from scipy.spatial import ConvexHull


def detect_outliers(data):
    """
    Outlierek detektálása az OPTICS algoritmussal.
    Az outlierek indexeit eltárolja és opcionálisan fájlba menti.

    :param data: A bemeneti adathalmaz (numpy array)
    :param save_to_file: Ha True, az outlierek indexeit fájlba menti
    :return: Az outlierek indexei
    
    OPTICS-alapú outlier detektálás
    """
    optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
    labels = optics.fit_predict(data)

    optics_outliers = labels == -1  # OPTICS által megjelölt outlierek

    outlier_indices = np.where(optics_outliers)[0]  # Az outlierek sorindexei

    # Vizualizáció
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c="purple", label="Klaszterezett adatok")
    plt.scatter(
        data[optics_outliers, 0],
        data[optics_outliers, 1],
        c="red",
        label="OPTICS Outlierek",
    )
    plt.legend()
    plt.title("OPTICS - Outlier Detekció")
    plt.show()

    return outlier_indices