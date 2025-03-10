import numpy as np
import matplotlib.pyplot as plt


def generate_clustered_data(
    n_samples=300, n_clusters=9, cluster_std=10, random_state=42, n_features=8
):
    """
    Generál klaszterezett tesztadatokat a filter_by_distance függvényhez.

    n_samples: Összes adatpont száma
    n_clusters: Klaszterek (manőverek) száma
    cluster_std: Klaszterek szórása
    random_state: Random állapot a replikálhatóság érdekében
    n_features: Az adatok dimenziója (pl. 8, hogy illeszkedjen az autoencoder bottleneck dimenzióhoz)

    Return: bottleneck_data (listák listája), labels (listák listája), label_mapping (dict)
    """
    np.random.seed(random_state)
    centers = np.random.uniform(
        -70, 70, (n_clusters, n_features)
    )  # Klaszterközéppontok

    data = []
    labels = []
    for i in range(n_clusters):
        cluster_data = np.random.normal(
            loc=centers[i],
            scale=cluster_std,
            size=(n_samples // n_clusters, n_features),
        )
        data.append(cluster_data)
        labels.extend([i] * (n_samples // n_clusters))

    data = np.vstack(data)
    labels = np.array(labels)

    label_mapping = {i: f"manoeuvre_{i}" for i in range(n_clusters)}

    return data, labels, label_mapping


def plot_clusters(bottleneck_data, labels):
    """
    2D scatter plot a generált adatokhoz.
    """
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        bottleneck_data[:, 0], bottleneck_data[:, 1], c=labels, cmap="tab10", alpha=0.6, edgecolors="k"
    )
    plt.colorbar(scatter, label="Cluster Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Klaszterezett adatok vizualizációja")
    plt.show()


def generate_advanced_sinusoidal_spiral_data(
    n_samples=500, n_clusters=5, noise=1.5, curve_strength=2.5, random_state=42
):
    np.random.seed(random_state)
    data = []
    labels = []

    for i in range(n_clusters):
        t = np.linspace(
            0, 6 * np.pi, n_samples // n_clusters
        )  # Hosszabb és kanyargósabb görbék
        x = (
            (curve_strength * t + np.random.normal(scale=noise, size=t.shape))
            * np.cos(t)
            * 12
        )  # Erősebb spirál X koordináta
        y = (
            (curve_strength * t + np.random.normal(scale=noise, size=t.shape))
            * np.sin(t)
            * 12
        )  # Erősebb spirál Y koordináta

        # Véletlenszerű eltolás, hogy ne legyenek teljesen szimmetrikusak
        x += np.random.uniform(-80, 80)
        y += np.random.uniform(-80, 80)

        cluster_data = np.vstack((x, y)).T
        data.append(cluster_data)
        labels.extend([i] * len(t))

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels
