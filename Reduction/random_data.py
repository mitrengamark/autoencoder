import numpy as np


def generate_clustered_data(
    n_samples=300, n_clusters=8, cluster_std=10, random_state=42
):
    np.random.seed(random_state)
    centers = np.random.uniform(-70, 70, (n_clusters, 2))  # Klaszterközéppontok

    data = []
    labels = []
    for i in range(n_clusters):
        cluster_data = np.random.normal(
            loc=centers[i], scale=cluster_std, size=(n_samples // n_clusters, 2)
        )
        data.append(cluster_data)
        labels.extend([i] * (n_samples // n_clusters))

    data = np.vstack(data)
    labels = np.array(labels)

    label_mapping = {i: f"manoeuvre_{i}" for i in range(n_clusters)}

    return data, labels, label_mapping


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
