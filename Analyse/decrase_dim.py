import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

def load_bottleneck_data(file_path):
    """
    Betölti a VAE bottleneck adatait tartalmazó fájlt.
    Feltételezzük, hogy a fájl tartalmazza az egyes változókhoz tartozó bottleneck értékeket.

    :param file_path: A fájl elérési útja.
    :return: Az adatok numpy array-ként.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Adatok betöltve: {file_path}")
        return data.values  # Adatok numpy array-ként
    except FileNotFoundError:
        print(f"Hiba: A fájl nem található: {file_path}")
        return None
    
def visualize_bottleneck(bottleneck_outputs, labels):
        """
        Vizualizálja a bottleneck kimeneteket PCA vagy T-SNE segítségével.

        :param bottleneck_outputs: A VAE bottleneck által generált adatok (numpy array).
        :param labels: Címkék az egyes mintákhoz.
        """
        print("PCA Visualization:")
        visualize_with_pca(bottleneck_outputs, labels, title="PCA - VAE Bottleneck")
        
        # print("T-SNE Visualization:")
        # visualize_with_tsne(bottleneck_outputs, labels, title="T-SNE - VAE Bottleneck")

def visualize_with_pca(data, labels, title="PCA Visualization"):
    """
    Adatok vizualizálása PCA használatával.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list).
    :param title: A grafikon címe.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Változók címkéi")
    plt.title(title)
    plt.xlabel("Főkomponens 1")
    plt.ylabel("Főkomponens 2")
    plt.grid(True)
    plt.show()

def visualize_with_lda(data, labels, title="LDA Visualization"):
    """
    Adatok vizualizálása LDA használatával.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list).
    :param title: A grafikon címe.
    """
    lda = LDA(n_components=2)
    reduced_data = lda.fit_transform(data, labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Változók címkéi")
    plt.title(title)
    plt.xlabel("LDA Komponens 1")
    plt.ylabel("LDA Komponens 2")
    plt.grid(True)
    plt.show()

def visualize_with_tsne(data, labels, title="T-SNE Visualization"):
    """
    Adatok vizualizálása T-SNE használatával.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list).
    :param title: A grafikon címe.
    """
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Változók címkéi")
    plt.title(title)
    plt.xlabel("T-SNE Komponens 1")
    plt.ylabel("T-SNE Komponens 2")
    plt.grid(True)
    plt.show()

# # Fájl betöltése és vizualizálás
# file_path = "data2/manoeuvre_bottleneck.csv"  # Cseréld le a megfelelő fájlra
# data = load_bottleneck_data(file_path)

# if data is not None:
#     labels = np.arange(data.shape[0])  # Címkék generálása (pl. indexek)

#     # PCA vizualizáció
#     visualize_with_pca(data, labels, title="PCA Visualization - VAE Bottleneck")

#     # LDA vizualizáció
#     # Megjegyzés: Az LDA-hoz a címkék (labels) szükségesek, amelyek általában osztályok
#     # visualize_with_lda(data, labels, title="LDA Visualization - VAE Bottleneck")

#     # T-SNE vizualizáció
#     visualize_with_tsne(data, labels, title="T-SNE Visualization - VAE Bottleneck")