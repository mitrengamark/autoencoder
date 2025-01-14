import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
    
def visualize_bottleneck(bottleneck_outputs, model_name, bottleneck_type=None):
    """
    Vizualizálja a bottleneck kimeneteket PCA vagy T-SNE segítségével.

    :param bottleneck_outputs: A bottleneck által generált adatok (numpy array).
    :param model_name: A modell neve (pl. "VAE" vagy "MAE").
    :param bottleneck_type: A bottleneck típusa (pl. "z_mean", "z") - csak VAE esetén.
    """
    # Címképzés dinamikusan
    if model_name == "VAE":
        pca_title = f"PCA - {model_name} Bottleneck ({bottleneck_type})"
        tsne_title = f"T-SNE - {model_name} Bottleneck ({bottleneck_type})"
    else:  # MAE esetén egyszerűsített cím
        pca_title = f"PCA - {model_name} Bottleneck"
        tsne_title = f"T-SNE - {model_name} Bottleneck"

    print("PCA Visualization:")
    visualize_with_pca(bottleneck_outputs, title=pca_title)
    
    print("T-SNE Visualization:")
    visualize_with_tsne(bottleneck_outputs, title=tsne_title)

def visualize_with_pca(data, title="PCA Visualization"):
    """
    Adatok vizualizálása PCA használatával.

    :param data: A bemeneti adatok (numpy array).
    :param title: A grafikon címe.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    alphas = np.linspace(0.1, 1.0, len(reduced_data))

    plt.figure(figsize=(8, 6))
    for i in range(len(reduced_data)):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], alpha=alphas[i], color='blue')
    plt.title(title)
    plt.xlabel("Főkomponens 1")
    plt.ylabel("Főkomponens 2")
    plt.grid(True)
    plt.tight_layout()
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

def visualize_with_tsne(data, title="T-SNE Visualization"):
    """
    Adatok vizualizálása T-SNE használatával.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list).
    :param title: A grafikon címe.
    """
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=42)
    reduced_data = tsne.fit_transform(data)

    alphas = np.linspace(0.1, 1.0, len(reduced_data))

    plt.figure(figsize=(8, 6))
    for i in range(len(reduced_data)):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], alpha=alphas[i], color='blue')
    plt.title(title)
    plt.xlabel("T-SNE Komponens 1")
    plt.ylabel("T-SNE Komponens 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_latent_space(z_mean, z_log_var, epoch):
    plt.figure(figsize=(6, 6))
    plt.scatter(z_mean.cpu().detach().numpy(), z_log_var.cpu().detach().numpy())
    plt.title(f"Latent Space (Epoch {epoch})")
    plt.xlabel("z_mean")
    plt.ylabel("z_log_var")
    plt.grid(True)
    plt.show()