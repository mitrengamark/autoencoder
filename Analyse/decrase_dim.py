import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
    
def visualize_bottleneck(bottleneck_outputs, labels, model_name, test_mode, bottleneck_type=None):
    """
    Vizualizálja a bottleneck kimeneteket PCA vagy T-SNE segítségével.

    :param bottleneck_outputs: A bottleneck által generált adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list vagy numpy array).
    :param model_name: A modell neve (pl. "VAE" vagy "MAE").
    :param bottleneck_type: A bottleneck típusa (pl. "z_mean", "z") - csak VAE esetén.
    """
    assert len(bottleneck_outputs) == len(labels), "A bottleneck kimenetek és a címkék mérete nem egyezik!"

    # Címképzés dinamikusan
    if model_name == "VAE":
        pca_title = f"PCA - {model_name} Bottleneck ({bottleneck_type})"
        lda_title = f"LDA - {model_name} Bottleneck ({bottleneck_type})"
        tsne_title = f"T-SNE - {model_name} Bottleneck ({bottleneck_type})"
    else:  # MAE esetén egyszerűsített cím
        pca_title = f"PCA - {model_name} Bottleneck"
        lda_title = f"LDA - {model_name} Bottleneck"
        tsne_title = f"T-SNE - {model_name} Bottleneck"

    if test_mode == 1:
        print("PCA Visualization:")
        visualize_with_pca(bottleneck_outputs, labels=labels, title=pca_title)

        print("LDA Visualization:")
        visualize_with_lda(bottleneck_outputs, labels=labels, title=lda_title)
        
        print("T-SNE Visualization:")
        visualize_with_tsne(bottleneck_outputs, labels=labels, title=tsne_title)

def visualize_with_pca(data, labels, title="PCA Visualization"):
    """
    Adatok vizualizálása PCA használatával.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list vagy numpy array).
    :param title: A grafikon címe.
    """
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    assert data.shape[0] == labels.shape[0], "Data és Labels mérete nem egyezik!"

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = cm.get_cmap("tab10", len(unique_labels))

    # Címkék szerinti szétválasztás
    for i, label in enumerate(unique_labels):
        mask = labels == label  # Boolean maszk
        label_data = reduced_data[mask]
        alphas = np.linspace(0.1, 1.0, len(label_data))  # Alpha értékek lineárisan növekednek
        for j in range(len(label_data)):
            plt.scatter(label_data[j, 0], label_data[j, 1], label=f"Manoeuvre {label}" if j == 0 else "",
                        color=colors(i), alpha=alphas[j])
    
    handles, _ = plt.gca().get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1.0)  # Legendben alpha érték kikapcsolása
        
    plt.title(title)
    plt.xlabel("Főkomponens 1")
    plt.ylabel("Főkomponens 2")
    plt.legend(title="Címkék")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_with_lda(data, labels, title="LDA Visualization"):
    """
    Adatok vizualizálása LDA használatával, külön színezéssel és folyamatosan növekvő alpha értékekkel.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list vagy numpy array).
    :param title: A grafikon címe.
    """
    n_features = data.shape[1]
    n_classes = len(np.unique(labels))
    max_components = min(n_features, n_classes - 1)  # LDA maximális dimenziószáma

    if max_components < 1:
        print(f"[INFO] LDA nem alkalmazható: túl kevés osztály ({n_classes}) vagy feature ({n_features}).")
        return  # Kilépés a függvényből

    lda = LDA(n_components=max_components)
    reduced_data = lda.fit_transform(data, labels)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_data = reduced_data[mask]
        alphas = np.linspace(0.1, 1.0, len(label_data))  # Alpha értékek lineárisan növekvő
        for j in range(len(label_data)):
            # Ha csak egy dimenzió van, a második koordináta konstans
            if max_components == 1:
                plt.scatter(label_data[j, 0], 0, label=f"Manoeuvre {label}" if j == 0 else "",
                            color=colors(i), alpha=alphas[j])

            else:
                plt.scatter(label_data[j, 0], label_data[j, 1], label=f"Manoeuvre {label}" if j == 0 else "",
                            color=colors(i), alpha=alphas[j])

    # Legend alpha értékének kikapcsolása
    handles, _ = plt.gca().get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1.0)

    plt.title(title)
    plt.xlabel("LDA Komponens 1")
    if max_components > 1:
        plt.ylabel("LDA Komponens 2")
    plt.legend(title="Címkék")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_with_tsne(data, labels, title="T-SNE Visualization"):
    """
    Adatok vizualizálása T-SNE használatával.

    :param data: A bemeneti adatok (numpy array).
    :param labels: Az egyes mintákhoz tartozó címkék (list).
    :param title: A grafikon címe.
    """
    perplexity = min(50, max(5, data.shape[0] // 10))

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, max_iter=1000, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_data = reduced_data[mask]
        alphas = np.linspace(0.1, 1.0, label_data.shape[0])  # Alpha értékek lineárisan növekednek
        for j in range(label_data.shape[0]):
            plt.scatter(label_data[j, 0], label_data[j, 1], label=f"Manoeuvre {label}" if j == 0 else "",
                        color=colors(i), alpha=alphas[j])
    
    handles, _ = plt.gca().get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1.0)  # Legendben alpha érték kikapcsolása

    plt.title(title)
    plt.xlabel("T-SNE Komponens 1")
    plt.ylabel("T-SNE Komponens 2")
    plt.legend(title="Címkék")
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