import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from Config.load_config import grid_size, save_fig, selected_manoeuvres, folder_name


def create_comparison_heatmaps(
    original_data, filtered_data, grid_size=grid_size, file_name=None
):
    """
    Az eredeti és a szűrt látenstér heatmapjének összehasonlítása.

    :param original_data: Eredeti 2D adatok.
    :param filtered_data: Szűrt 2D adatok.
    :param grid_size: A heatmap felbontása.
    """
    # Eredeti és szűrt heatmap generálása
    x_grid_orig, y_grid_orig, heatmap_orig = create_heatmap(original_data, grid_size)
    x_grid_filt, y_grid_filt, heatmap_filt = create_heatmap(filtered_data, grid_size)

    # Skálázás az adatmérethez
    heatmap_orig *= len(original_data) / len(filtered_data)

    # Skálázás az új minimum és maximum értékekhez
    vmin = min(heatmap_orig.min(), heatmap_filt.min())
    vmax = max(heatmap_orig.max(), heatmap_filt.max())

    # Subplot létrehozása
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Eredeti heatmap
    im1 = axes[0].imshow(
        heatmap_orig,
        extent=[
            x_grid_orig.min(),
            x_grid_orig.max(),
            y_grid_orig.min(),
            y_grid_orig.max(),
        ],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Eredeti Látenstér Heatmap")
    axes[0].set_xlabel("T-SNE Komponens 1")
    axes[0].set_ylabel("T-SNE Komponens 2")
    fig.colorbar(im1, ax=axes[0], label="Intenzitás")

    # Szűrt heatmap
    im2 = axes[1].imshow(
        heatmap_filt,
        extent=[
            x_grid_filt.min(),
            x_grid_filt.max(),
            y_grid_filt.min(),
            y_grid_filt.max(),
        ],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Szűrt Látenstér Heatmap")
    axes[1].set_xlabel("T-SNE Komponens 1")
    axes[1].set_ylabel("T-SNE Komponens 2")
    fig.colorbar(im2, ax=axes[1], label="Intenzitás")

    plt.tight_layout()

    if save_fig:
        filename = f"Results/{folder_name}/{selected_manoeuvres[0]}/{file_name}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    # plt.show()


def create_heatmap(data, grid_size=grid_size):
    """
    Heatmap generálása
    :param data: 2D adathalmaz (numpy array, shape: [n_samples, 2]).
    :param grid_size: A heatmap felbontása (rács mérete).
    :return: Heatmap (numpy array).
    """
    x = data[:, 0]
    y = data[:, 1]

    # 2D grid létrehozása
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # Kernel Density Estimation (KDE) az adatokra
    kde = gaussian_kde(np.vstack([x, y]), weights=np.ones(len(x)) / len(x))
    z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(grid_size, grid_size)

    return x_grid, y_grid, z
