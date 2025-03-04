import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde


class DetectOverlap:
    def __init__(self):
        # JSON fájl betöltése
        json_path = "valtozo_v_savvaltas_fek_tsne_data.json"

        with open(json_path, "r") as file:
            tsne_data_json = json.load(file)

        # TSNE adatok betöltése numpy tömbbé alakítva
        self.tsne_data = np.array(tsne_data_json["tsne_data"])
        self.labels = np.array(tsne_data_json["labels"])

        # TSNE adatok átalakítása síkba
        self.flattened_tsne_data = self.tsne_data.reshape(-1, 2)

        # Új label lista létrehozása, hogy minden ponthoz tartozzon egy címke
        self.expanded_labels = np.repeat(self.labels, self.tsne_data.shape[1])

        # Egyedi címkék azonosítása
        self.unique_labels = np.unique(self.labels)

    def detect_overlap_by_grid(self):
        """
        Átfedések detektálása egy egyszerű grid alapú módszerrel
        """
        grid_size = 10
        grid_counts = {}
        x_min, x_max = self.flattened_tsne_data[:, 0].min(), self.flattened_tsne_data[:, 0].max()
        y_min, y_max = self.flattened_tsne_data[:, 1].min(), self.flattened_tsne_data[:, 1].max()

        x_step = (x_max - x_min) / grid_size
        y_step = (y_max - y_min) / grid_size

        for i, (x, y) in enumerate(self.flattened_tsne_data):
            x_idx = int((x - x_min) / x_step)
            y_idx = int((y - y_min) / y_step)
            cell = (x_idx, y_idx)
            
            if cell not in grid_counts:
                grid_counts[cell] = set()
            grid_counts[cell].add(self.expanded_labels[i])

        # Átfedések meghatározása
        overlap_points = np.array([
            (x, y) for (x, y), cell in zip(self.flattened_tsne_data, [(int((x - x_min) / x_step), int((y - y_min) / y_step)) for x, y in self.flattened_tsne_data])
            if len(grid_counts[cell]) > 1  # Több különböző címke egy cellában
        ])

        # Vizualizáció az átfedésekkel
        plt.figure(figsize=(12, 8))
        for label in self.unique_labels:
            mask = self.expanded_labels == label
            plt.scatter(self.flattened_tsne_data[mask, 0], self.flattened_tsne_data[mask, 1], label=label, alpha=0.5, s=10)

        # Átfedések kiemelése
        if len(overlap_points) > 0:
            plt.scatter(overlap_points[:, 0], overlap_points[:, 1], color="red", label="Átfedés", s=30, edgecolors="black")

        plt.legend()
        plt.title("T-SNE Vizualizáció Átfedések Megjelölésével")
        plt.xlabel("T-SNE Komponens 1")
        plt.ylabel("T-SNE Komponens 2")
        plt.grid(True)
        plt.show()

    def detect_overlap_by_dbscan(self):
        """
        DBSCAN-alapú átfedés detektálás
        """
        eps = 2.0  # Maximális távolság két pont között (finomhangolható)
        min_samples = 5  # Minimális pontszám egy klaszterben

        # DBSCAN futtatása
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_dbscan = dbscan.fit_predict(self.flattened_tsne_data)

        # Az outlier (zaj) pontok indexei (-1 jelöli az outliereket)
        outlier_mask_dbscan = labels_dbscan == -1
        outlier_points_dbscan = self.flattened_tsne_data[outlier_mask_dbscan]

        plt.figure(figsize=(12, 8))

        # Alap pontok
        plt.scatter(self.flattened_tsne_data[:, 0], self.flattened_tsne_data[:, 1], color="lightgray", s=5, alpha=0.5, label="Adatok")

        # DBSCAN által detektált outlierek
        if len(outlier_points_dbscan) > 0:
            plt.scatter(outlier_points_dbscan[:, 0], outlier_points_dbscan[:, 1], color="blue", s=20, label="DBSCAN átfedések")

        plt.legend()
        plt.title("T-SNE Vizualizáció DBSCAN és KDE Átfedésekkel")
        plt.xlabel("T-SNE Komponens 1")
        plt.ylabel("T-SNE Komponens 2")
        plt.grid(True)
        plt.show()

    def detect_overlap_by_kde(self):
        """
        Kernel Density Estimation (KDE) alapú átfedés detektálás
        """
        kde = gaussian_kde(self.flattened_tsne_data.T)  # KDE becslés
        density = kde(self.flattened_tsne_data.T)  # Sűrűség számítása minden pontnál

        # Egy küszöbérték meghatározása az átfedések jelölésére (pl. felső 10%)
        density_threshold = np.percentile(density, 90)
        overlap_mask_kde = density > density_threshold
        overlap_points_kde = self.flattened_tsne_data[overlap_mask_kde]

        plt.figure(figsize=(12, 8))

        # Alap pontok
        plt.scatter(self.flattened_tsne_data[:, 0], self.flattened_tsne_data[:, 1], color="lightgray", s=5, alpha=0.5, label="Adatok")

        # KDE által talált sűrű helyek
        if len(overlap_points_kde) > 0:
            plt.scatter(overlap_points_kde[:, 0], overlap_points_kde[:, 1], color="red", s=20, label="KDE átfedések")

        plt.legend()
        plt.title("T-SNE Vizualizáció DBSCAN és KDE Átfedésekkel")
        plt.xlabel("T-SNE Komponens 1")
        plt.ylabel("T-SNE Komponens 2")
        plt.grid(True)
        plt.show()

DetectOverlap().detect_overlap_by_grid()
DetectOverlap().detect_overlap_by_dbscan()
DetectOverlap().detect_overlap_by_kde()