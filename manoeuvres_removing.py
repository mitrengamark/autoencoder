import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Mappa, ahol az adatok találhatók
data_dir = "Reduced_bottleneck_data/single_manoeuvres/"

# Adatok beolvasása
file_paths = [
    data_dir + "valtozo_v_savvaltas_gas_alacsony_pedal0_2.npy",
    data_dir + "valtozo_v_savvaltas_gas_alacsony_pedal0_5.npy",
    data_dir + "valtozo_v_savvaltas_gas_alacsony_pedal1_0.npy",
    data_dir + "valtozo_v_savvaltas_gas_kozepes_pedal0_2.npy",
    data_dir + "valtozo_v_savvaltas_gas_kozepes_pedal0_5.npy",
    data_dir + "valtozo_v_savvaltas_gas_kozepes_pedal1_0.npy",
    data_dir + "valtozo_v_savvaltas_gas_magas_pedal0_2.npy",
    data_dir + "valtozo_v_savvaltas_gas_magas_pedal0_5.npy",
    data_dir + "valtozo_v_savvaltas_gas_magas_pedal1_0.npy"
]

# Fájlok beolvasása
data_dict = {os.path.basename(path): np.load(path) for path in file_paths}

# Összes adat összefűzése
all_data = np.vstack(list(data_dict.values()))

reduced_data = all_data

# # t-SNE alkalmazása 2D-re csökkentéshez
# tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
# reduced_data = tsne.fit_transform(all_data)

# # PCA alkalmazása 2D-re csökkentéshez
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(all_data)

# # PCA variancia arányok kiírása
# print("PCA magyarázott variancia arányok:", pca.explained_variance_ratio_)

# t-SNE eredmény vizualizálása
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5, s=1)
plt.xlabel("t-SNE dimenzió 1")
plt.ylabel("t-SNE dimenzió 2")
plt.title("t-SNE dimenziócsökkentés eredménye")
plt.savefig("Manoeuvres_removing/t-SNE_dimenziocsokkentes.png")
plt.show()

# K-Means klaszterezés alkalmazása (5 klaszter)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(reduced_data)

# Klaszterek vizualizálása
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label="Klaszterközéppontok")
plt.xlabel("t-SNE dimenzió 1")
plt.ylabel("t-SNE dimenzió 2")
plt.title("K-Means klaszterezés eredménye t-SNE után")
plt.legend()
plt.savefig("Manoeuvres_removing/K-Means_klaszterezes.png")
plt.show()

# Hozzárendeljük a klaszter címkéket az egyes fájlokhoz
cluster_results = {}
start_idx = 0

for file_name, data in data_dict.items():
    end_idx = start_idx + len(data)
    cluster_results[file_name] = np.bincount(clusters[start_idx:end_idx], minlength=5)
    start_idx = end_idx

# Az eredmények kiírása DataFrame-be
df_clusters = pd.DataFrame(cluster_results, index=[f"Klaszter {i}" for i in range(5)]).T

# Klasztereloszlás megjelenítése
print("\nKlasztereloszlás az egyes fájlokban:")
print(df_clusters)
