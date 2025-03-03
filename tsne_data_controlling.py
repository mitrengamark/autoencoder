import pickle
import numpy as np
import matplotlib.pyplot as plt

# Betöltés függvény
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# A két fájl elérési útja
file1 = "TSNE_data/single_manoeuvers/tsne_allando_v_chirp_a1_v5_ceef3b266632e4e42dc518d812a9dc90dc08cd36d669f062daff3fccf54199d0.pkl"
file2 = "TSNE_data/single_manoeuvers/tsne_allando_v_chirp_a1_v5_decd4371fb6a503ce8996de493fbc30cf61463ac96755e5b1eb9ced073a1fcdb.pkl"

# Betöltjük az adatokat
data1 = load_pkl(file1)
data2 = load_pkl(file2)

# Ellenőrizzük, hogy a két adat azonos-e
if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
    are_equal = np.array_equal(data1, data2)
else:
    are_equal = data1 == data2

# Kiírjuk az eredményt
if are_equal:
    print("A két .pkl fájl tartalma teljesen azonos.")
else:
    print("A két .pkl fájl különböző adatokat tartalmaz.")

print(f"Szótár 1 kulcsai: {data1.keys()}")
print(f"Szótár 2 kulcsai: {data2.keys()}")

tsne_data1 = np.array(data1["tsne_data"])
tsne_data2 = np.array(data2["tsne_data"])

# TSNE adatok betöltése és plotolása
plt.figure(figsize=(10, 5))
plt.scatter(tsne_data1[:, 0], tsne_data1[:, 1], label="TSNE 1", alpha=0.5, marker="o")
plt.scatter(tsne_data2[:, 0], tsne_data2[:, 1], label="TSNE 2", alpha=0.5, marker="x")

plt.title("TSNE Adatok Összehasonlítása")
plt.xlabel("T-SNE Komponens 1")
plt.ylabel("T-SNE Komponens 2")
plt.legend()
plt.grid(True)
plt.show()