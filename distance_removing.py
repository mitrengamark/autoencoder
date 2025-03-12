import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from concurrent.futures import ProcessPoolExecutor

class RedundancyFilter:
    def __init__(self, bottleneck_data, labels, reverse_label_mapping=None):
        """
        Inicializálja az osztályt a manőveradatokkal és a hozzájuk tartozó címkékkel.

        bottleneck_data: np.array, az összes manőver adat (shape: [n_samples, n_features])
        labels: np.array, az adatokhoz tartozó címkék
        reverse_label_mapping: dict, ha a címkék numerikusak, akkor ez segít visszafejteni az eredeti neveket
        """
        self.bottleneck_data = bottleneck_data
        self.labels = labels
        self.reverse_label_mapping = reverse_label_mapping if reverse_label_mapping else {}

    def compute_similarities(self, batch_indices, full_data, threshold, labels, metric="cosine"):
        """
        Egy batch cosine similarity vagy euklideszi távolság mátrixát számolja ki 
        és visszaadja a redundáns párokat.
        """
        batch_data = full_data[batch_indices]  # Kiválasztjuk a batch adatokat
        
        if metric == "cosine":
            similarities = cosine_similarity(batch_data, full_data)
            condition = lambda sim: sim > threshold
        elif metric == "euclidean":
            distances = euclidean_distances(batch_data, full_data)
            condition = lambda dist: dist < threshold  # Kisebb távolság → nagyobb hasonlóság

        redundant_pairs = set()
        
        for row_idx, row in enumerate(similarities if metric == "cosine" else distances):
            for j in range(len(labels)):  # Kerüld az out-of-bounds hibát
                if condition(row[j]) and labels[batch_indices[row_idx]] != labels[j]:
                    redundant_pairs.add(
                        tuple(sorted((labels[batch_indices[row_idx]], labels[j])))
                    )


        return redundant_pairs

    def filter_by_distance(self, threshold=0.9, batch_size=1000, metric="cosine", num_workers=8):
        """
        Cosine Similarity vagy euklideszi távolság alapján kiszűri a redundáns manővereket.
        
        threshold: Az a küszöbérték, amely felett két manőver redundánsnak számít.
        metric: "cosine" vagy "euclidean"
        batch_size: Hány adatot dolgozzunk fel egy batch-ben
        num_workers: Párhuzamosan futtatandó folyamatok száma
        """
        print(f"Redundancia szűrés {metric} alapján...")

        n = self.bottleneck_data.shape[0]
        redundant_pairs = set()

        # Párhuzamos számítás elindítása
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, n, batch_size):
                batch_indices = list(range(i, min(i + batch_size, n)))
                futures.append(
                    executor.submit(
                        self.compute_similarities,
                        batch_indices,
                        self.bottleneck_data,
                        threshold,
                        self.labels,
                        metric
                    )
                )

            # Eredmények összegyűjtése
            for future in futures:
                redundant_pairs.update(future.result())

        # Redundáns manőverek kiírása
        redundant_manoeuvre_names = {
            (
                self.reverse_label_mapping.get(pair[0], f"Unknown_{pair[0]}"),
                self.reverse_label_mapping.get(pair[1], f"Unknown_{pair[1]}")
            )
            for pair in redundant_pairs
        }

        print(f"Redundáns manőver párok ({metric}): {redundant_manoeuvre_names}")

        return redundant_pairs

# Mappa, ahol az adatok találhatók
data_dir = "Reduced_bottleneck_data/single_manoeuvres/"

# Adatok beolvasása
file_basenames = [
    "allando_v_savvaltas_alacsony_v5",
    "allando_v_savvaltas_kozepes_v5",
    "allando_v_savvaltas_magas_v5",
]

file_paths = [os.path.join(data_dir, f"{basename}.npy") for basename in file_basenames]

data_dict = {path: np.load(path) for path in file_paths}
all_data = np.vstack(list(data_dict.values()))

# Labels (fájlnevek)
labels = np.array(list(data_dict.keys()))

print("Adathalmaz mérete:", all_data.shape)
print("Címkék száma:", labels.shape)

labels = np.concatenate([np.full(data.shape[0], i) for i, data in enumerate(data_dict.values())])
print("Új labels méret:", labels.shape)

# Opcionális: címkék visszafejtése (pl. ha numerikus ID-k vannak)
reverse_label_mapping = {i: labels[i] for i in range(len(labels))}

# Redundancia szűrő létrehozása
redundancy_filter = RedundancyFilter(all_data, labels, reverse_label_mapping)

redundant_pairs = redundancy_filter.filter_by_distance(threshold=0.5, metric="cosine")
# redundant_pairs = redundancy_filter.filter_by_distance(threshold=0.5, metric="euclidean")
