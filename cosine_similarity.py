import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class CosineSimilarity:
    def __init__(self, directory):
        self.directory = directory
        self.similarity_matrices = {}

    def compute_cosine_similarity_within_groups(self, grouped_manoeuvres):
        for idx, group in enumerate(grouped_manoeuvres):
            valid_manoeuvres = [
                m
                for m in group
                if os.path.exists(os.path.join(self.directory, m + ".npy"))
            ]

            if len(valid_manoeuvres) < 2:
                print(
                    f"Figyelmeztetés: Group {idx+1} csoportban nincs elég adat az összehasonlításhoz."
                )
                continue

            group_vectors = [
                np.load(os.path.join(directory, m + ".npy")) for m in valid_manoeuvres
            ]
            similarity_matrix = cosine_similarity(group_vectors)

            self.similarity_matrices[idx + 1] = (valid_manoeuvres, similarity_matrix)

            self.plot_confusion_matrix(
                valid_manoeuvres, similarity_matrix, f"Group {idx+1}"
            )

    def plot_confusion_matrix(self, labels, similarity_matrix, title):
        plt.figure(figsize=(12, 10))
        annot_flag = True if len(labels) <= 24 else False
        sns.heatmap(
            similarity_matrix,
            annot=annot_flag,
            xticklabels=labels,
            yticklabels=labels,
            cmap="coolwarm",
            fmt=".2f",
        )
        plt.title(f"Cosine Similarity Matrix - {title}")
        plt.xlabel("Manoeuvres")
        plt.ylabel("Manoeuvres")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        plt.savefig(f"cosine_similarity_matrix_{title}.png")

        # plt.show()

    def detect_redundancy(self, threshold=0.95):
        redundant_pairs = {}

        for group_idx, (labels, similarity_matrix) in self.similarity_matrices.items():
            similar_pairs = []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):  # Csak a felső háromszöget nézzük
                    if similarity_matrix[i, j] >= threshold:
                        similar_pairs.append(
                            [labels[i], labels[j]]
                        )

            if similar_pairs:
                redundant_pairs[group_idx] = (
                    similar_pairs  # Csak a nem üres listákat tároljuk
                )

        return redundant_pairs  # Egy dictionary, ahol a kulcs a csoport, az érték a hasonló párok listája
    
    def remove_redundancy(self, redundant_maneuvers_by_group):
        # Csoportonként külön-külön tároljuk az eltávolított manővereket
        removed_manoeuvres_by_group = {}

        for group_idx, redundant_maneuvers in redundant_maneuvers_by_group.items():
            removed_manoeuvres = []

            # Addig futtatjuk az optimalizációt, amíg maradnak redundáns párok az adott csoportban
            while redundant_maneuvers:
                # Gyakoriság számítása az adott csoport manővereire
                counter = Counter(maneuver for pair in redundant_maneuvers for maneuver in pair)
                
                # Leggyakoribb manőver kiválasztása
                most_common_maneuver, _ = counter.most_common(1)[0]
                
                # Mentjük a listába
                removed_manoeuvres.append(most_common_maneuver)
                
                # Kiszűrjük azokat a párokat, amelyek tartalmazzák a kiválasztott manővert
                redundant_maneuvers = [pair for pair in redundant_maneuvers if most_common_maneuver not in pair]

            removed_manoeuvres_by_group[group_idx] = removed_manoeuvres

        #  # Az eltávolított manőverek mentése JSON fájlba
        # with open("manoeuvres_for_removing_95.json", "w") as file:
        #     json.dump(removed_manoeuvres_by_group, file, indent=4)

        return removed_manoeuvres_by_group



# Mappa elérési útvonala
directory = "Bottleneck_data/averaged_manoeuvres"
cos_sim = CosineSimilarity(directory)

# Felhasználó által meghatározott csoportok
basic_maneuvers_list = [
    [
        "allando_v_savvaltas_alacsony_v5",
        "allando_v_savvaltas_kozepes_v5",
        "allando_v_savvaltas_magas_v5",
        "allando_v_savvaltas_alacsony_v10",
        "allando_v_savvaltas_kozepes_v10",
        "allando_v_savvaltas_magas_v10",
        "allando_v_savvaltas_alacsony_v15",
        "allando_v_savvaltas_kozepes_v15",
        "allando_v_savvaltas_magas_v15",
        "allando_v_savvaltas_alacsony_v20",
        "allando_v_savvaltas_kozepes_v20",
        "allando_v_savvaltas_magas_v20",
        "allando_v_savvaltas_alacsony_v25",
        "allando_v_savvaltas_kozepes_v25",
        "allando_v_savvaltas_magas_v25",
        "allando_v_savvaltas_alacsony_v30",
        "allando_v_savvaltas_kozepes_v30",
        "allando_v_savvaltas_magas_v30",
        "allando_v_savvaltas_alacsony_v35",
        "allando_v_savvaltas_kozepes_v35",
        "allando_v_savvaltas_magas_v35",
        "allando_v_savvaltas_alacsony_v40",
        "allando_v_savvaltas_kozepes_v40",
        "allando_v_savvaltas_magas_v40",
        "allando_v_savvaltas_alacsony_v45",
        "allando_v_savvaltas_kozepes_v45",
        "allando_v_savvaltas_magas_v45",
        "allando_v_savvaltas_alacsony_v50",
        "allando_v_savvaltas_kozepes_v50",
        "allando_v_savvaltas_magas_v50",
        "allando_v_savvaltas_alacsony_v55",
        "allando_v_savvaltas_kozepes_v55",
        "allando_v_savvaltas_magas_v55",
        "allando_v_savvaltas_alacsony_v60",
        "allando_v_savvaltas_kozepes_v60",
        "allando_v_savvaltas_magas_v60",
        "allando_v_savvaltas_alacsony_v65",
        "allando_v_savvaltas_kozepes_v65",
        "allando_v_savvaltas_magas_v65",
        "allando_v_savvaltas_alacsony_v70",
        "allando_v_savvaltas_kozepes_v70",
        "allando_v_savvaltas_magas_v70",
        "allando_v_savvaltas_alacsony_v75",
        "allando_v_savvaltas_kozepes_v75",
        "allando_v_savvaltas_magas_v75",
        "allando_v_savvaltas_alacsony_v80",
        "allando_v_savvaltas_kozepes_v80",
        "allando_v_savvaltas_magas_v80",
        "allando_v_savvaltas_alacsony_v85",
        "allando_v_savvaltas_kozepes_v85",
        "allando_v_savvaltas_magas_v85",
        "allando_v_savvaltas_alacsony_v90",
        "allando_v_savvaltas_kozepes_v90",
        "allando_v_savvaltas_magas_v90",
        "allando_v_savvaltas_alacsony_v95",
        "allando_v_savvaltas_kozepes_v95",
        "allando_v_savvaltas_magas_v95",
        "allando_v_savvaltas_alacsony_v100",
        "allando_v_savvaltas_kozepes_v100",
        "allando_v_savvaltas_magas_v100",
        "allando_v_savvaltas_alacsony_v105",
        "allando_v_savvaltas_kozepes_v105",
        "allando_v_savvaltas_magas_v105",
        "allando_v_savvaltas_alacsony_v110",
        "allando_v_savvaltas_kozepes_v110",
        "allando_v_savvaltas_magas_v110",
        "allando_v_savvaltas_alacsony_v115",
        "allando_v_savvaltas_kozepes_v115",
        "allando_v_savvaltas_magas_v115",
        "allando_v_savvaltas_alacsony_v120",
        "allando_v_savvaltas_kozepes_v120",
        "allando_v_savvaltas_magas_v120",
        "allando_v_savvaltas_alacsony_v125",
        "allando_v_savvaltas_kozepes_v125",
        "allando_v_savvaltas_magas_v125",
        "allando_v_savvaltas_alacsony_v130",
        "allando_v_savvaltas_kozepes_v130",
        "allando_v_savvaltas_magas_v130",
        "allando_v_savvaltas_alacsony_v135",
        "allando_v_savvaltas_kozepes_v135",
        "allando_v_savvaltas_magas_v135",
        "allando_v_savvaltas_alacsony_v140",
        "allando_v_savvaltas_kozepes_v140",
        "allando_v_savvaltas_magas_v140",
    ],
    [
        "allando_v_sin_a2_f1_v5",
        "allando_v_sin_a8_f1_v5",
        "allando_v_sin_a2_f3_v5",
        "allando_v_sin_a8_f3_v5",
        "allando_v_sin_a2_f5_v5",
        "allando_v_sin_a8_f5_v5",
        "allando_v_sin_a2_f7_v5",
        "allando_v_sin_a8_f7_v5",
        "allando_v_sin_a2_f1_v10",
        "allando_v_sin_a8_f1_v10",
        "allando_v_sin_a2_f3_v10",
        "allando_v_sin_a8_f3_v10",
        "allando_v_sin_a2_f5_v10",
        "allando_v_sin_a8_f5_v10",
        "allando_v_sin_a2_f7_v10",
        "allando_v_sin_a8_f7_v10",
        "allando_v_sin_a2_f1_v15",
        "allando_v_sin_a8_f1_v15",
        "allando_v_sin_a2_f3_v15",
        "allando_v_sin_a8_f3_v15",
        "allando_v_sin_a2_f5_v15",
        "allando_v_sin_a8_f5_v15",
        "allando_v_sin_a2_f7_v15",
        "allando_v_sin_a8_f7_v15",
        "allando_v_sin_a2_f1_v20",
        "allando_v_sin_a8_f1_v20",
        "allando_v_sin_a2_f3_v20",
        "allando_v_sin_a8_f3_v20",
        "allando_v_sin_a2_f5_v20",
        "allando_v_sin_a8_f5_v20",
        "allando_v_sin_a2_f7_v20",
        "allando_v_sin_a8_f7_v20",
        "allando_v_sin_a2_f1_v25",
        "allando_v_sin_a8_f1_v25",
        "allando_v_sin_a2_f3_v25",
        "allando_v_sin_a8_f3_v25",
        "allando_v_sin_a2_f5_v25",
        "allando_v_sin_a8_f5_v25",
        "allando_v_sin_a2_f7_v25",
        "allando_v_sin_a8_f7_v25",
        "allando_v_sin_a2_f1_v30",
        "allando_v_sin_a8_f1_v30",
        "allando_v_sin_a2_f3_v30",
        "allando_v_sin_a8_f3_v30",
        "allando_v_sin_a2_f5_v30",
        "allando_v_sin_a8_f5_v30",
        "allando_v_sin_a2_f7_v30",
        "allando_v_sin_a8_f7_v30",
        "allando_v_sin_a2_f1_v35",
        "allando_v_sin_a8_f1_v35",
        "allando_v_sin_a2_f3_v35",
        "allando_v_sin_a8_f3_v35",
        "allando_v_sin_a2_f5_v35",
        "allando_v_sin_a8_f5_v35",
        "allando_v_sin_a2_f7_v35",
        "allando_v_sin_a8_f7_v35",
        "allando_v_sin_a2_f1_v40",
        "allando_v_sin_a8_f1_v40",
        "allando_v_sin_a2_f3_v40",
        "allando_v_sin_a8_f3_v40",
        "allando_v_sin_a2_f5_v40",
        "allando_v_sin_a8_f5_v40",
        "allando_v_sin_a2_f7_v40",
        "allando_v_sin_a8_f7_v40",
        "allando_v_sin_a2_f1_v45",
        "allando_v_sin_a8_f1_v45",
        "allando_v_sin_a2_f3_v45",
        "allando_v_sin_a8_f3_v45",
        "allando_v_sin_a2_f5_v45",
        "allando_v_sin_a8_f5_v45",
        "allando_v_sin_a2_f7_v45",
        "allando_v_sin_a8_f7_v45",
        "allando_v_sin_a2_f1_v50",
        "allando_v_sin_a8_f1_v50",
        "allando_v_sin_a2_f3_v50",
        "allando_v_sin_a8_f3_v50",
        "allando_v_sin_a2_f5_v50",
        "allando_v_sin_a8_f5_v50",
        "allando_v_sin_a2_f7_v50",
        "allando_v_sin_a8_f7_v50",
        "allando_v_sin_a2_f1_v55",
        "allando_v_sin_a8_f1_v55",
        "allando_v_sin_a2_f3_v55",
        "allando_v_sin_a8_f3_v55",
        "allando_v_sin_a2_f5_v55",
        "allando_v_sin_a8_f5_v55",
        "allando_v_sin_a2_f7_v55",
        "allando_v_sin_a8_f7_v55",
        "allando_v_sin_a2_f1_v60",
        "allando_v_sin_a8_f1_v60",
        "allando_v_sin_a2_f3_v60",
        "allando_v_sin_a8_f3_v60",
        "allando_v_sin_a2_f5_v60",
        "allando_v_sin_a8_f5_v60",
        "allando_v_sin_a2_f7_v60",
        "allando_v_sin_a8_f7_v60",
        "allando_v_sin_a2_f1_v65",
        "allando_v_sin_a8_f1_v65",
        "allando_v_sin_a2_f3_v65",
        "allando_v_sin_a8_f3_v65",
        "allando_v_sin_a2_f5_v65",
        "allando_v_sin_a8_f5_v65",
        "allando_v_sin_a2_f7_v65",
        "allando_v_sin_a8_f7_v65",
        "allando_v_sin_a2_f1_v70",
        "allando_v_sin_a8_f1_v70",
        "allando_v_sin_a2_f3_v70",
        "allando_v_sin_a8_f3_v70",
        "allando_v_sin_a2_f5_v70",
        "allando_v_sin_a8_f5_v70",
        "allando_v_sin_a2_f7_v70",
        "allando_v_sin_a8_f7_v70",
        "allando_v_sin_a2_f1_v75",
        "allando_v_sin_a8_f1_v75",
        "allando_v_sin_a2_f3_v75",
        "allando_v_sin_a8_f3_v75",
        "allando_v_sin_a2_f5_v75",
        "allando_v_sin_a8_f5_v75",
        "allando_v_sin_a2_f7_v75",
        "allando_v_sin_a8_f7_v75",
        "allando_v_sin_a2_f1_v80",
        "allando_v_sin_a8_f1_v80",
        "allando_v_sin_a2_f3_v80",
        "allando_v_sin_a8_f3_v80",
        "allando_v_sin_a2_f5_v80",
        "allando_v_sin_a8_f5_v80",
        "allando_v_sin_a2_f7_v80",
        "allando_v_sin_a8_f7_v80",
        "allando_v_sin_a2_f1_v85",
        "allando_v_sin_a8_f1_v85",
        "allando_v_sin_a2_f3_v85",
        "allando_v_sin_a8_f3_v85",
        "allando_v_sin_a2_f5_v85",
        "allando_v_sin_a8_f5_v85",
        "allando_v_sin_a2_f7_v85",
        "allando_v_sin_a8_f7_v85",
        "allando_v_sin_a2_f1_v90",
        "allando_v_sin_a8_f1_v90",
        "allando_v_sin_a2_f3_v90",
        "allando_v_sin_a8_f3_v90",
        "allando_v_sin_a2_f5_v90",
        "allando_v_sin_a8_f5_v90",
        "allando_v_sin_a2_f7_v90",
        "allando_v_sin_a8_f7_v90",
        "allando_v_sin_a2_f1_v95",
        "allando_v_sin_a8_f1_v95",
        "allando_v_sin_a2_f3_v95",
        "allando_v_sin_a8_f3_v95",
        "allando_v_sin_a2_f5_v95",
        "allando_v_sin_a8_f5_v95",
        "allando_v_sin_a2_f7_v95",
        "allando_v_sin_a8_f7_v95",
        "allando_v_sin_a2_f1_v100",
        "allando_v_sin_a8_f1_v100",
        "allando_v_sin_a2_f3_v100",
        "allando_v_sin_a8_f3_v100",
        "allando_v_sin_a2_f5_v100",
        "allando_v_sin_a8_f5_v100",
        "allando_v_sin_a2_f7_v100",
        "allando_v_sin_a8_f7_v100",
        "allando_v_sin_a2_f1_v105",
        "allando_v_sin_a8_f1_v105",
        "allando_v_sin_a2_f3_v105",
        "allando_v_sin_a8_f3_v105",
        "allando_v_sin_a2_f5_v105",
        "allando_v_sin_a8_f5_v105",
        "allando_v_sin_a2_f7_v105",
        "allando_v_sin_a8_f7_v105",
        "allando_v_sin_a2_f1_v110",
        "allando_v_sin_a8_f1_v110",
        "allando_v_sin_a2_f3_v110",
        "allando_v_sin_a8_f3_v110",
        "allando_v_sin_a2_f5_v110",
        "allando_v_sin_a8_f5_v110",
        "allando_v_sin_a2_f7_v110",
        "allando_v_sin_a8_f7_v110",
        "allando_v_sin_a2_f1_v115",
        "allando_v_sin_a8_f1_v115",
        "allando_v_sin_a2_f3_v115",
        "allando_v_sin_a8_f3_v115",
        "allando_v_sin_a2_f5_v115",
        "allando_v_sin_a8_f5_v115",
        "allando_v_sin_a2_f7_v115",
        "allando_v_sin_a8_f7_v115",
        "allando_v_sin_a2_f1_v120",
        "allando_v_sin_a8_f1_v120",
        "allando_v_sin_a2_f3_v120",
        "allando_v_sin_a8_f3_v120",
        "allando_v_sin_a2_f5_v120",
        "allando_v_sin_a8_f5_v120",
        "allando_v_sin_a2_f7_v120",
        "allando_v_sin_a8_f7_v120",
        "allando_v_sin_a2_f1_v125",
        "allando_v_sin_a8_f1_v125",
        "allando_v_sin_a2_f3_v125",
        "allando_v_sin_a8_f3_v125",
        "allando_v_sin_a2_f5_v125",
        "allando_v_sin_a8_f5_v125",
        "allando_v_sin_a2_f7_v125",
        "allando_v_sin_a8_f7_v125",
        "allando_v_sin_a2_f1_v130",
        "allando_v_sin_a8_f1_v130",
        "allando_v_sin_a2_f3_v130",
        "allando_v_sin_a8_f3_v130",
        "allando_v_sin_a2_f5_v130",
        "allando_v_sin_a8_f5_v130",
        "allando_v_sin_a2_f7_v130",
        "allando_v_sin_a8_f7_v130",
        "allando_v_sin_a2_f1_v135",
        "allando_v_sin_a8_f1_v135",
        "allando_v_sin_a2_f3_v135",
        "allando_v_sin_a8_f3_v135",
        "allando_v_sin_a2_f5_v135",
        "allando_v_sin_a8_f5_v135",
        "allando_v_sin_a2_f7_v135",
        "allando_v_sin_a8_f7_v135",
        "allando_v_sin_a2_f1_v140",
        "allando_v_sin_a8_f1_v140",
        "allando_v_sin_a2_f3_v140",
        "allando_v_sin_a8_f3_v140",
        "allando_v_sin_a2_f5_v140",
        "allando_v_sin_a8_f5_v140",
        "allando_v_sin_a2_f7_v140",
        "allando_v_sin_a8_f7_v140",
    ],
    [
        "allando_v_chirp_a1_v5",
        "allando_v_chirp_a3_v5",
        "allando_v_chirp_a5_v5",
        "allando_v_chirp_a1_v10",
        "allando_v_chirp_a3_v10",
        "allando_v_chirp_a5_v10",
        "allando_v_chirp_a1_v15",
        "allando_v_chirp_a3_v15",
        "allando_v_chirp_a5_v15",
        "allando_v_chirp_a1_v20",
        "allando_v_chirp_a3_v20",
        "allando_v_chirp_a5_v20",
        "allando_v_chirp_a1_v25",
        "allando_v_chirp_a3_v25",
        "allando_v_chirp_a5_v25",
        "allando_v_chirp_a1_v30",
        "allando_v_chirp_a3_v30",
        "allando_v_chirp_a5_v30",
        "allando_v_chirp_a1_v35",
        "allando_v_chirp_a3_v35",
        "allando_v_chirp_a5_v35",
        "allando_v_chirp_a1_v40",
        "allando_v_chirp_a3_v40",
        "allando_v_chirp_a5_v40",
        "allando_v_chirp_a1_v45",
        "allando_v_chirp_a3_v45",
        "allando_v_chirp_a5_v45",
        "allando_v_chirp_a1_v50",
        "allando_v_chirp_a3_v50",
        "allando_v_chirp_a5_v50",
        "allando_v_chirp_a1_v55",
        "allando_v_chirp_a3_v55",
        "allando_v_chirp_a5_v55",
        "allando_v_chirp_a1_v60",
        "allando_v_chirp_a3_v60",
        "allando_v_chirp_a5_v60",
        "allando_v_chirp_a1_v65",
        "allando_v_chirp_a3_v65",
        "allando_v_chirp_a5_v65",
        "allando_v_chirp_a1_v70",
        "allando_v_chirp_a3_v70",
        "allando_v_chirp_a5_v70",
        "allando_v_chirp_a1_v75",
        "allando_v_chirp_a3_v75",
        "allando_v_chirp_a5_v75",
        "allando_v_chirp_a1_v80",
        "allando_v_chirp_a3_v80",
        "allando_v_chirp_a5_v80",
        "allando_v_chirp_a1_v85",
        "allando_v_chirp_a3_v85",
        "allando_v_chirp_a5_v85",
        "allando_v_chirp_a1_v90",
        "allando_v_chirp_a3_v90",
        "allando_v_chirp_a5_v90",
        "allando_v_chirp_a1_v95",
        "allando_v_chirp_a3_v95",
        "allando_v_chirp_a5_v95",
        "allando_v_chirp_a1_v100",
        "allando_v_chirp_a3_v100",
        "allando_v_chirp_a5_v100",
        "allando_v_chirp_a1_v105",
        "allando_v_chirp_a3_v105",
        "allando_v_chirp_a5_v105",
        "allando_v_chirp_a1_v110",
        "allando_v_chirp_a3_v110",
        "allando_v_chirp_a5_v110",
        "allando_v_chirp_a1_v115",
        "allando_v_chirp_a3_v115",
        "allando_v_chirp_a5_v115",
        "allando_v_chirp_a1_v120",
        "allando_v_chirp_a3_v120",
        "allando_v_chirp_a5_v120",
        "allando_v_chirp_a1_v125",
        "allando_v_chirp_a3_v125",
        "allando_v_chirp_a5_v125",
        "allando_v_chirp_a1_v130",
        "allando_v_chirp_a3_v130",
        "allando_v_chirp_a5_v130",
        "allando_v_chirp_a1_v135",
        "allando_v_chirp_a3_v135",
        "allando_v_chirp_a5_v135",
        "allando_v_chirp_a1_v140",
        "allando_v_chirp_a3_v140",
        "allando_v_chirp_a5_v140",
    ],
    [
        "valtozo_v_savvaltas_gas_alacsony_pedal0_2",
        "valtozo_v_savvaltas_gas_kozepes_pedal0_2",
        "valtozo_v_savvaltas_gas_magas_pedal0_2",
        "valtozo_v_savvaltas_gas_alacsony_pedal0_5",
        "valtozo_v_savvaltas_gas_kozepes_pedal0_5",
        "valtozo_v_savvaltas_gas_magas_pedal0_5",
        "valtozo_v_savvaltas_gas_alacsony_pedal1_0",
        "valtozo_v_savvaltas_gas_kozepes_pedal1_0",
        "valtozo_v_savvaltas_gas_magas_pedal1_0",
    ],
    [
        "valtozo_v_savvaltas_fek_alacsony_pedal0_2",
        "valtozo_v_savvaltas_fek_kozepes_pedal0_2",
        "valtozo_v_savvaltas_fek_magas_pedal0_2",
        "valtozo_v_savvaltas_fek_alacsony_pedal0_5",
        "valtozo_v_savvaltas_fek_kozepes_pedal0_5",
        "valtozo_v_savvaltas_fek_magas_pedal0_5",
        "valtozo_v_savvaltas_fek_alacsony_pedal1_0",
        "valtozo_v_savvaltas_fek_kozepes_pedal1_0",
        "valtozo_v_savvaltas_fek_magas_pedal1_0",
    ],
    [
        "valtozo_v_sin_gas_a2_f1_pedal0_2",
        "valtozo_v_sin_gas_a2_f1_pedal0_5",
        "valtozo_v_sin_gas_a2_f1_pedal1_0",
        "valtozo_v_sin_gas_a2_f3_pedal0_2",
        "valtozo_v_sin_gas_a2_f3_pedal0_5",
        "valtozo_v_sin_gas_a2_f3_pedal1_0",
        "valtozo_v_sin_gas_a2_f5_pedal0_2",
        "valtozo_v_sin_gas_a2_f5_pedal0_5",
        "valtozo_v_sin_gas_a2_f5_pedal1_0",
        "valtozo_v_sin_gas_a2_f7_pedal0_2",
        "valtozo_v_sin_gas_a2_f7_pedal0_5",
        "valtozo_v_sin_gas_a2_f7_pedal1_0",
        "valtozo_v_sin_gas_a8_f1_pedal0_2",
        "valtozo_v_sin_gas_a8_f1_pedal0_5",
        "valtozo_v_sin_gas_a8_f1_pedal1_0",
        "valtozo_v_sin_gas_a8_f3_pedal0_2",
        "valtozo_v_sin_gas_a8_f3_pedal0_5",
        "valtozo_v_sin_gas_a8_f3_pedal1_0",
        "valtozo_v_sin_gas_a8_f5_pedal0_2",
        "valtozo_v_sin_gas_a8_f5_pedal0_5",
        "valtozo_v_sin_gas_a8_f5_pedal1_0",
        "valtozo_v_sin_gas_a8_f7_pedal0_2",
        "valtozo_v_sin_gas_a8_f7_pedal0_5",
        "valtozo_v_sin_gas_a8_f7_pedal1_0",
    ],
    [
        "valtozo_v_sin_fek_a2_f1_pedal0_2",
        "valtozo_v_sin_fek_a2_f1_pedal0_5",
        "valtozo_v_sin_fek_a2_f1_pedal1_0",
        "valtozo_v_sin_fek_a2_f3_pedal0_2",
        "valtozo_v_sin_fek_a2_f3_pedal0_5",
        "valtozo_v_sin_fek_a2_f3_pedal1_0",
        "valtozo_v_sin_fek_a2_f5_pedal0_2",
        "valtozo_v_sin_fek_a2_f5_pedal0_5",
        "valtozo_v_sin_fek_a2_f5_pedal1_0",
        "valtozo_v_sin_fek_a2_f7_pedal0_2",
        "valtozo_v_sin_fek_a2_f7_pedal0_5",
        "valtozo_v_sin_fek_a2_f7_pedal1_0",
        "valtozo_v_sin_fek_a8_f1_pedal0_2",
        "valtozo_v_sin_fek_a8_f1_pedal0_5",
        "valtozo_v_sin_fek_a8_f1_pedal1_0",
        "valtozo_v_sin_fek_a8_f3_pedal0_2",
        "valtozo_v_sin_fek_a8_f3_pedal0_5",
        "valtozo_v_sin_fek_a8_f3_pedal1_0",
        "valtozo_v_sin_fek_a8_f5_pedal0_2",
        "valtozo_v_sin_fek_a8_f5_pedal0_5",
        "valtozo_v_sin_fek_a8_f5_pedal1_0",
        "valtozo_v_sin_fek_a8_f7_pedal0_2",
        "valtozo_v_sin_fek_a8_f7_pedal0_5",
        "valtozo_v_sin_fek_a8_f7_pedal1_0",
    ],
]

velocity_maneuvers_list = [
    [
        "allando_v_savvaltas_alacsony_v5",
        "allando_v_savvaltas_kozepes_v5",
        "allando_v_savvaltas_magas_v5",
        "allando_v_sin_a2_f1_v5",
        "allando_v_sin_a8_f1_v5",
        "allando_v_sin_a2_f3_v5",
        "allando_v_sin_a8_f3_v5",
        "allando_v_sin_a2_f5_v5",
        "allando_v_sin_a8_f5_v5",
        "allando_v_sin_a2_f7_v5",
        "allando_v_sin_a8_f7_v5",
        "allando_v_chirp_a1_v5",
        "allando_v_chirp_a3_v5",
        "allando_v_chirp_a5_v5",
    ],
    [
        "allando_v_savvaltas_alacsony_v10",
        "allando_v_savvaltas_kozepes_v10",
        "allando_v_savvaltas_magas_v10",
        "allando_v_sin_a2_f1_v10",
        "allando_v_sin_a8_f1_v10",
        "allando_v_sin_a2_f3_v10",
        "allando_v_sin_a8_f3_v10",
        "allando_v_sin_a2_f5_v10",
        "allando_v_sin_a8_f5_v10",
        "allando_v_sin_a2_f7_v10",
        "allando_v_sin_a8_f7_v10",
        "allando_v_chirp_a1_v10",
        "allando_v_chirp_a3_v10",
        "allando_v_chirp_a5_v10",
    ],
    [
        "allando_v_savvaltas_alacsony_v15",
        "allando_v_savvaltas_kozepes_v15",
        "allando_v_savvaltas_magas_v15",
        "allando_v_sin_a2_f1_v15",
        "allando_v_sin_a8_f1_v15",
        "allando_v_sin_a2_f3_v15",
        "allando_v_sin_a8_f3_v15",
        "allando_v_sin_a2_f5_v15",
        "allando_v_sin_a8_f5_v15",
        "allando_v_sin_a2_f7_v15",
        "allando_v_sin_a8_f7_v15",
        "allando_v_chirp_a1_v15",
        "allando_v_chirp_a3_v15",
        "allando_v_chirp_a5_v15",
    ],
    [
        "allando_v_savvaltas_alacsony_v20",
        "allando_v_savvaltas_kozepes_v20",
        "allando_v_savvaltas_magas_v20",
        "allando_v_sin_a2_f1_v20",
        "allando_v_sin_a8_f1_v20",
        "allando_v_sin_a2_f3_v20",
        "allando_v_sin_a8_f3_v20",
        "allando_v_sin_a2_f5_v20",
        "allando_v_sin_a8_f5_v20",
        "allando_v_sin_a2_f7_v20",
        "allando_v_sin_a8_f7_v20",
        "allando_v_chirp_a1_v20",
        "allando_v_chirp_a3_v20",
        "allando_v_chirp_a5_v20",
    ],
    [
        "allando_v_savvaltas_alacsony_v25",
        "allando_v_savvaltas_kozepes_v25",
        "allando_v_savvaltas_magas_v25",
        "allando_v_sin_a2_f1_v25",
        "allando_v_sin_a8_f1_v25",
        "allando_v_sin_a2_f3_v25",
        "allando_v_sin_a8_f3_v25",
        "allando_v_sin_a2_f5_v25",
        "allando_v_sin_a8_f5_v25",
        "allando_v_sin_a2_f7_v25",
        "allando_v_sin_a8_f7_v25",
        "allando_v_chirp_a1_v25",
        "allando_v_chirp_a3_v25",
        "allando_v_chirp_a5_v25",
    ],
    [
        "allando_v_savvaltas_alacsony_v30",
        "allando_v_savvaltas_kozepes_v30",
        "allando_v_savvaltas_magas_v30",
        "allando_v_sin_a2_f1_v30",
        "allando_v_sin_a8_f1_v30",
        "allando_v_sin_a2_f3_v30",
        "allando_v_sin_a8_f3_v30",
        "allando_v_sin_a2_f5_v30",
        "allando_v_sin_a8_f5_v30",
        "allando_v_sin_a2_f7_v30",
        "allando_v_sin_a8_f7_v30",
        "allando_v_chirp_a1_v30",
        "allando_v_chirp_a3_v30",
        "allando_v_chirp_a5_v30",
    ],
    [
        "allando_v_savvaltas_alacsony_v35",
        "allando_v_savvaltas_kozepes_v35",
        "allando_v_savvaltas_magas_v35",
        "allando_v_sin_a2_f1_v35",
        "allando_v_sin_a8_f1_v35",
        "allando_v_sin_a2_f3_v35",
        "allando_v_sin_a8_f3_v35",
        "allando_v_sin_a2_f5_v35",
        "allando_v_sin_a8_f5_v35",
        "allando_v_sin_a2_f7_v35",
        "allando_v_sin_a8_f7_v35",
        "allando_v_chirp_a1_v35",
        "allando_v_chirp_a3_v35",
        "allando_v_chirp_a5_v35",
    ],
    [
        "allando_v_savvaltas_alacsony_v40",
        "allando_v_savvaltas_kozepes_v40",
        "allando_v_savvaltas_magas_v40",
        "allando_v_sin_a2_f1_v40",
        "allando_v_sin_a8_f1_v40",
        "allando_v_sin_a2_f3_v40",
        "allando_v_sin_a8_f3_v40",
        "allando_v_sin_a2_f5_v40",
        "allando_v_sin_a8_f5_v40",
        "allando_v_sin_a2_f7_v40",
        "allando_v_sin_a8_f7_v40",
        "allando_v_chirp_a1_v40",
        "allando_v_chirp_a3_v40",
        "allando_v_chirp_a5_v40",
    ],
    [
        "allando_v_savvaltas_alacsony_v45",
        "allando_v_savvaltas_kozepes_v45",
        "allando_v_savvaltas_magas_v45",
        "allando_v_sin_a2_f1_v45",
        "allando_v_sin_a8_f1_v45",
        "allando_v_sin_a2_f3_v45",
        "allando_v_sin_a8_f3_v45",
        "allando_v_sin_a2_f5_v45",
        "allando_v_sin_a8_f5_v45",
        "allando_v_sin_a2_f7_v45",
        "allando_v_sin_a8_f7_v45",
        "allando_v_chirp_a1_v45",
        "allando_v_chirp_a3_v45",
        "allando_v_chirp_a5_v45",
    ],
    [
        "allando_v_savvaltas_alacsony_v50",
        "allando_v_savvaltas_kozepes_v50",
        "allando_v_savvaltas_magas_v50",
        "allando_v_sin_a2_f1_v50",
        "allando_v_sin_a8_f1_v50",
        "allando_v_sin_a2_f3_v50",
        "allando_v_sin_a8_f3_v50",
        "allando_v_sin_a2_f5_v50",
        "allando_v_sin_a8_f5_v50",
        "allando_v_sin_a2_f7_v50",
        "allando_v_sin_a8_f7_v50",
        "allando_v_chirp_a1_v50",
        "allando_v_chirp_a3_v50",
        "allando_v_chirp_a5_v50",
    ],
    [
        "allando_v_savvaltas_alacsony_v55",
        "allando_v_savvaltas_kozepes_v55",
        "allando_v_savvaltas_magas_v55",
        "allando_v_sin_a2_f1_v55",
        "allando_v_sin_a8_f1_v55",
        "allando_v_sin_a2_f3_v55",
        "allando_v_sin_a8_f3_v55",
        "allando_v_sin_a2_f5_v55",
        "allando_v_sin_a8_f5_v55",
        "allando_v_sin_a2_f7_v55",
        "allando_v_sin_a8_f7_v55",
        "allando_v_chirp_a1_v55",
        "allando_v_chirp_a3_v55",
        "allando_v_chirp_a5_v55",
    ],
    [
        "allando_v_savvaltas_alacsony_v60",
        "allando_v_savvaltas_kozepes_v60",
        "allando_v_savvaltas_magas_v60",
        "allando_v_sin_a2_f1_v60",
        "allando_v_sin_a8_f1_v60",
        "allando_v_sin_a2_f3_v60",
        "allando_v_sin_a8_f3_v60",
        "allando_v_sin_a2_f5_v60",
        "allando_v_sin_a8_f5_v60",
        "allando_v_sin_a2_f7_v60",
        "allando_v_sin_a8_f7_v60",
        "allando_v_chirp_a1_v60",
        "allando_v_chirp_a3_v60",
        "allando_v_chirp_a5_v60",
    ],
    [
        "allando_v_savvaltas_alacsony_v65",
        "allando_v_savvaltas_kozepes_v65",
        "allando_v_savvaltas_magas_v65",
        "allando_v_sin_a2_f1_v65",
        "allando_v_sin_a8_f1_v65",
        "allando_v_sin_a2_f3_v65",
        "allando_v_sin_a8_f3_v65",
        "allando_v_sin_a2_f5_v65",
        "allando_v_sin_a8_f5_v65",
        "allando_v_sin_a2_f7_v65",
        "allando_v_sin_a8_f7_v65",
        "allando_v_chirp_a1_v65",
        "allando_v_chirp_a3_v65",
        "allando_v_chirp_a5_v65",
    ],
    [
        "allando_v_savvaltas_alacsony_v70",
        "allando_v_savvaltas_kozepes_v70",
        "allando_v_savvaltas_magas_v70",
        "allando_v_sin_a2_f1_v70",
        "allando_v_sin_a8_f1_v70",
        "allando_v_sin_a2_f3_v70",
        "allando_v_sin_a8_f3_v70",
        "allando_v_sin_a2_f5_v70",
        "allando_v_sin_a8_f5_v70",
        "allando_v_sin_a2_f7_v70",
        "allando_v_sin_a8_f7_v70",
        "allando_v_chirp_a1_v70",
        "allando_v_chirp_a3_v70",
        "allando_v_chirp_a5_v70",
    ],
    [
        "allando_v_savvaltas_alacsony_v75",
        "allando_v_savvaltas_kozepes_v75",
        "allando_v_savvaltas_magas_v75",
        "allando_v_sin_a2_f1_v75",
        "allando_v_sin_a8_f1_v75",
        "allando_v_sin_a2_f3_v75",
        "allando_v_sin_a8_f3_v75",
        "allando_v_sin_a2_f5_v75",
        "allando_v_sin_a8_f5_v75",
        "allando_v_sin_a2_f7_v75",
        "allando_v_sin_a8_f7_v75",
        "allando_v_chirp_a1_v75",
        "allando_v_chirp_a3_v75",
        "allando_v_chirp_a5_v75",
    ],
    [
        "allando_v_savvaltas_alacsony_v80",
        "allando_v_savvaltas_kozepes_v80",
        "allando_v_savvaltas_magas_v80",
        "allando_v_sin_a2_f1_v80",
        "allando_v_sin_a8_f1_v80",
        "allando_v_sin_a2_f3_v80",
        "allando_v_sin_a8_f3_v80",
        "allando_v_sin_a2_f5_v80",
        "allando_v_sin_a8_f5_v80",
        "allando_v_sin_a2_f7_v80",
        "allando_v_sin_a8_f7_v80",
        "allando_v_chirp_a1_v80",
        "allando_v_chirp_a3_v80",
        "allando_v_chirp_a5_v80",
    ],
    [
        "allando_v_savvaltas_alacsony_v85",
        "allando_v_savvaltas_kozepes_v85",
        "allando_v_savvaltas_magas_v85",
        "allando_v_sin_a2_f1_v85",
        "allando_v_sin_a8_f1_v85",
        "allando_v_sin_a2_f3_v85",
        "allando_v_sin_a8_f3_v85",
        "allando_v_sin_a2_f5_v85",
        "allando_v_sin_a8_f5_v85",
        "allando_v_sin_a2_f7_v85",
        "allando_v_sin_a8_f7_v85",
        "allando_v_chirp_a1_v85",
        "allando_v_chirp_a3_v85",
        "allando_v_chirp_a5_v85",
    ],
    [
        "allando_v_savvaltas_alacsony_v90",
        "allando_v_savvaltas_kozepes_v90",
        "allando_v_savvaltas_magas_v90",
        "allando_v_sin_a2_f1_v90",
        "allando_v_sin_a8_f1_v90",
        "allando_v_sin_a2_f3_v90",
        "allando_v_sin_a8_f3_v90",
        "allando_v_sin_a2_f5_v90",
        "allando_v_sin_a8_f5_v90",
        "allando_v_sin_a2_f7_v90",
        "allando_v_sin_a8_f7_v90",
        "allando_v_chirp_a1_v90",
        "allando_v_chirp_a3_v90",
        "allando_v_chirp_a5_v90",
    ],
    [
        "allando_v_savvaltas_alacsony_v95",
        "allando_v_savvaltas_kozepes_v95",
        "allando_v_savvaltas_magas_v95",
        "allando_v_sin_a2_f1_v95",
        "allando_v_sin_a8_f1_v95",
        "allando_v_sin_a2_f3_v95",
        "allando_v_sin_a8_f3_v95",
        "allando_v_sin_a2_f5_v95",
        "allando_v_sin_a8_f5_v95",
        "allando_v_sin_a2_f7_v95",
        "allando_v_sin_a8_f7_v95",
        "allando_v_chirp_a1_v95",
        "allando_v_chirp_a3_v95",
        "allando_v_chirp_a5_v95",
    ],
    [
        "allando_v_savvaltas_alacsony_v100",
        "allando_v_savvaltas_kozepes_v100",
        "allando_v_savvaltas_magas_v100",
        "allando_v_sin_a2_f1_v100",
        "allando_v_sin_a8_f1_v100",
        "allando_v_sin_a2_f3_v100",
        "allando_v_sin_a8_f3_v100",
        "allando_v_sin_a2_f5_v100",
        "allando_v_sin_a8_f5_v100",
        "allando_v_sin_a2_f7_v100",
        "allando_v_sin_a8_f7_v100",
        "allando_v_chirp_a1_v100",
        "allando_v_chirp_a3_v100",
        "allando_v_chirp_a5_v100",
    ],
    [
        "allando_v_savvaltas_alacsony_v105",
        "allando_v_savvaltas_kozepes_v105",
        "allando_v_savvaltas_magas_v105",
        "allando_v_sin_a2_f1_v105",
        "allando_v_sin_a8_f1_v105",
        "allando_v_sin_a2_f3_v105",
        "allando_v_sin_a8_f3_v105",
        "allando_v_sin_a2_f5_v105",
        "allando_v_sin_a8_f5_v105",
        "allando_v_sin_a2_f7_v105",
        "allando_v_sin_a8_f7_v105",
        "allando_v_chirp_a1_v105",
        "allando_v_chirp_a3_v105",
        "allando_v_chirp_a5_v105",
    ],
    [
        "allando_v_savvaltas_alacsony_v110",
        "allando_v_savvaltas_kozepes_v110",
        "allando_v_savvaltas_magas_v110",
        "allando_v_sin_a2_f1_v110",
        "allando_v_sin_a8_f1_v110",
        "allando_v_sin_a2_f3_v110",
        "allando_v_sin_a8_f3_v110",
        "allando_v_sin_a2_f5_v110",
        "allando_v_sin_a8_f5_v110",
        "allando_v_sin_a2_f7_v110",
        "allando_v_sin_a8_f7_v110",
        "allando_v_chirp_a1_v110",
        "allando_v_chirp_a3_v110",
        "allando_v_chirp_a5_v110",
    ],
    [
        "allando_v_savvaltas_alacsony_v115",
        "allando_v_savvaltas_kozepes_v115",
        "allando_v_savvaltas_magas_v115",
        "allando_v_sin_a2_f1_v115",
        "allando_v_sin_a8_f1_v115",
        "allando_v_sin_a2_f3_v115",
        "allando_v_sin_a8_f3_v115",
        "allando_v_sin_a2_f5_v115",
        "allando_v_sin_a8_f5_v115",
        "allando_v_sin_a2_f7_v115",
        "allando_v_sin_a8_f7_v115",
        "allando_v_chirp_a1_v115",
        "allando_v_chirp_a3_v115",
        "allando_v_chirp_a5_v115",
    ],
    [
        "allando_v_savvaltas_alacsony_v120",
        "allando_v_savvaltas_kozepes_v120",
        "allando_v_savvaltas_magas_v120",
        "allando_v_sin_a2_f1_v120",
        "allando_v_sin_a8_f1_v120",
        "allando_v_sin_a2_f3_v120",
        "allando_v_sin_a8_f3_v120",
        "allando_v_sin_a2_f5_v120",
        "allando_v_sin_a8_f5_v120",
        "allando_v_sin_a2_f7_v120",
        "allando_v_sin_a8_f7_v120",
        "allando_v_chirp_a1_v120",
        "allando_v_chirp_a3_v120",
        "allando_v_chirp_a5_v120",
    ],
    [
        "allando_v_savvaltas_alacsony_v125",
        "allando_v_savvaltas_kozepes_v125",
        "allando_v_savvaltas_magas_v125",
        "allando_v_sin_a2_f1_v125",
        "allando_v_sin_a8_f1_v125",
        "allando_v_sin_a2_f3_v125",
        "allando_v_sin_a8_f3_v125",
        "allando_v_sin_a2_f5_v125",
        "allando_v_sin_a8_f5_v125",
        "allando_v_sin_a2_f7_v125",
        "allando_v_sin_a8_f7_v125",
        "allando_v_chirp_a1_v125",
        "allando_v_chirp_a3_v125",
        "allando_v_chirp_a5_v125",
    ],
    [
        "allando_v_savvaltas_alacsony_v130",
        "allando_v_savvaltas_kozepes_v130",
        "allando_v_savvaltas_magas_v130",
        "allando_v_sin_a2_f1_v130",
        "allando_v_sin_a8_f1_v130",
        "allando_v_sin_a2_f3_v130",
        "allando_v_sin_a8_f3_v130",
        "allando_v_sin_a2_f5_v130",
        "allando_v_sin_a8_f5_v130",
        "allando_v_sin_a2_f7_v130",
        "allando_v_sin_a8_f7_v130",
        "allando_v_chirp_a1_v130",
        "allando_v_chirp_a3_v130",
        "allando_v_chirp_a5_v130",
    ],
    [
        "allando_v_savvaltas_alacsony_v135",
        "allando_v_savvaltas_kozepes_v135",
        "allando_v_savvaltas_magas_v135",
        "allando_v_sin_a2_f1_v135",
        "allando_v_sin_a8_f1_v135",
        "allando_v_sin_a2_f3_v135",
        "allando_v_sin_a8_f3_v135",
        "allando_v_sin_a2_f5_v135",
        "allando_v_sin_a8_f5_v135",
        "allando_v_sin_a2_f7_v135",
        "allando_v_sin_a8_f7_v135",
        "allando_v_chirp_a1_v135",
        "allando_v_chirp_a3_v135",
        "allando_v_chirp_a5_v135",
    ],
    [
        "allando_v_savvaltas_alacsony_v140",
        "allando_v_savvaltas_kozepes_v140",
        "allando_v_savvaltas_magas_v140",
        "allando_v_sin_a2_f1_v140",
        "allando_v_sin_a8_f1_v140",
        "allando_v_sin_a2_f3_v140",
        "allando_v_sin_a8_f3_v140",
        "allando_v_sin_a2_f5_v140",
        "allando_v_sin_a8_f5_v140",
        "allando_v_sin_a2_f7_v140",
        "allando_v_sin_a8_f7_v140",
        "allando_v_chirp_a1_v140",
        "allando_v_chirp_a3_v140",
        "allando_v_chirp_a5_v140",
    ],
]

cos_sim.compute_cosine_similarity_within_groups(velocity_maneuvers_list)

# Redundáns párok keresése
redundant_pairs = cos_sim.detect_redundancy()

removed_manoeuvres_by_group = cos_sim.remove_redundancy(redundant_pairs)
