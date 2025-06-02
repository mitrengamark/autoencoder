import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import re
import pandas as pd


def compare_json_sets_by_threshold_from_selected_subfolders(root_dir, subfolders):
    # Step 1: Collect JSON files by threshold across selected subfolders
    threshold_files = defaultdict(list)  # {threshold: [file_paths]}
    threshold_pattern = re.compile(
        r"removing_(\d+)_"
    )  # To capture threshold like 90, 95, 98

    for subfolder in subfolders:
        folder_path = Path(root_dir) / subfolder
        if not folder_path.is_dir():
            continue
        for file in folder_path.glob("*.json"):
            match = threshold_pattern.search(file.stem)
            if match:
                threshold = match.group(1)
                threshold_files[threshold].append(file)

    # Step 2: Compare all files within each threshold group
    comparison_results = []

    for threshold, file_list in threshold_files.items():
        for f1, f2 in combinations(file_list, 2):
            # Load JSON data
            with open(f1, "r") as file1, open(f2, "r") as file2:
                data1 = json.load(file1)
                data2 = json.load(file2)

            # Flatten manoeuvres
            set1 = {item for sublist in data1.values() for item in sublist}
            set2 = {item for sublist in data2.values() for item in sublist}

            # Calculate metrics
            common = set1 & set2
            union = set1 | set2
            jaccard_index = len(common) / len(union) if union else 0
            precision = len(common) / len(set2) if set2 else 0
            recall = len(common) / len(set1) if set1 else 0
            f1_score = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall)
                else 0
            )
            overlap_coefficient = (
                len(common) / min(len(set1), len(set2))
                if min(len(set1), len(set2))
                else 0
            )

            comparison_results.append(
                {
                    "threshold": threshold,
                    "file1": f1.parent.name,
                    "file2": f2.parent.name,
                    "count_1": len(set1),
                    "count_2": len(set2),
                    "common": len(common),
                    "jaccard_index": round(jaccard_index, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1_score, 4),
                    "overlap_coefficient": round(overlap_coefficient, 4),
                }
            )

    return comparison_results, pd.DataFrame(comparison_results)


results, df_results = compare_json_sets_by_threshold_from_selected_subfolders(
    "cosine_similarity_matrices",
    ["bmw_OG_remake", "OG_remake", "bmw_model_tesla_data", "tesla_model_bmw_data"],
)
output_path = "Results/redundancy_comparison_results.csv"
df_results.to_csv(output_path, index=False)

print(json.dumps(results, indent=4))
