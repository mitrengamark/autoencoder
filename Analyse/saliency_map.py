import torch
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from Config.load_config import save_saliency


def compute_saliency_map(input, output, device):
    input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    reconstructed_tensor = (
        torch.tensor(output, dtype=torch.float32).unsqueeze(0).to(device)
    )

    loss = torch.nn.functional.mse_loss(reconstructed_tensor, input_tensor)
    loss.backward()
    saliency = input_tensor.grad.abs().squeeze(0).detach().cpu()
    return saliency


def plot_saliency_map(
    all_columns, avg_saliency, maneouvre_group_name, save_dir="Saliency_Results"
):
    if isinstance(avg_saliency, torch.Tensor):
        avg_saliency = avg_saliency.numpy()

    # Rendezés csökkenő sorrendben
    saliency_df = pd.DataFrame(
        {"Feature": all_columns, "Saliency": avg_saliency}
    ).sort_values(by="Saliency", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(saliency_df["Feature"], saliency_df["Saliency"])
    plt.title("Saliency map - átlagolt (rekonstrukciós gradiens alapján)")
    plt.ylabel("Absz. gradiens (fontosság)")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    if save_saliency == 1:
        # Kép mentése
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"{maneouvre_group_name}_saliency_map.png")
        plt.savefig(image_path)
        print(f"Saliency map ábra elmentve ide: {image_path}")

        # CSV mentés
        csv_path = os.path.join(save_dir, f"{maneouvre_group_name}_saliency_values.csv")
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Feature", "Saliency"])
            for feature, value in zip(all_columns, avg_saliency):
                writer.writerow([feature, value])
        print(f"Saliency értékek CSV-be mentve ide: {csv_path}")

    plt.show()


def saved_saliency_map_data():
    df = pd.read_csv("Saliency_Results/saliency_values.csv")
    all_columns = df["Feature"].tolist()
    saliency_values = df["Saliency"].tolist()

    return all_columns, saliency_values
