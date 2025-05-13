import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fájlnevek és elérési útvonal
base_path = "data_plot"
filenames = {
    "All data": "All data.csv",
    "Chirp (constant speed)": "Chirp (constant speed).csv",
    "Lane change (constant speed)": "Lane change (constant speed).csv",
    "Sinusoidal (constant speed)": "Sinusoidal (constant speed).csv",
    "Lane change (acceleration)": "Lane change (acceleration).csv",
    "Lane change (slow down)": "Lane change (slow down).csv",
    "Sinusoidal (acceleration)": "Sinusoidal (acceleration).csv",
    "Sinusoidal (slow down)": "Sinusoidal (slow down).csv",
}

custom_colors = {
    "All data": "#D60000",  # fekete
    "Chirp (constant speed)": "#3AB2EE",
    "Lane change (constant speed)": "#115D97",
    "Sinusoidal (constant speed)": "#5A47EB",
    "Lane change (acceleration)": "#115D97",
    "Lane change (slow down)": "#3AB2EE",
    "Sinusoidal (acceleration)": "#610891",
    "Sinusoidal (slow down)": "#5A47EB",
}

dataframes = {}
for label, file in filenames.items():
    df = pd.read_csv(os.path.join(base_path, file), header=None)
    if df.shape[1] >= 3:
        dataframes[label] = df.iloc[:, 2]

# Csoportosítás
constant_speed = ["All data", "Chirp (constant speed)", "Lane change (constant speed)", "Sinusoidal (constant speed)"]
variable_speed = ["All data", "Lane change (acceleration)", "Lane change (slow down)", "Sinusoidal (acceleration)", "Sinusoidal (slow down)"]

# Plotoló függvény
def plot_losses(group, output_file):
    plt.figure(figsize=(10, 6))
    for label in group:
        if label in dataframes:
            sns.lineplot(
                x=range(len(dataframes[label])),
                y=dataframes[label],
                label=label,
                color=custom_colors.get(label, "#333333")  # fallback szín
            )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# Plottok mentése fájlba
plot_losses(constant_speed, "Jurnal_plots/constant_speed_losses.png")
plot_losses(variable_speed, "Jurnal_plots/variable_speed_losses.png")
