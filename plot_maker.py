import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fájlnevek és elérési útvonal
base_path = "data_plot/BMW"
filenames = {
    "All data": "All data.csv",
    # "All data threshold 0.9": "All data threshold 0.9.csv",
    # "All data threshold 0.95": "All data threshold 0.95.csv",
    # "All data threshold 0.98": "All data threshold 0.98.csv",
    "Chirp (constant speed)": "Chirp (constant speed).csv",
    # "Chirp (constant speed) threshold 0.9": "Chirp (constant speed) threshold 0.9.csv",
    # "Chirp (constant speed) threshold 0.95": "Chirp (constant speed) threshold 0.95.csv",
    # "Chirp (constant speed) threshold 0.98": "Chirp (constant speed) threshold 0.98.csv",
    "Lane change (constant speed)": "Lane change (constant speed).csv",
    # "Lane change (constant speed) threshold 0.9": "Lane change (constant speed) threshold 0.9.csv",
    # "Lane change (constant speed) threshold 0.95": "Lane change (constant speed) threshold 0.95.csv",
    # "Lane change (constant speed) threshold 0.98": "Lane change (constant speed) threshold 0.98.csv",
    "Sinusoidal (constant speed)": "Sinusoidal (constant speed).csv",
    # "Sinusoidal (constant speed) threshold 0.9": "Sinusoidal (constant speed) threshold 0.9.csv",
    # "Sinusoidal (constant speed) threshold 0.95": "Sinusoidal (constant speed) threshold 0.95.csv",
    # "Sinusoidal (constant speed) threshold 0.98": "Sinusoidal (constant speed) threshold 0.98.csv",
    "Lane change (acceleration)": "Lane change (acceleration).csv",
    # "Lane change (acceleration) threshold 0.9": "Lane change (acceleration) threshold 0.9.csv",
    # "Lane change (acceleration) threshold 0.95": "Lane change (acceleration) threshold 0.95.csv",
    # "Lane change (acceleration) threshold 0.98": "Lane change (acceleration) threshold 0.98.csv",
    "Lane change (slow down)": "Lane change (slow down).csv",
    # "Lane change (slow down) threshold 0.9": "Lane change (slow down) threshold 0.9.csv",
    # "Lane change (slow down) threshold 0.95": "Lane change (slow down) threshold 0.95.csv",
    # "Lane change (slow down) threshold 0.98": "Lane change (slow down) threshold 0.98.csv",
    "Sinusoidal (acceleration)": "Sinusoidal (acceleration).csv",
    # "Sinusoidal (acceleration) threshold 0.9": "Sinusoidal (acceleration) threshold 0.9.csv",
    # "Sinusoidal (acceleration) threshold 0.95": "Sinusoidal (acceleration) threshold 0.95.csv",
    # "Sinusoidal (acceleration) threshold 0.98": "Sinusoidal (acceleration) threshold 0.98.csv",
    "Sinusoidal (slow down)": "Sinusoidal (slow down).csv",
    # "Sinusoidal (slow down) threshold 0.9": "Sinusoidal (slow down) threshold 0.9.csv",
    # "Sinusoidal (slow down) threshold 0.95": "Sinusoidal (slow down) threshold 0.95.csv",
    # "Sinusoidal (slow down) threshold 0.98": "Sinusoidal (slow down) threshold 0.98.csv",
}

custom_colors = {
    "All data": "#D60000",
    # "All data threshold 0.9": "#3AB2EE",
    # "All data threshold 0.95": "#115D97",
    # "All data threshold 0.98": "#5A47EB",
    "Chirp (constant speed)": "#3AB2EE",
    # "Chirp (constant speed) threshold 0.9": "#3AB2EE",
    # "Chirp (constant speed) threshold 0.95": "#115D97",
    # "Chirp (constant speed) threshold 0.98": "#5A47EB",
    "Lane change (constant speed)": "#115D97",
    # "Lane change (constant speed) threshold 0.9": "#3AB2EE",
    # "Lane change (constant speed) threshold 0.95": "#115D97",
    # "Lane change (constant speed) threshold 0.98": "#5A47EB",
    "Sinusoidal (constant speed)": "#5A47EB",
    # "Sinusoidal (constant speed) threshold 0.9": "#3AB2EE",
    # "Sinusoidal (constant speed) threshold 0.95": "#115D97",
    # "Sinusoidal (constant speed) threshold 0.98": "#5A47EB",
    "Lane change (acceleration)": "#3AB2EE",
    # "Lane change (acceleration) threshold 0.9": "#3AB2EE",
    # "Lane change (acceleration) threshold 0.95": "#115D97",
    # "Lane change (acceleration) threshold 0.98": "#5A47EB",
    "Lane change (slow down)": "#115D97",
    # "Lane change (slow down) threshold 0.9": "#3AB2EE",
    # "Lane change (slow down) threshold 0.95": "#115D97",
    # "Lane change (slow down) threshold 0.98": "#5A47EB",
    "Sinusoidal (acceleration)": "#5A47EB",
    # "Sinusoidal (acceleration) threshold 0.9": "#3AB2EE",
    # "Sinusoidal (acceleration) threshold 0.95": "#115D97",
    # "Sinusoidal (acceleration) threshold 0.98": "#5A47EB",
    "Sinusoidal (slow down)": "#610891",
    # "Sinusoidal (slow down) threshold 0.9": "#3AB2EE",
    # "Sinusoidal (slow down) threshold 0.95": "#115D97",
    # "Sinusoidal (slow down) threshold 0.98": "#5A47EB",
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
plot_losses(constant_speed, "Jurnal_plots/BMW/constant_speed_losses.png")
plot_losses(variable_speed, "Jurnal_plots/BMW/variable_speed_losses.png")
