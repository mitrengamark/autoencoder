import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Adatok
data = {
    "Manővercsoport": [
        "Áll. csirp",
        "Áll. sávváltás",
        "Áll. szinusz",
        "Vált. sávv. gyors.",
        "Vált. sávv. lass.",
        "Vált. szin. gyors.",
        "Vált. szin. lass.",
    ],
    "BMW": [702, 704, 1864, 76, 75, 201, 198],
    "Tesla": [632, 633, 1679, 68, 68, 180, 179],
}

df = pd.DataFrame(data)
df_melted = df.melt(
    id_vars="Manővercsoport", var_name="Jármű", value_name="Adatméret (MB)"
)

# Stílus
sns.set(style="whitegrid", font_scale=1.1)
palette = {"BMW": "#115D97", "Tesla": "#3AB2EE"}

# Ábra
fig, ax = plt.subplots(figsize=(10, 6))
barplot = sns.barplot(
    x="Manővercsoport",
    y="Adatméret (MB)",
    hue="Jármű",
    data=df_melted,
    palette=palette,
    ax=ax,
)

# Grid és tengelyek
ax.yaxis.grid(True, linestyle="-", color="lightgray", alpha=0.7)
ax.xaxis.grid(False)
ax.set_xlabel("")
ax.set_ylabel("Adatméret (MB)")
ax.set_yticklabels([])

# Oszlopcímkék
for container in barplot.containers:
    barplot.bar_label(container, fmt="%d MB", label_type="edge", fontsize=9)

plt.xticks(rotation=15)
plt.legend(title="")
plt.tight_layout()

# PNG mentés
plt.savefig("manovercsoport_bmw_tesla.png", dpi=300, format="png")
plt.close()
