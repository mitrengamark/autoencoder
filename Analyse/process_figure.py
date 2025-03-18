import os
import pandas as pd
import matplotlib.pyplot as plt
from Config.load_config import selected_manoeuvres

def process_figures():
    # Mappák elérési útjai
    input_folder = "data3"
    difference_folder = "data_difference"
    output_folder = "Process_figures"

    # Ha a kimeneti mappa nem létezik, létrehozzuk
    os.makedirs(output_folder, exist_ok=True)

    # Összes manőver azonosítása (CSV fájlok listázása)
    maneuver_files = [f.replace(".csv", "") for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Feldolgozás minden manőverre
    for maneuver_name in maneuver_files:
        print(f"\nFeldolgozás: {maneuver_name}")

        # Mappák létrehozása az adott manőverhez
        maneuver_output_folder = os.path.join(output_folder, maneuver_name)
        os.makedirs(maneuver_output_folder, exist_ok=True)

        # Fájlok elérési útjai
        original_path = os.path.join(input_folder, f"{maneuver_name}.csv")
        difference_path = os.path.join(difference_folder, f"{maneuver_name}_difference.csv")

        # Ellenőrizzük, hogy mindkét fájl létezik-e
        if not os.path.exists(original_path) or not os.path.exists(difference_path):
            print(f"HIBA: Nem található az adathalmaz {maneuver_name}-hez!")
            continue

        # Adatok beolvasása
        df_original = pd.read_csv(original_path)
        df_difference = pd.read_csv(difference_path)

        # Időindex létrehozása
        time_index = range(len(df_original))

        # Az összes változóhoz plot készítése és mentése
        for column in df_original.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(time_index, df_original[column], label=f"{column} (Eredeti)", color="blue")
            plt.plot(time_index[1:], df_difference[column], label=f"{column} változása (Δ)", color="red", linestyle="dashed")
            plt.xlabel("Időlépés")
            plt.ylabel(column)
            plt.title(f"{column} és annak változása - {maneuver_name}")
            plt.legend()
            plt.grid()

            # Kép mentése
            figure_path = os.path.join(maneuver_output_folder, f"{maneuver_name}_{column}.png")
            plt.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close()  # Nem jelenítjük meg a plotokat, csak mentjük
            
            print(f"Mentve: {figure_path}")

    print("\nMinden manőver feldolgozása kész! Az eredmények a 'Process_figures' mappában találhatók.")
