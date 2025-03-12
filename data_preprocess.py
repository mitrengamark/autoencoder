import scipy.io
import pandas as pd
import os
import glob
import numpy as np
import re


def mat_to_csv():
    data_dir = "data"
    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

    for mat_file in mat_files:
        mat_file_path = os.path.join(data_dir, mat_file)
        mat_content = scipy.io.loadmat(
            mat_file_path, struct_as_record=False, squeeze_me=True
        )

        # Ellenőrizzük a betöltött adatok szerkezetét
        for key, value in mat_content.items():
            if not key.startswith("__"):  # A metaadatok kihagyása
                print(f"Key: {key}, Value: {value}")

                # Ha az adat egy struktúra, hagyjuk figyelmen kívül
                if hasattr(value, "_fieldnames"):
                    print(
                        f"A '{key}' kulcs alatt található adat egy struktúra, kihagyva."
                    )
                    continue

                # Ha az adat egy oszlopvektor vagy 2D tömb, alakítsuk át DataFrame-é
                if isinstance(value, (list, tuple)) or hasattr(value, "shape"):
                    df = pd.DataFrame(value)
                    csv_file_name = os.path.splitext(mat_file)[0] + f"_{key}.csv"
                    csv_file_path = os.path.join(data_dir, csv_file_name)
                    df.to_csv(csv_file_path, index=False)
                    print(f"Az adatok sikeresen mentve lettek: {csv_file_path}")
                else:
                    print(
                        f"A '{key}' kulcs alatt található adatok nem megfelelő formátumúak."
                    )


def delete_csv_with_keyword(keyword):
    """
    Törli a .csv fájlokat a 'data' mappából, amelyek nevében szerepel a megadott kulcsszó.

    :param keyword: Az a szövegrészlet, amely alapján a fájlokat törölni kell.
    """
    data_dir = "data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    deleted_files = []

    for csv_file in csv_files:
        if keyword in os.path.basename(csv_file):
            os.remove(csv_file)
            deleted_files.append(csv_file)

    if deleted_files:
        for file in deleted_files:
            print(f"- {file}")

        print(f"Törölt fájlok ({len(deleted_files)}):")
    else:
        print(
            f"Nem található olyan fájl a '{data_dir}' mappában, amely tartalmazza a(z) '{keyword}' kulcsszót."
        )


def collect_maoeuver_names():
    manoeuver_names = []

    v_values = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        105,
        110,
        115,
        120,
        125,
        130,
        135,
        140,
    ]
    ay_values = ["alacsony", "kozepes", "magas"]
    f_sin_values = [1, 3, 5, 7]
    a_sin_values = [2, 8]
    a_chirp_values = [1, 3, 5]
    pedal_values = [0.2, 0.5, 1.0]

    # Állandó sebesség

    # Kettős sávváltás
    for v_val in v_values:
        for ay_val in ay_values:
            manoeuver_names.append(f"allando_v_savvaltas_{ay_val}_v{v_val}")

    # Szinuszos kormányszög
    for v_val in v_values:
        for f_val in f_sin_values:
            for a_val in a_sin_values:
                manoeuver_names.append(f"allando_v_sin_a{a_val}_f{f_val}_v{v_val}")

    # Chirp signal
    for v_val in v_values:
        for a_val in a_chirp_values:
            manoeuver_names.append(f"allando_v_chirp_a{a_val}_v{v_val}")

    # Változó sebesség

    # Kettős sávváltás gyorsulással
    for pedal_val in pedal_values:
        for ay_val in ay_values:
            # String formázás, és "." helyett "_" használata
            pedal_val_str = str(pedal_val).replace(".", "_")
            manoeuver_names.append(
                f"valtozo_v_savvaltas_gas_{ay_val}_pedal{pedal_val_str}"
            )

    # Kettős sávváltás lassítással
    for pedal_val in pedal_values:
        for ay_val in ay_values:
            # String formázás, és "." helyett "_" használata
            pedal_val_str = str(pedal_val).replace(".", "_")
            manoeuver_names.append(
                f"valtozo_v_savvaltas_fek_{ay_val}_pedal{pedal_val_str}"
            )

    # Szinuszos kormányszög gyorsulással
    for a_val in a_sin_values:
        for f_val in f_sin_values:
            for pedal_val in pedal_values:
                # String formázás, és "." helyett "_" használata
                pedal_val_str = str(pedal_val).replace(".", "_")
                manoeuver_names.append(
                    f"valtozo_v_sin_gas_a{a_val}_f{f_val}_pedal{pedal_val_str}"
                )

    # Szinuszos kormányszög lassítással
    for a_val in a_sin_values:
        for f_val in f_sin_values:
            for pedal_val in pedal_values:
                # String formázás, és "." helyett "_" használata
                pedal_val_str = str(pedal_val).replace(".", "_")
                manoeuver_names.append(
                    f"valtozo_v_sin_fek_a{a_val}_f{f_val}_pedal{pedal_val_str}"
                )

    return manoeuver_names


def merge_csv_for_manoeuvres(
    manoeuvre_names, input_dir="data", output_dir="data2", save=False
):
    """
    Az egyes manőverekhez tartozó .csv fájlokat egy közös .csv fájlba egyesíti.

    :param manoeuvre_names: Lista az egyes manőverek neveiről.
    :param input_dir: Az input fájlok mappájának elérési útja.
    :param output_dir: A kimeneti fájlok mappájának elérési útja.
    """
    # Létrehozza az output mappát, ha nem létezik
    os.makedirs(output_dir, exist_ok=True)

    print(f"Manoeuver_names type: {type(manoeuvre_names)}")
    print(f"Manoeuver_names: {manoeuvre_names}")

    for manoeuvre_name in manoeuvre_names:
        # Az adott manőverhez tartozó fájlok keresése
        pattern = re.compile(rf"^{re.escape(manoeuvre_name)}(_.*|\.csv)$")
        matching_files = [
            f
            for f in glob.glob(os.path.join(input_dir, "*.csv"))
            if pattern.search(os.path.basename(f))
        ]

        if not matching_files:
            print(f"Nincs találat a {manoeuvre_name} manőverhez.")
            continue

        print(f"Feldolgozás: {manoeuvre_name}")
        list_of_vectors = []
        file_lengths = []
        column_names = []

        # Az összes fájl méretének meghatározása és változónevek kinyerése
        for file in matching_files:
            df = pd.read_csv(file)
            file_lengths.append(len(df))

            # A változó neve a fájlnév utolsó "_" utáni része ".csv" nélkül
            variable_name = re.search(
                r"_(.+)_([^_]+)\.csv$", os.path.basename(file)
            ).group(2)
            column_names.append(variable_name)

        # A legrövidebb fájl hossza
        # min_length = min(file_lengths)
        # print(f"Legrövidebb fájl hossza: {min_length}")

        # Azonos méretre vágás és összegyűjtés
        for file in matching_files:
            df = pd.read_csv(file)
            trimmed_vector = df.values[:10805]  # Azonos méretre vágás
            list_of_vectors.append(trimmed_vector)

        # A listát numpy array-é alakítjuk és egyesítjük oszlopokként
        combined_vectors = np.hstack(list_of_vectors)

        # Közös .csv fájl mentése
        if save:
            output_path = os.path.join(output_dir, f"{manoeuvre_name}_combined.csv")
            combined_df = pd.DataFrame(combined_vectors, columns=column_names)
            combined_df.to_csv(output_path, index=False)
            print(f"{manoeuvre_name} manőverhez tartozó fájl mentve: {output_path}")


def rename_files(variable):
    """
    Az adott változót tartalmazó fájlok nevét úgy módosítja, hogy az utolsó "_" karakter törlődik belőlük.

    :param variable: Az a string, amelyet a fájlok nevében keresünk.
    """
    # A 'data' mappában lévő fájlok keresése, amelyek tartalmazzák a 'variable'-t
    matching_files = [f for f in glob.glob("data/*") if variable in os.path.basename(f)]

    # Minden egyező fájl feldolgozása
    for file_path in matching_files:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        # Új név generálása az utolsó "_" törlésével
        if "_" in base_name:
            parts = base_name.rsplit("_", 1)  # Az utolsó "_" mentén bontás
            new_name = parts[0] + parts[1].replace(".csv", "") + ".csv"  # Az új fájlnév
            new_file_path = os.path.join(dir_name, new_name)

            # Fájl átnevezése az eredeti helyén
            os.rename(file_path, new_file_path)
            print(f"{file_path} átnevezve erre: {new_file_path}")
        else:
            print(f"A fájlnévben nincs '_', így nem módosítottam: {file_path}")


def aposztrof_nincs(manouver_names):
    modified_list = [item.replace("'", "") for item in manouver_names]
    with open("modified_list.txt", "w") as file:
        file.write("\n".join(modified_list))
    print(modified_list)

def sort_columns_in_files(folder_path, test_mode=True):
    # Ellenőrizzük, hogy a mappa létezik-e
    if not os.path.exists(folder_path):
        print(f"Hiba: A megadott mappa nem létezik: {folder_path}")
        return
    
    # Mappa összes CSV fájljának listázása
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not file_paths:
        print("Nincsenek CSV fájlok a megadott mappában.")
        return
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        
        # Oszlopok ABC sorrendbe rendezése
        df = df[sorted(df.columns)]
        
        # Ha tesztmódban fut, akkor új fájlba menti az eredményt
        if test_mode:
            new_file_path = file_path.replace(".csv", "_sorted_abc.csv")
            df.to_csv(new_file_path, index=False)
            print(f"Oszlopok ABC sorrendben rendezve és mentve: {new_file_path}")
        else:
            df.to_csv(file_path, index=False)
            print(f"Oszlopok ABC sorrendben rendezve: {file_path}")

def load_and_average_manoeuvres(directory, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    averaged_manoeuvres = {}
    
    for file in os.listdir(directory):
        if file.endswith(".npy"):  # Csak .npy fájlokat dolgozunk fel
            file_path = os.path.join(directory, file)
            data = np.load(file_path)  # Betöltés
            data = data[2500:]
            
            if data.shape[1] != 8:
                print(f"Figyelmeztetés: {file} nem megfelelő alakú ({data.shape}), kihagyva.")
                continue
            
            average_vector = np.mean(data, axis=0)  # Átlagolás soronként
            averaged_manoeuvres[file] = average_vector
            
            # Mentés az új mappába
            save_path = os.path.join(save_directory, file)
            np.save(save_path, average_vector)
    
    return averaged_manoeuvres

# Mappa elérési útvonala
directory = "Bottleneck_data/single_manoeuvres"
save_directory = "Bottleneck_data/averaged_manoeuvres"
averaged_results = load_and_average_manoeuvres(directory, save_directory)

# Példa egy fájl eredményének kiírására
for filename, avg_vector in averaged_results.items():
    print(f"{filename}: {avg_vector}")

print(f"Összesen {len(averaged_results)} fájl átlagolt eredménye lett kiszámolva és mentve.")