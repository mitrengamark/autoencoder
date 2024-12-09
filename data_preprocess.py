import scipy.io
import pandas as pd
import os


class DataPreProcess:
    def __init__(self):
        pass

    def print_keys(self, a, v):
        mat_file_path = os.path.join("data", f"allando_v_chirp_a{a}_v{v}.mat")  # A fájl elérési útja
        mat_content = scipy.io.loadmat(mat_file_path)

        # Kulcsok kilistázása
        print("A fájlban található kulcsok:")
        for key in mat_content.keys():
            print(key)

    def mat_to_csv(self):
        data_dir = "data"
        mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

        for mat_file in mat_files:
            mat_file_path = os.path.join(data_dir, mat_file)
            mat_content = scipy.io.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)

            # Ellenőrizzük a betöltött adatok szerkezetét
            for key, value in mat_content.items():
                if not key.startswith("__"):  # A metaadatok kihagyása
                    print(f"Key: {key}, Value: {value}")

                    # Ha az adat egy struktúra, hagyjuk figyelmen kívül
                    if hasattr(value, '_fieldnames'):
                        print(f"A '{key}' kulcs alatt található adat egy struktúra, kihagyva.")
                        continue

                    # Ha az adat egy oszlopvektor vagy 2D tömb, alakítsuk át DataFrame-é
                    if isinstance(value, (list, tuple)) or hasattr(value, "shape"):
                        df = pd.DataFrame(value)
                        csv_file_name = os.path.splitext(mat_file)[0] + f"_{key}.csv"
                        csv_file_path = os.path.join(data_dir, csv_file_name)
                        df.to_csv(csv_file_path, index=False)
                        print(f"Az adatok sikeresen mentve lettek: {csv_file_path}")
                    else:
                        print(f"A '{key}' kulcs alatt található adatok nem megfelelő formátumúak.")

df = DataPreProcess()
df.mat_to_csv()
