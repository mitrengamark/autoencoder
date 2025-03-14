import os
import shutil
import json

def copy_filtered_manoeuvres(source_dir, destination_dir, removed_manoeuvres_file="Redundant_manoeuvres/manoeuvres_for_removing_velocity_95.json"):
    """
    Átmásolja a manővereket a source_dir-ből a destination_dir-be, 
    kivéve azokat, amelyek szerepelnek a removed_manoeuvres.json fájlban.

    :param source_dir: Az eredeti mappa, ahonnan a fájlokat másoljuk.
    :param destination_dir: A célmappa, ahová a szűrt fájlokat másoljuk.
    :param removed_manoeuvres_file: A JSON fájl neve, amely az eltávolítandó manővereket tartalmazza.
    """

    # Ellenőrizzük, hogy létezik-e a removed_manoeuvres.json fájl
    if not os.path.exists(removed_manoeuvres_file):
        print(f"Hiba: Nem található a {removed_manoeuvres_file} fájl.")
        return
    
    # Beolvassuk a JSON fájl tartalmát
    with open(removed_manoeuvres_file, "r") as file:
        removed_manoeuvres_by_group = json.load(file)

    # Kinyerjük az összes eltávolított manőver nevét (csoportoktól függetlenül)
    removed_manoeuvres = set()
    for group in removed_manoeuvres_by_group.values():
        removed_manoeuvres.update(group)

    # Ha nem létezik a célmappa, létrehozzuk
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Fájlok másolása a célmappába, kihagyva a törölt manővereket
    for filename in os.listdir(source_dir):
        manoeuvre_name, ext = os.path.splitext(filename)  # Fájlnév és kiterjesztés szétválasztása

        if manoeuvre_name in removed_manoeuvres:
            print(f"Kihagyva: {filename}")
            continue  # Ne másoljuk ezt a fájlt

        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(destination_dir, filename)

        shutil.copy2(src_path, dest_path)
        print(f"Átmásolva: {filename}")

    print("\nManőverek sikeresen átmásolva a szűrés után!")

source_directory = "data/mat_files"
destination_directory = "filtered_data/velocity95/whole_mat_files"

copy_filtered_manoeuvres(source_directory, destination_directory)
