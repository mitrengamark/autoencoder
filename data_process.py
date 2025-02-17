import os
import numpy as np
import pandas as pd
import torch
import random
from Config.load_config import (
    selected_manoeuvres,
    num_manoeuvres,
    data_dir,
    parameter,
    training_model,
    coloring_method,
    train_size,
    val_size,
    batch_size,
    num_workers,
    basic_method,
    parameters,
    normalization,
)


class DataProcess:
    def __init__(self, single_file=None):
        self.data_dir = data_dir
        self.parameter = parameter
        self.training_model = training_model
        self.coloring_method = coloring_method
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        selected_manoeuvres_list = [
            m.strip() + "_combined.csv" for m in selected_manoeuvres if m.strip()
        ]

        all_files = [
            file for file in os.listdir(self.data_dir) if file.endswith(".csv")
        ]

        if single_file:
            # **Csak egy fájl feldolgozása**
            self.file_paths = [os.path.join(self.data_dir, single_file)]
            self.labels = [os.path.splitext(os.path.basename(single_file))[0]]
        else:
            if (
                selected_manoeuvres_list and selected_manoeuvres_list[0]
            ):  # Ha a felhasználó explicit megadott manővereket
                self.file_paths = [
                    os.path.join(self.data_dir, file.strip())
                    for file in selected_manoeuvres_list
                    if file.strip() in all_files
                ]
            else:
                # Véletlenszerű fájlválasztás
                selected_files = random.sample(
                    all_files, min(num_manoeuvres, len(all_files))
                )
                self.file_paths = [
                    os.path.join(self.data_dir, file) for file in selected_files
                ]

            self.labels = [
                os.path.splitext(os.path.basename(file))[0] for file in self.file_paths
            ]

        print("File paths:", self.file_paths)
        print("Labels:", self.labels)

    def compute_global_stats(self):
        """
        Az összes manőver együttes min-max és mean-std értékeinek kiszámítása.
        """
        all_data = []
        for file_path in self.file_paths:
            df = pd.read_csv(file_path)
            data_tensor = torch.tensor(df.values, dtype=torch.float32)
            all_data.append(data_tensor)

        all_data = torch.cat(all_data, dim=0)  # Összes manőver összevonása

        # Globális min-max és z-score értékek
        self.min = all_data.min(dim=0).values
        self.max = all_data.max(dim=0).values
        self.mean = all_data.mean(dim=0)
        self.std = all_data.std(dim=0) + 1e-8  # Stabilitás

    def z_score_normalize(self, data):
        """
        Globális Z-score standardizálás a megadott adatokra.
        """
        data_standardized = (data - self.mean) / self.std
        return data_standardized

    def z_score_denormalize(self, data, data_mean, data_std):
        """
        Az adatok denormalizálása (Z-score alapján visszatranszformálás az eredeti skálára).

        data: torch.Tensor - A standardizált adatok
        mean: torch.Tensor - Az eredeti adatok oszloponkénti átlaga
        std: torch.Tensor - Az eredeti adatok oszloponkénti szórása

        Visszaadja: torch.Tensor - Az eredeti skálára visszaállított adatok
        """
        data_denormalized = data * data_std + data_mean
        return data_denormalized

    def normalize(self, data):
        """
        Globális min-max normalizálás a megadott adatokra.
        """
        data_normalized = (data - self.min) / (self.max - self.min + 1e-8)
        return data_normalized

    def denormalize(self, data, data_min, data_max):
        """
        Globális min-max alapján történő visszaskálázás.
        """
        data_denormalized = data * (data_max - data_min) + data_min
        return data_denormalized

    def load_and_label_data(self):
        # Először számítsuk ki a globális statisztikákat
        self.compute_global_stats()

        combined_data = {label: [] for label in self.labels}
        self.sign_change_indices = {}

        for file_path, label in zip(self.file_paths, self.labels):
            df = pd.read_csv(file_path)

            self.selected_columns = [
                df.columns.get_loc(param) for param in parameters if param in df.columns
            ]

            if not self.selected_columns:
                raise ValueError(
                    "A megadott paraméterek egyike sem található a fájlban!"
                )

            print(f"Kiválasztott oszlopindexek {label}-hez: {self.selected_columns}")

            # Ellenőrizzük, hogy a megadott paraméter létezik-e az adathalmazban
            if self.parameter and self.parameter not in df.columns:
                raise ValueError(
                    f"A megadott paraméter '{self.parameter}' nem található az adathalmazban!"
                )

            data_tensor = torch.tensor(df.values, dtype=torch.float32)
            self.data = data_tensor  # [2500:]

            if normalization == "min_max":
                normalized_data = self.normalize(data_tensor)
            elif normalization == "z_score":
                normalized_data = self.z_score_normalize(data_tensor)
            else:
                raise ValueError(
                    "Unsupported normalization method. Expected 'min_max' or 'z_score'!"
                )

            combined_data[label].append(normalized_data)

            if self.parameter:
                param_values = df[self.parameter].values
                if self.coloring_method == "sign_change":
                    # Előjelváltások detektálása a kiválasztott paraméteren
                    sign_changes = np.diff(np.sign(param_values)) != 0
                    indices = np.where(sign_changes)[0] + 1  # Az előjelváltás indexei
                    self.sign_change_indices[label] = indices
                elif self.coloring_method == "local_extrema":
                    # Helyi szélsőértékek keresése
                    first_derivative = np.diff(param_values)
                    second_derivative = np.diff(np.sign(first_derivative))
                    local_extrema_indices = (
                        np.where(second_derivative != 0)[0] + 1
                    )  # Helyi min./max.
                    self.sign_change_indices[label] = local_extrema_indices
                elif self.coloring_method == "inflection_points":
                    # Első és második derivált kiszámítása
                    first_derivative = np.diff(param_values)
                    second_derivative = np.diff(first_derivative)

                    # Második derivált előjelváltásainak detektálása
                    inflexion_points = (
                        np.where(np.diff(np.sign(second_derivative)) != 0)[0] + 1
                    )  # Az inflexiós pontok indexei

                    self.sign_change_indices[label] = inflexion_points
                else:
                    raise ValueError(
                        f"Unsupported coloring method. Expected 'sign_change' or 'local_extrema'!"
                    )

        self.data = {
            label: torch.cat(data_list, dim=0)
            for label, data_list in combined_data.items()
        }

    def get_manoeuvre_specific_data(self):
        manoeuvre_data = {label: [] for label in torch.unique(self.data_labels)}
        for label in manoeuvre_data.keys():
            manoeuvre_data[label] = self.data[self.data_labels == label]
        return manoeuvre_data

    def train_test_split(self):
        self.load_and_label_data()

        all_data = []
        all_labels = []
        label_mapping = {
            label: idx for idx, label in enumerate(self.labels)
        }  # Címkék indexelése

        for label, data in self.data.items():
            all_data.append(data)
            all_labels.append(
                torch.full((data.shape[0],), label_mapping[label], dtype=torch.int64)
            )

        # Összesített tensor
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Train-val-test split
        n = all_data.size(0)
        train_size = int(self.train_size * n)
        if basic_method == 1:
            val_size = int(self.val_size * n)

            indices = torch.randperm(n)
            train_indices = indices[:train_size]
            val_indices = indices[train_size : train_size + val_size]
            test_indices = indices[train_size + val_size :]

            train_data = all_data[train_indices]
            train_labels = all_labels[train_indices]
            val_data = all_data[val_indices]
            val_labels = all_labels[val_indices]
            test_data = all_data[test_indices]
            test_labels = all_labels[test_indices]
        else:
            train_indices = torch.arange(train_size)
            val_indices = torch.arange(train_size, n)

            train_data = all_data[train_indices]
            train_labels = all_labels[train_indices]
            val_data = all_data[val_indices]
            val_labels = all_labels[val_indices]
            test_data = all_data
            test_labels = all_labels

        print(f"Train data shape: {train_data.shape}, Labels: {train_labels.shape}")
        print(f"Validation data shape: {val_data.shape}, Labels: {val_labels.shape}")
        print(f"Test data shape: {test_data.shape}, Labels: {test_labels.shape}")

        trainloader = torch.utils.data.DataLoader(
            list(zip(train_data, train_labels)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        valloader = torch.utils.data.DataLoader(
            list(zip(val_data, val_labels)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        testloader = torch.utils.data.DataLoader(
            list(zip(test_data, test_labels)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return (
            trainloader,
            valloader,
            testloader,
            self.min,
            self.max,
            self.mean,
            self.std,
            all_labels,
            label_mapping,
            self.sign_change_indices,
            self.selected_columns,
        )
