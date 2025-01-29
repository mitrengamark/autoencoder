import os
import numpy as np
import pandas as pd
import torch
import configparser


class DataProcess:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")

        self.coloring_method = config.get("Plot", "coloring_method")
        self.parameter = config.get("Plot", "parameter")
        self.num_workers = int(config["Data"]["num_workers"])
        self.batch_size = int(config["Hyperparameters"]["batch_size"])
        self.train_size = float(config["Data"]["train_size"])
        self.val_size = float(config["Data"]["val_size"])
        self.seed = int(config["Data"]["seed"])
        self.training_model = config.get("Model", "training_model")
        self.data_dir = config.get("Data", "data_dir")
        selected_manoeuvres = config.get(
            "Data", "selected_manoeuvres", fallback=""
        ).split(",")
        selected_manoeuvres = [
            m.strip() + "_combined.csv" for m in selected_manoeuvres if m.strip()
        ]

        all_files = [
            os.path.join(self.data_dir, file)
            for file in os.listdir(self.data_dir)
            if file.endswith(".csv")
        ]
        if selected_manoeuvres and selected_manoeuvres[0]:  # Ha nem üres a lista
            self.file_paths = [
                os.path.join(self.data_dir, file.strip())
                for file in selected_manoeuvres
                if file.strip() in os.listdir(self.data_dir)
            ]
        else:
            num_manoeuvres = int(config["Data"]["num_manoeuvres"])
            self.file_paths = all_files[:num_manoeuvres]

        self.labels = [
            os.path.splitext(os.path.basename(file))[0] for file in self.file_paths
        ]

        print("File paths:", self.file_paths)
        print("Labels:", self.labels)

    def z_score_normalize(self):
        """
        Z-score standardizálás az adatokhoz.
        """
        self.data_mean = self.data.mean(dim=0)
        self.data_std = self.data.std(dim=0)
        # self.data_mean = self.data.mean().mean()
        # self.data_std = self.data.std().std()
        data_standardized = (self.data - self.data_mean) / (self.data_std + 1e-8)
        return data_standardized

    def z_score_denormalize(self, data, data_mean, data_std):
        """
        Az adatok denormalizálása (visszatranszformálás az eredeti skálára).
        """
        print("Data Type:", type(data))
        print("Data Mean Type:", type(data_mean))
        print("Data Std Type:", type(data_std))
        data = torch.tensor(data) if isinstance(data, np.ndarray) else data
        print("Data Type 2:", type(data))
        data_denormalized = (data * data_std) + data_mean
        return data_denormalized

    def normalize(self):
        """
        Min-Max normalizálás az adatokhoz oszloponként.
        """
        self.data_min = self.data.min(dim=0).values  # Oszloponkénti minimum
        self.data_max = self.data.max(dim=0).values  # Oszloponkénti maximum

        # Min-Max normalizálás oszloponként
        data_normalized = (self.data - self.data_min) / (
            self.data_max - self.data_min + 1e-8
        )
        return data_normalized

    def denormalize(self, data, data_min, data_max):
        """
        Az adatok denormalizálása (visszatranszformálás az eredeti skálára).
        """
        data_denormalized = data * (data_max - data_min) + data_min
        return data_denormalized

    def load_and_label_data(self):
        combined_data = {label: [] for label in self.labels}
        self.sign_change_indices = {}

        for file_path, label in zip(self.file_paths, self.labels):
            df = pd.read_csv(file_path)

            # Ellenőrizzük, hogy a megadott paraméter létezik-e az adathalmazban
            if self.parameter and self.parameter not in df.columns:
                raise ValueError(
                    f"A megadott paraméter '{self.parameter}' nem található az adathalmazban!"
                )

            data_tensor = torch.tensor(df.values, dtype=torch.float32)
            self.data = data_tensor

            if self.training_model == "VAE":
                normalized_data = self.normalize()
            elif self.training_model == "MAE":
                normalized_data = self.z_score_normalize()
            else:
                raise ValueError("Unsupported model type. Expected 'VAE' or 'MAE'!")

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
        print(
            f"Data by manoeuvre: { {label: data.shape for label, data in self.data.items()} }"
        )

        # Debugging: Ellenőrzés
        if self.parameter:
            print(
                f"Sign change indices for parameter '{self.parameter}': {self.sign_change_indices}"
            )

        print(
            f"Data by manoeuvre: { {label: data.shape for label, data in self.data.items()} }"
        )

    def get_manoeuvre_specific_data(self):
        manoeuvre_data = {label: [] for label in torch.unique(self.data_labels)}
        for label in manoeuvre_data.keys():
            manoeuvre_data[label] = self.data[self.data_labels == label]
        return manoeuvre_data

    def train_test_split(self):
        self.load_and_label_data()

        torch.manual_seed(self.seed)

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

        print(f"Train data shape: {train_data.shape}, Labels: {train_labels.shape}")
        print(f"Validation data shape: {val_data.shape}, Labels: {val_labels.shape}")
        print(f"Test data shape: {test_data.shape}, Labels: {test_labels.shape}")

        rest_data = train_data.shape[0] % self.batch_size
        rest_data_procent = (rest_data / self.batch_size) * 100

        if rest_data_procent >= 70:
            drop_last = False
        else:
            drop_last = True

        trainloader = torch.utils.data.DataLoader(
            list(zip(train_data, train_labels)),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )
        valloader = torch.utils.data.DataLoader(
            list(zip(val_data, val_labels)),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )
        testloader = torch.utils.data.DataLoader(
            list(zip(test_data, test_labels)),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )

        if self.training_model == "VAE":
            return (
                trainloader,
                valloader,
                testloader,
                self.data_min,
                self.data_max,
                all_labels,
                label_mapping,
                self.sign_change_indices,
            )
        elif self.training_model == "MAE":
            return (
                trainloader,
                valloader,
                testloader,
                self.data_mean,
                self.data_std,
                all_labels,
                label_mapping,
                self.sign_change_indices,
            )
