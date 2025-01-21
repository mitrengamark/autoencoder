import glob
import os
import numpy as np
import pandas as pd
import torch
import configparser
from sklearn.model_selection import train_test_split
import random


class DataProcess:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.num_workers = int(config['Data']['num_workers'])
        self.batch_size = int(config['Hyperparameters']['batch_size'])
        self.train_size = float(config['Data']['train_size'])
        self.val_size = float(config['Data']['val_size'])
        self.seed = int(config['Data']['seed'])
        self.training_model = config.get('Model', 'training_model')
        self.data_dir = config.get('Data', 'data_dir')
        self.file_paths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.csv')]
        num_manoeuvres = int(config['Data']['num_manoeuvres'])

        # Csak az első 2 fájlt tartjuk meg:
        self.file_paths = self.file_paths[:num_manoeuvres]

        self.labels = [os.path.splitext(os.path.basename(file))[0] for file in self.file_paths]

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
        Min-Max normalizálás az adatokhoz.
        """
        self.data_min = self.data.min().min()
        self.data_max = self.data.max().max()
        data_normalized = (self.data - self.data_min) / (self.data_max - self.data_min)
        return data_normalized
    
    def denormalize(self, data, data_min, data_max):
        """
        Az adatok denormalizálása (visszatranszformálás az eredeti skálára).
        """
        data_denormalized = data * (data_max - data_min) + data_min
        return data_denormalized

    def load_single_manoeuvre(self, file_path, batch_size=None):
        """
        Egyetlen .csv fájl betöltése, standardizálása és numpy array-é alakítása.
        """
        # Adatok betöltése
        df = pd.read_csv(file_path)
        self.data = df
        if self.training_model == "VAE":
            normalized_data = self.normalize()
        elif self.training_model == "MAE":
            normalized_data = self.z_score_normalize()
        return normalized_data.values
    
    def load_manoeuvres(self, file_paths, batch_size=None):
        """
        Több .csv fájl betöltése, standardizálása és numpy array-é alakítása.

        :param file_paths: Lista az egyes .csv fájlok elérési útvonalairól.
        :param batch_size: Nem kötelező, a betöltési batch méret (ha releváns).
        """
        all_data = []
        for file_path in file_paths:
            # Adatok betöltése
            df = pd.read_csv(file_path)
            self.data = df
            if self.training_model == "VAE":
                normalized_data = self.normalize()
            elif self.training_model == "MAE":
                normalized_data = self.z_score_normalize()
            all_data.append(normalized_data.values)

        return np.concatenate(all_data, axis=0)
    
    def load_and_label_data(self):
        combined_data = {label: [] for label in self.labels}

        for file_path, label in zip(self.file_paths, self.labels):
            df = pd.read_csv(file_path)
            data_tensor = torch.tensor(df.values, dtype=torch.float32)
            self.data = data_tensor


            if self.training_model == "VAE":
                normalized_data = self.normalize()
            elif self.training_model == "MAE":
                normalized_data = self.z_score_normalize()
            else:
                raise ValueError("Unsupported model type. Expected 'VAE' or 'MAE'!")

            combined_data[label].append(normalized_data)

        self.data = {label: torch.cat(data_list, dim=0) for label, data_list in combined_data.items()}
        print(f"Data by manoeuvre: { {label: data.shape for label, data in self.data.items()} }")
        
    def get_manoeuvre_specific_data(self):
        manoeuvre_data = {label: [] for label in torch.unique(self.data_labels)}
        for label in manoeuvre_data.keys():
            manoeuvre_data[label] = self.data[self.data_labels == label]
        return manoeuvre_data

    
    def train_test_split(self, file_path=None):

        # if file_path:
        #     self.data = self.load_single_manoeuvre(file_path)
        # else:
        #     raise ValueError("Egy fájlt kell megadni a `file_path` paraméterben!")

        # self.data = torch.tensor(self.data, dtype=torch.float32)

        self.load_and_label_data()

        torch.manual_seed(self.seed)

        all_data = []
        all_labels = []
        label_mapping = {label: idx for idx, label in enumerate(self.labels)}  # Címkék indexelése

        for label, data in self.data.items():
            all_data.append(data)
            all_labels.append(torch.full((data.shape[0],), label_mapping[label], dtype=torch.int64))

        # Összesített tensor
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Train-val-test split
        n = all_data.size(0)
        train_size = int(self.train_size * n)
        val_size = int(self.val_size * n)

        indices = torch.randperm(n)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

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
            list(zip(train_data, train_labels)), batch_size=self.batch_size, shuffle=False, drop_last=drop_last, num_workers=self.num_workers
        )
        valloader = torch.utils.data.DataLoader(
            list(zip(val_data, val_labels)), batch_size=self.batch_size, shuffle=False, drop_last=drop_last, num_workers=self.num_workers
        )
        testloader = torch.utils.data.DataLoader(
            list(zip(test_data, test_labels)), batch_size=self.batch_size, shuffle=False, drop_last=drop_last, num_workers=self.num_workers
        )

        if self.training_model == "VAE":
            return trainloader, valloader, testloader, self.data_min, self.data_max, all_labels
        elif self.training_model == "MAE":
            return trainloader, valloader, testloader, self.data_mean, self.data_std, all_labels # self.data_mean, self.data_std,