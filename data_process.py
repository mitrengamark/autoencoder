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

        self.batch_size = int(config['Hyperparameters']['batch_size'])
        self.train_size = float(config['Data']['train_size'])
        self.val_size = float(config['Data']['val_size'])
        self.seed = int(config['Data']['seed'])
        self.training_model = config.get('Model', 'training_model')

    def z_score_normalize(self):
        """
        Z-score standardizálás az adatokhoz.
        """
        self.data_mean = self.data.mean().mean()
        self.data_std = self.data.std().std()
        data_standardized = (self.data - self.data_mean) / self.data_std
        return data_standardized
    
    def z_score_denormalize(self, data, data_mean, data_std):
        """
        Az adatok denormalizálása (visszatranszformálás az eredeti skálára).
        """
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

    def load_single_manoeuvre(self, file_path):
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
    
    def train_test_split(self, file_path=None):
        if file_path:
            self.data = self.load_single_manoeuvre(file_path)
        else:
            raise ValueError("Egy fájlt kell megadni a `file_path` paraméterben!")

        self.data = torch.tensor(self.data, dtype=torch.float32)

        n = len(self.data)
        torch.manual_seed(self.seed)

        train_size = int(self.train_size * n)
        val_size = int(self.val_size * n)

        indices = torch.randperm(n)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_data = self.data[train_indices]
        val_data = self.data[val_indices]
        test_data = self.data[test_indices]

        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Test data shape: {test_data.shape}")
            
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        if self.training_model == "VAE":
            return trainloader, valloader, testloader, self.data_min, self.data_max
        elif self.training_model == "MAE":
            return trainloader, valloader, testloader, self.data_mean, self.data_std