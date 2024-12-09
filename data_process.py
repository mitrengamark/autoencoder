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
        ...

    def normalize(self):
        self.data_min = self.data.min().min()
        self.data_max = self.data.max().max()
        data_normalized = (self.data - self.data_min) / (self.data_max - self.data_min)
        return data_normalized
    
    def denormalize(self, data, data_min, data_max):
        data_denormalized = data * (data_max - data_min) + data_min
        return data_denormalized

    def group_manoeuvre_vectors(self, manoeuvre_name, save_csv=False):
        """
        Összegyűjti az azonos manőverhez tartozó paramétereket, és opcionálisan egy CSV fájlba menti.

        :param manoeuvre_name: A manőver neve (vagy egyedi azonosítója), amely alapján fájlokat keresünk.
        :param save_csv: Ha True, akkor az eredményt egy CSV fájlba menti. Alapértelmezés szerint False.
        :return: Egy numpy array, amely az azonos manőverhez tartozó paramétereket tartalmazza.
        """
        # Az azonos manőverhez tartozó fájlok keresése
        matching_files = [f for f in glob.glob("data/*") if manoeuvre_name in os.path.basename(f)]
        if not matching_files:
            print(f"Nincs találat a {manoeuvre_name} manőverhez.")
            return None

        list_of_vectors = []
        file_lengths = []

        # Az összes fájl méretének meghatározása
        for file in matching_files:
            df = pd.read_csv(file)
            file_lengths.append(len(df))

        # A legrövidebb fájl hossza
        min_length = min(file_lengths)
        shortest_file_index = file_lengths.index(min_length)
        shortest_file_name = matching_files[shortest_file_index]
        print(f"Min length: {min_length}")
        print(f"Shortest file: {shortest_file_name}")

        # Azonos méretre vágás és összegyűjtés
        for file in matching_files:
            df = pd.read_csv(file)
            trimmed_vector = df.values[:min_length]  # Azonos méretre vágás
            list_of_vectors.append(trimmed_vector)

        # A listát numpy array-é alakítjuk
        combined_vectors = np.array(list_of_vectors)

        # Opcionális mentés CSV fájlba
        if save_csv:
            combined_df = pd.DataFrame(np.hstack(combined_vectors))
            combined_df.to_csv(f"{manoeuvre_name}_combined.csv", index=False)
            print(f"Mentve a {manoeuvre_name}_combined.csv fájlba.")

        return combined_vectors
    
    def vector_collector(self, metric):
        matching_files = [f for f in glob.glob("data/*") if metric in os.path.basename(f)]
        list_of_vectors = []
        file_lengths = []
        self.file_names = matching_files

        for file in matching_files:
            df = pd.read_csv(file)
            file_lengths.append(len(df))

        min_length = min(file_lengths)
        min_length_index = file_lengths.index(min_length)
        shortest_file = matching_files[min_length_index]
        
        for file in matching_files:
            df = pd.read_csv(file)
            list_of_vectors.append(df.values)
            list_of_vectors[-1] = list_of_vectors[-1][:min_length]
        list_of_vectors = np.array(list_of_vectors)
        # shortest_vector = list_of_vectors[min_length_index]
        print(f"Shortest file: {shortest_file}")
        print(f"Shortest file length: {min_length}")
        return list_of_vectors
    
    def select_random_maneuver(self):        
        random_index = random.randint(0, len(self.data) - 1)
        selected_maneuver = self.data[random_index]
        selected_file = self.file_names[random_index]
        
        print(f"Selected maneuver: {selected_file}")

        if selected_maneuver.dim() == 1:
            selected_maneuver = selected_maneuver.unsqueeze(0)
        
        return selected_maneuver

    def train_test_split(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        metric = config.get('Data', 'metric')
        batch_size = int(config['Model']['batch_size'])
        test_size = float(config['Data']['test_size'])
        seed = int(config['Hyperparameters']['seed'])
        test_mode = int(config['Data']['test_mode'])
        batch_size = int(config['Model']['batch_size'])
        training_model = config.get('Agent', 'training_model')

        self.data = self.vector_collector(metric)
        # if training_model == "VAE":
        #     self.data = self.normalize()

        self.data = self.data.reshape(-1, self.data.shape[1])

        if training_model == "MAE":
            self.data = self.data[:, :self.data.shape[1] // 2]

        # print(f"Data type: {type(self.data)}.")
        # print(f"Data type: {type(self.data[0])}.")
        # print(f"Data: {self.data[0]}")
        # print(f"Data shape: {self.data[0].shape}")

        self.data = [torch.tensor(vector, dtype=torch.float32) for vector in self.data]

        if test_mode == 1:
            self.data = self.select_random_maneuver()
            if self.data.dim() == 1:
                self.data = self.data.unsqueeze(0)
                
            # print(f"Data shape: {self.data.shape}")

            n = self.data.size(1)  # Az egyetlen minta elemszáma
            train_size = int((1 - test_size) * n)
            torch.manual_seed(seed)
            
            indices = torch.randperm(n)  # Véletlenszerű sorrend
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            train_data = self.data[:, train_indices].squeeze(0)
            test_data = self.data[:, test_indices].squeeze(0)

            train_input_dim = train_data.shape[-1]
            test_input_dim = test_data.shape[-1]

            if train_input_dim > test_input_dim:
                padding = train_input_dim - test_input_dim
                test_data = torch.nn.functional.pad(test_data, (0, padding))  # Jobb oldali padding
            elif train_input_dim < test_input_dim:
                test_data = test_data[:, :train_input_dim]

            # DataLoader-ek létrehozása
            trainloader = torch.utils.data.DataLoader([train_data], batch_size=batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader([test_data], batch_size=batch_size, shuffle=False)

            # print(f"Train data shape: {train_data.shape}")
            # print(f"Test data shape: {test_data.shape}")
            if training_model == "VAE":
                return trainloader, testloader, train_data, test_data#, self.data_min, self.data_max
            elif training_model == "MAE":
                return trainloader, testloader, train_data, test_data

        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=seed)

        # print(f"Train data shape: {len(train_data)}")
        # print(f"Test data shape: {len(test_data)}")
            
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # print((f"trainloader type: {type(trainloader)}"))
        # print((f"testloader type: {type(testloader)}"))
        # print((f"train_data type: {type(train_data)}"))
        # print((f"train_data shape: {train_data[0].shape}"))
        # print(f"train_data: {train_data}")

        if training_model == "VAE":
            return trainloader, testloader, train_data, test_data# , self.data_min, self.data_max
        elif training_model == "MAE":
            return trainloader, testloader, train_data, test_data
    
dp = DataProcess()
# # list_of_vectors = dp.vector_collector('')
# combined_vectors = dp.group_manoeuvre_vectors("allando_v_chirp_a1_v5")
# # tranindata, testdata, traindata, _, _, _ = dp.train_test_split()
combined_vectors = dp.group_manoeuvre_vectors("allando_v_chirp_a1_v5")
print(f"Combined vectors shape: {combined_vectors.shape}")