import sys
import os

# Adja hozzá a projekt gyökérkönyvtárát a Python keresési útvonalához
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import configparser
from itertools import product
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.optimizer import optimizer_maker
from Factory.scheduler import scheduler_maker
from train_test import Training
from data_process import DataProcess

# Paramétertér definiálása
param_space = {
    "model_type": ["VAE"],
    "latent_dim": [4, 8, 16], # 3
    "hidden_dim_0": [16, 32, 64], # 3
    "hidden_dim_1": [8, 16, 32], # 3
    "beta": [0.1, 0.5, 1], # 3
    "dropout": [0.1, 0.2, 0.3], # 3
    # "mask_ratio": [0.75, 0.8], # 2
    "scheduler": ["WarmupCosine"],
    # "gamma": [0.5, 0.75, 0.9], # 3
    "num_epochs": [100, 1000, 10000], # 3
    # "patience": [5, 20, 50, 100], # 4
    "initial_lr": [1e-3, 1e-4, 1e-5], # 3
    "max_lr": [1e-2, 1e-3, 1e-4], # 3
    "final_lr": [1e-4, 1e-5, 1e-6], # 3
    "batch_size": [16, 32, 64, 128, 256, 512], # 6
    "optimizer": ['SGD', 'Adam', 'AdamW', 'Adagrad', 'RMSprop'], # 5
} # 590 490

# Grid készítése a paraméterek kombinációjához
grid = list(product(*param_space.values()))
param_keys = list(param_space.keys())

config = configparser.ConfigParser()
config.read('config.ini')

seed = int(config['Data']['seed'])
training_model = config.get('Model', 'training_model')
file_path = config.get('Data', 'file_path')
tolerance = float(config['Callbacks']['tolerance'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

dp = DataProcess()

if training_model == "VAE":
    trainloader, valloader, testloader, data_min, data_max = dp.train_test_split(file_path=file_path)
    data_mean = None
    data_std = None
elif training_model == "MAE":
    trainloader, valloader, testloader, data_mean, data_std = dp.train_test_split(file_path=file_path)
    data_min = None
    data_max = None
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

def grid_search(grid, param_keys, trainloader, valloader, testloader, device, tolerance):
    """
    Grid search implementáció hyperparaméter optimalizációhoz.
    
    :param grid: Az összes paraméter kombináció listája.
    :param param_keys: A paraméterek neveinek listája.
    :param trainloader: Az edzési adatok betöltője.
    :param valloader: A validációs adatok betöltője.
    :param testloader: A teszt adatok betöltője.
    :param device: A használt eszköz (CPU vagy GPU).
    """
    best_params = None
    best_val_accuracy = 0.0

    for trial, param_values in enumerate(grid):
        params = dict(zip(param_keys, param_values))
        print(f"Trial {trial + 1}/{len(grid)} - Current Parameters: {params}")

        # Modell inicializálása
        if params["model_type"] == "VAE":
            model = VariationalAutoencoder(
                input_dim=trainloader.dataset[0].shape[0],
                latent_dim=params["latent_dim"],
                hidden_dim_0=params["hidden_dim_0"],
                hidden_dim_1=params["hidden_dim_1"],
                beta=params["beta"],
                dropout=params["dropout"]
            ).to(device)
        elif params["model_type"] == "MAE":
            model = MaskedAutoencoder(
                input_dim=trainloader.dataset[0].shape[0],
                mask_ratio=params["mask_ratio"]
            ).to(device)
        else:
            raise ValueError("Unsupported model type!")
        
        # Optimizer és Scheduler inicializálása
        optimizer = optimizer_maker(
            optimizer_type=params["optimizer"],
            model_params=model.parameters()
        )

        warmup_epochs = params["num_epochs"] / 10

        scheduler = scheduler_maker(
            scheduler=params["scheduler"],
            optimizer=optimizer,
            # gamma=params["gamma"],
            num_epochs=params["num_epochs"],
            # patience=params["patience"],
            warmup_epochs=warmup_epochs,
            initial_lr=params["initial_lr"],
            max_lr=params["max_lr"],
            final_lr=params["final_lr"]
        )

        # Tréning
        training = Training(
            trainloader, valloader, testloader, optimizer, model, params["num_epochs"],
            device, scheduler, warmup_epochs=warmup_epochs, initial_lr=params["initial_lr"], max_lr=params["max_lr"],
            final_lr=params["final_lr"], hyperopt=1, tolerance=tolerance #, gamma=params["gamma"], patience=params["patience"]
        )
        training.train()

        # Validációs pontosság kiértékelése
        _, val_accuracy = training.validate()
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Legjobb paraméterek mentése
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = params
    
        print(f"Best Parameters: {best_params}")
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    
    return best_params

best_params = grid_search(
    grid=grid,
    param_keys=param_keys,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    device=device,
    tolerance=tolerance
)
