import sys
import os

# Adja hozzá a projekt gyökérkönyvtárát a Python keresési útvonalához
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import torch
from itertools import product
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.optimizer import optimizer_maker
from Factory.scheduler import scheduler_maker
from train_test import Training
from data_process import DataProcess
from load_config import (
    seed,
    training_model,
    initial_lr,
    max_lr,
    final_lr,
    tolerance,
    hyperopt,
    n_trials,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

dp = DataProcess()

if training_model == "VAE":
    trainloader, valloader, testloader, data_min, data_max = dp.train_test_split()
    data_mean = None
    data_std = None
elif training_model == "MAE":
    trainloader, valloader, testloader, data_mean, data_std = dp.train_test_split()
    data_min = None
    data_max = None
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")


def objective(trial):
    # Hiperparaméterek kiválasztása Optuna segítségével
    model_type = trial.suggest_categorical("model_type", ["VAE"])
    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16])
    hidden_dim_0 = trial.suggest_categorical("hidden_dim_0", [16, 32, 64])
    hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [8, 16, 32])
    beta = trial.suggest_categorical("beta", [0.1, 0.5, 1])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    scheduler_type = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau"])
    # initial_lr = trial.suggest_categorical("initial_lr", [1e-3, 1e-4, 1e-5])
    # max_lr = trial.suggest_categorical("max_lr", [1e-2, 1e-3, 1e-4])
    # final_lr = trial.suggest_categorical("final_lr", [1e-4, 1e-5, 1e-6])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    optimizer_type = trial.suggest_categorical(
        "optimizer", ["SGD", "Adam", "AdamW", "Adagrad", "RMSprop"]
    )
    # gamma = trial.suggest_categorical("gamma", [0.5, 0.75, 0.9])
    num_epochs = trial.suggest_categorical("num_epochs", [100, 1000, 10000])
    # patience = trial.suggest_categorical("patience", [5, 20, 50, 100])

    if model_type == "VAE":
        model = VariationalAutoencoder(
            input_dim=trainloader.dataset[0].shape[0],
            latent_dim=latent_dim,
            hidden_dim_0=hidden_dim_0,
            hidden_dim_1=hidden_dim_1,
            beta=beta,
            dropout=dropout,
        ).to(device)
    elif model_type == "MAE":
        model = MaskedAutoencoder(
            input_dim=trainloader.dataset[0].shape[0],
            mask_ratio=trial.suggest_categorical("mask_ratio", [0.75, 0.8]),
        ).to(device)
    else:
        raise ValueError("Unsupported model type!")

    optimizer = optimizer_maker(
        optimizer_type=optimizer_type, model_params=model.parameters()
    )

    warmup_epochs = num_epochs * 0.1

    scheduler = scheduler_maker(
        scheduler=scheduler_type,
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        initial_lr=initial_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    training = Training(
        trainloader,
        valloader,
        testloader,
        optimizer,
        model,
        num_epochs,
        device,
        scheduler,
        warmup_epochs=warmup_epochs,
        initial_lr=initial_lr,
        max_lr=max_lr,
        final_lr=final_lr,
        hyperopt=hyperopt,
        tolerance=tolerance,  # gamma=gamma, patience=patience
    )

    training.train()
    _, val_accuracy = training.validate()

    return val_accuracy


# Optuna tanulmány létrehozása
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials)

# Legjobb paraméterek kiíratása
print("Best trial:")
print(study.best_trial)
print("Best parameters:")
print(study.best_params)
