import torch
import torch.utils.data
import numpy as np
import random
from data_process import DataProcess
from train_test import Training
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.optimizer import optimizer_maker
from Analyse.neptune_utils import init_neptune
from load_config import (
    seed,
    latent_dim,
    hidden_dims,
    num_epochs,
    dropout,
    mask_ratio,
    num_heads,
    beta_min,
    initial_lr,
    max_lr,
    final_lr,
    scheduler,
    step_size,
    gamma,
    patience,
    opt_name,
    training_model,
    save_model,
    test_mode,
    model_path,
    num_manoeuvres,
    n_clusters,
    use_cosine_similarity,
    plot,
    project_name,
    api_token,
    tolerance,
    hyperopt,
    bottleneck_dim,
    model_path,
    warmup_epochs,
    saved_model,
)

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Több GPU esetén
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parameters = {
    "latent_dim": latent_dim,
    "hidden_dims": hidden_dims,
    "num_epochs": num_epochs,
    "training_model": training_model,
    "mask_ratio": mask_ratio,
    "scheduler": scheduler,
    "step_size": step_size,
    "gamma": gamma,
    "patience": patience,
    "initial_lr": initial_lr,
    "max_lr": max_lr,
    "final_lr": final_lr,
    "model": model_path,
    "tolerance": tolerance,
}

# Neptune inicializáció
if plot == 1 and test_mode == 0:
    run = init_neptune(project_name, api_token, parameters)
else:
    run = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

dp = DataProcess()

if training_model == "VAE":
    (
        trainloader,
        valloader,
        testloader,
        data_min,
        data_max,
        labels,
        label_mapping,
        sign_change_indices,
    ) = dp.train_test_split()
    data_mean = None
    data_std = None
elif training_model == "MAE":
    (
        trainloader,
        valloader,
        testloader,
        data_mean,
        data_std,
        labels,
        label_mapping,
        sign_change_indices,
    ) = dp.train_test_split()
    data_min = None
    data_max = None
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

train_input_dim = trainloader.dataset[0][0].shape[0]
print(f"Train input dim: {train_input_dim}")

if training_model == "VAE":
    model = VariationalAutoencoder(
        train_input_dim, latent_dim, hidden_dims, dropout
    ).to(device)
elif training_model == "MAE":
    model = MaskedAutoencoder(
        train_input_dim, bottleneck_dim, mask_ratio, num_heads, dropout
    ).to(device)
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

model_params = model.parameters()
optimizer = optimizer_maker(opt_name, model_params, initial_lr, scheduler)
training = Training(
    trainloader,
    valloader,
    testloader,
    test_mode,
    optimizer,
    model,
    labels,
    num_epochs,
    device,
    scheduler,
    beta_min,
    step_size,
    gamma,
    patience,
    warmup_epochs,
    initial_lr,
    max_lr,
    final_lr,
    saved_model,
    run=run,
    data_min=data_min,
    data_max=data_max,
    data_mean=data_mean,
    data_std=data_std,
    hyperopt=hyperopt,
    tolerance=tolerance,
    label_mapping=label_mapping,
    sign_change_indices=sign_change_indices,
    num_manoeuvres=num_manoeuvres,
    n_clusters=n_clusters,
    use_cosine_similarity=use_cosine_similarity,
    model_name=training_model,
)

if test_mode == 0:
    training.train()
    if save_model == 1:
        training.save_model(model_path)
elif test_mode == 1:
    training.test()
