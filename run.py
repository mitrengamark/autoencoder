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
from Config.load_config import seed, training_model, save_model, test_mode, plot

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Több GPU esetén
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Neptune inicializáció
if plot == 1 and test_mode == 0:
    run = init_neptune()
else:
    run = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

dp = DataProcess()

(
    trainloader,
    valloader,
    testloader,
    data_min,
    data_max,
    data_mean,
    data_std,
    labels,
    label_mapping,
    sign_change_indices,
    selected_columns,
) = dp.train_test_split()

train_input_dim = trainloader.dataset[0][0].shape[0]

if training_model == "VAE":
    model = VariationalAutoencoder(train_input_dim).to(device)
elif training_model == "MAE":
    model = MaskedAutoencoder(train_input_dim).to(device)
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

model_params = model.parameters()
optimizer = optimizer_maker(model_params)
training = Training(
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    optimizer=optimizer,
    model=model,
    labels=labels,
    device=device,
    run=run,
    data_min=data_min,
    data_max=data_max,
    data_mean=data_mean,
    data_std=data_std,
    sign_change_indices=sign_change_indices,
    label_mapping=label_mapping,
    selected_columns=selected_columns,
)

if test_mode == 0:
    training.train()
    if save_model == 1:
        training.save_model()
elif test_mode == 1:
    training.test()
