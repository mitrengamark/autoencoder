import torch
import torch.optim as optim
import configparser
import torch.utils.data
from data_process import DataProcess
from train_test import Training
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Analyse.neptune_utils import init_neptune

torch.cuda.empty_cache()

config = configparser.ConfigParser()
config.read('config.ini')

latent_dim = int(config['Dims']['latent_dim'])
hidden_dim_0 = int(config['Dims']['hidden_dim_0'])
hidden_dim_1 = int(config['Dims']['hidden_dim_1'])
lr = float(config['Hyperparameters']['lr'])
max_lr = float(config['Hyperparameters']['max_lr'])
num_epochs = int(config['Hyperparameters']['num_epochs'])
beta = float(config['Hyperparameters']['beta'])
seed = int(config['Data']['seed'])
training_model = config.get('Model', 'training_model')
mask_ratio = float(config['Hyperparameters']['mask_ratio'])
save_model = int(config['Model']['save_model'])
project_name = config.get('Callbacks', 'neptune_project')
api_token = config.get('Callbacks', 'neptune_token')
dropout = float(config['Hyperparameters']['dropout'])
scheduler = config.get('Hyperparameters', 'scheduler')
step_size = int(config['Hyperparameters']['step_size'])
gamma = float(config['Hyperparameters']['gamma'])
patience = int(config['Hyperparameters']['patience'])
warmup_epochs = int(config['Hyperparameters']['warmup_epochs'])
plot = int(config['Callbacks']['plot'])
file_path = config.get('Data', 'file_path')
test_mode = int(config['Model']['test_mode'])

parameters = {
    "latent_dim": latent_dim,
    "hidden_dim_0": hidden_dim_0,
    "hidden_dim_1": hidden_dim_1,
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "training_model": training_model,
    "mask_ratio": mask_ratio,
    "scheduler": scheduler,
    "step_size": step_size,
    "gamma": gamma,
    "patience": patience,
    "warmup_epochs": warmup_epochs
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
    trainloader, testloader, data_min, data_max = dp.train_test_split(file_path=file_path)
    data_mean = None
    data_std = None
elif training_model == "MAE":
    trainloader, testloader, data_mean, data_std = dp.train_test_split(file_path=file_path)
    data_min = None
    data_max = None
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

train_input_dim = trainloader.dataset[0].shape[0]
print(f"Train input dim: {train_input_dim}")

if training_model == "VAE":
    model = VariationalAutoencoder(train_input_dim, latent_dim, hidden_dim_0, hidden_dim_1, beta, dropout).to(device)
    model_path = 'Models/vae.pth'
elif training_model == "MAE":
    model = MaskedAutoencoder(train_input_dim, mask_ratio).to(device)
    model_path = 'Models/mae.pth'
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

optimizer = optim.Adam(model.parameters(), lr)
training = Training(trainloader, testloader, optimizer, model, num_epochs, device, scheduler, step_size, gamma, patience,
                    warmup_epochs, max_lr, run=run, data_min=data_min, data_max=data_max, data_mean=data_mean, data_std=data_std)

if test_mode == 0:
    training.train()
    if save_model == 1:
        training.save_model(model_path)
elif test_mode == 1:
    training.test()