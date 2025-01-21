import torch
import torch.optim as optim
import configparser
import torch.utils.data
import datetime
from data_process import DataProcess
from train_test import Training
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.optimizer import optimizer_maker
from Analyse.neptune_utils import init_neptune

torch.cuda.empty_cache()

config = configparser.ConfigParser()
config.read('config.ini')

latent_dim = int(config['Dims']['latent_dim'])
bottleneck_dim = int(config['Dims']['bottleneck_dim'])
hidden_dims_str = config.get('Dims', 'hidden_dims')
hidden_dims = [int(dim) for dim in hidden_dims_str.strip('[]').split(', ')]
initial_lr = float(config['Hyperparameters']['initial_lr'])
max_lr = float(config['Hyperparameters']['max_lr'])
final_lr = float(config['Hyperparameters']['final_lr'])
num_epochs = int(config['Hyperparameters']['num_epochs'])
beta_min = 1 / float(config['Hyperparameters']['beta_min'])
seed = int(config['Data']['seed'])
training_model = config.get('Model', 'training_model')
mask_ratio = float(config['Hyperparameters']['mask_ratio'])
num_heads = int(config['Hyperparameters']['num_heads'])
save_model = int(config['Model']['save_model'])
project_name = config.get('Callbacks', 'neptune_project')
api_token = config.get('Callbacks', 'neptune_token')
dropout = float(config['Hyperparameters']['dropout'])
scheduler = config.get('Hyperparameters', 'scheduler')
step_size = int(config['Hyperparameters']['step_size'])
gamma = float(config['Hyperparameters']['gamma'])
patience = int(config['Hyperparameters']['patience'])
plot = int(config['Callbacks']['plot'])
file_path = config.get('Data', 'file_path')
test_mode = int(config['Model']['test_mode'])
saved_model = config.get('Model', 'model_path')
warmup_epochs = num_epochs * 0.1
current_date = datetime.datetime.now().strftime("%m_%d_%H_%M")
model_path = f'Models/{training_model}_{num_epochs}_{current_date}.pth'
opt_name = config.get('Hyperparameters', 'optimizer')
hyperopt = int(config['Hyperparameters']['hyperopt'])
tolerance = float(config['Callbacks']['tolerance'])

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
    "tolerance": tolerance
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
    trainloader, valloader, testloader, data_min, data_max, labels = dp.train_test_split(file_path=file_path)
    data_mean = None
    data_std = None
elif training_model == "MAE":
    trainloader, valloader, testloader, data_mean, data_std, labels = dp.train_test_split(file_path=file_path)
    data_min = None
    data_max = None
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

train_input_dim = trainloader.dataset[0][0].shape[0]
print(f"Train input dim: {train_input_dim}")

if training_model == "VAE":
    model = VariationalAutoencoder(train_input_dim, latent_dim, hidden_dims, dropout).to(device)
elif training_model == "MAE":
    model = MaskedAutoencoder(train_input_dim, bottleneck_dim, mask_ratio, num_heads, dropout).to(device)
else:
    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

model_params = model.parameters()
optimizer = optimizer_maker(opt_name, model_params, initial_lr)
training = Training(trainloader, valloader, testloader, test_mode, optimizer, model, labels, num_epochs, device, scheduler, beta_min, step_size, gamma, patience,
                    warmup_epochs, initial_lr, max_lr, final_lr, saved_model, run=run, data_min=data_min, data_max=data_max, data_mean=data_mean, data_std=data_std, hyperopt=hyperopt, tolerance=tolerance)

if test_mode == 0:
    training.train()
    if save_model == 1:
        training.save_model(model_path)
elif test_mode == 1:
       training.test()