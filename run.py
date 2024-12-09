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
num_epochs = int(config['Hyperparameters']['num_epochs'])
beta = float(config['Hyperparameters']['beta'])
seed = int(config['Hyperparameters']['seed'])
training_model = config.get('Agent', 'training_model')
mask_ratio = float(config['Agent']['mask_ratio'])
save_model = int(config['Agent']['save_model'])
test_mode = int(config['Data']['test_mode'])
project_name = config.get('callbacks', 'neptune_project')
api_token = config.get('callbacks', 'neptune_token')

parameters = {
    "latent_dim": latent_dim,
    "hidden_dim_0": hidden_dim_0,
    "hidden_dim_1": hidden_dim_1,
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "training_model": training_model,
    "mask_ratio": mask_ratio
}

# Neptune inicializáció
# run = init_neptune(project_name, api_token, parameters)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

dp = DataProcess()

if training_model == "VAE":
    trainloader, testloader, train_input_size, test_input_size  = dp.train_test_split() #data_min, data_max
    if test_mode == 1:
        train_input_dim = train_input_size.shape[0]
    else:
        train_input_dim = len(train_input_size[0])

    print(f"Train input dim: {train_input_dim}")
    model = VariationalAutoencoder(train_input_dim, latent_dim, hidden_dim_0, hidden_dim_1).to(device)
    model_path = 'Models/vae.pth'
    optimizer = optim.Adam(model.parameters(), lr)
    training = Training(trainloader, testloader, optimizer, model, num_epochs, device) #, data_min, data_max) #, run)
elif training_model == "MAE":
    trainloader, testloader, train_input_size, test_input_size = dp.train_test_split()
    train_input_dim = len(train_input_size)
    model = MaskedAutoencoder(train_input_dim, mask_ratio).to(device)
    model_path = 'Models/mae.pth'
    optimizer = optim.Adam(model.parameters(), lr)
    training = Training(trainloader, testloader, optimizer, model, num_epochs, device) #run=run)

training.train()
if save_model == 1:
    training.save_model(model_path)

inputs, denorm_outputs, outputs = training.test()
# ev = Evaluation(inputs, denorm_outputs, outputs)
# ev.mean_absolute_error()