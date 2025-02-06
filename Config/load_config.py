import configparser
import datetime

config = configparser.ConfigParser()
config.read("Config/config.ini")

# Dims
latent_dim = int(config["Dims"]["latent_dim"])
bottleneck_dim = int(config["Dims"]["bottleneck_dim"])
hidden_dims_str = config.get("Dims", "hidden_dims")

# Hyperparameters
hyperopt = int(config["Hyperparameters"]["hyperopt"])
n_trials = int(config["Hyperparameters"]["n_trials"])

num_epochs = int(config["Hyperparameters"]["num_epochs"])
dropout = float(config["Hyperparameters"]["dropout"])
batch_size = int(config["Hyperparameters"]["batch_size"])

mask_ratio = float(config["Hyperparameters"]["mask_ratio"])
num_heads = int(config["Hyperparameters"]["num_heads"])

beta_min = 1 / float(config["Hyperparameters"]["beta_min"])

initial_lr = float(config["Hyperparameters"]["initial_lr"])
max_lr = float(config["Hyperparameters"]["max_lr"])
final_lr = float(config["Hyperparameters"]["final_lr"])
scheduler = config.get("Hyperparameters", "scheduler")
step_size = int(config["Hyperparameters"]["step_size"])
gamma = float(config["Hyperparameters"]["gamma"])
patience = int(config["Hyperparameters"]["patience"])
opt_name = config.get("Hyperparameters", "optimizer")

# Parameters
eps = float(config["Parameters"]["eps"])
min_samples = int(config["Parameters"]["min_samples"])
n_clusters = int(config["Parameters"]["n_clusters"])
method = config.get("Parameters", "method")

grid_size = int(config["Parameters"]["grid_size"])

grid = int(config["Parameters"]["grid"])
max_sample = int(config["Parameters"]["max_sample"])

# Model
training_model = config.get("Model", "training_model")
save_model = int(config["Model"]["save_model"])
test_mode = int(config["Model"]["test_mode"])
saved_model = config.get("Model", "model_path")

# Data
num_workers = int(config["Data"]["num_workers"])
data_dir = config.get("Data", "data_dir")
num_manoeuvres = int(config["Data"]["num_manoeuvres"])
train_size = float(config["Data"]["train_size"])
val_size = float(config["Data"]["val_size"])
basic_method = int(config["Data"]["basic_method"])
seed = int(config["Data"]["seed"])
selected_manoeuvres = config.get("Data", "selected_manoeuvres", fallback="").split(",")

# Plot
parameter = config.get("Plot", "parameter")
coloring_method = config.get("Plot", "coloring_method")
coloring = int(config["Plot"]["coloring"])
n_clusters = int(config["Plot"]["n_clusters"])
use_cosine_similarity = int(config["Plot"]["use_cosine_similarity"])
dimension = int(config["Plot"]["dimension"])
tsneplot = int(config["Plot"]["tsneplot"])
step = int(config["Plot"]["step"])
save_fig = int(config["Plot"]["save_fig"])

# Callbacks
plot = int(config["Callbacks"]["plot"])
project_name = config.get("Callbacks", "neptune_project")
api_token = config.get("Callbacks", "neptune_token")
tolerance = float(config["Callbacks"]["tolerance"])

# Egyéb változók
hidden_dims = [int(dim) for dim in hidden_dims_str.strip("[]").split(", ")]
warmup_epochs = num_epochs * 0.1
current_date = datetime.datetime.now().strftime("%m_%d_%H_%M")
model_path = f"Models/{training_model}_{num_epochs}_{current_date}.pth"
