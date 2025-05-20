import configparser
import datetime
import sys
import os

config_path = os.environ.get("CONFIG_PATH", None)

if config_path is None and len(sys.argv) > 1:
    config_path = sys.argv[1]

if config_path is None:
    config_path = "Config/config.ini"

config = configparser.ConfigParser()
config.read(config_path)

print(f"Beolvasott config fájl: {config_path}")
print(f"Szereplő szekciók: {config.sections()}")

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

beta_min = float(config["Hyperparameters"]["beta_min"])
beta_max = float(config["Hyperparameters"]["beta_max"])
tau = int(config["Hyperparameters"]["tau"])
beta_multiplier = float(config["Hyperparameters"]["beta_multiplier"])
beta_warmup_epochs = int(config["Hyperparameters"]["beta_warmup_epochs"])
slope = int(config["Hyperparameters"]["slope"])
delay_epochs = int(config["Hyperparameters"]["delay_epochs"])
beta_scheduler_name = config.get("Hyperparameters", "beta_scheduler_name")

initial_lr = float(config["Hyperparameters"]["initial_lr"])
max_lr = float(config["Hyperparameters"]["max_lr"])
final_lr = float(config["Hyperparameters"]["final_lr"])
scheduler_name = config.get("Hyperparameters", "scheduler_name")
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
normalization = config.get("Data", "normalization")
parallel = int(config["Data"]["parallel"])
num_workers = int(config["Data"]["num_workers"])
data_dir = config.get("Data", "data_dir")
tsne_dir = config.get("Data", "tsne_dir")
manoeuvers_tsne_dir = config.get("Data", "manoeuvers_tsne_dir")
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
folder_name = config.get("Plot", "folder_name")
overlay_multiple_manoeuvres = int(config["Plot"]["overlay_multiple_manoeuvres"])
save_saliency = int(config["Plot"]["save_saliency"])
load_saliency = int(config["Plot"]["load_saliency"])

# Filtering
filtering = int(config["Filtering"]["filtering"])
removing_steps = int(config["Filtering"]["removing_steps"])
inconsistent_points_distance = float(
    config["Filtering"]["inconsistent_points_distance"]
)
time_difference = int(config["Filtering"]["time_difference"])
remove_start = int(config["Filtering"]["remove_start"])
distance_metric = config.get("Filtering", "distance_metric")

# Callbacks
plot = int(config["Callbacks"]["plot"])
project_name = config.get("Callbacks", "neptune_project")
api_token = config.get("Callbacks", "neptune_token")

# Validation
parameters = config.get("Validation", "parameters").split(", ")
validation_method = config.get("Validation", "validation_method")

# Egyéb változók
hidden_dims = [int(dim) for dim in hidden_dims_str.strip("[]").split(", ")]
warmup_epochs = num_epochs * 0.1
current_date = datetime.datetime.now().strftime("%m_%d_%H_%M")
model_path = f"Models/{training_model}_{num_epochs}_{current_date}.pth"
