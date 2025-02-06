import os
import sys
import torch
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Config.load_config import data_dir, seed
from data_process import DataProcess
from Factory.variational_autoencoder import VariationalAutoencoder
from train_test import Training

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Több GPU esetén
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

for file_name in os.listdir(data_dir):
    dp = DataProcess()

    (
        _,
        _,
        testloader,
        _,
        _,
        labels,
        label_mapping,
        _,
    ) = dp.train_test_split()

    test_input_dim = testloader.dataset[0][0].shape[0]
    print(f"Test input dim: {test_input_dim}")

    model = VariationalAutoencoder(test_input_dim).to(device)

    training = Training(
        testloader=testloader,
        model=model,
        labels=labels,
        device=device,
        label_mapping=label_mapping,
    )
    print(f"{file_name} feldolgozása...")
    training.test()
