import torch
import torch.utils.data
import numpy as np
import random
import json
import sys
import os
from data_process import DataProcess
from train_test import Training
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.optimizer import optimizer_maker
from Analyse.wandb_utils import init_wandb, finish_wandb
from Analyse.process_figure import process_figures
from Analyse.efficiency_metrics import (
    EfficiencyTracker,
    build_efficiency_metrics,
    write_efficiency_json,
    append_efficiency_csv,
    log_efficiency_to_wandb,
)
from Config.load_config import (
    seed,
    training_model,
    save_model,
    test_mode,
    plot,
    batch_size,
    num_epochs,
    data_dir,
    selected_manoeuvres,
    config_name,
)

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Több GPU esetén
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Weights & Biases inicializáció
config_path = os.environ.get("CONFIG_PATH", "Config/config.ini")
if plot == 1 and test_mode == 0:
    run = init_wandb(config_path)
else:
    run = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    print(f"CUDA elérhető: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA nem elérhető, CPU használatban.")

dp = DataProcess()

(
    trainloader,
    valloader,
    testloader,
    data_min,
    data_max,
    data_mean,
    data_std,
    scaler,
    labels,
    label_mapping,
    sign_change_indices,
    selected_columns,
    all_columns,
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
    scaler=scaler,
    sign_change_indices=sign_change_indices,
    label_mapping=label_mapping,
    selected_columns=selected_columns,
    all_columns=all_columns,
)

if test_mode == 0:
    tracker = EfficiencyTracker()
    training.efficiency_tracker = tracker
    tracker.start()
    training.train()
    train_wall_sec, peak_rss_mb, peak_cuda_mb = tracker.stop()

    metrics = build_efficiency_metrics(
        config_path=config_path,
        config_name=config_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        selected_manoeuvres=selected_manoeuvres,
        train_wall_sec=train_wall_sec,
        epoch_times=training.epoch_times,
        peak_rss_mb=peak_rss_mb,
        peak_cuda_mb=peak_cuda_mb,
    )
    json_path = write_efficiency_json(metrics)
    csv_path = append_efficiency_csv(metrics)
    log_efficiency_to_wandb(run, metrics)
    print(f"Efficiency metrics saved: {json_path}, {csv_path}")
    print(
        f"  manoeuvres={metrics['num_manoeuvres']}, "
        f"train_wall_sec={metrics['train_wall_sec']}, "
        f"peak_rss_mb={metrics['peak_rss_mb']}, "
        f"peak_cuda_mb={metrics['peak_cuda_allocated_mb']}"
    )

    if run:
        finish_wandb(run)

    if save_model == 1:
        training.save_model()
elif test_mode == 1:
    # latent_data, label, bottleneck_outputs, labels, avg_saliency = training.test()
    training.test()

    # latent_data = latent_data[2500:]
    # # Ellenőrzés
    # print("latent_data type:", type(latent_data))
    # print("label type:", type(label))
    # print("bottleneck_outputs type:", type(bottleneck_outputs))
    # print("labels type:", type(labels))

    # # Konvertálás megfelelő formátumba
    # latent_list = latent_data.tolist()
    # label_list = label.tolist() if isinstance(label, np.ndarray) else label
    # bottleneck_outputs_list = bottleneck_outputs.tolist()
    # labels_list = labels.tolist() if isinstance(labels, np.ndarray) else labels

    # # JSON mentése fájlba
    # output_file = "tsne_output.json"
    # output_file_2 = "bottleneck_output.json"
    # with open(output_file, "w") as f:
    #     json.dump({"latent_data": latent_list, "labels": label_list}, f)

    # with open(output_file_2, "w") as f:
    #     json.dump(
    #         {
    #             "bottleneck_outputs": bottleneck_outputs_list,
    #             "labels": labels_list,
    #             "label_mapping": label_mapping,
    #         },
    #         f,
    #     )

    # print(f"TSNE adatok elmentve: {output_file}")
    # print(f"Bottleneck adatok elmentve: {output_file_2}")

    # Mentés JSON fájlba
#     with open("saliency_output.json", "w") as f:
#         json.dump({
#             "saliency": avg_saliency.tolist(),
#             "features": all_columns
#         }, f)

#     print("Saliency elmentve: saliency_output.json")
# else:
#     process_figures()
