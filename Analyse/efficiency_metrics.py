import csv
import json
import os
import time
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    psutil = None

import torch

CSV_FIELDS = [
    "timestamp",
    "config_name",
    "data_dir",
    "num_manoeuvres",
    "n_train",
    "n_val",
    "n_test",
    "n_total_points",
    "train_wall_sec",
    "mean_epoch_sec",
    "samples_per_sec",
    "peak_rss_mb",
    "peak_cuda_allocated_mb",
    "device",
    "batch_size",
    "num_epochs",
]


class EfficiencyTracker:
    """Track wall-clock time and peak memory during training."""

    def __init__(self):
        self._start = None
        self._peak_rss = 0
        self._process = psutil.Process(os.getpid()) if psutil else None

    def start(self):
        self._start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.sample()

    def sample(self):
        if self._process is not None:
            rss = self._process.memory_info().rss
            self._peak_rss = max(self._peak_rss, rss)

    def stop(self):
        self.sample()
        elapsed = time.perf_counter() - self._start if self._start is not None else 0.0
        peak_cuda_mb = None
        if torch.cuda.is_available():
            peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_rss_mb = self._peak_rss / (1024**2) if self._peak_rss else None
        return elapsed, peak_rss_mb, peak_cuda_mb


def build_efficiency_metrics(
    config_path,
    config_name,
    data_dir,
    batch_size,
    num_epochs,
    device,
    trainloader,
    valloader,
    testloader,
    selected_manoeuvres,
    train_wall_sec,
    epoch_times,
    peak_rss_mb=None,
    peak_cuda_mb=None,
):
    n_train = len(trainloader.dataset)
    n_val = len(valloader.dataset)
    n_test = len(testloader.dataset)
    n_total = n_train + n_val + n_test

    num_manoeuvres_kept = len([m for m in selected_manoeuvres if m.strip()])
    mean_epoch_sec = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    total_samples = n_train * num_epochs
    samples_per_sec = total_samples / train_wall_sec if train_wall_sec > 0 else 0.0

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "config_name": config_name,
        "data_dir": data_dir,
        "device": str(device),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_manoeuvres": num_manoeuvres_kept,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total_points": n_total,
        "train_wall_sec": round(train_wall_sec, 3),
        "mean_epoch_sec": round(mean_epoch_sec, 3),
        "samples_per_sec": round(samples_per_sec, 2),
        "peak_rss_mb": round(peak_rss_mb, 2) if peak_rss_mb is not None else None,
        "peak_cuda_allocated_mb": round(peak_cuda_mb, 2)
        if peak_cuda_mb is not None
        else None,
    }
    return metrics


def write_efficiency_json(metrics, output_dir="Results/efficiency"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{metrics['config_name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def append_efficiency_csv(metrics, csv_path="Results/efficiency_summary.csv"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: metrics.get(k) for k in CSV_FIELDS})
    return csv_path


def log_efficiency_to_wandb(run, metrics):
    if run is None:
        return
    import wandb

    efficiency = {}
    for key in CSV_FIELDS:
        if key in {"timestamp", "config_name", "data_dir", "device"}:
            continue
        value = metrics.get(key)
        if value is not None:
            efficiency[f"efficiency/{key}"] = value
    wandb.log(efficiency)
