import os

import wandb
from dotenv import load_dotenv

from Config.load_config import (
    latent_dim,
    hidden_dims,
    num_epochs,
    mask_ratio,
    initial_lr,
    max_lr,
    final_lr,
    scheduler_name,
    training_model,
    step_size,
    gamma,
    patience,
    model_path,
    wandb_project,
)

load_dotenv()


def init_wandb(config_path=None):
    config_path = config_path or os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError(
            "CONFIG_PATH környezeti változó nincs beállítva és nem adtál meg paramétert sem!"
        )

    run_name = os.path.splitext(os.path.basename(config_path))[0]
    project = os.environ.get("WANDB_PROJECT") or wandb_project

    config = {
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "num_epochs": num_epochs,
        "training_model": training_model,
        "mask_ratio": mask_ratio,
        "scheduler": scheduler_name,
        "step_size": step_size,
        "gamma": gamma,
        "patience": patience,
        "initial_lr": initial_lr,
        "max_lr": max_lr,
        "final_lr": final_lr,
        "model": model_path,
        "config_path": config_path,
    }

    run = wandb.init(
        project=project,
        name=run_name,
        tags=[run_name],
        config=config,
    )
    if os.path.isfile(config_path):
        wandb.save(config_path, base_path=os.path.dirname(config_path) or ".")
    return run


def log_epoch_metrics(run, metrics, step):
    if run is not None:
        wandb.log(metrics, step=step)


def finish_wandb(run):
    if run is not None:
        wandb.finish()
