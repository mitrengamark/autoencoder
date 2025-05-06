import neptune
import os
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
    project_name,
    api_token,
    model_path,
)


def init_neptune(config_name=None):
    """
    Inicializálja a Neptune.ai run-t és beállítja az alap paramétereket.

    :param project_name: Neptune projekt neve (pl. "workspace/project").
    :param api_token: Neptune API token.
    :param parameters: Opcionális szótár, amely tartalmazza a futtatás paramétereit.
    :return: Neptune run objektum.
    """
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH környezeti változó nincs beállítva!")
    
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    parameters = {
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
    }
    run = neptune.init_run(
        project=project_name,
        api_token=api_token,
        name=config_name,
        tags=[config_name] if config_name else [],
    )
    if parameters:
        run["parameters"] = parameters

    run["config_file"].upload(config_path)
    return run
