import neptune
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


def init_neptune():
    """
    Inicializálja a Neptune.ai run-t és beállítja az alap paramétereket.

    :param project_name: Neptune projekt neve (pl. "workspace/project").
    :param api_token: Neptune API token.
    :param parameters: Opcionális szótár, amely tartalmazza a futtatás paramétereit.
    :return: Neptune run objektum.
    """
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
    )
    if parameters:
        run["parameters"] = parameters
    return run
