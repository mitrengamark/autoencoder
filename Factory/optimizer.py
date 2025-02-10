import torch
from Config.load_config import opt_name, scheduler_name, initial_lr


def optimizer_maker(model_params):
    """
    Creates a PyTorch optimizer based on the specified type.

    Parameters:
    - optimizer_type (str): The type of optimizer ('SGD', 'Adam', 'AdamW', 'Adagrad', 'RMSprop').
    - model_params: Parameters of the model to optimize.

    Returns:
    - optimizer: The instantiated optimizer.
    """
    lr = initial_lr
    if scheduler_name == "WarmupCosine":
        lr = 1

    if opt_name == "SGD":
        optimizer = torch.optim.SGD(model_params, lr)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model_params, lr)
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model_params, lr)
    elif opt_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model_params, lr)
    elif opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model_params, lr)
    else:
        raise ValueError(
            f"Unsupported optimizer type: {opt_name}. Expected SGD, Adam, AdamW, Adagrad, or RMSprop."
        )

    return optimizer
