import torch

def optimizer_maker(optimizer_type, model_params, lr, scheduler):
    """
    Creates a PyTorch optimizer based on the specified type.

    Parameters:
    - optimizer_type (str): The type of optimizer ('SGD', 'Adam', 'AdamW', 'Adagrad', 'RMSprop').
    - model_params: Parameters of the model to optimize.
    
    Returns:
    - optimizer: The instantiated optimizer.
    """
    if scheduler == "WarmupCosine":
        lr = 1

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model_params, lr)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model_params, lr)
    elif optimizer_type == 'Adagrad':
        optimizer = torch.optim.Adagrad(model_params, lr)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(model_params, lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Expected SGD, Adam, AdamW, Adagrad, or RMSprop.")
    
    return optimizer
