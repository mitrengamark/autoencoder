import torch
import numpy as np
from Config.load_config import (
    step_size,
    gamma,
    num_epochs,
    patience,
    warmup_epochs,
    initial_lr,
    max_lr,
    final_lr,
    scheduler_name,
)


def scheduler_maker(optimizer=None):
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=patience
        )
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif scheduler_name == "WarmupCosine":
        lr_lambda = warmup_cosine_lr()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(
            f"Unsupported scheduler type. Expected StepLR, CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR or WarmupCosine!"
        )

    return scheduler


def warmup_cosine_lr():
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            cosine_epochs = epoch - warmup_epochs
            cosine_total = num_epochs - warmup_epochs
            lr = final_lr + (max_lr - final_lr) * 0.5 * (
                1 + np.cos(cosine_epochs / cosine_total * np.pi)
            )
        return lr

    return lr_lambda
