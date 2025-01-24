import torch
import numpy as np


def scheduler_maker(
    scheduler=None,
    optimizer=None,
    step_size=None,
    gamma=None,
    num_epochs=None,
    patience=None,
    warmup_epochs=None,
    initial_lr=None,
    max_lr=None,
    final_lr=None,
):
    if scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=patience
        )
    elif scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif scheduler == "WarmupCosine":
        lr_lambda = warmup_cosine_lr(
            num_epochs, warmup_epochs, initial_lr, max_lr, final_lr
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(
            f"Unsupported scheduler type. Expected StepLR, CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR or WarmupCosine!"
        )

    return scheduler


def warmup_cosine_lr(num_epochs, warmup_epochs, initial_lr, max_lr, final_lr):
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
