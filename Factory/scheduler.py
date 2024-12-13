import torch


def scheduler_maker(scheduler, optimizer, step_size, gamma, num_epochs, patience, warmup_epochs, max_lr):
    if scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=patience)
    elif scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif scheduler == 'WarmupCosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr(num_epochs, warmup_epochs, max_lr))
    else:
        raise ValueError(f"Unsupported scheduler type. Expected StepLR, CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR or WarmupCosine!")
    
    return scheduler

def warmup_cosine_lr(num_epochs, warmup_epochs, max_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Lineáris melegítés
            return (epoch / warmup_epochs) * max_lr
        else:
            # Koszinusz annealing
            cosine_epochs = epoch - warmup_epochs
            cosine_total = num_epochs - warmup_epochs
            return max_lr * 0.5 * (1 + torch.cos(torch.tensor(cosine_epochs / cosine_total * 3.14159265359)))
    return lr_lambda