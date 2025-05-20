import numpy as np
from Config.load_config import (
    beta_min,
    beta_max,
    beta_multiplier,
    tau,
    beta_warmup_epochs,
    slope,
    delay_epochs,
)


def beta_scheduler(
    epoch,
    strategy,
):
    if epoch < delay_epochs:
        return 0.0

    adjusted_epoch = epoch - delay_epochs

    if strategy == "sigmoid":
        x = (adjusted_epoch - beta_warmup_epochs / 2) / (beta_warmup_epochs / slope)
        return min(beta_max, beta_max / (1 + np.exp(-x)))

    elif strategy == "exponential":
        return beta_max * (1 - np.exp(-adjusted_epoch / tau))

    elif strategy == "linear":
        return min(beta_max, beta_min + adjusted_epoch * beta_multiplier)

    elif strategy == "inverse":
        if adjusted_epoch == 0:
            return 1 / beta_min
        return min(1.0, adjusted_epoch / beta_min)

    elif strategy == "constant":
        return beta_min

    else:
        raise ValueError(f"Unknown beta scheduling strategy: {strategy}")
