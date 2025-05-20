import numpy as np
from Config.load_config import (
    beta_min,
    beta_max,
    beta_multiplier,
    tau,
    beta_warmup_epochs,
    slope,
)


def beta_scheduler(
    epoch,
    strategy,
):
    if strategy == "sigmoid":
        x = (epoch - beta_warmup_epochs / 2) / (beta_warmup_epochs / slope)
        return min(beta_max, beta_max / (1 + np.exp(-x)))

    elif strategy == "exponential":
        return beta_max * (1 - np.exp(-epoch / tau))

    elif strategy == "linear":
        return min(beta_max, beta_min + epoch * beta_multiplier)

    elif strategy == "inverse":
        if epoch == 0:
            return 1 / beta_min
        return min(1.0, epoch / beta_min)

    elif strategy == "constant":
        return beta_min

    else:
        raise ValueError(f"Unknown beta scheduling strategy: {strategy}")
