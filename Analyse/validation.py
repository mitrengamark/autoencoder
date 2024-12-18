import torch
import numpy as np

def reconstruction_accuracy(inputs, outputs, tolerance=0.01):
    """
    Számolja a rekonstrukciós pontosságot a bemenet és a kimenet között.
    :param inputs: Eredeti bemenet
    :param outputs: Rekonstruált kimenet
    :param tolerance: A megengedett eltérés a bemenet és kimenet között
    :return: Rekonstrukciós pontosság százalékosan
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    if isinstance(outputs, np.ndarray):
        outputs = torch.tensor(outputs)

    differences = torch.abs(inputs - outputs)  # Eltérések kiszámítása
    accurate_reconstructions = (differences <= tolerance).float()  # Eltérés a tolerancián belül
    accuracy = accurate_reconstructions.mean().item() * 100  # Pontosság százalékban
    return accuracy
