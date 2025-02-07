import torch
import numpy as np
from Config.load_config import tolerance, parameters


def reconstruction_accuracy(inputs, outputs, selected_columns=None):
    """
    Rekonstrukciós pontosság kiszámítása, csak bizonyos oszlopokat figyelembe véve.

    :param inputs: Az eredeti bemenet (numpy array vagy torch tensor).
    :param outputs: A generált kimenet (numpy array vagy torch tensor).
    :param selected_columns: Az oszlopok indexeinek listája, amelyeket figyelembe kell venni.
    :return: A rekonstrukciós pontosság százalékosan.
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    if isinstance(outputs, np.ndarray):
        outputs = torch.tensor(outputs, dtype=torch.float32)

    column_mapping = {selected_columns[i]: parameters[i] for i in range(len(parameters))}

    if selected_columns is not None:
        inputs = inputs[:, selected_columns]  # Csak a megadott oszlopokat használjuk
        outputs = outputs[:, selected_columns]

    differences = torch.abs(inputs - outputs)  # Eltérések kiszámítása
    diff_1, diff_2, diff_3, diff_4, diff_5, diff_6, diff_7 = [
        differences[:, i] for i in range(7)
    ]

    # accurate_reconstructions = (
    #     differences <= tolerance
    # ).float()  # Eltérés a tolerancián belül
    # accuracy = accurate_reconstructions.mean().item() * 100  # Pontosság százalékban
    # return accuracy
