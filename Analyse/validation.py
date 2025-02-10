import torch
import numpy as np
from Config.load_config import parameters


def reconstruction_accuracy(inputs, outputs, selected_columns=None):
    """
    Az egyes oszlopok eltérésének átlagát és az összes eltérés átlagát számolja ki.

    :param inputs: Az eredeti bemenet (numpy array vagy torch tensor).
    :param outputs: A generált kimenet (numpy array vagy torch tensor).
    :param selected_columns: Az oszlopok indexeinek listája.
    :return: Dictionary az egyes oszlopok eltéréseinek átlagával + az összes eltérés átlagával.
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    if isinstance(outputs, np.ndarray):
        outputs = torch.tensor(outputs, dtype=torch.float32)

    column_mapping = {
        selected_columns[i]: parameters[i] for i in range(len(parameters))
    }

    if selected_columns is not None:
        inputs = inputs[:, selected_columns]  # Csak a megadott oszlopokat használjuk
        outputs = outputs[:, selected_columns]

    differences = torch.abs(inputs - outputs)  # Eltérések kiszámítása
    differences_dict = {
        column_mapping[selected_columns[i]]: differences[:, i].mean().item()
        for i in range(len(selected_columns))
    }

    # Az összes oszlop eltérésének átlaga
    diff_average = differences.mean().item()
    differences_dict["diff_average"] = diff_average

    return differences_dict
