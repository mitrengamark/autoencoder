import torch
import matplotlib.pyplot as plt


def compute_saliency_map(input, output, device):
    input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    reconstructed_tensor = (
        torch.tensor(output, dtype=torch.float32).unsqueeze(0).to(device)
    )

    loss = torch.nn.functional.mse_loss(reconstructed_tensor, input_tensor)
    loss.backward()
    saliency = input_tensor.grad.abs().squeeze(0).detach().cpu()
    return saliency


def plot_saliency_map(all_columns, avg_saliency):
    plt.figure(figsize=(10, 5))
    plt.bar(all_columns, avg_saliency.numpy())
    plt.title("Saliency map - átlagolt (rekonstrukciós gradiens alapján)")
    plt.ylabel("Absz. gradiens (fontosság)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
