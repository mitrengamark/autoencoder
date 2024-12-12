import torch
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from data_process import DataProcess
from Analyse.decrase_dim import visualize_bottleneck, plot_latent_space
import matplotlib.pyplot as plt
import numpy as np

class Training():
    def __init__(self, trainloader, testloader, optimizer, model, num_epochs, device, data_min=None, data_max=None, run=None):
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.model = model
        self.num_epochs = num_epochs
        self.losses = []
        self.reconst_losses = []
        self.kl_losses = []
        self.device = device
        self.data_min = data_min
        self.data_max = data_max
        self.run = run

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            loss_per_episode = 0
            reconst_loss_per_epoch = 0
            kl_loss_per_epoch = 0

            for data in self.trainloader:
                inputs = data.to(self.device)
                # print(f"inputs dim: {inputs.shape}")
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(1)
                elif inputs.dim() != 2:
                    raise ValueError(f"Unexpected input dimension: {inputs.dim()}. Expected 2D tensor.")
                
                # print(f"Input shape: {inputs.shape}")
                
                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, reconst_loss, kl_div = self.model.loss(inputs, outputs, z_mean, z_log_var)
                    reconst_loss_per_epoch += reconst_loss.item()
                    kl_loss_per_epoch += kl_div.item()
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _ = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_per_episode += loss.item()
                
            average_loss = loss_per_episode / len(self.trainloader)
            self.losses.append(round(average_loss, 4))
            if isinstance(self.model, VariationalAutoencoder):
                self.reconst_losses.append(reconst_loss_per_epoch / len(self.trainloader))
                self.kl_losses.append(kl_loss_per_epoch / len(self.trainloader))

            if self.run:
                self.run[f"train/loss"].append(average_loss)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {average_loss:.4f}')

        plot_latent_space(z_mean, z_log_var, epoch)
        bottleneck_output = z_mean.cpu().detach().numpy()
        visualize_bottleneck(bottleneck_output)

        self.plot_losses()
            
    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs = data.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, _, _ = self.model.loss(inputs, outputs, z_mean, z_log_var)
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, masked_input = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")
                
                test_loss += loss.item()

        print(f'Test Loss: {test_loss / len(self.testloader):.4f}')
        bottleneck_output = z_mean.cpu().detach().numpy()
        visualize_bottleneck(bottleneck_output)

        if isinstance(self.model, VariationalAutoencoder):
            dp = DataProcess()
            # denorm_outputs = dp.denormalize(outputs, self.data_min, self.data_max)
            # print(f"Denormalized output: {denorm_outputs}")
        elif isinstance(self.model, MaskedAutoencoder):
            print(f"Masked input: {masked_input}")
            print(f"Reconstructed output: {outputs}")

        return inputs, _, outputs
        
    #     dataiter = iter(self.testloader)
    #     images = next(dataiter)
    #     images = images.to(self.device)
    #     outputs, _, _ = self.model.forward(images)

    #     # self.imgshow(images, outputs)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    # def imgshow(self, images, outputs):
    #     images = images.cpu()
    #     outputs = outputs.cpu()
    #     fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    #     for i in range(10):
    #         axes[0, i].imshow(images[i].view(input_size, 1).numpy(), cmap='gray')
    #         axes[0, i].axis('off')
    #         axes[1, i].imshow(outputs[i].detach().view(input_size, 1).numpy(), cmap='gray')
    #         axes[1, i].axis('off')
    #     plt.show()

    def plot_losses(self):
        if isinstance(self.model, VariationalAutoencoder):
            plt.figure(figsize=(10, 6))
            plt.plot(self.reconst_losses, label="Reconstruction Loss", marker='x')
            plt.plot(self.kl_losses, label="KL Divergence Loss", marker='s')
            plt.plot(self.losses, label="Total Loss", marker='o')
            plt.title("VAE Losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
        elif isinstance(self.model, MaskedAutoencoder):
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses, label="Total Loss", marker='o', color='orange')
            plt.title("MAE Losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()