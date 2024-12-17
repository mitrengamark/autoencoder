import torch
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.scheduler import scheduler_maker
from data_process import DataProcess
from Analyse.decrase_dim import visualize_bottleneck, plot_latent_space
import matplotlib.pyplot as plt
import numpy as np

class Training():
    def __init__(self, trainloader, testloader, optimizer, model, num_epochs, device, scheduler, step_size, gamma, patience,
                 warmup_epochs, max_lr, data_min=None, data_max=None, run=None, data_mean=None, data_std=None):
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
        self.data_mean = data_mean
        self.data_std = data_std
        self.run = run
        self.scheduler = scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr

    def train(self):
        self.model.train()
        scheduler = scheduler_maker(self.scheduler, self.optimizer, self.step_size, self.gamma, self.num_epochs, self.patience,
                                    self.warmup_epochs, self.max_lr)
        for epoch in range(self.num_epochs):
            loss_per_episode = 0
            reconst_loss_per_epoch = 0
            kl_loss_per_epoch = 0

            for data in self.trainloader:
                inputs = data.to(self.device)
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(1)
                elif inputs.dim() != 2:
                    raise ValueError(f"Unexpected input dimension: {inputs.dim()}. Expected 2D tensor.")
                                
                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, reconst_loss, kl_div = self.model.loss(inputs, outputs, z_mean, z_log_var)
                    reconst_loss_per_epoch += reconst_loss.item()
                    kl_loss_per_epoch += kl_div.item()
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, encoded = self.model.forward(inputs)
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

            if self.scheduler == 'ReduceLROnPlateau':
                scheduler.step(average_loss)
            else:
                scheduler.step()

            if self.run:
                self.run[f"train/loss"].append(average_loss)
                self.run[f"learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {average_loss:.4f}')

        if self.run:
            self.run.stop()

        if isinstance(self.model, VariationalAutoencoder):
            z = self.model.reparameterize(z_mean, z_log_var)
            plot_latent_space(z_mean, z_log_var, epoch)
            bottleneck_output = z.cpu().detach().numpy()
            visualize_bottleneck(bottleneck_output)
        elif isinstance(self.model, MaskedAutoencoder):
            bottleneck_output = encoded.cpu().detach().numpy()
            visualize_bottleneck(bottleneck_output)

        self.plot_losses()
            
    def test(self):
        if isinstance(self.model, VariationalAutoencoder):
            self.model.load_state_dict(torch.load("Models/vae.pth", map_location=self.device))
        elif isinstance(self.model, MaskedAutoencoder):
            self.model.load_state_dict(torch.load("Models/mae.pth", map_location=self.device))
        else:
            raise ValueError("Unsupported model type. Expected VAE or MAE!")
        
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        bottleneck_outputs = []
        whole_output = []
        whole_masked_input = []
        with torch.no_grad():
            for data in self.testloader:
                inputs = data.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, _, _ = self.model.loss(inputs, outputs, z_mean, z_log_var)
                    bottleneck_output = z_mean.cpu().detach().numpy()
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, masked_input, encoded = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                    bottleneck_output = encoded.cpu().detach().numpy()
                    whole_masked_input.append(masked_input.cpu().detach().numpy())
                else:
                    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")
                
                test_loss += loss.item()

                bottleneck_outputs.append(bottleneck_output)
                whole_output.append(outputs.cpu().detach().numpy())

        bottleneck_outputs = np.vstack(bottleneck_outputs)
        whole_output = np.vstack(whole_output)

        print(f'Test Loss: {test_loss / len(self.testloader):.4f}')
        print(f'Bottleneck outputs shape: {bottleneck_outputs.shape}')

        dp = DataProcess()
        if isinstance(self.model, VariationalAutoencoder):
            bottleneck_outputs = dp.denormalize(bottleneck_outputs, self.data_min, self.data_max)
            denorm_outputs = dp.denormalize(whole_output, self.data_min, self.data_max)
            print(f"Denormalized output: {denorm_outputs}")
        elif isinstance(self.model, MaskedAutoencoder):
            bottleneck_outputs = dp.z_score_denormalize(bottleneck_outputs, self.data_mean, self.data_std)
            whole_masked_input = np.vstack(whole_masked_input)
            destandardized_outputs = dp.z_score_denormalize(whole_output, self.data_mean, self.data_std)
            print(f"Masked input: {whole_masked_input}")
            print(f"Reconstructed output: {destandardized_outputs}")
            print(f"Reconstructed output shape: {destandardized_outputs.shape}")

        visualize_bottleneck(bottleneck_outputs)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

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