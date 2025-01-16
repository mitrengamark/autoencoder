import torch
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.scheduler import scheduler_maker
from data_process import DataProcess
from Analyse.decrase_dim import visualize_bottleneck, plot_latent_space
from Analyse.validation import reconstruction_accuracy
import matplotlib.pyplot as plt
import numpy as np

class Training():
    def __init__(self, trainloader, valloader, testloader, optimizer, model, labels, num_epochs, device, scheduler, step_size=None, gamma=None, patience=None,
                 warmup_epochs=None, initial_lr=None, max_lr=None, final_lr=None, model_path=None, data_min=None, data_max=None, run=None, data_mean=None, data_std=None, hyperopt=None, tolerance=None):
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.model = model
        self.num_epochs = num_epochs
        self.losses = []
        self.reconst_losses = []
        self.kl_losses = []
        self.val_losses = []
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
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.model_path = model_path
        self.hyperopt = hyperopt
        self.tolerance = tolerance
        self.labels = labels

    def train(self):
        self.model.train()
        if self.hyperopt == 0:
            scheduler = scheduler_maker(self.scheduler, self.optimizer, self.step_size, self.gamma, self.num_epochs, self.patience,
                                        self.warmup_epochs, self.initial_lr, self.max_lr, self.final_lr)
        
        for epoch in range(self.num_epochs):
            loss_per_episode = 0
            reconst_loss_per_epoch = 0
            kl_loss_per_epoch = 0
            train_accuracy = 0
            
            for data in self.trainloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
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
                    
                accuracy = reconstruction_accuracy(inputs, outputs, self.tolerance)
                train_accuracy += accuracy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_per_episode += loss.item()
                
            average_loss = loss_per_episode / len(self.trainloader)
            average_accuracy = train_accuracy / len(self.trainloader)
            self.losses.append(round(average_loss, 4))
            if isinstance(self.model, VariationalAutoencoder):
                self.reconst_losses.append(reconst_loss_per_epoch / len(self.trainloader))
                self.kl_losses.append(kl_loss_per_epoch / len(self.trainloader))

            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)

            if self.hyperopt == 0:
                if self.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(average_loss)
                else:
                    scheduler.step()
            elif self.hyperopt == 1:
                self.scheduler.step()
            else:
                raise ValueError("Unsupported hyperopt value. Expected 0 or 1.")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr:.6f}")

            if self.run:
                self.run[f"train/loss"].append(average_loss)
                self.run[f"train/accuracy"].append(average_accuracy)
                self.run[f"learning_rate"].append(self.optimizer.param_groups[0]['lr'])
                self.run[f"validation/loss"].append(val_loss)
                self.run[f"validation/accuracy"].append(val_accuracy)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {average_loss:.4f}, Train Accuracy: {average_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
            
        if self.run:
            self.run.stop()

        if self.hyperopt == 0:
            if isinstance(self.model, VariationalAutoencoder):
                z = self.model.reparameterize(z_mean, z_log_var)
                # plot_latent_space(z_mean, z_log_var, epoch)
                bottleneck_output_z_mean = z_mean.cpu().detach().numpy() # A latens térben lévő átlagok.
                # Használható a latens tér szerkezetének elemzésére, pl. klaszterek vizsgálatára.
                bottleneck_output_z = z.cpu().detach().numpy() # A mintavételezett tényleges latens értékek.
                                                            # Ez a modell valódi bemenete a decoder számára.
                visualize_bottleneck(bottleneck_output_z_mean, self.labels, "VAE", "z_mean")
                visualize_bottleneck(bottleneck_output_z, self.labels, "VAE", "z")
            elif isinstance(self.model, MaskedAutoencoder):
                bottleneck_output = encoded.cpu().detach().numpy()
                visualize_bottleneck(bottleneck_output, self.labels, "MAE")

            if isinstance(self.model, VariationalAutoencoder):
                with torch.no_grad():
                    bottleneck_outputs = {label: [] for label in torch.unique(self.labels)}
                    for data in self.trainloader:
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        _, z_mean, _ = self.model.forward(inputs)
                        for label in torch.unique(labels):
                            bottleneck_outputs[label.item()].append(z_mean[labels == label])
                    bottleneck_outputs = {label: torch.cat(outputs, dim=0).cpu().numpy() for label, outputs in bottleneck_outputs.items()}
                    visualize_bottleneck(bottleneck_outputs, self.labels, "VAE")
            elif isinstance(self.model, MaskedAutoencoder):
                # MAE esetén
                with torch.no_grad():
                    bottleneck_outputs = {label: [] for label in torch.unique(self.labels)}
                    for data in self.trainloader:
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        _, _, encoded = self.model.forward(inputs)
                        for label in torch.unique(labels):
                            bottleneck_outputs[label.item()].append(encoded[labels == label])
                    bottleneck_outputs = {label: torch.cat(outputs, dim=0).cpu().numpy() for label, outputs in bottleneck_outputs.items()}
                    visualize_bottleneck(bottleneck_outputs, self.labels, "MAE")


            self.plot_losses()

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data in self.valloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, _, _ = self.model.loss(inputs, outputs, z_mean, z_log_var)
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, _ = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError("Unsupported model type. Expected VAE or MAE!")
                
                val_loss += loss.item()
                val_accuracy += reconstruction_accuracy(inputs, outputs, self.tolerance)
        return val_loss / len(self.valloader), val_accuracy / len(self.valloader)
            
    def test(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        bottleneck_outputs_z_mean = []
        bottleneck_outputs_z = []
        bottleneck_outputs = []
        whole_input = []
        whole_output = []
        whole_masked_input = []
        with torch.no_grad():
            for data in self.testloader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, _, _ = self.model.loss(inputs, outputs, z_mean, z_log_var)
                    z = self.model.reparameterize(z_mean, z_log_var)
                    bottleneck_output_z_mean = z_mean.cpu().detach().numpy()
                    bottleneck_output_z = z.cpu().detach().numpy()
                    bottleneck_outputs_z_mean.append(bottleneck_output_z_mean)
                    bottleneck_outputs_z.append(bottleneck_output_z)
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, masked_input, encoded = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                    bottleneck_output = encoded.cpu().detach().numpy()
                    bottleneck_outputs.append(bottleneck_output)
                    whole_masked_input.append(masked_input.cpu().detach().numpy())
                else:
                    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")
                
                test_loss += loss.item()

                whole_input.append(inputs.cpu().detach().numpy())
                whole_output.append(outputs.cpu().detach().numpy())

        whole_input = np.vstack(whole_input)
        whole_output = np.vstack(whole_output)

        print(f'Test Loss: {test_loss / len(self.testloader):.4f}')
        # print(f'Bottleneck outputs shape: {bottleneck_outputs.shape}')

        dp = DataProcess()
        if isinstance(self.model, VariationalAutoencoder):
            bottleneck_outputs_z_mean = np.vstack(bottleneck_outputs_z_mean)
            bottleneck_outputs_z = np.vstack(bottleneck_outputs_z)
            bottleneck_outputs_z_mean = dp.denormalize(bottleneck_outputs_z_mean, self.data_min, self.data_max)
            bottleneck_outputs_z = dp.denormalize(bottleneck_outputs_z, self.data_min, self.data_max)
            denorm_outputs = dp.denormalize(whole_output, self.data_min, self.data_max)
            model_name = "VAE"
            print(f"Denormalized output: {denorm_outputs}")
            visualize_bottleneck(bottleneck_outputs_z_mean, self.labels, model_name, "z_mean")
            visualize_bottleneck(bottleneck_outputs_z, self.labels, model_name, "z")
        elif isinstance(self.model, MaskedAutoencoder):
            bottleneck_outputs = np.vstack(bottleneck_outputs)
            bottleneck_outputs = dp.z_score_denormalize(bottleneck_outputs, self.data_mean, self.data_std)
            whole_masked_input = np.vstack(whole_masked_input)
            destandardized_outputs = dp.z_score_denormalize(whole_output, self.data_mean, self.data_std)
            model_name = "MAE"
            print(f"Masked input: {whole_masked_input}")
            print(f"Reconstructed output: {destandardized_outputs}")
            print(f"Reconstructed output shape: {destandardized_outputs.shape}")
            visualize_bottleneck(bottleneck_outputs, self.labels, model_name)

        if isinstance(self.model, VariationalAutoencoder):
            with torch.no_grad():
                bottleneck_outputs_z_mean = {label: [] for label in torch.unique(self.labels)}
                for data in self.testloader:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    _, z_mean, _ = self.model.forward(inputs)
                    for label in torch.unique(labels):
                        bottleneck_outputs_z_mean[label.item()].append(z_mean[labels == label])
                bottleneck_outputs_z_mean = {label: torch.cat(outputs, dim=0).cpu().numpy() for label, outputs in bottleneck_outputs_z_mean.items()}
                visualize_bottleneck(bottleneck_outputs_z_mean, self.labels, "VAE")


        accuracy = reconstruction_accuracy(whole_input, whole_output, self.tolerance)
        print(f"Test Accuracy: {accuracy:.2f}%")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def plot_losses(self):
        if isinstance(self.model, VariationalAutoencoder):
            plt.figure(figsize=(10, 6))
            plt.plot(self.reconst_losses, label="Reconstruction Loss", marker='x')
            plt.plot(self.kl_losses, label="KL Divergence Loss", marker='s')
            plt.plot(self.losses, label="Total Loss", marker='o')
            plt.plot(self.val_losses, label="Validation Loss", marker='x', color='red')
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