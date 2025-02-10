import torch
import matplotlib.pyplot as plt
import numpy as np
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.scheduler import scheduler_maker
from Analyse.decrase_dim import Visualise
from Analyse.validation import reconstruction_accuracy
from Synthesis.data_synthesis import remove_redundant_data, plot_removed_data
from Synthesis.heat_map import create_comparison_heatmaps
from Synthesis.manoeuvres_filtering import ManoeuvresFiltering
from Config.load_config import (
    num_manoeuvres,
    training_model,
    num_epochs,
    scheduler_name,
    beta_min,
    hyperopt,
    model_path,
    saved_model,
    save_fig,
    parameters,
)


class Training:
    def __init__(
        self,
        trainloader=None,
        valloader=None,
        testloader=None,
        optimizer=None,
        model=None,
        labels=None,
        device=None,
        data_min=None,
        data_max=None,
        run=None,
        data_mean=None,
        data_std=None,
        sign_change_indices=None,
        label_mapping=None,
        selected_columns=None,
    ):

        # Adatbetöltők
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # Modell és optimalizáló
        self.model = model
        self.model_name = training_model
        self.optimizer = optimizer

        # Hiperparaméterek és tanítási konfigurációk
        self.beta = 0  # Dinamikus érték lesz tanítás során
        self.scheduler_name = scheduler_name

        # Címkék
        self.labels = labels
        self.label_mapping = label_mapping
        self.selected_columns = selected_columns

        # Adatok normalizálásához szükséges statisztikák
        self.data_min = data_min
        self.data_max = data_max
        self.data_mean = data_mean
        self.data_std = data_std

        # Egyéb
        self.run = run
        self.device = device
        self.sign_change_indices = sign_change_indices

        # Loss értékek tárolása
        self.losses = []
        self.reconst_losses = []
        self.kl_losses = []
        self.val_losses = []

    def train(self):
        self.model.train()
        if hyperopt == 0:
            scheduler = scheduler_maker(optimizer=self.optimizer)

        for epoch in range(num_epochs):
            loss_per_episode = 0
            reconst_loss_per_epoch = 0
            kl_loss_per_epoch = 0
            train_differences = {param: [] for param in parameters}  # Listát tárolunk
            train_total_differences = []

            self.beta = min(1.0, epoch / beta_min)
            if epoch == 0:
                self.beta = 1 / beta_min

            for data in self.trainloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(1)
                elif inputs.dim() != 2:
                    raise ValueError(
                        f"Unexpected input dimension: {inputs.dim()}. Expected 2D tensor."
                    )

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, reconst_loss, kl_div = self.model.loss(
                        inputs, outputs, z_mean, z_log_var, self.beta
                    )
                    reconst_loss_per_epoch += reconst_loss.item()
                    kl_loss_per_epoch += kl_div.item()
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, encoded = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

                # Eltérések kiszámítása minden egyes batch-re
                batch_differences = reconstruction_accuracy(
                    inputs, outputs, self.selected_columns
                )

                # Eltérések összeadása az epizódon belül
                for param in parameters:  
                    if param in batch_differences:  
                        train_differences[param].append(batch_differences[param])

                if "diff_average" in batch_differences:  
                    train_total_differences.append(batch_differences["diff_average"]) 

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                loss_per_episode += loss.item()

            train_differences = {param: np.mean(train_differences[param]) for param in parameters}
            train_total_difference = np.mean(train_total_differences)

            average_loss = loss_per_episode / len(self.trainloader)
            self.losses.append(round(average_loss, 4))

            for param, avg_value in train_differences.items():
                globals()[f"train_diff_{param}_average"] = avg_value  

            globals()["train_diff_average"] = train_total_difference

            if isinstance(self.model, VariationalAutoencoder):
                self.reconst_losses.append(
                    reconst_loss_per_epoch / len(self.trainloader)
                )
                self.kl_losses.append(kl_loss_per_epoch / len(self.trainloader))

            val_loss, val_differences = self.validate()
            self.val_losses.append(val_loss)

            if hyperopt == 0:
                if self.scheduler_name == "ReduceLROnPlateau":
                    scheduler.step(average_loss)
                else:
                    scheduler.step()
            elif hyperopt == 1:
                self.scheduler_name.step()
            else:
                raise ValueError("Unsupported hyperopt value. Expected 0 or 1.")

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.run:
                self.run[f"train/loss"].append(average_loss)
                self.run[f"learning_rate"].append(self.optimizer.param_groups[0]["lr"])
                self.run[f"validation/loss"].append(val_loss)

                for param, avg_value in train_differences.items():
                    self.run[f"train/{param}_average"].append(avg_value)

                for param, avg_value in val_differences.items():
                    self.run[f"validation/{param}_average"].append(avg_value)

                self.run[f"train/diff_average"].append(train_total_difference)
                self.run[f"validation/diff_average"].append(val_differences["diff_average"])

            # Kiírás az egyes train és validation eltérésekre
            train_differences_str = " | ".join([f"{param}: {train_differences[param]:.6f}" for param in parameters])
            val_differences_str = " | ".join([f"{param}: {val_differences[param]:.6f}" for param in parameters])

            print(f"Epoch [{epoch+1}/{num_epochs}], Current Learning Rate: {current_lr:.6f}")
            print(f"Train Loss: {average_loss:.4f}")
            print(f"Train Differences: {train_differences_str}\n **Total Avg: {train_total_difference:.6f}**")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Differences: {val_differences_str}\n **Total Avg: {val_differences['diff_average']:.6f}**")

        if self.run:
            self.run.stop()
            self.plot_losses()

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_differences = {param: [] for param in parameters}  
        val_total_differences = []

        with torch.no_grad():
            for data in self.valloader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, _, _ = self.model.loss(
                        inputs, outputs, z_mean, z_log_var, self.beta
                    )
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, _ = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError("Unsupported model type. Expected VAE or MAE!")

                val_loss += loss.item()
                batch_differences = reconstruction_accuracy(inputs, outputs, self.selected_columns)

                for param in parameters:  
                    if param in batch_differences:  
                        val_differences[param].append(batch_differences[param])

                if "diff_average" in batch_differences:  
                    val_total_differences.append(batch_differences["diff_average"])   

        val_differences = {param: np.mean(val_differences[param]) for param in parameters}
        val_total_difference = np.mean(val_total_differences)

        globals()["val_diff_average"] = val_total_difference

        for param, avg_value in val_differences.items():
            globals()[f"val_diff_{param}_average"] = avg_value

        val_differences["diff_average"] = val_total_difference

        return val_loss / len(self.valloader), val_differences

    def test(self):
        self.model.load_state_dict(
            torch.load(saved_model, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        bottleneck_outputs = []
        labels_list = []
        whole_input = []
        whole_output = []

        with torch.no_grad():
            for data in self.testloader:
                inputs, batch_labels = data
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, _, _ = self.model.loss(
                        inputs, outputs, z_mean, z_log_var, self.beta
                    )
                    bottleneck_outputs.append(z_mean.cpu())
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, encoded = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                    bottleneck_outputs.append(encoded.cpu())
                else:
                    raise ValueError(f"Unsupported model type. Expected VAE or MAE!")

                test_loss += loss.item()
                labels_list.append(batch_labels.cpu())
                whole_input.append(inputs.cpu().detach().numpy())
                whole_output.append(outputs.cpu().detach().numpy())

        print(f"Test Loss: {test_loss / len(self.testloader):.4f}")

        bottleneck_outputs = torch.cat(bottleneck_outputs, dim=0)
        labels = torch.cat(labels_list, dim=0).numpy()
        whole_input = np.vstack(whole_input)
        whole_output = np.vstack(whole_output)

        if isinstance(self.model, VariationalAutoencoder):
            bottleneck_outputs = bottleneck_outputs.numpy()
        elif isinstance(self.model, MaskedAutoencoder):
            bottleneck_outputs = bottleneck_outputs.mean(dim=1).numpy()
            # bottleneck_outputs_flattened = bottleneck_outputs.view(-1, bottleneck_outputs.size(-1)).numpy()  # [batch_size * seq_len, feature_dim]

        # Vizualizáció
        vs = Visualise(
            bottleneck_outputs=bottleneck_outputs,
            labels=labels,
            model_name=self.model_name,
            label_mapping=self.label_mapping,
            sign_change_indices=self.sign_change_indices,
        )
        latent_data = vs.visualize_bottleneck()
        if save_fig == 0:
            if num_manoeuvres == 1:
                vs.kmeans_clustering()
                filtered_latent_data = remove_redundant_data(latent_data)
                create_comparison_heatmaps(latent_data, filtered_latent_data)
            else:
                mf = ManoeuvresFiltering(
                    reduced_data=latent_data,
                    labels=labels,
                    label_mapping=self.label_mapping,
                )
                filtered_reduced_data = mf.filter_manoeuvres()
                plot_removed_data(latent_data, filtered_reduced_data)
                create_comparison_heatmaps(latent_data, filtered_reduced_data)

        # Denormalizáció

        reconstruction_accuracy(whole_input, whole_output, self.selected_columns)

    def save_model(self):
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def plot_losses(self):
        if isinstance(self.model, VariationalAutoencoder):
            plt.figure(figsize=(10, 6))
            plt.plot(self.reconst_losses, label="Reconstruction Loss", marker="x")
            plt.plot(self.kl_losses, label="KL Divergence Loss", marker="s")
            plt.plot(self.losses, label="Total Loss", marker="o")
            plt.plot(self.val_losses, label="Validation Loss", marker="x", color="red")
            plt.title("VAE Losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
        elif isinstance(self.model, MaskedAutoencoder):
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses, label="Total Loss", marker="o", color="orange")
            plt.title("MAE Losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
