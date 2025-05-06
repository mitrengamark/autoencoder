import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.scheduler import scheduler_maker
from Analyse.decrase_dim import Visualise
from Analyse.validation import reconstruction_accuracy
from Analyse.saliency_map import compute_saliency_map, plot_saliency_map
from Reduction.data_removing import (
    remove_redundant_data,
    plot_removed_data,
    remove_data_step_by_step,
)
from Reduction.heat_map import create_comparison_heatmaps
from Reduction.manoeuvres_filtering import ManoeuvresFiltering
from Reduction.outlier_detection import detect_outliers
from Reduction.inconsistent_points import (
    filter_inconsistent_points,
    filter_outliers_by_grid,
)
from data_process import DataProcess
from Config.load_config import (
    num_manoeuvres,
    training_model,
    num_epochs,
    scheduler_name,
    beta_min,
    beta_max,
    tau,
    hyperopt,
    model_path,
    saved_model,
    save_fig,
    parameters,
    validation_method,
    normalization,
    removing_steps,
    folder_name,
    beta_multiplier,
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
        scaler=None,
        run=None,
        data_mean=None,
        data_std=None,
        sign_change_indices=None,
        label_mapping=None,
        selected_columns=None,
        all_columns=None,
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
        self.beta_min = 1 / beta_min
        self.scheduler_name = scheduler_name

        # Címkék
        self.labels = labels
        self.label_mapping = label_mapping
        self.selected_columns = selected_columns
        self.all_columns = all_columns

        # Egyéb
        self.run = run
        self.device = device
        self.sign_change_indices = sign_change_indices

        # Adatok normalizálásához szükséges statisztikák
        self.data_min = data_min.to(self.device)
        self.data_max = data_max.to(self.device)
        self.data_mean = data_mean.to(self.device)
        self.data_std = data_std.to(self.device)
        self.scaler = scaler

        # Loss értékek tárolása
        self.losses = []
        self.reconst_losses = []
        self.kl_losses = []
        self.val_losses = []

        self.dp = DataProcess()

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

            if tau > 0:
                self.beta = beta_max * (1 - np.exp(-epoch / tau))
            elif beta_max > 0:
                self.beta = min(beta_max, beta_min + epoch * beta_multiplier)
            elif beta_min > 0:
                self.beta = min(1.0, epoch / self.beta_min)
                if epoch == 0:
                    self.beta = 1 / self.beta_min
            else:
                self.beta = beta_min

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

                if validation_method == "denormalized":
                    if normalization == "z_score":
                        inputs = self.dp.z_score_denormalize(
                            data=inputs,
                            data_mean=self.data_mean,
                            data_std=self.data_std,
                        )
                        outputs = self.dp.z_score_denormalize(
                            data=outputs,
                            data_mean=self.data_mean,
                            data_std=self.data_std,
                        )
                    elif normalization == "min_max":
                        inputs = self.dp.denormalize(
                            data=inputs, data_min=self.data_min, data_max=self.data_max
                        )
                        outputs = self.dp.denormalize(
                            data=outputs, data_min=self.data_min, data_max=self.data_max
                        )
                    elif normalization == "robust_scaler":
                        inputs = self.dp.robust_scaler_denormalize(
                            data=inputs, scaler=self.scaler
                        )
                        outputs = self.dp.robust_scaler_denormalize(
                            data=outputs, scaler=self.scaler
                        )
                    elif normalization == "log_transform":
                        inputs = self.dp.log_transform_denormalize(data=inputs)
                        outputs = self.dp.log_transform_denormalize(data=outputs)
                    else:
                        raise ValueError(
                            "Unsupported normalization method. Expected 'min_max', 'z_score', 'robust_scaler' or 'log_transform'!"
                        )

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

            train_differences = {
                param: np.mean(train_differences[param]) for param in parameters
            }
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

            val_loss, val_differences, val_reconst_losses, val_kl_losses = (
                self.validate()
            )
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

            if isinstance(self.model, VariationalAutoencoder):
                with torch.no_grad():
                    _, z_mean, _ = self.model.forward(inputs)
                    z_norm = z_mean.norm(dim=1).mean().item()
                    z_std = torch.std(z_mean, dim=0).mean().item()
                    z_abs_mean = torch.mean(torch.abs(z_mean), dim=0).mean().item()

                if self.run:
                    self.run[f"latent/mean_norm"].append(
                        z_norm
                    )  # ha < 0.1, akkor baj van (összeomlik a látenstér)
                    self.run[f"latent/avg_std_dim"].append(
                        z_std
                    )  # ha ez 0.0 körül van, nem használja a dimenziókat
                    self.run[f"latent/avg_abs_mean"].append(
                        z_abs_mean
                    )  # ha minden z_mean közel 0 ez is jelezhet összeomlást

                    self.run[f"train/total_loss"].append(average_loss)
                    self.run[f"train/reconstruction_loss"].append(reconst_loss.item())
                    self.run[f"train/KL_divergence_loss"].append(kl_div.item())
                    self.run[f"train/KL_scaled_loss"].append((self.beta * kl_div).item())
                    self.run[f"learning_rate"].append(
                        self.optimizer.param_groups[0]["lr"]
                    )
                    self.run[f"validation/total_loss"].append(val_loss)
                    self.run[f"validation/reconstruction_loss"].append(
                        np.mean(val_reconst_losses)
                    )
                    self.run[f"validation/KL_divergence_loss"].append(
                        np.mean(val_kl_losses)
                    )
                    self.run[f"beta"].append(self.beta)

                    for param, avg_value in train_differences.items():
                        self.run[f"train/{param}_average"].append(avg_value)

                    for param, avg_value in val_differences.items():
                        self.run[f"validation/{param}_average"].append(avg_value)

                    self.run[f"train/diff_average"].append(train_total_difference)
                    self.run[f"validation/diff_average"].append(
                        val_differences["diff_average"]
                    )
            elif isinstance(self.model, MaskedAutoencoder):
                if self.run:
                    self.run[f"train/loss"].append(average_loss)
                    self.run[f"learning_rate"].append(
                        self.optimizer.param_groups[0]["lr"]
                    )
                    self.run[f"validation/loss"].append(val_loss)
                    self.run[f"beta"].append(self.beta)

                    for param, avg_value in train_differences.items():
                        self.run[f"train/{param}_average"].append(avg_value)

                    for param, avg_value in val_differences.items():
                        self.run[f"validation/{param}_average"].append(avg_value)

                    self.run[f"train/diff_average"].append(train_total_difference)
                    self.run[f"validation/diff_average"].append(
                        val_differences["diff_average"]
                    )

            # Kiírás az egyes train és validation eltérésekre
            train_differences_str = " | ".join(
                [f"{param}: {train_differences[param]:.6f}" for param in parameters]
            )
            val_differences_str = " | ".join(
                [f"{param}: {val_differences[param]:.6f}" for param in parameters]
            )

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Current Learning Rate: {current_lr:.6f}"
            )
            print(f"Train Loss: {average_loss:.4f}")
            print(
                f"Train Differences: {train_differences_str}\n **Total Avg: {train_total_difference:.6f}**"
            )
            print(f"Validation Loss: {val_loss:.4f}")
            print(
                f"Validation Differences: {val_differences_str}\n **Total Avg: {val_differences['diff_average']:.6f}**"
            )

        if self.run:
            self.run.stop()

        self.plot_losses()

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_reconst_losses = []
        val_kl_losses = []
        val_differences = {param: [] for param in parameters}
        val_total_differences = []

        with torch.no_grad():
            for data in self.valloader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    outputs, z_mean, z_log_var = self.model.forward(inputs)
                    loss, reconst_loss, kl_div = self.model.loss(
                        inputs, outputs, z_mean, z_log_var, self.beta
                    )
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, _ = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError("Unsupported model type. Expected VAE or MAE!")

                val_reconst_losses.append(reconst_loss.item())
                val_kl_losses.append(kl_div.item())

                if validation_method == "denormalized":
                    if normalization == "z_score":
                        inputs = self.dp.z_score_denormalize(
                            data=inputs,
                            data_mean=self.data_mean,
                            data_std=self.data_std,
                        )
                        outputs = self.dp.z_score_denormalize(
                            data=outputs,
                            data_mean=self.data_mean,
                            data_std=self.data_std,
                        )
                    elif normalization == "min_max":
                        inputs = self.dp.denormalize(
                            data=inputs, data_min=self.data_min, data_max=self.data_max
                        )
                        outputs = self.dp.denormalize(
                            data=outputs, data_min=self.data_min, data_max=self.data_max
                        )
                    elif normalization == "robust_scaler":
                        inputs = self.dp.robust_scaler_denormalize(
                            data=inputs, scaler=self.scaler
                        )
                        outputs = self.dp.robust_scaler_denormalize(
                            data=outputs, scaler=self.scaler
                        )
                    elif normalization == "log_transform":
                        inputs = self.dp.log_transform_denormalize(data=inputs)
                        outputs = self.dp.log_transform_denormalize(data=outputs)
                    else:
                        raise ValueError(
                            "Unsupported normalization method. Expected 'min_max', 'z_score', 'robust_scaler' or 'log_transform'!"
                        )

                val_loss += loss.item()
                batch_differences = reconstruction_accuracy(
                    inputs, outputs, self.selected_columns
                )

                for param in parameters:
                    if param in batch_differences:
                        val_differences[param].append(batch_differences[param])

                if "diff_average" in batch_differences:
                    val_total_differences.append(batch_differences["diff_average"])

        val_differences = {
            param: np.mean(val_differences[param]) for param in parameters
        }
        val_total_difference = np.mean(val_total_differences)

        globals()["val_diff_average"] = val_total_difference

        for param, avg_value in val_differences.items():
            globals()[f"val_diff_{param}_average"] = avg_value

        val_differences["diff_average"] = val_total_difference

        return (
            val_loss / len(self.valloader),
            val_differences,
            val_reconst_losses,
            val_kl_losses,
        )

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

        save_path = f"data_bottleneck/VAE_1/single_manoeuvres_z_score/{folder_name}.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bottleneck_outputs)  # Mentés NumPy formátumban

        # Vizualizáció
        vs = Visualise(
            bottleneck_outputs=bottleneck_outputs,
            labels=labels,
            model_name=self.model_name,
            label_mapping=self.label_mapping,
            sign_change_indices=self.sign_change_indices,
        )
        latent_data, label = vs.visualize_with_tsne()
        if num_manoeuvres == 1:
            outlier_indices = detect_outliers(latent_data[2500:])
            filtered_data = np.delete(latent_data[2500:], outlier_indices, axis=0)
            print(f"Reduced data shape: {filtered_data.shape}")
            time_labels = np.arange(len(filtered_data))
            filtered_data, filtered_labels = filter_inconsistent_points(
                filtered_data, time_labels
            )
            print(f"Reduced data shape: {filtered_data.shape}")
            filtered_latent_data, filtered_labels = filter_outliers_by_grid(
                filtered_data, filtered_labels
            )
            print(f"Reduced data shape: {filtered_data.shape}")
            # filtered_latent_data = remove_data_step_by_step(
            #     filtered_data, file_name="rész_szűrés"
            # )
            # print(f"Reduced data shape: {filtered_latent_data.shape}")
            # filtered_latent_data = remove_redundant_data(filtered_data)
            # create_comparison_heatmaps(
            #     filtered_data, filtered_latent_data, file_name="rész_szűrés_heatmap"
            # )
            plot_removed_data(
                latent_data, filtered_latent_data, file_name="teljes_szűrés"
            )
            create_comparison_heatmaps(
                latent_data, filtered_latent_data, file_name="teljes_szűrés_heatmap"
            )
        else:
            mf = ManoeuvresFiltering(
                reduced_data=latent_data,
                bottleneck_data=bottleneck_outputs,
                labels=labels,
                label_mapping=self.label_mapping,
            )
            filtered_reduced_data, filtered_labels = mf.filter_manoeuvres()
            if filtered_reduced_data == 2:
                filtered_latent_data = filtered_reduced_data
            else:
                filtered_latent_data = vs.calculate_tsne(filtered_reduced_data)
            plot_removed_data(latent_data, filtered_reduced_data)
            create_comparison_heatmaps(latent_data, filtered_reduced_data)

        removed_data_procentage = (
            (bottleneck_outputs.shape[0] - filtered_latent_data.shape[0]) * 100
        ) / bottleneck_outputs.shape[0]
        print(
            f"Final reduced bottleneck shape: {filtered_latent_data.shape}, Removed data: {removed_data_procentage:.2f}%"
        )

        if validation_method == "denormalized":
            if normalization == "min_max":
                whole_input = self.dp.denormalize(
                    data=whole_input, data_min=self.data_min, data_max=self.data_max
                )
                whole_output = self.dp.denormalize(
                    data=whole_output, data_min=self.data_min, data_max=self.data_max
                )
            elif normalization == "z_score":
                whole_input = self.dp.z_score_denormalize(
                    data=whole_input, data_mean=self.data_mean, data_std=self.data_std
                )
                whole_output = self.dp.z_score_denormalize(
                    data=whole_output, data_mean=self.data_mean, data_std=self.data_std
                )
            elif normalization == "robust_scaler":
                whole_input = self.dp.robust_scaler_denormalize(
                    data=whole_input, scaler=self.scaler
                )
                whole_output = self.dp.robust_scaler_denormalize(
                    data=whole_output, scaler=self.scaler
                )
            elif normalization == "log_transform":
                whole_input = self.dp.log_transform_denormalize(data=whole_input)
                whole_output = self.dp.log_transform_denormalize(data=whole_output)
            else:
                raise ValueError(
                    "Unsupported normalization method. Expected 'min_max', 'z_score', 'robust_scaler' or 'log_transform'!"
                )

        # Saliency map
        print("\n--- Saliency map számítása ---")
        saliencies = []
        for i in range(whole_input.shape[0]):
            saliency = compute_saliency_map(
                whole_input[i], whole_output[i], self.device
            )
            saliencies.append(saliency)

        avg_saliency = torch.stack(saliencies).mean(dim=0)
        plot_saliency_map(self.all_columns, avg_saliency.numpy())

        # Accuracy
        reconstruction_accuracy(whole_input, whole_output, self.selected_columns)
        return latent_data, label, bottleneck_outputs, labels, avg_saliency

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
