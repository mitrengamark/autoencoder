import torch
import matplotlib.pyplot as plt
import numpy as np
from Factory.variational_autoencoder import VariationalAutoencoder
from Factory.masked_autoencoder import MaskedAutoencoder
from Factory.scheduler import scheduler_maker
from Analyse.decrase_dim import Visualise
from Analyse.validation import reconstruction_accuracy
from Synthesis.data_synthesis import remove_redundant_data
from Synthesis.heat_map import create_comparison_heatmaps
from Synthesis.manoeuvres_filtering import ManoeuvresFiltering


class Training:
    def __init__(
        self,
        trainloader=None,
        valloader=None,
        testloader=None,
        test_mode=None,
        optimizer=None,
        model=None,
        labels=None,
        num_epochs=None,
        device=None,
        scheduler=None,
        beta_min=None,
        step_size=None,
        gamma=None,
        patience=None,
        warmup_epochs=None,
        initial_lr=None,
        max_lr=None,
        final_lr=None,
        model_path=None,
        data_min=None,
        data_max=None,
        run=None,
        data_mean=None,
        data_std=None,
        hyperopt=None,
        tolerance=None,
        label_mapping=None,
        sign_change_indices=None,
        num_manoeuvres=None,
        n_clusters=None,
        use_cosine_similarity=None,
        model_name=None,
    ):

        # Adatbetöltők
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # Modell és optimalizáló
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer

        # Hiperparaméterek és tanítási konfigurációk
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.beta_min = beta_min
        self.beta = 0  # Dinamikus érték lesz tanítás során
        self.tolerance = tolerance
        self.hyperopt = hyperopt
        self.test_mode = test_mode

        # Modell mentéséhez szükséges útvonal
        self.model_path = model_path

        # Címkék
        self.labels = labels
        self.label_mapping = label_mapping

        # Adatok normalizálásához szükséges statisztikák
        self.data_min = data_min
        self.data_max = data_max
        self.data_mean = data_mean
        self.data_std = data_std

        # Egyéb
        self.run = run
        self.device = device
        self.sign_change_indices = sign_change_indices
        self.num_manoeuvres = num_manoeuvres
        self.n_clusters = n_clusters
        self.use_cosine_similarity = use_cosine_similarity

        # Loss értékek tárolása
        self.losses = []
        self.reconst_losses = []
        self.kl_losses = []
        self.val_losses = []

    def train(self):
        self.model.train()
        if self.hyperopt == 0:
            scheduler = scheduler_maker(
                scheduler=self.scheduler,
                optimizer=self.optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
                num_epochs=self.num_epochs,
                patience=self.patience,
                warmup_epochs=self.warmup_epochs,
                initial_lr=self.initial_lr,
                max_lr=self.max_lr,
                final_lr=self.final_lr,
            )

        for epoch in range(self.num_epochs):
            loss_per_episode = 0
            reconst_loss_per_epoch = 0
            kl_loss_per_epoch = 0
            train_accuracy = 0
            self.beta = min(1.0, epoch / self.beta_min)
            if epoch == 0:
                self.beta = 1 / self.beta_min

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

                accuracy = reconstruction_accuracy(inputs, outputs, self.tolerance)
                train_accuracy += accuracy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                loss_per_episode += loss.item()

            average_loss = loss_per_episode / len(self.trainloader)
            average_accuracy = train_accuracy / len(self.trainloader)
            self.losses.append(round(average_loss, 4))
            if isinstance(self.model, VariationalAutoencoder):
                self.reconst_losses.append(
                    reconst_loss_per_epoch / len(self.trainloader)
                )
                self.kl_losses.append(kl_loss_per_epoch / len(self.trainloader))

            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)

            if self.hyperopt == 0:
                if self.scheduler == "ReduceLROnPlateau":
                    scheduler.step(average_loss)
                else:
                    scheduler.step()
            elif self.hyperopt == 1:
                self.scheduler.step()
            else:
                raise ValueError("Unsupported hyperopt value. Expected 0 or 1.")

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr:.6f}")

            if self.run:
                self.run[f"train/loss"].append(average_loss)
                self.run[f"train/accuracy"].append(average_accuracy)
                self.run[f"learning_rate"].append(self.optimizer.param_groups[0]["lr"])
                self.run[f"validation/loss"].append(val_loss)
                self.run[f"validation/accuracy"].append(val_accuracy)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {average_loss:.4f}, Train Accuracy: {average_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
            )

        if self.run:
            self.run.stop()
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
                    loss, _, _ = self.model.loss(
                        inputs, outputs, z_mean, z_log_var, self.beta
                    )
                elif isinstance(self.model, MaskedAutoencoder):
                    outputs, _, _ = self.model.forward(inputs)
                    loss = self.model.loss(inputs, outputs)
                else:
                    raise ValueError("Unsupported model type. Expected VAE or MAE!")

                val_loss += loss.item()
                val_accuracy += reconstruction_accuracy(inputs, outputs, self.tolerance)
        return val_loss / len(self.valloader), val_accuracy / len(self.valloader)

    def test(self):
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
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
            num_manoeuvres=self.num_manoeuvres,
            n_clusters=self.n_clusters,
            use_cosine_similarity=self.use_cosine_similarity,
        )
        latent_data = vs.visualize_bottleneck()
        # vs.kmeans_clustering()

        mf = ManoeuvresFiltering(reduced_data=latent_data, labels=labels)
        mf.filter_manoeuvres()

        # Adat eltávolítás és szintetizálás
        filtered_latent_data = remove_redundant_data(latent_data)
        create_comparison_heatmaps(latent_data, filtered_latent_data)

        # Denormalizáció

        accuracy = reconstruction_accuracy(whole_input, whole_output, self.tolerance)
        print(f"Test Accuracy: {accuracy:.2f}%")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

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
