import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_0, hidden_dim_1):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_0 = hidden_dim_0
        self.hidden_dim_1 = hidden_dim_1
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim_0),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim_0,self.hidden_dim_1),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim_1, self.latent_dim * 2))

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim_1),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim_1, self.hidden_dim_0),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim_0, self.input_dim),
                                     nn.Sigmoid())
        

    def encode(self, x):
        encoder = self.encoder(x)
        z_mean = encoder[:, :self.latent_dim]
        z_log_var = encoder[:, self.latent_dim:]
        return z_mean, z_log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(z_log_var / 2)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_ = self.decode(z)
        return x_, z_mean, z_log_var

    def loss(self, x, x_, z_mean, z_log_var):
        reconst_loss = F.mse_loss(x_, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss = reconst_loss + kl_div
        return loss, reconst_loss, kl_div