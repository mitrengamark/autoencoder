import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout):
        """
        :param input_dim: Bemeneti dimenziók száma.
        :param latent_dim: Latens tér dimenziója.
        :param hidden_dims: A rejtett rétegek méreteit tartalmazó lista.
        :param dropout: Dropout arány.
        """
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(self.dropout))
            prev_dim = h_dim
        self.shared_encoder = nn.Sequential(*encoder_layers)

        self.encoder_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = self.latent_dim
        for h_dim in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        

    def encode(self, x):
        shared_output = self.shared_encoder(x)
        z_mean = self.encoder_mu(shared_output)
        z_log_var = self.encoder_logvar(shared_output)
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

    def loss(self, x, x_, z_mean, z_log_var, beta):
        reconst_loss = F.mse_loss(x_, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss = reconst_loss + beta * kl_div
        return loss, reconst_loss, beta * kl_div