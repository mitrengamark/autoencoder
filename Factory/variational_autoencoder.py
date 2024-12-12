import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_0, hidden_dim_1, beta, dropout):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_0 = hidden_dim_0
        self.hidden_dim_1 = hidden_dim_1
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout = dropout

        self.shared_encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim_0),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(self.hidden_dim_0,self.hidden_dim_1),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout))
        
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.hidden_dim_1, self.latent_dim))
        
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim_1, self.latent_dim))

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim_1),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim_1, self.hidden_dim_0),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim_0, self.input_dim),
                                     nn.Sigmoid())
        

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

    def loss(self, x, x_, z_mean, z_log_var):
        reconst_loss = F.mse_loss(x_, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss = reconst_loss + self.beta * kl_div
        return loss, reconst_loss, self.beta * kl_div