import torch
import torch.nn as nn
from Factory.self_attention_factory import SelfAttention

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, mask_ratio):
        super(MaskedAutoencoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.self_attention = SelfAttention(input_dim)
        self.decoder_self_attention = SelfAttention(input_dim)
        self.reconstruction_layer = nn.Linear(input_dim, input_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim)) 

    def masking(self, x):
        mask = torch.rand_like(x) > self.mask_ratio
        masked_input = x * mask
        return masked_input, mask

    def encoder(self, x):
        masked_input, mask = self.masking(x)
        encoded = self.self_attention(masked_input)
        return encoded, mask, masked_input

    def decoder(self, encoded, mask, original_input):
        encoded += self.positional_encoding
        decoded = self.decoder_self_attention(encoded)
        reconstructed = self.reconstruction_layer(decoded)
        reconstructed_input = torch.where(mask, original_input, reconstructed)
        return reconstructed_input

    def forward(self, x):
        encoded, mask, masked_input = self.encoder(x)
        reconstructed = self.decoder(encoded, mask, x)
        return reconstructed, masked_input, encoded

    def loss(self, input, reconstructed):
        loss_fn = nn.SmoothL1Loss()
        return loss_fn(input, reconstructed)