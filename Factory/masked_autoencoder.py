import torch
import torch.nn as nn
from Factory.self_attention_factory import SelfAttention
from Config.load_config import mask_ratio, bottleneck_dim, num_heads, dropout


class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(MaskedAutoencoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.bottleneck_dim = bottleneck_dim
        self.self_attention = SelfAttention(input_dim)
        # self.encoder_bottleneck = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(input_dim, bottleneck_dim),
        #     nn.ReLU()
        # )
        # self.decoder_bottleneck = nn.Sequential(
        #     nn.Linear(bottleneck_dim, input_dim),
        #     nn.ReLU()
        # )
        self.decoder_self_attention = SelfAttention(input_dim, num_heads, dropout)
        self.positional_encoding = PositionalEncoding(input_dim)
        self.reconstruction_layer = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, input_dim)
        )

    def masking(self, x):
        dynamic_mask_ratio = torch.rand(1).item() * self.mask_ratio
        mask = torch.rand_like(x) > dynamic_mask_ratio
        masked_input = x * mask
        return masked_input, mask

    def encoder(self, x):
        masked_input, mask = self.masking(x)
        encoded = self.self_attention(
            masked_input,
        )
        # bottleneck_output = self.encoder_bottleneck(encoded)  # Dimenziócsökkentés
        return encoded, mask, masked_input

    def decoder(self, encoded, mask, original_input):
        # expanded_output = self.decoder_bottleneck(bottleneck_output)  # Dimenzió visszaállítása
        expanded_output = self.positional_encoding(encoded)
        decoded = self.decoder_self_attention(expanded_output)
        reconstructed = self.reconstruction_layer(decoded)
        reconstructed_input = torch.where(mask, original_input, reconstructed)
        return reconstructed_input

    def forward(self, x):
        bottleneck_output, mask, masked_input = self.encoder(x)
        reconstructed = self.decoder(bottleneck_output, mask, x)
        return reconstructed, masked_input, bottleneck_output

    def loss(self, input, reconstructed):
        loss_fn = nn.SmoothL1Loss()
        return loss_fn(input, reconstructed)


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / input_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
