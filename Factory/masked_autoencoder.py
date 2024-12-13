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
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim))  # Javítás: Méret explicit kezelése.
        # Indok: Dinamikusan kezeli a bemenet méretének megfelelő inicializációt.

    def masking(self, x):
        mask = torch.rand_like(x) > self.mask_ratio  # Javítás: Maszkolás explicit bemenettel.
        # Indok: Az eredeti input helyett dinamikusan a 'forward' bemenetét használjuk.
        masked_input = x * mask
        return masked_input, mask

    def encoder(self, x):
        masked_input, mask = self.masking(x)
        # print(masked_input)
        # print(mask)
        encoded = self.self_attention(masked_input)
        return encoded, mask, masked_input

    def decoder(self, encoded, mask, original_input):
        encoded += self.positional_encoding  # Javítás: Positional encoding explicit hozzáadása.
        # Indok: Ez a dekódoláshoz kontextuális információkat biztosít.
        decoded = self.decoder_self_attention(encoded)
        reconstructed = self.reconstruction_layer(decoded)
        reconstructed_input = torch.where(mask, original_input, reconstructed)  # Javítás: Maszkolás megfelelő kezelése.
        # Indok: Ez biztosítja, hogy csak a maszkolt értékek kerülnek helyettesítésre.
        return reconstructed_input

    def forward(self, x):
        encoded, mask, masked_input = self.encoder(x)  # Javítás: Bemenet dinamikus kezelése.
        # Indok: Az eredeti osztály fix 'self.input'-ot használt, ami nem dinamikus.
        reconstructed = self.decoder(encoded, mask, x)
        return reconstructed, masked_input, encoded

    def loss(self, input, reconstructed):
        loss_fn = nn.MSELoss()  # Javítás: Loss funkció inicializálása a 'forward'-ban kívül.
        # Indok: Jobb olvashatóság és újrafelhasználhatóság érdekében.
        return loss_fn(input, reconstructed)


# # Tesztelés
# input_tensor = torch.randn(1, 10)
# mae = MaskedAutoencoder(input_dim=10, mask_ratio=0.75)
# reconstructed_output = mae(input_tensor)  # Javítás: Dinamikus bemenet a 'forward'-ban.
# # Indok: Ez rugalmassá teszi a modellt többféle bemenettel való használatra.
# print("Bemenet:\n", input_tensor)
# print("Helyreállított bemenet:\n", reconstructed_output)