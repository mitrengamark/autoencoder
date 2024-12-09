import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        # Javítás: Az input dimenzió explicit megadása helyett a bemenet méretét várjuk el.
        # Indok: Ez általánosabbá teszi az osztályt különböző méretű bemenetekhez.
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        # Javítás: Korrekt transzponálás a mátrixszorzáshoz (key.T helyett key.transpose).
        # Indok: Ez a PyTorch standardja, és biztosítja a dimenziók helyes kezelését.
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention, value)
        return output