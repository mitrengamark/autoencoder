import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention, value)
        return output