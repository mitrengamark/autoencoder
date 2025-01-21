import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, max_len=5000):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.positional_encoding = self.create_positional_encoding(input_dim, max_len)
        self.output_layer = nn.Linear(input_dim, input_dim)
        # self.query_layer = nn.Linear(input_dim, input_dim)
        # self.key_layer = nn.Linear(input_dim, input_dim)
        # self.value_layer = nn.Linear(input_dim, input_dim)

    def create_positional_encoding(self, input_dim, max_len):
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # query = self.query_layer(x)
        # key = self.key_layer(x)
        # value = self.value_layer(x)
        # attention = torch.matmul(query, key.transpose(-2, -1))
        # attention = attention / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        # attention = torch.softmax(attention, dim=-1)
        # output = torch.matmul(attention, value)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x + self.positional_encoding[:, :x.size(0), :].to(x.device)
        output, _ = self.multi_head_attention(x, x, x)
        output = self.output_layer(output)  # Kimenet tovább transzformálása
        output = output.squeeze(1)
        return output