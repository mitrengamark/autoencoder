import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
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
        # x = x.unsqueeze(0)  # MultiheadAttention batch dimenziót vár
        # output, _ = self.multi_head_attention(x, x, x)
        # return output.squeeze(0)
        return output