import math
import torch
import torch.nn as nn


def create_future_mask(size):
    """
    Create a future mask of the given size.

    Args:
    size (int): The size of the square mask.

    Returns:
    torch.Tensor: A boolean tensor of shape (size, size) representing the future mask.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


def create_mask(inputs):
    """
    Create a mask for the input data.

    Args:
    _inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).

    Returns:
    torch.Tensor: A boolean tensor of shape (batch_size, seq_length) representing the mask.
    """
    mask = (torch.sum(inputs, dim=-1) < -100).bool()
    return mask


class SimplePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()

        # Register the positional encoding as a buffer
        encoding = torch.arange(max_len).unsqueeze(1).repeat(1, d_model).float()
        encoding /= math.sqrt(d_model)
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        # Adjusting for batch-first tensors
        batch_size, seq_len, _ = x.size()
        return x + self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)


# Simplified Transformer Model
class Transformer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.2,
    ):
        super().__init__()
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers

        self.dropout = dropout
        # Add MLP embedding
        self.mlp = nn.Linear(input_size, hidden_size)
        self.pos_encoder = SimplePositionalEncoding(hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="relu",
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.outlayer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp(x)
        x = self.relu(x)
        x = self.pos_encoder(x)
        # Creating masks
        future_mask = create_future_mask(x.size(1)).to(x.device)
        padding_mask = create_mask(x)

        x = self.transformer_encoder(
            x, mask=future_mask, src_key_padding_mask=padding_mask
        )
        x = self.outlayer(x)

        return x
