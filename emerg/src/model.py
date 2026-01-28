"""Multimodal fusion model"""

import torch
import torch.nn as nn

from preduce.emerg.config import ModelConfig


class MLPBlock(nn.Module):
    """MLP block with BatchNorm, activation, and dropout."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    

class TabularEncoder(nn.Module):
    """MLP encoder for tabular features.

    Learn embeddings for categorical columns.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int], 
        dropout: float,
        categ_sizes: dict[str, int], # {column_name: num_unique_values}
    ):
        """
        Args:
            input_dim: Total number of columns, including the categorical ones
            categ_sizes: Mapping of categorical columns to their cardinalities
        """
        super().__init__()

        # embedding for categorical columns
        self.categ_embs = nn.ModuleDict({
            col: nn.Embedding(nunique, self.embedding_size(nunique))
            for col, nunique in categ_sizes.items()
        })

        # adjust the input dimension to include categorical embeddings
        total_emb_dim = sum(emb.embedding_dim for emb in self.categ_embs.values())
        input_dim = input_dim - len(categ_sizes) + total_emb_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(
        self, 
        cont_feats: torch.Tensor,
        categ_feats: dict[str, torch.Tensor],  # {column_name: indices}
    ) -> torch.Tensor:
        """
        Args:
            cont_feats: Continuous column features
            categ_feats: Mapping of categorical columns to their category features, represented as indices
        """
        categ_embed = [emb(categ_feats[col]) for col, emb in self.categ_embs.items()]
        x = torch.cat([cont_feats, *categ_embed], dim=-1)
        return self.encoder(x)

    @staticmethod
    def embedding_size(cardinality: int, max_dim: int = 600) -> int:
        """Compute embedding size for categorical features based on cardinality.
        This Heuristics is based on fast.ai
        """
        return min(max_dim, round(1.6 * cardinality ** 0.56))