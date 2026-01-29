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
    

class EmbeddingEncoder(nn.Module):
    """MLP encoder for pre-computed text embeddings"""
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionModel(nn.Module):
    """Multimodal fusion model combining tabular and embedding encoders.

    Architecture:
        Tabular data        Embedding data
            |                   |
        TabularEncoder      EmbeddingEncoder
            |                   |
        [hidden_dim]        [hidden_dim]
            \                   /
             \                 /
              \               /
                    concat
                        |
                    PredictionHead
                        |
                    Probability
    """

    def __init__(
        self,
        tabular_input_dim: int,
        embedding_input_dim: int,
        categ_sizes: dict[str, int], # {column_name: num_unique_values}
        model_config: ModelConfig | None = None,
    ):
        """
        Args:
            tabular_input_dim: Total number of columns in tabular data, including the categorical ones
            categ_sizes: Mapping of categorical columns to their cardinalities
        """
        super().__init__()

        self.config = ModelConfig() if model_config is None else model_config

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=self.config.tabular_hidden_dims,
            dropout=self.config.tabular_dropout,
            categ_sizes=categ_sizes
        )

        # Embedding encoder
        self.embedding_encoder = EmbeddingEncoder(
            input_dim=embedding_input_dim,
            hidden_dims=self.config.embedding_hidden_dims,
            dropout=self.config.embedding_dropout,
        )

        # Learnable embedding for missing text embeddings
        self.missing_embedding = nn.Parameter(
            torch.randn(embedding_input_dim)
        )

        # Compute fusion input dimension
        fusion_input_dim = (
            self.tabular_encoder.output_dim + self.embedding_encoder.output_dim
        )

        # Prediction head
        layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in self.config.fusion_hidden_dims:
            layers.append(MLPBlock(prev_dim, hidden_dim, self.config.fusion_dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.prediction_head = nn.Sequential(*layers)


    def forward(
        self,
        tabular_cont_feats: torch.Tensor,
        tabular_categ_feats: dict[str, torch.Tensor],
        embedding_feats: torch.Tensor,
        has_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Encode tabular features
        tab_hidden = self.tabular_encoder(tabular_cont_feats, tabular_categ_feats)

        # Encode embedding features
        embedding_feats = self._fill_missing_embeddings(embedding_feats, has_embedding)
        emb_hidden = self.embedding_encoder(embedding_feats)

        # Fuse and predict
        fused = torch.cat([tab_hidden, emb_hidden], dim=-1)
        logits = self.prediction_head(fused)
        return logits.squeeze(-1)


    def _fill_missing_embeddings(
        self,
        embedding_feats: torch.Tensor,  # (batch, emb_dim)
        has_embedding: torch.Tensor,    # (batch,) bool
    ) -> torch.Tensor:
        """Replace missing embeddings with learnable parameter."""
        # (emb_dim,) -> (batch, emb_dim)
        missing = self.missing_embedding.expand(embedding_feats.shape[0], -1)
        # (batch,) -> (batch, emb_dim)
        mask = has_embedding.unsqueeze(-1).expand_as(embedding_feats)
        return torch.where(mask, embedding_feats, missing)