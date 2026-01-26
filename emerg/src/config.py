"""Configuration classes for the multimodal fusion model."""

from dataclasses import dataclass, field

EMBEDDING_SECTIONS = [
    "active_symptoms",
    "recent_complications",
    "healthcare_utilization",
    "functional_status",
    "medication_risks",
    "psychosocial_risks",
    "clinical_uncertainty",
    "acuity_assessment",
]


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Tabular data encoder
    tabular_hidden_dim: list[int] = field(default_factory=lambda: [256, 128])
    tabular_dropout: float = 0.3

    # Text embedding encoder
    embedding_hidden_dim: list[int] = field(default_factory=lambda: [512, 128])
    embedding_dropout: float = 0.3

    # Fusion
    fusion_hidden_dim: list[int] = field(default_factory=lambda: [64])
    fusion_dropout: float = 0.3


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10  # early stopping
    pos_weight: float | None = None  # computed from data if None
