"""Training loop and utilities for multimodal ED prediction."""
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml_common.eval import auc_scores
from preduce.emerg.config import TrainConfig
from preduce.emerg.model import FusionModel
    

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: FusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        target = batch.pop("target")
        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(
    model: FusionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> dict:
    """Evaluate model on a dataset.

    Returns:
        Dictionary containing loss, preds, labels, and metrics.
    """
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            target = batch.pop("target")
            logits = model(**batch)
            loss = criterion(logits, target)
            total_loss += loss.item()

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            labels.append(target.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = auc_scores(labels, preds)

    return {
        "loss": total_loss / len(dataloader),
        "preds": preds,
        "labels": labels,
        **metrics,
    }


def train(
    model: FusionModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    train_config: TrainConfig | None = None,
    save_dir: str | None = None,
) -> dict:
    """Full training loop with early stopping.

    Returns:
        Dictionary containing training history and best model path.
    """
    config = TrainConfig() if train_config is None else train_config
    save_dir = './' if save_dir is None else save_dir

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Setup loss function with optional class weighting
    if config.pos_weight is not None:
        pos_weight = torch.tensor([config.pos_weight])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    best_auroc = 0.0
    patience_counter = 0
    best_model_path = f"{save_dir}/best_model.pt"

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        history["train_loss"].append(train_loss)

        # Validate
        val_metrics = evaluate(model, valid_loader, criterion)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val AUROC: {val_metrics['auroc']:.4f}, "
            f"Val AUPRC: {val_metrics['auprc']:.4f}"
        )

        # Check for improvement
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            patience_counter = 0

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": best_auroc,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return {
        "best_auroc": best_auroc,
        "best_model_path": best_model_path,
        "history": history,
    }
