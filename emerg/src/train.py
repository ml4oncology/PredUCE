"""Training loop and utilities for multimodal ED prediction.

TODO: modality dropout
TODO: auxiliary losses (i.e. contrastive loss)
TODO: modality-specific learning rates
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_common.eval import auc_scores
from preduce.emerg.config import TrainConfig
from preduce.emerg.model import FusionModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%I:%M:%S'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multimodal fusion model."""

    def __init__(
        self,
        model: FusionModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: TrainConfig | None = None,
        save_dir: str | Path | None = None,
    ):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config or TrainConfig()
        self.save_dir = Path(save_dir) if save_dir else Path(".")

        # Setup components
        self._setup_optimizer()
        self._setup_schedulers()
        self._setup_criterion()

        # Training state
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": [],
            "lr": [],
        }
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.current_epoch = 0


    def _setup_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


    def _setup_schedulers(self) -> None:
        # Warmup scheduler (linear warmup from start_factor to 1.0)
        self.warmup_scheduler = None
        if self.config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.config.warmup_epochs,
            )

        # Main scheduler (ReduceLROnPlateau, applied after warmup)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.lr_min,
        )


    def _setup_criterion(self) -> None:
        # Use reduction='none' for per-element loss (needed for multi-task masking)
        if self.config.pos_weight is not None:
            pos_weight = torch.tensor([self.config.pos_weight])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')


    def _compute_masked_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        invalid_value: float = -1,
    ) -> torch.Tensor:
        """Compute loss with per-target masking for multi-task learning.

        Args:
            logits: Model predictions, shape [batch_size, num_tasks]
            target: Ground truth labels, shape [batch_size, num_tasks]
            invalid_value: Value indicating missing/invalid targets (default: -1)

        Returns:
            Scalar loss averaged over valid targets only
        """
        mask = target != invalid_value

        # Handle case where all targets are invalid
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute per-element loss and mask
        loss = self.criterion(logits, target)
        masked_loss = (loss * mask).sum() / mask.sum()
        return masked_loss


    def train(self) -> dict:
        """Run full training loop with early stopping.

        Returns:
            Dictionary containing training history and best/last model paths.
        """
        best_model_path = self.save_dir / "best_model.pt"
        last_model_path = self.save_dir / "last_checkpoint.pt"

        for epoch in tqdm(range(self.current_epoch, self.config.epochs), leave=False):
            self.current_epoch = epoch

            # Train
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self.evaluate(self.valid_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_auprc"].append(val_metrics["auprc"])

            # Step the appropriate learning rate scheduler
            self._step_scheduler(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Log progress
            warmup_indicator = " [warmup]" if epoch < self.config.warmup_epochs else ""
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs}{warmup_indicator} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUROC: {val_metrics['auroc']:.4f}, "
                f"Val AUPRC: {val_metrics['auprc']:.4f}, "
                f"LR: {current_lr:.2e}"
            )

            # Check for improvement and save best model
            if val_metrics["auroc"] > self.best_auroc:
                self.best_auroc = val_metrics["auroc"]
                self.patience_counter = 0
                self.save_checkpoint(best_model_path)
            else:
                self.patience_counter += 1

            # Always save last checkpoint for resuming
            self.save_checkpoint(last_model_path)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "best_auroc": self.best_auroc,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "history": self.history,
        }


    def _train_epoch(self) -> float:
        """Train for a single epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            target = batch.pop("target").float()

            # Skip batch if no valid targets
            if (target == -1).all():
                continue

            self.optimizer.zero_grad()
            logits = self.model(**batch)
            loss = self._compute_masked_loss(logits, target)
            loss.backward()

            # Gradient balancing (before clipping)
            if self.config.grad_balance:
                self._balance_gradients()

            # Gradient clipping
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)


    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model on a dataset.

        Returns:
            Dictionary containing loss, preds, labels, and per-task metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        preds = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                target = batch.pop("target").float()

                # Skip batch if no valid targets
                if (target == -1).all():
                    continue

                logits = self.model(**batch)
                loss = self._compute_masked_loss(logits, target)
                total_loss += loss.item()
                num_batches += 1

                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())
                labels.append(target.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        # Compute metrics per task
        num_tasks = self.model.num_tasks
        metrics = []
        for t in range(num_tasks):
            task_preds = preds[:, t]
            task_labels = labels[:, t]
            mask = task_labels != -1
            if mask.sum() > 0:
                metrics.append(auc_scores(task_labels[mask], task_preds[mask]))

        return {
            "loss": total_loss / max(num_batches, 1),
            "preds": preds,
            "labels": labels,
            "metrics": metrics,
        }


    def _step_scheduler(self, val_loss: float) -> None:
        """Step the appropriate scheduler based on current epoch."""
        if (
            self.warmup_scheduler is not None
            and self.current_epoch < self.config.warmup_epochs
        ):
            self.warmup_scheduler.step()
        else:
            self.scheduler.step(val_loss)


    @torch.no_grad()
    def _balance_gradients(self, epsilon: float = 1e-8) -> None:
        """Balance gradients across modalities.

        Scales each encoder's gradients to the mean norm, preventing
        one modality from dominating the gradient updates.
        """
        tab_norm = self._compute_grad_norm(self.model.tabular_encoder.parameters())
        emb_norm = self._compute_grad_norm(self.model.embedding_encoder.parameters())

        if tab_norm < epsilon or emb_norm < epsilon:
            return

        target_norm = (tab_norm + emb_norm) / 2

        tab_scale = target_norm / tab_norm
        emb_scale = target_norm / emb_norm

        for p in self.model.tabular_encoder.parameters():
            if p.grad is not None:
                p.grad.data.mul_(tab_scale)

        for p in self.model.embedding_encoder.parameters():
            if p.grad is not None:
                p.grad.data.mul_(emb_scale)


    @staticmethod
    def _compute_grad_norm(params) -> float:
        """Compute the total L2 gradient norm for parameters."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auroc": self.best_auroc,
            "patience_counter": self.patience_counter,
            "history": self.history,
        }
        if self.warmup_scheduler is not None:
            checkpoint["warmup_scheduler_state_dict"] = self.warmup_scheduler.state_dict()
        torch.save(checkpoint, path)


    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint to resume training."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1  # resume from next epoch
        self.best_auroc = checkpoint["best_auroc"]
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.history = checkpoint.get("history", self.history)

        if (
            self.warmup_scheduler is not None
            and "warmup_scheduler_state_dict" in checkpoint
        ):
            self.warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
