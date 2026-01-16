"""PyTorch Lightning model module for sentiment classification.

This module defines the SentimentClassifier, a LightningModule wrapper around
a HuggingFace transformer model fine-tuned for binary sentiment classification
on movie reviews.

The model:
- Uses a pre-trained DistilBERT (or similar) as the backbone
- Adds a classification head for binary sentiment (positive/negative)
- Tracks accuracy metrics for train/val/test splits
- Saves misclassified examples for error analysis
"""

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification

from ml_ops_project.visualize import log_mismatches_to_wandb


class SentimentClassifier(pl.LightningModule):
    """Lightning module for binary sentiment classification using transformers.

    Wraps a HuggingFace AutoModelForSequenceClassification with PyTorch Lightning
    training logic, metric tracking, and mismatch logging for debugging.

    Args:
        model_name: HuggingFace model identifier (e.g., "distilbert-base-uncased").
        inference_mode: Whether to load in inference mode (from checkpoint) or training mode.
        optimizer_cfg: Hydra/OmegaConf config for optimizer instantiation.
            Must contain `_target_` key pointing to optimizer class. Required in training mode.

    Attributes:
        model: The underlying HuggingFace transformer model.
        criterion: Loss function (CrossEntropyLoss for classification).
        train_acc, val_acc, test_acc: TorchMetrics Accuracy instances per split.
    """

    def __init__(self, model_name: str, inference_mode: bool, optimizer_cfg: DictConfig = None) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg

        # Save hyperparameters to checkpoint for reproducibility and easy loading
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        if not inference_mode:  # Load the pre-trained model for binary classification (2 labels)
            if not optimizer_cfg:
                raise AttributeError("A 'NoneType' cannot be passed as optimizer config, when in training mode.")

            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            self.model.train()
            self.criterion = torch.nn.CrossEntropyLoss()

        else:
            # Fetch the saved model.
            path = "../models/epoch-epoch=01-val_accuracy=0.842.ckpt"
            self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
            self.model.eval()

        # Separate metric instances per split to avoid state leakage between train/val/test
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer model.

        Args:
            input_ids: Tokenized input IDs, shape [batch_size, seq_len].
            attention_mask: Mask for padding tokens, shape [batch_size, seq_len].
            labels: Optional ground truth labels for loss computation, shape [batch_size].

        Returns:
            HuggingFace model output containing logits and (if labels provided) loss.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute one training step.

        Args:
            batch: Dictionary with keys 'input_ids', 'attention_mask', 'labels'.
            batch_idx: Index of the current batch (unused but required by Lightning).

        Returns:
            Scalar loss tensor for backpropagation.
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)  # [batch_size]

        # Update and log metrics
        acc = self.train_acc(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute one validation step.

        Args:
            batch: Dictionary with keys 'input_ids', 'attention_mask', 'labels'.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor (for logging, not backprop).
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)  # [batch_size]

        # Update and log metrics (computed at epoch end)
        # Log to WandB Table (only for the first batch of validation)
        if batch_idx == 0 and self.logger is not None:
            log_mismatches_to_wandb(
                batch=batch, 
                preds=preds, 
                tokenizer=self.tokenizer, 
                table_name="val_mismatches"
            )
            
        # Log validation accuracy and loss
        self.val_acc(preds, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", self.val_acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute one test step.

        Args:
            batch: Dictionary with keys 'input_ids', 'attention_mask', 'labels'.
            batch_idx: Index of the current batch (unused but required by Lightning).

        Returns:
            Scalar loss tensor (for logging).
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)  # [batch_size]

        # Save all misclassified test examples for thorough error analysis
        save_mismatches(batch, preds, folder="mismatches_test")

        # Update and log metrics
        self.test_acc(preds, batch["labels"])
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Instantiate the optimizer from Hydra config.

        Uses hydra.utils.instantiate to create the optimizer specified in
        the config (e.g., Adam, SGD) with the configured hyperparameters.

        Returns:
            Configured PyTorch optimizer instance.
        """
        return hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
