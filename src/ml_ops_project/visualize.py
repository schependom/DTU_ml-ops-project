"""Visualization utilities for model error analysis.

This module provides functions for saving and analyzing model predictions,
particularly focusing on misclassified examples for debugging and improving
model performance.
"""

import os

import pandas as pd
import torch
import wandb


def save_mismatches(
    batch: dict[str, torch.Tensor],
    preds: torch.Tensor,
    folder: str = "mismatches",
) -> None:
    """Save misclassified examples to a CSV file for error analysis.

    Compares predictions against ground truth labels and appends any
    mismatches to a CSV file. Useful for understanding what types of
    examples the model struggles with.

    Args:
        batch: Dictionary containing at minimum:
            - 'input_ids': Token IDs tensor, shape [batch_size, seq_len]
            - 'labels': Ground truth labels tensor, shape [batch_size]
        preds: Model predictions tensor, shape [batch_size].
        folder: Directory to save the errors CSV. Created if it doesn't exist.

    Note:
        Results are appended to `{folder}/errors.csv`. The CSV contains:
        - input_ids: List of token IDs (can be decoded back to text)
        - label: Ground truth label (0 or 1)
        - predicted: Model's prediction (0 or 1)
    """
    # Move tensors to CPU for pandas compatibility
    labels = batch["labels"].detach().cpu()
    preds = preds.detach().cpu()

    # Find indices where prediction doesn't match ground truth
    mask = preds != labels

    # Only write if there are actual mismatches in this batch
    if mask.any():
        wrong_input_ids = batch["input_ids"][mask].detach().cpu()
        wrong_labels = labels[mask]
        wrong_preds = preds[mask]

        df = pd.DataFrame(
            {
                "input_ids": [ids.tolist() for ids in wrong_input_ids],
                "label": wrong_labels.tolist(),
                "predicted": wrong_preds.tolist(),
            }
        )

        # Create folder if needed and append to CSV (write header only on first write)
        os.makedirs(folder, exist_ok=True)
        df.to_csv(f"{folder}/errors.csv", mode="a", header=not os.path.exists(f"{folder}/errors.csv"), index=False)


def save_mismatches_to_wandb(
    batch: dict[str, torch.Tensor],
    preds: torch.Tensor,
    tokenizer,
    table_name: str = "mismatches",
) -> None:
    """Log misclassified examples to a W&B Table."""
    labels = batch["labels"].detach().cpu()
    preds = preds.detach().cpu()
    input_ids = batch["input_ids"].detach().cpu()

    mask = preds != labels

    if mask.any():
        columns = ["Input Text", "Label", "Predicted"]
        data = []

        wrong_input_ids = input_ids[mask]
        wrong_labels = labels[mask]
        wrong_preds = preds[mask]

        for i in range(len(wrong_labels)):
            text = tokenizer.decode(wrong_input_ids[i], skip_special_tokens=True)
            data.append([text, int(wrong_labels[i]), int(wrong_preds[i])])

        wandb.log({table_name: wandb.Table(columns=columns, data=data)})


def log_mismatches_to_wandb(
    batch: dict[str, torch.Tensor],
    preds: torch.Tensor,
    tokenizer,
    table_name: str = "mismatches",
) -> None:
    """Backward-compatible alias for W&B mismatch logging."""
    save_mismatches_to_wandb(batch=batch, preds=preds, tokenizer=tokenizer, table_name=table_name)
