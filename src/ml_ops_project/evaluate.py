"""Evaluation script for running inference on a trained sentiment classification model.

This module provides a standalone evaluation entry point that:
- Loads a trained model from a checkpoint file
- Runs the test set through the model
- Reports test metrics (accuracy, loss)

Usage:
    # Use default checkpoint directory from config:
    python -m ml_ops_project.evaluate

    # Specify a specific checkpoint:
    python -m ml_ops_project.evaluate ckpt_path="checkpoints/epoch-05-0.850.ckpt"

    # Or run as script:
    uv run src/ml_ops_project/evaluate.py
"""

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

# Allow running as a script without package installation.
# When run directly (e.g., `python src/ml_ops_project/evaluate.py`),
# __package__ is None and imports would fail. This adds the src directory
# to sys.path so that `from ml_ops_project.X import Y` works.
if __package__ is None:
    src_dir = Path(__file__).resolve().parents[1]
    if (src_dir / "ml_ops_project").exists():
        sys.path.insert(0, str(src_dir))

from ml_ops_project.data import DataConfig, RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the most recently modified checkpoint file in a directory.

    Args:
        checkpoint_dir: Path to directory containing .ckpt files.

    Returns:
        Path to the most recently modified checkpoint file.

    Raises:
        ValueError: If no .ckpt files are found in the directory.
    """
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No .ckpt files found in directory: {checkpoint_dir}")

    # Sort by modification time and return the latest
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig) -> None:
    """Main evaluation function for testing a trained model.

    Loads a model checkpoint and evaluates it on the test set.
    The checkpoint can be specified via CLI override or defaults to
    searching the configured checkpoint directory.

    Args:
        cfg: Hydra DictConfig populated from configs/config.yaml and CLI overrides.
            Expected keys: training, model, data_dir.
            Optional key: ckpt_path (path to specific checkpoint).
    """
    # Seed all RNGs for reproducible evaluation results
    pl.seed_everything(cfg.training.seed)

    # --- Data Module Setup ---
    config = DataConfig(
        data_dir=cfg.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        model_name=cfg.model.name,
    )
    data_module = RottenTomatoesDataModule(config)
    # Note: trainer.test() automatically calls data_module.setup('test')

    # --- Checkpoint Resolution ---
    # Users can override via CLI: python evaluate.py ckpt_path="path/to/model.ckpt"
    ckpt_path = cfg.get("ckpt_path")

    if not ckpt_path:
        # Fall back to default checkpoint directory from config
        ckpt_path = cfg.training.checkpoint_dir
        print(f"No specific checkpoint provided. Searching in: {ckpt_path}")

    ckpt_path_obj = Path(ckpt_path)

    # Handle both directory (find latest) and direct file paths
    if ckpt_path_obj.is_dir():
        latest_checkpoint = find_latest_checkpoint(ckpt_path_obj)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        ckpt_path = str(latest_checkpoint)
    elif not ckpt_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")

    # --- Model Loading ---
    # load_from_checkpoint restores model weights and hyperparameters saved during training
    model = SentimentClassifier.load_from_checkpoint(ckpt_path)

    # --- Trainer Setup ---
    # Minimal trainer config for evaluation (no logging, single device)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,  # Disable logging for standalone evaluation
    )

    # --- Run Evaluation ---
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    evaluate()
