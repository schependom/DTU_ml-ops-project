"""Training script for the Sentiment Classification model.

This module provides the main training entry point, orchestrating:
- Hydra-based configuration loading
- Weights & Biases experiment tracking (optional)
- PyTorch Lightning training with checkpointing and early stopping

Usage:
    python -m ml_ops_project.train
    # or with Hydra overrides:
    python -m ml_ops_project.train training.max_epochs=5 training.batch_size=32
"""

import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ml_ops_project.data import DataConfig, RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier
from ml_ops_project.pl_logging import LoguruLightningLogger

# Load environment variables from .env file (e.g., WANDB_API_KEY)
load_dotenv()

# Configure file-based logging: main log + alerts-only log for warnings/errors
logger.add("logs/train.log", rotation="500 MB")
logger.add("logs/train_alerts.log", level="WARNING", rotation="500 MB")


def setup_wandb(cfg: DictConfig) -> bool:
    """Initialize Weights & Biases for experiment tracking.

    Attempts to log in and initialize a W&B run using API key from environment.
    Gracefully skips if disabled in config or credentials are missing.

    Args:
        cfg: Hydra DictConfig containing the full experiment configuration.
            Must have `cfg.wandb.enabled` and `cfg.wandb.tags`.

    Returns:
        True if W&B was successfully initialized, False otherwise.
    """
    if not cfg.wandb.enabled:
        logger.info("WandB disabled via config.")
        return False

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not found. Skipping WandB.")
        return False

    try:
        wandb.login(key=api_key)

        init_kwargs = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "tags": cfg.wandb.tags,
        }

        # During a sweep, W&B agent sets project/entity automatically;
        # setting them manually would conflict with the sweep configuration.
        if not os.getenv("WANDB_SWEEP_ID"):
            entity = os.getenv("WANDB_ENTITY")
            project = os.getenv("WANDB_PROJECT")
            if entity:
                init_kwargs["entity"] = entity
            if project:
                init_kwargs["project"] = project

        wandb.init(**init_kwargs)
        logger.success("Initialized Weights & Biases.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return False


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig) -> None:
    """Main training function orchestrating the full training pipeline.

    This function:
    1. Seeds all random generators for reproducibility
    2. Optionally initializes W&B logging
    3. Prepares the data module (download/load, tokenization)
    4. Initializes the model and callbacks (checkpointing, early stopping)
    5. Runs training via PyTorch Lightning Trainer
    6. Evaluates on the test set using the best checkpoint

    Args:
        cfg: Hydra DictConfig populated from configs/config.yaml and CLI overrides.
            Expected keys: training, model, optimizer, wandb, data_dir.
    """
    # Seed all RNGs (Python, NumPy, PyTorch) for reproducible experiments
    pl.seed_everything(cfg.training.seed)

    # Log the device being used
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Training on device: {device}")

    # --- W&B Logger Setup ---
    using_wandb = setup_wandb(cfg)
    if using_wandb:
        # log_model="all" uploads all checkpoints saved by ModelCheckpoint to W&B Artifacts
        wandb_logger = WandbLogger(
            project="MLOps-Project",
            name="experiment_run_name",
            log_model="all",
        )
    else:
        wandb_logger = None

    # --- Data Module Setup ---
    # DataConfig holds all data loading parameters; passed to the LightningDataModule
    config = DataConfig(
        data_dir=cfg.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        model_name=cfg.model.name,
        persistent_workers=cfg.training.persistent_workers,
        pin_memory=cfg.training.pin_memory,
    )
    logger.info("Setting up data-loaders with the following configuration:")
    logger.info(print(config))

    data_module = RottenTomatoesDataModule(config, logger=logger)

    # --- Model Setup ---
    logger.info(f"Configuring model: {cfg.model.name}, with optimizer: {cfg.optimizer}")
    model = SentimentClassifier(model_name=cfg.model.name, optimizer_cfg=cfg.optimizer)

    # --- Callbacks ---
    # Callbacks are hooks that execute at specific points in the training loop.
    callbacks = [
        # ModelCheckpoint: saves model weights when monitored metric improves
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="{epoch:02d}-{val_accuracy:.3f}",
            monitor="val_accuracy",
            mode="max",  # higher accuracy is better
            save_top_k=1,  # keep only the single best checkpoint
            save_weights_only=True,  # Only save model weights (no optimizer state)
        ),
        # EarlyStopping: halts training if metric doesn't improve for `patience` epochs
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
        ),
    ]

    # Adapter to route Lightning logs through Loguru for consistent formatting
    loguru_adapter = LoguruLightningLogger(logger)

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",  # auto-select GPU/CPU/TPU
        devices=1,
        logger=[loguru_adapter, wandb_logger],
        default_root_dir="hydra_logs",
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # --- Training ---
    logger.info("Training model")
    trainer.fit(model, data_module)
    logger.success("Done!")

    # --- Testing ---
    # ckpt_path="best" loads the checkpoint with highest val_accuracy from ModelCheckpoint
    logger.info("Testing model")
    checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_cb is not None and hasattr(checkpoint_cb, "best_model_path"):
        best_path = checkpoint_cb.best_model_path
        logger.info(f"Best model path: {best_path}")

        # DEBUG: Verify file existence to diagnose Vertex AI failure
        if os.path.exists(best_path):
            logger.info(f"Checkpoint file exists at {best_path}")
        else:
            logger.error(f"Checkpoint file MISSING at {best_path}")
            # List contents of the parent directory
            parent_dir = os.path.dirname(best_path)
            if os.path.exists(parent_dir):
                logger.info(f"Contents of {parent_dir}: {os.listdir(parent_dir)}")
            else:
                logger.error(f"Parent directory {parent_dir} does not exist!")

    trainer.test(model=model, datamodule=data_module, ckpt_path="best")

    # Finalize W&B run after all logging is complete (fit + test)
    if wandb_logger is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
