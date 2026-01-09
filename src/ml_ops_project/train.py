import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Allow running as a script: `uv run src/ml_ops_project/train.py`
# (src-layout packages otherwise need `python -m ml_ops_project.train`)
if __package__ is None:
    src_dir = Path(__file__).resolve().parents[1]  # .../src
    if (src_dir / "ml_ops_project").exists():
        sys.path.insert(0, str(src_dir))

from ml_ops_project.data import RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier


# 1. Add the Hydra Decorator
@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    # Optional: Weights & Biases logging (enabled via configs/wandb/default.yaml)
    wandb_logger = None
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is not None and bool(getattr(wandb_cfg, "enabled", False)):
        try:
            from pytorch_lightning.loggers import WandbLogger
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "W&B logging is enabled (wandb.enabled=true) but WandbLogger could not be imported. "
                "Ensure `wandb` is installed and compatible with your pytorch-lightning version."
            ) from e

        mode = getattr(wandb_cfg, "mode", None)
        if mode:
            # Respect an explicitly set environment variable (useful on CI/containers)
            os.environ.setdefault("WANDB_MODE", str(mode))

        wandb_logger = WandbLogger(
            project=str(getattr(wandb_cfg, "project", "MLOps")),
            entity=getattr(wandb_cfg, "entity", None),
            name=str(getattr(cfg, "experiment_name", "train")),
            tags=list(getattr(wandb_cfg, "tags", []) or []),
            notes=getattr(wandb_cfg, "notes", None),
            log_model=bool(getattr(wandb_cfg, "log_model", False)),
        )

        # Store full Hydra config in W&B (so runs are reproducible from the UI)
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.experiment.config.update(resolved_cfg, allow_val_change=True)

    config = DataConfig(
        data_dir=cfg.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        model_name=cfg.model.name,
    )

    # Initialize Data Module
    data_module = RottenTomatoesDataModule(config)

    # Initialize Model
    model = SentimentClassifier(model_name=cfg.model.name, learning_rate=cfg.training.learning_rate)

    # Callbacks are plugins that execute code at certain points in the training loop.
    # ModelCheckpoint automatically saves the best model based on a monitored metric.
    callbacks = [
        ModelCheckpoint(
            # Default: after each validation epoch
            # more often: every_n_epochs=1, every_n_train_steps=100, etc.
            # less often: every_n_epochs=5, every_n_train_steps=500, etc.
            dirpath=cfg.training.checkpoint_path,
            filename="epoch-{epoch:02d}-{val_accuracy:.3f}",  # Save with epoch and val_accuracy in filename
            monitor="val_accuracy",  # Monitor validation accuracy
            mode="max",  # Higher is better (for accuracy)
            save_top_k=1,  # Keep only the best checkpoint
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,  # Stop training if no improvement in 3 epochs
            mode="max",  # Higher is better
        ),
    ]

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        default_root_dir="hydra_logs",  # Clean logs
        # logger=CSVLogger(save_dir="outputs", name="rotten_tomatoes"),  # Log metrics to CSV files
        callbacks=callbacks,  # Attach our checkpoint callback
        log_every_n_steps=10,  # Log metrics every 10 batches
    )

    # Train
    trainer.fit(model, data_module)

    if wandb_logger is not None:
        # Ensure the run is closed cleanly even if the process exits quickly after training
        try:
            import importlib

            wandb = importlib.import_module("wandb")
            wandb.finish()
        except Exception:
            pass
    # Test
    trainer.test(model=model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    train()
