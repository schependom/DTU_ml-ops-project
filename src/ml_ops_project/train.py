import sys
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

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
    # Optional: Print config to verify (useful in logs)
    print(OmegaConf.to_yaml(cfg))

    # 2. Use parameters from the config (cfg.training.batch_size, etc.)
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

    # Initialize Data Module
    data_module = RottenTomatoesDataModule(model_name=cfg.model.name, batch_size=cfg.training.batch_size)

    # Initialize Model
    model = SentimentClassifier(model_name=cfg.model.name, learning_rate=cfg.training.learning_rate)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        default_root_dir="hydra_logs",  # Clean logs
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


if __name__ == "__main__":
    train()
