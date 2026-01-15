import os

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from ml_ops_project.data import DataConfig, RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier

load_dotenv()


def setup_wandb(cfg: DictConfig) -> bool:
    """Initializes Weights & Biases."""
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

        # Only set project and entity if not running a sweep (WandB handles this automatically in sweeps)
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


# 1. Add the Hydra Decorator
@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    using_wandb = setup_wandb(cfg)
    if using_wandb:
        # TODO customize WandbLogger params as needed
        # log_model="all" to log all checkpoints saved by ModelCheckpoint
        wandb_logger = WandbLogger(log_model="all", checkpoint_name="model")
    else:
        wandb_logger = None

    # Prepare Data Module
    config = DataConfig(
        data_dir=cfg.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        model_name=cfg.model.name,
        persistent_workers=cfg.training.persistent_workers,
        pin_memory=cfg.training.pin_memory,
    )

    # Initialize Data Module
    data_module = RottenTomatoesDataModule(config)

    # Initialize Model
    model = SentimentClassifier(model_name=cfg.model.name, optimizer_cfg=cfg.optimizer)

    # Callbacks are plugins that execute code at certain points in the training loop.
    # ModelCheckpoint automatically saves the best model based on a monitored metric.
    callbacks = [
        ModelCheckpoint(
            # Default: after each validation epoch
            # more often: every_n_epochs=1, every_n_train_steps=100, etc.
            # less often: every_n_epochs=5, every_n_train_steps=500, etc.
            dirpath=cfg.training.checkpoint_dir,
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
        callbacks=callbacks,  # Attach our checkpoint callback
        log_every_n_steps=10,  # Log metrics every 10 batches
    )

    # Train
    trainer.fit(model, data_module)

    # Test (uses best checkpoint from ModelCheckpoint above)
    trainer.test(model=model, datamodule=data_module, ckpt_path="best")

    # Close the W&B run *after* all Lightning stages are done (fit + test),
    # otherwise Lightning may still try to log hyperparams/metrics during test.
    if wandb_logger is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
