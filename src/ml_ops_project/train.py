import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from ml_ops_project.data import DataConfig, RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.training.seed)

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
        default_root_dir="hydra_logs",  # Clean logs
        logger=CSVLogger(save_dir="outputs", name="rotten_tomatoes"),  # Log metrics to CSV files
        callbacks=callbacks,  # Attach our checkpoint callback
        log_every_n_steps=10,  # Log metrics every 10 batches
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model=model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    train()
