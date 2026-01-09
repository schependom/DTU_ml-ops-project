import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from ml_ops_project.data import RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier


# 1. Add the Hydra Decorator
@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # Optional: Print config to verify (useful in logs)
    print(OmegaConf.to_yaml(cfg))

    # 2. Use parameters from the config (cfg.training.batch_size, etc.)
    pl.seed_everything(cfg.training.seed)

    # Initialize Data Module
    data_module = RottenTomatoesDataModule(model_name=cfg.model.name, batch_size=cfg.training.batch_size)

    # Initialize Model
    model = SentimentClassifier(model_name=cfg.model.name, learning_rate=cfg.training.learning_rate)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir="hydra_logs",  # Clean logs
    )

    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
