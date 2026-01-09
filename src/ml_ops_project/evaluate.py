import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

# Allow running as a script: `uv run src/ml_ops_project/evaluate.py`
if __package__ is None:
    src_dir = Path(__file__).resolve().parents[1]
    if (src_dir / "ml_ops_project").exists():
        sys.path.insert(0, str(src_dir))

from ml_ops_project.data import DataConfig, RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    # Prepare Data Module
    config = DataConfig(
        data_dir=cfg.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        model_name=cfg.model.name,
    )
    data_module = RottenTomatoesDataModule(config)
    # Ensure data is ready (downloaded/tokenized) - Lightning handles this in setup(),
    # but instantiating it usually doesn't trigger setup until trainer runs.
    # However, trainer.test() handles calling setup('test').

    # Check for checkpoint path
    # Users can run: python src/ml_ops_project/evaluate.py ckpt_path="path/to/model.ckpt"
    # or rely on default checkpoint_dir to find latest.
    ckpt_path = cfg.get("ckpt_path")

    if not ckpt_path:
        ckpt_path = cfg.training.checkpoint_dir
        print(f"No specific checkpoint provided. Searching in: {ckpt_path}")

    ckpt_path_obj = Path(ckpt_path)

    # If it's a directory, find the latest .ckpt file
    if ckpt_path_obj.is_dir():
        checkpoints = list(ckpt_path_obj.glob("*.ckpt"))
        if not checkpoints:
            raise ValueError(f"No .ckpt files found in directory: {ckpt_path_obj}")

        # Sort by modification time (latest first)
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        ckpt_path = str(latest_checkpoint)
    elif not ckpt_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")

    # Load model from checkpoint
    # Note: load_from_checkpoint handles loading hparams saved in the checkpoint
    model = SentimentClassifier.load_from_checkpoint(ckpt_path)

    # Initialize Trainer (only need basic settings for evaluation)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,  # Usually don't need a logger for simple eval, or can reuse WandB if preferred
    )

    # Run Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    evaluate()
