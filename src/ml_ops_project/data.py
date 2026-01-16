"""Data loading and preprocessing module for the Rotten Tomatoes sentiment dataset.

This module provides:
- DataConfig: dataclass holding all data loading parameters
- RottenTomatoesDataModule: PyTorch Lightning DataModule for the Rotten Tomatoes dataset

The data pipeline:
1. Downloads the dataset from Hugging Face (if not already cached)
2. Cleans and renames columns for compatibility with the model
3. Tokenizes text on-the-fly during setup (model-agnostic storage)
4. Returns DataLoaders with dynamic padding via DataCollatorWithPadding

Usage:
    python -m ml_ops_project.data  # downloads and prepares data
"""

import os
from dataclasses import dataclass
from typing import Any

import hydra
import pytorch_lightning as pl
from datasets import load_dataset, load_from_disk
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class DataConfig:
    """Configuration dataclass for the data module.

    Attributes:
        data_dir: Path to store/load the dataset. Can be local or GCS bucket path.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading (0 = main process only).
        model_name: HuggingFace model name for tokenizer initialization.
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).
        pin_memory: Pin memory for faster GPU transfer (useful when training on GPU).
    """

    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 0
    model_name: str = "distilbert-base-uncased"
    persistent_workers: bool = True
    pin_memory: bool = False

    def __str__(self) -> str:
        """Return a formatted string representation for logging."""
        header = "Data Loader Configuration"
        lines = [header, "-" * len(header) * 2]

        for key, value in self.__dict__.items():
            lines.append(f"{key:20}: {value}")

        lines.append("-" * len(header) * 2)

        return "\n".join(lines)


class RottenTomatoesDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Rotten Tomatoes sentiment dataset.

    Handles downloading, preprocessing, tokenization, and batching of the
    Rotten Tomatoes movie review dataset for binary sentiment classification.

    Args:
        config: DataConfig instance with all data loading parameters.
        logger: Optional logger instance. Falls back to module-level loguru logger.
    """

    def __init__(self, config: DataConfig, logger: Any = None) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        self.batch_size = config.batch_size
        # Initialize tokenizer here so it's available for both prepare_data and setup
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def prepare_data(self, force_download: bool = False) -> None:
        """Download and preprocess the dataset (runs on main process only).

        This method is called once per node in distributed training.
        It downloads the dataset from Hugging Face, filters empty samples,
        renames columns for model compatibility, and saves to disk.

        Args:
            force_download: If True, re-download even if data exists locally.
        """
        log = self.logger or logger

        # Check if data already exists: directory must exist and contain more than just .gitkeep
        if force_download or not (os.path.exists(self.config.data_dir) and len(os.listdir(self.config.data_dir)) > 1):
            print(f"Downloading and processing data to {self.config.data_dir}...")

            # Download from Hugging Face Hub
            dataset = load_dataset("rotten_tomatoes")
            raw_length = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
            log.info(f"Downloaded {raw_length} samples.")

            # Remove samples with missing or empty text to prevent tokenization errors
            dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)
            cleaned_length = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
            log.info(f"After cleaning, {cleaned_length} samples remain.")
            log.info(f"Dropped {raw_length - cleaned_length} samples with missing/empty text.")

            # Rename 'label' to 'labels' for compatibility with HuggingFace model forward signature
            if "label" in dataset["train"].column_names:
                dataset = dataset.rename_column("label", "labels")

            # Persist to disk (or GCS bucket if data_dir points there)
            dataset.save_to_disk(self.config.data_dir)
            log.success("Data saved successfully.")
        else:
            log.info(
                f"Data found at {self.config.data_dir}, skipping download. Use force_download=True to re-download."
            )

    def setup(self, stage: str | None = None) -> None:
        """Load and tokenize the dataset (runs on every process/GPU).

        This method is called on every GPU in distributed training.
        It loads the preprocessed data from disk and applies tokenization.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
                   Currently unused but required by Lightning interface.
        """
        log = self.logger or logger
        log.info(f"Loading data from path: {self.config.data_dir}")
        dataset = load_from_disk(self.config.data_dir)

        # Tokenize on-the-fly rather than storing tokenized data
        # This keeps the saved dataset model-agnostic (can switch tokenizers without re-downloading)
        def tokenize_function(examples: dict) -> dict:
            return self.tokenizer(examples["text"], truncation=True)

        self.tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Set PyTorch tensor format for the columns needed by the model
        self.tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # DataCollatorWithPadding pads each batch to the longest sequence in that batch
        # This is more efficient than padding all sequences to a fixed max length
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with shuffling enabled."""
        return DataLoader(
            self.tokenized_datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data each epoch for better generalization
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (no shuffling for reproducible evaluation)."""
        return DataLoader(
            self.tokenized_datasets["validation"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader (no shuffling for reproducible evaluation)."""
        return DataLoader(
            self.tokenized_datasets["test"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def load_data(cfg: DictConfig) -> None:
    """Standalone entry point for data preparation.

    Can be run directly to download and preprocess the dataset
    without starting training.

    Args:
        cfg: Hydra DictConfig with at least `data_dir` key.
    """
    data_config = DataConfig(
        data_dir=cfg.data_dir,
    )

    data_module = RottenTomatoesDataModule(data_config)

    data_module.prepare_data()


if __name__ == "__main__":
    load_data()
