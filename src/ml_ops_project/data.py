import os
from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
from datasets import load_dataset, load_from_disk
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class DataConfig:
    """Configuration for the data module."""

    # Default path is local 'data' folder; can be overridden to point to GCP Bucket "/gcs/ml_ops_project_g7/data"
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 0
    model_name: str = "distilbert-base-uncased"
    persistent_workers: bool = True
    pin_memory: bool = False


class RottenTomatoesDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def prepare_data(self, force_download: bool = False):
        """
        Runs on the main process only.
        1. Checks if data exists at data_dir.
        2. If not, downloads from Hugging Face.
        3. Renames columns (label -> labels).
        4. Saves the cleaned data to data_dir (GCP Bucket).
        """
        # If data/ contains something else apart from .gitkeep
        if force_download or not (os.path.exists(self.config.data_dir) and len(os.listdir(self.config.data_dir)) > 1):
            print(f"Downloading and processing data to {self.config.data_dir}...")

            # Download
            dataset = load_dataset("rotten_tomatoes")
            raw_length = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
            logger.info(f"Downloaded {raw_length} samples.")

            dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)
            cleaned_length = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
            logger.info(f"After cleaning, {cleaned_length} samples remain.")
            logger.info(f"Dropped {raw_length - cleaned_length} samples with missing/empty text.")

            # Rename
            if "label" in dataset["train"].column_names:
                dataset = dataset.rename_column("label", "labels")

            # Save to disk/GCP
            dataset.save_to_disk(self.config.data_dir)
            print("Data saved successfully.")
        else:
            print(f"Data found at {self.config.data_dir}, skipping download. Use force_download=True to re-download.")

    def setup(self, stage=None):
        """
        Runs on every process/GPU.
        1. Loads the pre-processed data from disk.
        2. Tokenizes the text.
        """
        # Load the already renamed data from the bucket
        dataset = load_from_disk(self.config.data_dir)

        # Tokenize
        # We process on the fly here to keep the saved data "model-agnostic"
        # (i.e. if you switch BERT for RoBERTa, you don't need to re-download the raw data)
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        self.tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Set format for PyTorch
        self.tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Create the DataCollator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["validation"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["test"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def load_data(cfg: DictConfig):
    data_config = DataConfig(
        data_dir=cfg.data_dir,
    )

    data_module = RottenTomatoesDataModule(data_config)

    data_module.prepare_data()


if __name__ == "__main__":
    load_data()
