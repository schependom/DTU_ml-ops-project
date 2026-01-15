import os
from dataclasses import dataclass

import pytorch_lightning as pl
from datasets import load_dataset, load_from_disk
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


class RottenTomatoesDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def prepare_data(self):
        """
        Runs on the main process only.
        1. Checks if data exists at data_dir.
        2. If not, downloads from Hugging Face.
        3. Renames columns (label -> labels).
        4. Saves the cleaned data to data_dir (GCP Bucket).
        """
        # Check if the data is already saved to the bucket/disk
        if not os.path.exists(self.config.data_dir):
            print(f"Downloading and processing data to {self.config.data_dir}...")

            # 1. Download
            dataset = load_dataset("rotten_tomatoes")

            # 2. Rename (Preprocessing moved here)
            if "label" in dataset["train"].column_names:
                dataset = dataset.rename_column("label", "labels")

            # 3. Save to disk/GCP
            # This saves the Arrow files directly to the path
            dataset.save_to_disk(self.config.data_dir)
            print("Data saved successfully.")
        else:
            print(f"Data found at {self.config.data_dir}, skipping download and rename.")

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
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["validation"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["test"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
