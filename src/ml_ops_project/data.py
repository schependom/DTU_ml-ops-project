from dataclasses import dataclass

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class DataConfig:
    """Configuration for the data module.

    Using a dataclass makes configuration explicit and type-safe. This gets
    passed to CorruptMNISTDataModule to control batch size, data paths, etc.
    """

    data_dir: str = "data/processed"  # Directory with train_images.pt, etc.
    batch_size: int = 64  # Number of samples per batch
    num_workers: int = 0  # Number of processes for data loading (0 = main process)
    model_name: str = "distilbert-base-uncased"  # Pretrained model name


class RottenTomatoesDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def setup(self, stage=None):
        # Download the dataset
        dataset = load_dataset("rotten_tomatoes")
        dataset = dataset.rename_column("label", "labels")

        # Tokenize the data
        # We only tokenize here. Padding happens in the DataCollator (dynamic padding)
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        # Apply the tokenizer to the dataset
        self.tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Set format for PyTorch
        self.tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Create the DataCollator (handles padding)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["train"],  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["validation"],  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["test"],  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )
