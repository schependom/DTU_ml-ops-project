from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from ml_ops_project.data import DataConfig, RottenTomatoesDataModule


def _build_test_config() -> DataConfig:
    # Build a minimal config for data tests using the project Hydra config,
    # but with a small batch size and no worker processes.
    config_path = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.2", config_dir=config_path):
        cfg = compose(config_name="config")
        return DataConfig(
            batch_size=4,
            num_workers=0,
            data_dir=cfg.data_dir,
            persistent_workers=False,
        )


@pytest.fixture(scope="module")
def datamodule() -> RottenTomatoesDataModule:
    # Create and initialize the data module once per module to reduce setup time.
    dm = RottenTomatoesDataModule(_build_test_config())
    # Prepare and Setup (this mimics the Lightning Trainer flow)
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def test_datamodule_batch_dimensions(datamodule: RottenTomatoesDataModule):
    # Verify that a training batch has the expected shapes, dtypes, and keys.
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Print statements are for local debugging and are not required for assertions.
    print("\n--- Batch Debug Info ---")
    print(f"Keys in batch: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention Mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels dtypes: {batch['labels'].dtype}")
    print(f"Max token ID: {batch['input_ids'].max()}")

    # Assertions
    # Check batch size
    assert batch["input_ids"].shape[0] == 4, "Batch size mismatch in input_ids"
    assert batch["labels"].shape[0] == 4, "Batch size mismatch in labels"

    # NLP Classification models usually expect Long (Int64) for labels and input_ids
    assert batch["labels"].dtype == torch.int64, "Labels should be Long tensors for CrossEntropy"
    assert batch["input_ids"].dtype == torch.int64, "Input IDs should be Long tensors"

    # The collator should ensure all sequences in a single batch have the same length
    seq_len = batch["input_ids"].shape[1]
    assert batch["attention_mask"].shape[1] == seq_len, "Attention mask length must match input_ids length"

    # Check keys required by your SentimentClassifier forward pass
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    # Verify dimensions: input_ids should be [batch_size, seq_len]
    # Note: seq_len is dynamic due to DataCollatorWithPadding
    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_datamodule_splits_exist(datamodule: RottenTomatoesDataModule):
    # Ensure the expected dataset splits exist and are non-empty.
    splits = datamodule.tokenized_datasets
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0


def test_val_and_test_dataloaders_batch_size(datamodule: RottenTomatoesDataModule):
    # Validate batch sizes in val/test loaders match the configured batch size.
    val_batch = next(iter(datamodule.val_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))
    assert val_batch["input_ids"].shape[0] == 4
    assert test_batch["input_ids"].shape[0] == 4


def test_labels_are_binary(datamodule: RottenTomatoesDataModule):
    # Confirm labels are binary (0/1) for sentiment classification.
    batch = next(iter(datamodule.train_dataloader()))
    unique_labels = torch.unique(batch["labels"]).tolist()
    assert all(label in (0, 1) for label in unique_labels)
