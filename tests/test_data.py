import pytest
import torch
from hydra import compose, initialize
from ml_ops_project.data import DataConfig, RottenTomatoesDataModule


def test_datamodule_batch_dimensions():
    # Setup config (use a small batch for testing)
    with initialize(version_base="1.2", config_path="../../configs"):
        cfg = compose(config_name="config")
        config = DataConfig(batch_size=4, num_workers=0, data_dir=cfg.data_dir)

    dm = RottenTomatoesDataModule(config)
    
    # Prepare and Setup (this mimics the Lightning Trainer flow)
    dm.prepare_data()
    dm.setup(stage="fit")
    
    # Get one batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # Print Statements for Debugging 
    print(f"\n--- Batch Debug Info ---")
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
