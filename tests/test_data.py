import pytest
import torch
from ml_ops_project.data import DataConfig, RottenTomatoesDataModule
from ml_ops_project.models import SentimentClassifier


def test_model_forward_pass(batch_size, seq_length):
    # 1. Initialize model
    model = SentimentClassifier()
    
    # 2. Create mock data matching the dictionary format in your forward/training_step
    # DistilBERT vocabulary size is 30522
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, 2, (batch_size,))
    
    # 3. Run forward pass and check for crashes
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    except Exception as e:
        pytest.fail(f"Model failed forward pass with seq_length {seq_length}: {e}")
    
    # 4. Verify output dimensions
    # Sequence classification models return logits of shape [batch_size, num_labels]
    assert output.logits.shape == (batch_size, 2)



def test_datamodule_batch_dimensions():
    # 1. Setup config (use a small batch for testing)
    config = DataConfig(batch_size=4, num_workers=0)
    dm = RottenTomatoesDataModule(config)
    
    # 2. Prepare and Setup (this mimics the Lightning Trainer flow)
    dm.prepare_data()
    dm.setup(stage="fit")
    
    # 3. Get one batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # 4. Assertions
    # Check batch size
    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].shape[0] == 4
    
    # Check keys required by your SentimentClassifier forward pass
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    
    # Verify dimensions: input_ids should be [batch_size, seq_len]
    # Note: seq_len is dynamic due to DataCollatorWithPadding
    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
