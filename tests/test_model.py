from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from ml_ops_project.models import SentimentClassifier


@pytest.fixture(scope="module")
def model_and_cfg():
    config_path = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.2", config_dir=config_path):
        cfg = compose(config_name="config")
        model = SentimentClassifier(model_name=cfg.model.name, optimizer_cfg=cfg.optimizer)
    model.log = lambda *args, **kwargs: None  # type: ignore[assignment]
    return model, cfg


def _make_batch(batch_size: int, seq_length: int):
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int64)
    labels = torch.randint(0, 2, (batch_size,))
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@pytest.mark.parametrize("batch_size, seq_length", [(1, 16), (4, 128)])
def test_model_forward_pass(model_and_cfg, batch_size, seq_length):
    model, _ = model_and_cfg
    model.eval()

    batch = _make_batch(batch_size, seq_length)

    # Run forward pass
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    # 5. Assertions
    assert output.logits.shape == (batch_size, 2), "Output logits shape mismatch"
    assert output.loss is not None, "Loss was not calculated"
    print(f"\nForward pass successful for batch={batch_size}, seq={seq_length}")


def test_training_step_returns_loss(model_and_cfg):
    model, _ = model_and_cfg
    model.train()
    batch = _make_batch(batch_size=2, seq_length=8)
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    assert loss.dim() == 0


def test_validation_and_test_steps_return_loss(model_and_cfg):
    model, _ = model_and_cfg
    model.eval()
    batch = _make_batch(batch_size=2, seq_length=8)
    val_loss = model.validation_step(batch, batch_idx=0)
    test_loss = model.test_step(batch, batch_idx=0)
    assert val_loss is not None
    assert test_loss is not None


def test_configure_optimizers_returns_optimizer(model_and_cfg):
    model, _ = model_and_cfg
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
