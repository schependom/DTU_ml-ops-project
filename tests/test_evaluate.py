from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ml_ops_project.evaluate as evaluate_module


def _base_cfg(tmp_path: Path, ckpt_path: str | None = None):
    # Build a minimal Hydra-like config for evaluate() without loading full configs.
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "ckpt_path": ckpt_path,
        "model": {"name": "distilbert-base-uncased"},
        "training": {
            "seed": 123,
            "batch_size": 8,
            "num_workers": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }
    return OmegaConf.create(cfg)


class _FakeTrainer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.test_calls = []

    def test(self, model, datamodule=None):
        self.test_calls.append((model, datamodule))


def _mock_dependencies(monkeypatch, trainer: _FakeTrainer):
    # Avoid touching real data/model logic to keep tests fast and deterministic.
    monkeypatch.setattr(evaluate_module, "RottenTomatoesDataModule", lambda cfg: f"dm:{cfg.data_dir}")
    monkeypatch.setattr(evaluate_module, "DataConfig", lambda **kwargs: OmegaConf.create(kwargs))
    monkeypatch.setattr(
        evaluate_module.SentimentClassifier,
        "load_from_checkpoint",
        staticmethod(lambda path: f"model:{path}"),
    )
    monkeypatch.setattr(evaluate_module.pl, "Trainer", lambda **kwargs: trainer)


def test_evaluate_uses_explicit_checkpoint(tmp_path, monkeypatch):
    # If a specific checkpoint path is provided, evaluate() should use it directly.
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("dummy")
    cfg = _base_cfg(tmp_path, ckpt_path=str(ckpt))

    trainer = _FakeTrainer()
    _mock_dependencies(monkeypatch, trainer)

    evaluate_module.evaluate(cfg)

    assert trainer.test_calls == [(f"model:{ckpt}", "dm:" + str(tmp_path / "data"))]


def test_evaluate_picks_latest_checkpoint_in_dir(tmp_path, monkeypatch):
    # If no explicit ckpt_path is given, evaluate() should pick the newest .ckpt in the directory.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    older = ckpt_dir / "old.ckpt"
    newer = ckpt_dir / "new.ckpt"
    older.write_text("old")
    newer.write_text("new")
    # Ensure different mtimes
    older.touch()
    newer.touch()

    cfg = _base_cfg(tmp_path, ckpt_path=None)

    trainer = _FakeTrainer()
    _mock_dependencies(monkeypatch, trainer)

    evaluate_module.evaluate(cfg)

    assert trainer.test_calls[0][0] == f"model:{newer}"


def test_evaluate_errors_when_dir_has_no_checkpoints(tmp_path, monkeypatch):
    # If the checkpoint directory is empty, evaluate() should raise a ValueError.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    cfg = _base_cfg(tmp_path, ckpt_path=None)

    trainer = _FakeTrainer()
    _mock_dependencies(monkeypatch, trainer)

    with pytest.raises(ValueError, match="No .ckpt files found"):
        evaluate_module.evaluate(cfg)


def test_evaluate_errors_when_checkpoint_missing(tmp_path, monkeypatch):
    # If an explicit checkpoint path does not exist, evaluate() should raise FileNotFoundError.
    missing = tmp_path / "missing.ckpt"
    cfg = _base_cfg(tmp_path, ckpt_path=str(missing))

    trainer = _FakeTrainer()
    _mock_dependencies(monkeypatch, trainer)

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        evaluate_module.evaluate(cfg)
