from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ml_ops_project.train as train_module


def _base_cfg(tmp_path: Path):
    # Build a minimal Hydra-like config for train() without loading full configs.
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "model": {"name": "distilbert-base-uncased"},
        "optimizer": {"lr": 1e-3},
        "training": {
            "seed": 123,
            "batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "max_epochs": 1,
        },
        "wandb": {"enabled": True, "tags": []},
    }
    return OmegaConf.create(cfg)


class _FakeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        self.test_calls = []

    def fit(self, model, datamodule):
        self.fit_calls.append((model, datamodule))

    def test(self, model=None, datamodule=None, ckpt_path=None):
        self.test_calls.append((model, datamodule, ckpt_path))


def _mock_train_dependencies(monkeypatch, trainer: _FakeTrainer):
    # Avoid heavy Lightning, model, and data operations to keep tests fast.
    monkeypatch.setattr(train_module.pl, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "DataConfig", lambda **kwargs: OmegaConf.create(kwargs))
    monkeypatch.setattr(train_module, "RottenTomatoesDataModule", lambda cfg, logger=None: f"dm:{cfg.data_dir}")
    monkeypatch.setattr(
        train_module.SentimentClassifier,
        "__init__",
        lambda self, **_kwargs: None,
    )

    def _trainer_factory(**kwargs):
        trainer.kwargs = kwargs
        return trainer

    monkeypatch.setattr(train_module.pl, "Trainer", _trainer_factory)


def test_setup_wandb_disabled(monkeypatch):
    # If WandB is disabled in config, setup_wandb should return False.
    cfg = OmegaConf.create({"wandb": {"enabled": False}})
    assert train_module.setup_wandb(cfg) is False


def test_setup_wandb_missing_api_key(monkeypatch):
    # If the API key is missing, setup_wandb should skip initialization.
    cfg = OmegaConf.create({"wandb": {"enabled": True, "tags": []}})
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert train_module.setup_wandb(cfg) is False


def test_setup_wandb_sets_project_and_entity(monkeypatch):
    # When not in a sweep, setup_wandb should include entity/project if set.
    cfg = OmegaConf.create({"wandb": {"enabled": True, "tags": ["t1"]}})

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_ENTITY", "demo-entity")
    monkeypatch.setenv("WANDB_PROJECT", "demo-project")
    monkeypatch.delenv("WANDB_SWEEP_ID", raising=False)

    calls = {}

    def _login(key):
        calls["login_key"] = key

    def _init(**kwargs):
        calls["init_kwargs"] = kwargs

    monkeypatch.setattr(train_module.wandb, "login", _login)
    monkeypatch.setattr(train_module.wandb, "init", _init)

    assert train_module.setup_wandb(cfg) is True
    assert calls["login_key"] == "dummy"
    assert calls["init_kwargs"]["entity"] == "demo-entity"
    assert calls["init_kwargs"]["project"] == "demo-project"


def test_setup_wandb_omits_project_in_sweep(monkeypatch):
    # When running a sweep, setup_wandb should not set project/entity explicitly.
    cfg = OmegaConf.create({"wandb": {"enabled": True, "tags": []}})

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_SWEEP_ID", "sweep-123")
    monkeypatch.setenv("WANDB_ENTITY", "demo-entity")
    monkeypatch.setenv("WANDB_PROJECT", "demo-project")

    init_kwargs = {}

    monkeypatch.setattr(train_module.wandb, "login", lambda key: None)
    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: init_kwargs.update(kwargs))

    assert train_module.setup_wandb(cfg) is True
    assert "entity" not in init_kwargs
    assert "project" not in init_kwargs


def test_train_runs_fit_and_test_with_wandb(monkeypatch, tmp_path):
    # Ensure train() wires together data, model, trainer, and calls fit/test.
    cfg = _base_cfg(tmp_path)
    trainer = _FakeTrainer()

    _mock_train_dependencies(monkeypatch, trainer)
    monkeypatch.setattr(train_module, "setup_wandb", lambda _cfg: True)

    fake_logger = object()
    monkeypatch.setattr(train_module, "WandbLogger", lambda **_kwargs: fake_logger)

    finished = {"called": False}
    monkeypatch.setattr(train_module.wandb, "finish", lambda: finished.__setitem__("called", True))

    train_module.train(cfg)

    assert trainer.kwargs["logger"] is fake_logger
    assert trainer.fit_calls
    assert trainer.test_calls
    _, test_dm, test_ckpt = trainer.test_calls[0]
    assert test_dm == "dm:" + str(tmp_path / "data")
    assert test_ckpt == "best"
    assert finished["called"] is True


def test_train_runs_without_wandb(monkeypatch, tmp_path):
    # If WandB is disabled, train() should proceed with no logger and no finish().
    cfg = _base_cfg(tmp_path)
    trainer = _FakeTrainer()

    _mock_train_dependencies(monkeypatch, trainer)
    monkeypatch.setattr(train_module, "setup_wandb", lambda _cfg: False)
    monkeypatch.setattr(train_module, "WandbLogger", lambda **_kwargs: object())

    monkeypatch.setattr(train_module.wandb, "finish", lambda: pytest.fail("wandb.finish should not be called"))

    train_module.train(cfg)

    assert trainer.kwargs["logger"] is None
    assert trainer.fit_calls
    assert trainer.test_calls
    _, test_dm, test_ckpt = trainer.test_calls[0]
    assert test_dm == "dm:" + str(tmp_path / "data")
    assert test_ckpt == "best"
