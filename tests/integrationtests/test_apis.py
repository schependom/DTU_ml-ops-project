"""Integration tests for the FastAPI inference and metrics endpoints.

These tests:
1. Stub the model and tokenizer to avoid loading real artifacts
2. Exercise the inference endpoint for valid and invalid payloads
3. Confirm the metrics endpoint exposes request counters
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import ml_ops_project.api as api


# Minimal tokenizer stub to keep API handler lightweight in tests.
class _DummyTokenizer:
    def __call__(self, _text: str, return_tensors: str = "pt", padding: bool = True, truncation: bool = True):
        input_ids = torch.tensor([[101, 102, 103]])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Minimal model stub to return deterministic logits for sentiment.
class _DummyModel:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "_DummyModel":
        self.device = device
        return self

    def eval(self) -> "_DummyModel":
        return self

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_kwargs) -> SimpleNamespace:
        logits = torch.tensor([[0.2, 0.8]], device=input_ids.device)
        return SimpleNamespace(logits=logits)


# Configure the API app with dummy dependencies and a test client.
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "device", torch.device("cpu"))
    monkeypatch.setattr(api, "load_model", lambda _cfg: _DummyModel())
    monkeypatch.setattr(api, "save_prediction_to_gcp", lambda *args, **kwargs: None)

    with TestClient(api.app) as test_client:
        yield test_client


# Ensure inference returns a valid sentiment class on happy path.
def test_inference_returns_sentiment(client: TestClient):
    response = client.post("/inference", json={"statement": "Great movie!"})

    assert response.status_code == 200
    payload = response.json()
    assert "sentiment" in payload
    assert payload["sentiment"] in (0, 1)


# Validate missing payload fields are rejected by request validation.
def test_inference_requires_statement(client: TestClient):
    response = client.post("/inference", json={})

    assert response.status_code == 422


# Verify Prometheus metrics endpoint responds with counters.
def test_metrics_endpoint_available(client: TestClient):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "prediction_requests" in response.text
