from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
import torch
from torch import nn
from types import SimpleNamespace

from metrics.training_dev_metrics import MetricConfig, TrainingDevMetrics
from paligemma2.training.pipeline import CheckpointManager


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        logits = self.linear(input_ids.float())
        loss = torch.nn.functional.mse_loss(logits, torch.zeros_like(logits))
        return SimpleNamespace(loss=loss)


def test_training_metrics_logging(tmp_path: Path) -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = MetricConfig(log_dir=str(tmp_path), log_every=1, parquet_rollover_rows=5)
    metrics = TrainingDevMetrics(cfg, model=model, optimizer=optimizer)

    batch = torch.ones(2, 4)
    for step in range(3):
        metrics.on_step_end(loss=1.0 + step, batch_size=batch.size(0), lr=0.1)
    metrics.finalize()

    parquet = tmp_path / "METRICS" / "train_timeseries.parquet"
    assert parquet.exists()
    df = pd.read_parquet(parquet)
    assert set(["loss", "grad_norm", "weight_norm"]).issubset(df.columns)


def test_checkpoint_manifest(tmp_path: Path) -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = MetricConfig(log_dir=str(tmp_path), log_every=1)
    metrics = TrainingDevMetrics(cfg, model=model, optimizer=optimizer)
    manager = CheckpointManager(save_dir=str(tmp_path / "ckpts"))

    dummy_state = {"input_ids": torch.ones(2, 4)}
    metrics.global_step = 10
    manager.save(step=10, stage="stage1", model=model, optimizer=optimizer, scheduler=None, metrics=metrics)
    manifest = tmp_path / "CHECKPOINT_MANIFEST.parquet"
    assert manifest.exists()
    df = pd.read_parquet(manifest)
    assert df.shape[0] == 1
    assert df.iloc[0]["stage"] == "stage1"


def test_checkpoint_indexer_script(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "CHECKPOINT_MANIFEST.parquet"
    pd.DataFrame([
        {"ts": 0.0, "step": 1, "stage": "stage1", "ckpt_path": "a", "param_stats": {}},
        {"ts": 0.0, "step": 2, "stage": "stage1", "ckpt_path": "b", "param_stats": {}},
    ]).to_parquet(manifest)
    out_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "checkpoint_indexer.py",
            "--log-dir",
            str(tmp_path),
            "--out",
            str(out_path),
        ],
    )
    runpy.run_path("scripts/checkpoint_indexer.py")
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["num_checkpoints"] == 2
