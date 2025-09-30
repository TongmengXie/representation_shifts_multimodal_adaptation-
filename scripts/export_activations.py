"""Export activation snapshots from a checkpoint for post-hoc analysis."""
from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import torch

from metrics.training_dev_metrics import MetricConfig, TrainingDevMetrics
from paligemma2.data.registry import REGISTRY, register_stub_tasks
from paligemma2.modeling import PaliGemma2Config, build_paligemma2_model, load_paligemma2_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--layers", nargs="+", required=True, help="Layer names to hook.")
    parser.add_argument("--model-size", default="3b")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=10)
    return parser.parse_args()


def build_loader(resolution: int) -> Iterable:
    if not REGISTRY.names():
        register_stub_tasks()
    return REGISTRY.get("captioning").loader(resolution, batch_size=4)


def main() -> None:
    args = parse_args()
    model = build_paligemma2_model(PaliGemma2Config(model_size=args.model_size, resolution=args.resolution), device=args.device)
    load_paligemma2_checkpoint(model, str(args.checkpoint))
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = TrainingDevMetrics(MetricConfig(log_dir=args.out.parent.as_posix()), model=model, optimizer=dummy_optimizer)
    loader = build_loader(args.resolution)
    iterator = iter(loader)
    sliced = (batch for _, batch in zip(range(args.batches), iterator))
    metrics.export_activations(sliced, layers=args.layers, out_path=str(args.out))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
