"""Evaluate a saved PaliGemma 2 checkpoint on registered tasks."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import torch

from paligemma2.data.registry import REGISTRY, register_stub_tasks
from paligemma2.modeling import PaliGemma2Config, build_paligemma2_model, load_paligemma2_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--model-size", default="3b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--limit", type=int, default=100, help="Maximum batches per task.")
    return parser.parse_args()


def evaluate(model: torch.nn.Module, loaders: Dict[str, torch.utils.data.DataLoader], device: str, limit: int) -> Dict[str, float]:
    results: Dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for task, loader in loaders.items():
            losses = []
            for idx, batch in enumerate(loader):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                losses.append(float(loss.detach().cpu()))
                if idx + 1 >= limit:
                    break
            results[f"{task}/loss"] = sum(losses) / max(1, len(losses))
    return results


def main() -> None:
    args = parse_args()
    model = build_paligemma2_model(PaliGemma2Config(model_size=args.model_size, resolution=args.resolution), device=args.device)
    load_paligemma2_checkpoint(model, str(args.checkpoint))
    if not REGISTRY.names():
        register_stub_tasks()
    loaders = {name: spec.loader(args.resolution, batch_size=8) for name, spec in ((name, REGISTRY.get(name)) for name in REGISTRY.names())}
    metrics = evaluate(model, loaders, args.device, args.limit)
    print(json.dumps({"checkpoint": str(args.checkpoint), "metrics": metrics}, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
