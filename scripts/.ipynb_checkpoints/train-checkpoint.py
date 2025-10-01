"""Entry point for reproducing the three-stage PaliGemma 2 curriculum."""
from __future__ import annotations

import argparse
import pathlib

import torch

from metrics.training_dev_metrics import MetricConfig, TrainingDevMetrics
from paligemma2.config import StageSchedule
from paligemma2.modeling import PaliGemma2Config, build_paligemma2_model
from paligemma2.training.pipeline import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="3b", choices=["3b", "10b", "28b"])
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("runs"))
    parser.add_argument("--tb-dir", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("checkpoints"))
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per stage for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.precision]

    schedule = StageSchedule.default_schedule()
    if args.max_steps is not None:
        for stage in schedule.stages:
            stage.max_steps = min(stage.max_steps, args.max_steps)

    model = build_paligemma2_model(PaliGemma2Config(model_size=args.model_size, resolution=schedule.stages[0].resolution), dtype=dtype, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metric_cfg = MetricConfig(log_dir=str(args.log_dir), tb_dir=str(args.tb_dir) if args.tb_dir else None)
    metrics = TrainingDevMetrics(metric_cfg, model=model, optimizer=optimizer)

    trainer_cfg = TrainerConfig(
        device=args.device,
        precision=dtype,
        save_dir=str(args.save_dir),
        save_every=args.save_every,
    )
    trainer = Trainer(model=model, optimizer=optimizer, schedule=schedule, metrics=metrics, cfg=trainer_cfg)
    trainer.run()
    metrics.finalize()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
