"""Training pipeline approximating the PaliGemma 2 three-stage curriculum."""
from __future__ import annotations

import contextlib
import logging
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import StageSchedule, TrainingStageConfig
from ..data.registry import REGISTRY, register_stub_tasks
from metrics.training_dev_metrics import TrainingDevMetrics

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration knobs for the trainer."""

    device: str = "cuda"
    precision: torch.dtype = torch.bfloat16
    grad_clip: float = 1.0
    save_dir: str = "outputs"
    save_every: int = 500
    keep_last: int = 5
    use_fsdp: bool = False
    zero_stage: int = 0
    activation_checkpointing: bool = False
    resume_from: Optional[str] = None
    enable_ema: bool = False
    ema_decay: float = 0.9999


class CheckpointManager:
    """Persist high-frequency checkpoints with metadata."""

    def __init__(self, save_dir: str, keep_last: int = 5) -> None:
        self.root = pathlib.Path(save_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._checkpoints: List[pathlib.Path] = []

    def save(self, step: int, stage: str, model: nn.Module, optimizer: Optimizer, scheduler: Optional[LambdaLR], metrics: TrainingDevMetrics, ema_state: Optional[Dict[str, torch.Tensor]] = None) -> pathlib.Path:
        ckpt_path = self.root / f"ckpt_step{step:08d}.pt"
        state = {
            "step": step,
            "stage": stage,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "rng_state": torch.get_rng_state(),
        }
        if ema_state:
            state["ema"] = ema_state
        torch.save(state, ckpt_path)
        self._checkpoints.append(ckpt_path)
        self._checkpoints = self._checkpoints[-self.keep_last :]
        metrics.on_checkpoint(str(ckpt_path), stage=stage)
        return ckpt_path


class WeightedMultiTaskIterator:
    """Yield batches by sampling tasks proportionally to mixture weights."""

    def __init__(self, loaders: Dict[str, DataLoader], weights: Dict[str, float]):
        self.loaders = loaders
        self._iters: Dict[str, Iterator] = {}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Task weights must be positive")
        self.probs = {name: weight / total for name, weight in weights.items()}

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
        while True:
            task = random.choices(list(self.probs.keys()), weights=list(self.probs.values()))[0]
            batch = self._next_batch(task)
            yield task, batch

    def _next_batch(self, name: str):
        if name not in self._iters:
            self._iters[name] = iter(self.loaders[name])
        try:
            return next(self._iters[name])
        except StopIteration:
            self._iters[name] = iter(self.loaders[name])
            return next(self._iters[name])


class Trainer:
    """Minimal trainer supporting the required curriculum and metrics."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        schedule: StageSchedule,
        metrics: TrainingDevMetrics,
        cfg: TrainerConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.metrics = metrics
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        use_scaler = self.device.type == "cuda" and cfg.precision == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        self.checkpoints = CheckpointManager(cfg.save_dir, cfg.keep_last)
        if not REGISTRY.names():
            register_stub_tasks()

    def _build_scheduler(self, stage_cfg: TrainingStageConfig, total_steps: int) -> LambdaLR:
        def lr_lambda(step: int) -> float:
            if step < stage_cfg.warmup_steps:
                return step / max(1, stage_cfg.warmup_steps)
            progress = (step - stage_cfg.warmup_steps) / max(1, total_steps - stage_cfg.warmup_steps)
            return max(0.0, 1.0 - progress)

        return LambdaLR(self.optimizer, lr_lambda)

    def run(self) -> None:
        for stage_cfg in self.schedule.stages:
            LOGGER.info("Starting stage %s", stage_cfg.name)
            loaders = REGISTRY.loaders(stage_cfg.task_weights, stage_cfg.resolution, stage_cfg.batch_size)
            iterator = iter(WeightedMultiTaskIterator(loaders, stage_cfg.task_weights))
            scheduler = self._build_scheduler(stage_cfg, stage_cfg.max_steps)
            if loaders:
                # Use the first loader as LLC reference if requested.
                first_loader = next(iter(loaders.values()))
                self.metrics.register_llc_inputs(first_loader)
            for step in range(stage_cfg.max_steps):
                task_name, batch = next(iterator)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                autocast_ctx = (
                    torch.autocast(device_type=self.device.type, dtype=self.cfg.precision)
                    if self.device.type in {"cuda", "xpu"}
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                self.optimizer.zero_grad()
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]
                self.metrics.on_step_end(loss=float(loss.detach().cpu()), batch_size=batch["input_ids"].size(0), lr=lr, task=task_name)
                self.metrics.register_curvature_closure(lambda batch=batch: self._recompute_loss(batch))

                if self.metrics.global_step % self.cfg.save_every == 0:
                    self.checkpoints.save(self.metrics.global_step, stage_cfg.name, self.model, self.optimizer, scheduler, self.metrics)

            LOGGER.info("Completed stage %s", stage_cfg.name)

    def _recompute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        clone = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                clone[key] = value.detach().clone().requires_grad_(value.requires_grad)
            else:
                clone[key] = value
        outputs = self.model(**clone)
        return outputs.loss if hasattr(outputs, "loss") else outputs[0]


__all__ = [
    "Trainer",
    "TrainerConfig",
    "CheckpointManager",
]
