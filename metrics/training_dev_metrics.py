"""Training-time metric aggregation for developmental interpretability."""
from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

try:  # pragma: no cover - optional dependency
    from devinterp.slt.sampler import sample as devinterp_sample
    from devinterp.slt.llc import LLCEstimator
    from devinterp.optim import SGLD

    DEVINTERP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency missing
    DEVINTERP_AVAILABLE = False


class MetricModule(Protocol):
    """Interface for plugin metric modules."""

    def __call__(self, row: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:  # pragma: no cover - protocol
        ...


@dataclass
class MetricConfig:
    """Configuration parameters for :class:`TrainingDevMetrics`."""

    log_dir: str
    tb_dir: Optional[str] = None
    parquet_rollover_rows: int = 50_000
    # Cadences (in steps)
    log_every: int = 10
    ckpt_every: int = 200
    curvature_every: int = 1_000
    llc_every: int = 5_000
    activation_probe_every: int = 0
    # Geometry settings
    hutchinson_vecs: int = 1
    spectral_topk: int = 8
    # LLC settings
    llc_batch_size: int = 1_024
    llc_nbeta: Optional[int] = None
    llc_max_steps: int = 200
    llc_sampler_lr: float = 5e-4
    # Serialization
    manifest_name: str = "CHECKPOINT_MANIFEST.parquet"


@dataclass
class TrainingDevMetrics:
    """Collect and persist metrics for the PaliGemma 2 training pipeline."""

    cfg: MetricConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    global_step: int = 0
    _tb: Optional[SummaryWriter] = field(default=None, init=False)
    _parquet_path: pathlib.Path = field(default=None, init=False)
    _buffer: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _curvature_closure: Optional[callable] = field(default=None, init=False)
    _llc_loader: Optional[Iterable] = field(default=None, init=False)
    _llc_sampler_kwargs: Dict[str, Any] = field(default_factory=dict, init=False)
    _modules: List[MetricModule] = field(default_factory=list, init=False)
    _prev_flat_params: Optional[torch.Tensor] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.root = pathlib.Path(self.cfg.log_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.root / "METRICS"
        self.metrics_path.mkdir(exist_ok=True)
        self._parquet_path = self.metrics_path / "train_timeseries.parquet"
        if self.cfg.tb_dir:
            self._tb = SummaryWriter(self.cfg.tb_dir)
        self._init_manifest()

    # ---------- configuration hooks ----------
    def register_curvature_closure(self, closure: callable) -> None:
        """Register a closure that recomputes the loss for curvature probes."""

        self._curvature_closure = closure

    def register_llc_inputs(self, loader: Iterable, **sampler_kwargs: Any) -> None:
        """Register a dataloader and sampler kwargs for DevInterp LLC estimation."""

        self._llc_loader = loader
        self._llc_sampler_kwargs = sampler_kwargs

    def register_metric_module(self, module: MetricModule) -> None:
        """Attach a plugin metric module executed on every log row."""

        self._modules.append(module)

    # ---------- public API (called from the training loop) ----------
    def on_step_end(self, loss: float, batch_size: int, lr: float, **extras: Any) -> None:
        """Record fast-path metrics every step (controlled by ``cfg.log_every``)."""

        self.global_step += 1
        if self.global_step % self.cfg.log_every == 0:
            grad_norm = float(self._grad_global_norm())
            weight_norm = float(self._weight_global_norm())
            row: Dict[str, Any] = {
                "ts": time.time(),
                "step": self.global_step,
                "loss": float(loss),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "grad_norm": grad_norm,
                "weight_norm": weight_norm,
                "grad_to_weight_ratio": self._safe_ratio(grad_norm, weight_norm),
                "optimizer/momentum": self._get_momentum_stat(),
            }
            for k, v in extras.items():
                row[f"extra/{k}"] = self._to_float(v)
            self._log_row(row)

        if self.cfg.curvature_every and self.global_step % self.cfg.curvature_every == 0:
            self._log_curvature_proxies()

        if DEVINTERP_AVAILABLE and self.cfg.llc_every and self.global_step % self.cfg.llc_every == 0:
            self._log_llc_estimate()

    def on_checkpoint(self, ckpt_path: str, stage: str) -> None:
        """Record checkpoint metadata for post-hoc sweeps."""

        info = {
            "ts": time.time(),
            "step": self.global_step,
            "stage": stage,
            "ckpt_path": ckpt_path,
            "param_stats": self._param_space_summary(topk=self.cfg.spectral_topk),
        }
        self._append_manifest(info)

    # ---------- post-hoc (run in separate scripts) ----------
    @torch.inference_mode()
    def export_activations(self, dataloader: Iterable, layers: Iterable[str], out_path: str) -> None:
        """Export activation snapshots for later interpretability analysis."""

        records: List[Dict[str, Any]] = []
        hooks: List[Any] = []

        def _hook(name: str):
            def fn(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
                sample = out[:1].detach().cpu().numpy()
                records.append(
                    {
                        "layer": name,
                        "step": self.global_step,
                        "shape": tuple(out.shape),
                        "sample": sample,
                    }
                )

            return fn

        try:
            device = next(self.model.parameters()).device
            for name, module in self.model.named_modules():
                if name in layers:
                    hooks.append(module.register_forward_hook(_hook(name)))
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                _ = self.model(**batch)
        finally:
            for hook in hooks:
                hook.remove()

        out_file = pathlib.Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_parquet(out_file)

    # ---------- internals ----------
    def _grad_global_norm(self) -> float:
        total = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total += param.grad.data.float().pow(2).sum().item()
        return math.sqrt(total) if total > 0 else 0.0

    def _weight_global_norm(self) -> float:
        total = 0.0
        for param in self.model.parameters():
            total += param.data.float().pow(2).sum().item()
        return math.sqrt(total)

    def _get_momentum_stat(self) -> float:
        for group in self.optimizer.param_groups:
            momentum = group.get("betas")
            if momentum:
                return float(momentum[0])
            momentum = group.get("momentum")
            if momentum:
                return float(momentum)
        return float("nan")

    def _param_space_summary(self, topk: int = 8) -> Dict[str, Any]:
        layers: List[Dict[str, Any]] = []
        flat_params: List[torch.Tensor] = []
        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                weight = module.weight.detach()
                norm = float(weight.norm().item())
                sparsity = float((weight == 0).float().mean().item())
                sv1 = self._spectral_proxy(weight, power_iter=2)
                layers.append({"name": name, "norm": norm, "sv1_proxy": sv1, "sparsity": sparsity, "shape": tuple(weight.shape)})
                flat_params.append(weight.flatten().float().cpu())
        flat = torch.cat(flat_params) if flat_params else torch.tensor([], dtype=torch.float32)
        cosine = float("nan")
        if self._prev_flat_params is not None and flat.numel() == self._prev_flat_params.numel():
            cosine = torch.nn.functional.cosine_similarity(flat, self._prev_flat_params, dim=0).item()
        self._prev_flat_params = flat
        return {"layers": layers, "cosine_drift": cosine, "num_layers": len(layers)}

    def _spectral_proxy(self, weight: torch.Tensor, power_iter: int = 2) -> float:
        if weight.ndim > 2:
            w_mat = weight.flatten(1)
        else:
            w_mat = weight
        device = w_mat.device
        vec = torch.randn(w_mat.shape[-1], device=device)
        for _ in range(power_iter):
            vec = torch.mv(w_mat.t(), torch.mv(w_mat, vec))
            vec = vec / (vec.norm() + 1e-12)
        sv = torch.mv(w_mat, vec)
        return float(sv.norm().item())

    def _log_curvature_proxies(self) -> None:
        if self._curvature_closure is None:
            self._log_row({"step": self.global_step, "warn/curvature": "closure_not_registered"})
            return
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return
        estimates = []
        try:
            for _ in range(self.cfg.hutchinson_vecs):
                loss = self._curvature_closure()
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                vectors = []
                for grad in grads:
                    v = torch.randint_like(grad, low=0, high=2)
                    v = (v * 2 - 1).to(dtype=grad.dtype)
                    vectors.append(v)
                inner = torch.zeros((), device=params[0].device)
                for g, v in zip(grads, vectors):
                    inner = inner + (g * v).sum()
                hvps = torch.autograd.grad(inner, params, retain_graph=False)
                trace = sum((h * v).sum().item() for h, v in zip(hvps, vectors))
                estimates.append(trace)
            trace_est = float(sum(estimates) / max(1, len(estimates)))
            self._log_row({"step": self.global_step, "curvature/hutchinson_trace": trace_est})
        except Exception as exc:  # pragma: no cover - protective
            self._log_row({"step": self.global_step, "warn/curvature_error": str(exc)})

    def _log_llc_estimate(self) -> None:
        if not DEVINTERP_AVAILABLE:
            self._log_row({"step": self.global_step, "warn/llc_missing": True})
            return
        if self._llc_loader is None:
            self._log_row({"step": self.global_step, "warn/llc_loader_missing": True})
            return
        try:
            estimator = LLCEstimator(nbeta=self.cfg.llc_nbeta)
            sampler = SGLD(lr=self.cfg.llc_sampler_lr)
            devinterp_sample(
                model=self.model,
                data_loader=self._llc_loader,
                sampler=sampler,
                max_steps=self.cfg.llc_max_steps,
                batch_size=self.cfg.llc_batch_size,
                callbacks=[estimator],
                **self._llc_sampler_kwargs,
            )
            if hasattr(estimator, "summary"):
                results = estimator.summary()
            elif hasattr(estimator, "get_results"):
                results = estimator.get_results()
            else:
                results = {"mean": float("nan")}
            formatted = {f"llc/{k}": self._to_float(v) for k, v in results.items()}
            formatted["step"] = self.global_step
            self._log_row(formatted)
        except Exception as exc:  # pragma: no cover - optional path
            self._log_row({"step": self.global_step, "warn/llc_error": str(exc)})

    def _init_manifest(self) -> None:
        path = self.root / self.cfg.manifest_name
        if not path.exists():
            pd.DataFrame(columns=["ts", "step", "stage", "ckpt_path", "param_stats"]).to_parquet(path)

    def _append_manifest(self, record: Dict[str, Any]) -> None:
        path = self.root / self.cfg.manifest_name
        df_old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        df_new = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
        df_new.to_parquet(path)

    def _log_row(self, row: Dict[str, Any]) -> None:
        for module in self._modules:
            try:
                extra = module(row, self.global_step)
                if extra:
                    row.update(extra)
            except Exception as exc:  # pragma: no cover - plugin errors shouldn't crash training
                row[f"warn/plugin/{module.__class__.__name__}"] = str(exc)
        self._buffer.append(row)
        if self._tb:
            for key, value in row.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    self._tb.add_scalar(key, float(value), self.global_step)
        if len(self._buffer) >= self.cfg.parquet_rollover_rows or (self.global_step % self.cfg.log_every == 0):
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        if self._parquet_path.exists():
            df_prev = pd.read_parquet(self._parquet_path)
            df = pd.concat([df_prev, df], ignore_index=True)
        df.to_parquet(self._parquet_path)
        self._buffer.clear()

    def finalize(self) -> None:
        self._flush()
        if self._tb:
            self._tb.flush()
            self._tb.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.finalize()
        except Exception:
            pass

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    @staticmethod
    def _safe_ratio(a: float, b: float) -> float:
        return a / b if b not in {0.0, 0} else float("inf")


__all__ = ["TrainingDevMetrics", "MetricConfig", "MetricModule", "DEVINTERP_AVAILABLE"]
