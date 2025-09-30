"""Configuration primitives for PaliGemma 2 reproduction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelAssetConfig:
    """Describe a released PaliGemma 2 asset."""

    model_size: str  # e.g. "3b", "10b", "28b"
    resolution: int  # input resolution (224, 448, 896)
    tag: str  # huggingface or kaggle identifier
    file_name: str  # expected file name for weight archive
    sha256: Optional[str] = None
    description: str = ""


@dataclass
class TrainingStageConfig:
    """Hyperparameters for a coarse training stage."""

    name: str
    resolution: int
    num_examples: int
    batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    task_weights: Dict[str, float]
    sequence_length: int
    notes: str = ""


@dataclass
class StageSchedule:
    """Container capturing the three-stage curriculum described in the report."""

    stages: List[TrainingStageConfig] = field(default_factory=list)

    @classmethod
    def default_schedule(cls) -> "StageSchedule":
        """Return a coarse approximation of the public three-stage curriculum."""
        stages = [
            TrainingStageConfig(
                name="stage1",
                resolution=224,
                num_examples=1_000_000_000,
                batch_size=4096,
                max_steps=250_000,
                learning_rate=3e-4,
                weight_decay=0.1,
                warmup_steps=2_000,
                task_weights={
                    "captioning": 0.4,
                    "ocr": 0.3,
                    "grounded_captioning": 0.2,
                    "vision_language": 0.1,
                },
                sequence_length=1024,
                notes="Joint pre-training with mixed tasks at 224px",
            ),
            TrainingStageConfig(
                name="stage2a",
                resolution=448,
                num_examples=50_000_000,
                batch_size=2048,
                max_steps=20_000,
                learning_rate=2e-4,
                weight_decay=0.05,
                warmup_steps=1_000,
                task_weights={
                    "captioning": 0.3,
                    "ocr": 0.4,
                    "grounded_captioning": 0.2,
                    "vision_language": 0.1,
                },
                sequence_length=1536,
                notes="Resolution upshift to 448px with OCR emphasis",
            ),
            TrainingStageConfig(
                name="stage2b",
                resolution=896,
                num_examples=10_000_000,
                batch_size=1024,
                max_steps=10_000,
                learning_rate=1.5e-4,
                weight_decay=0.05,
                warmup_steps=500,
                task_weights={
                    "captioning": 0.2,
                    "ocr": 0.45,
                    "grounded_captioning": 0.25,
                    "vision_language": 0.1,
                },
                sequence_length=2048,
                notes="High-resolution phase with long-sequence decoding",
            ),
            TrainingStageConfig(
                name="stage3",
                resolution=896,
                num_examples=5_000_000,
                batch_size=512,
                max_steps=5_000,
                learning_rate=1e-4,
                weight_decay=0.02,
                warmup_steps=500,
                task_weights={
                    "captioning": 0.25,
                    "ocr": 0.35,
                    "grounded_captioning": 0.25,
                    "vision_language": 0.15,
                },
                sequence_length=2048,
                notes="Per-task fine-tuning sweeps; adjust LR per model scale",
            ),
        ]
        return cls(stages=stages)

    def to_dict(self) -> List[Dict[str, object]]:
        """Serialise the schedule to a list of dictionaries for logging."""
        return [vars(stage) for stage in self.stages]
