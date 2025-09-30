"""Model assembly utilities for PaliGemma 2 reproduction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

try:
    from transformers import AutoModelForCausalLM, SiglipVisionModel
except Exception as exc:  # pragma: no cover - import error surfaces at runtime
    raise ImportError(
        "transformers>=4.41.0 is required to construct the PaliGemma 2 model"
    ) from exc

from .config import ModelAssetConfig

LOGGER = logging.getLogger(__name__)


PALIGEMMA2_ASSETS: Dict[str, ModelAssetConfig] = {
    "3b-224": ModelAssetConfig(
        model_size="3b",
        resolution=224,
        tag="google/paligemma2-3b-pt-224",
        file_name="pytorch_model.bin",
        description="Base 3B checkpoint pretrained at 224px",
    ),
    "3b-448": ModelAssetConfig(
        model_size="3b",
        resolution=448,
        tag="google/paligemma2-3b-pt-448",
        file_name="pytorch_model.bin",
        description="Resolution-upshifted 3B checkpoint",
    ),
    "3b-896": ModelAssetConfig(
        model_size="3b",
        resolution=896,
        tag="google/paligemma2-3b-pt-896",
        file_name="pytorch_model.bin",
        description="High-resolution 3B checkpoint",
    ),
    "10b-224": ModelAssetConfig(
        model_size="10b",
        resolution=224,
        tag="google/paligemma2-10b-pt-224",
        file_name="pytorch_model.bin",
        description="Base 10B checkpoint",
    ),
    "10b-448": ModelAssetConfig(
        model_size="10b",
        resolution=448,
        tag="google/paligemma2-10b-pt-448",
        file_name="pytorch_model.bin",
        description="Intermediate resolution 10B",
    ),
    "10b-896": ModelAssetConfig(
        model_size="10b",
        resolution=896,
        tag="google/paligemma2-10b-pt-896",
        file_name="pytorch_model.bin",
        description="High-resolution 10B",
    ),
    "28b-224": ModelAssetConfig(
        model_size="28b",
        resolution=224,
        tag="google/paligemma2-28b-pt-224",
        file_name="pytorch_model.bin",
        description="Base 28B checkpoint",
    ),
    "28b-448": ModelAssetConfig(
        model_size="28b",
        resolution=448,
        tag="google/paligemma2-28b-pt-448",
        file_name="pytorch_model.bin",
        description="Intermediate resolution 28B",
    ),
    "28b-896": ModelAssetConfig(
        model_size="28b",
        resolution=896,
        tag="google/paligemma2-28b-pt-896",
        file_name="pytorch_model.bin",
        description="High-resolution 28B",
    ),
}

SIGLIP_SO400M_REPO = "google/siglip-so400m-patch14-384"
GEMMA2_TEXT_REPOS = {
    "3b": "google/gemma-2-2b-it",
    "10b": "google/gemma-2-9b-it",
    "28b": "google/gemma-2-27b-it",
}


@dataclass
class PaliGemma2Config:
    """Configuration for the composed multimodal model."""

    model_size: str
    resolution: int
    vision_repo: str = SIGLIP_SO400M_REPO
    text_repo: Optional[str] = None
    use_lora: bool = False
    freeze_vision: bool = False
    freeze_text: bool = False


class PaliGemma2ForConditionalGeneration(nn.Module):
    """Minimal SigLIP + Gemma 2 fusion resembling PaliGemma 2."""

    def __init__(self, vision_model: SiglipVisionModel, text_model: AutoModelForCausalLM, hidden_dim: int):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_projector = nn.Linear(vision_model.config.hidden_size, hidden_dim)
        nn.init.normal_(self.vision_projector.weight, std=0.02)
        nn.init.zeros_(self.vision_projector.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prepend_vision_token: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run a forward pass through the multimodal model."""

        vision_hidden = self.vision_model(pixel_values=pixel_values, output_hidden_states=True).last_hidden_state
        pooled = vision_hidden.mean(dim=1)
        vision_embedding = self.vision_projector(pooled)

        text_inputs = self.text_model.get_input_embeddings()(input_ids)
        if prepend_vision_token:
            vis_token = vision_embedding.unsqueeze(1)
            text_inputs = torch.cat([vis_token, text_inputs], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if labels is not None:
                pad_value = -100
                labels = torch.cat([torch.full((labels.size(0), 1), pad_value, dtype=labels.dtype, device=labels.device), labels], dim=1)

        outputs = self.text_model(
            inputs_embeds=text_inputs,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs


def build_paligemma2_model(cfg: PaliGemma2Config, device: Optional[str] = None, dtype: torch.dtype = torch.bfloat16) -> PaliGemma2ForConditionalGeneration:
    """Instantiate the multimodal model with released checkpoints."""

    text_repo = cfg.text_repo or GEMMA2_TEXT_REPOS.get(cfg.model_size)
    if text_repo is None:
        raise ValueError(f"No Gemma 2 text repo mapping for model size {cfg.model_size}")

    LOGGER.info("Loading SigLIP vision tower from %s", cfg.vision_repo)
    vision_model = SiglipVisionModel.from_pretrained(cfg.vision_repo, torch_dtype=dtype)
    LOGGER.info("Loading Gemma text decoder from %s", text_repo)
    text_model = AutoModelForCausalLM.from_pretrained(text_repo, torch_dtype=dtype)

    if cfg.freeze_vision:
        for p in vision_model.parameters():
            p.requires_grad = False
    if cfg.freeze_text:
        for p in text_model.parameters():
            p.requires_grad = False

    model = PaliGemma2ForConditionalGeneration(vision_model, text_model, hidden_dim=text_model.config.hidden_size)
    if device:
        model.to(device)
    return model


def load_paligemma2_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> Dict[str, torch.Tensor]:
    """Load a checkpoint produced by this repo's trainer."""

    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=strict)
    if missing or unexpected:
        LOGGER.warning("Missing keys: %s | Unexpected keys: %s", missing, unexpected)
    return state
