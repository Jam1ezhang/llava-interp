from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoConfig

from .base import VLMAdapter


def _normalize_model_name(value: Optional[str]) -> str:
    return (value or "").lower()


def _resolve_model_type(model_id: str) -> Optional[str]:
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None
    return getattr(config, "model_type", None)


def create_adapter(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    quantize: bool,
) -> VLMAdapter:
    model_name = _normalize_model_name(model_id)
    model_type = _normalize_model_name(_resolve_model_type(model_id))

    if model_name == "qwen2-vl" or model_type == "qwen2-vl":
        from .qwen2_vl import Qwen2VLAdapter

        return Qwen2VLAdapter(model_id, device, torch_dtype, quantize)

    llava_markers = ("llava", "onevision", "llava-next", "llava_video")
    if any(marker in model_name for marker in llava_markers) or any(
        marker in model_type for marker in llava_markers
    ):
        from .llava_hf import LlavaHFAdapter

        return LlavaHFAdapter(model_id, device, torch_dtype, quantize)

    raise ValueError(
        "Unsupported model_id or model_type for adapter selection. "
        f"model_id={model_id!r}, model_type={model_type!r}. "
        "Expected qwen2-vl or llava/onevision/llava-next/llava_video variants."
    )


def make_adapter(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    quantize: bool,
) -> VLMAdapter:
    return create_adapter(model_id, device, torch_dtype, quantize)
