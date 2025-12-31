from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn


class VLMAdapter(ABC):
    def __init__(self, model_id: str, device: str, torch_dtype: torch.dtype, quantize: bool):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.quantize = quantize
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def prepare_inputs(self, frames: List, prompt: str) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def get_text_model(self) -> nn.Module:
        ...

    @abstractmethod
    def get_layers(self) -> Sequence[nn.Module]:
        ...

    @abstractmethod
    def locate_visual_spans(
        self,
        inputs: Dict[str, torch.Tensor],
        inputs_embeds: torch.Tensor,
        num_frames: int,
    ) -> Tuple[slice, List[slice]]:
        """
        返回：(visual_span, frame_spans)
        visual_span: embedding 序列里整段视觉 tokens
        frame_spans: 每帧对应的 token slice（若无法精确分帧，也要返回近似均分）
        """
        ...
