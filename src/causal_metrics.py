from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch

from token_indexing import get_frame_token_spans, get_visual_token_span


TokenSlice = Union[slice, Sequence[slice]]


@dataclass
class MarginResult:
    margin: float
    logits: List[float]


def logit_margin(logits: Iterable[float], correct_index: int) -> float:
    logits_list = list(logits)
    correct = logits_list[correct_index]
    competitors = [logit for idx, logit in enumerate(logits_list) if idx != correct_index]
    if not competitors:
        return correct
    return correct - max(competitors)


def normalized_use_score(recovered: float, total_effect: float) -> Optional[float]:
    if total_effect == 0:
        return None
    return recovered / total_effect


def _parse_range_slice(spec: str, max_len: int) -> slice:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice spec: {spec}")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else max_len
    return slice(start, min(end, max_len))


def _parse_open_range(spec: str) -> Tuple[Optional[int], Optional[int]]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice spec: {spec}")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return start, end


def parse_slice(spec: str, max_len: int, inputs: Optional[Dict] = None, processor: Optional[object] = None) -> TokenSlice:
    if spec == "all":
        return slice(0, max_len)
    if spec == "visual":
        if inputs is None or processor is None:
            raise ValueError("parse_slice('visual') requires inputs and processor.")
        return get_visual_token_span(inputs, processor)
    if spec == "text":
        if inputs is None or processor is None:
            raise ValueError("parse_slice('text') requires inputs and processor.")
        visual_span = get_visual_token_span(inputs, processor)
        text_slices = [
            slice(0, visual_span.start),
            slice(visual_span.stop, max_len),
        ]
        text_slices = [sl for sl in text_slices if sl.stop > sl.start]
        if not text_slices:
            raise ValueError("No text tokens found outside the visual span.")
        return text_slices
    if spec.startswith("visual_frames:"):
        if inputs is None or processor is None:
            raise ValueError("parse_slice('visual_frames') requires inputs and processor.")
        frame_part = spec.split("visual_frames:", 1)[1]
        frame_spans = get_frame_token_spans(inputs, processor)
        start, end = _parse_open_range(frame_part)
        if start is None:
            start = 0
        if end is None or end > len(frame_spans):
            end = len(frame_spans)
        if start >= end:
            raise ValueError(f"Invalid visual_frames slice: {frame_part}")
        return list(frame_spans[start:end])
    return _parse_range_slice(spec, max_len)


def _gather_tokens(hidden_states: torch.Tensor, token_slice: TokenSlice) -> torch.Tensor:
    if isinstance(token_slice, slice):
        return hidden_states[:, token_slice, :]
    return torch.cat([hidden_states[:, sl, :] for sl in token_slice], dim=1)


def pool_hidden_states(hidden_states: torch.Tensor, token_slice: TokenSlice) -> torch.Tensor:
    return _gather_tokens(hidden_states, token_slice).mean(dim=1)
