from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch


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


def parse_slice(spec: str, max_len: int) -> slice:
    if spec == "all":
        return slice(0, max_len)
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice spec: {spec}")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else max_len
    return slice(start, min(end, max_len))


def pool_hidden_states(hidden_states: torch.Tensor, token_slice: slice) -> torch.Tensor:
    return hidden_states[:, token_slice, :].mean(dim=1)
