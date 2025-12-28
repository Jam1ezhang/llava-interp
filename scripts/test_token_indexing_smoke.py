from __future__ import annotations

import os
import sys

import torch

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))

from causal_metrics import parse_slice
from token_indexing import get_frame_token_spans, get_span_token_slice, get_visual_token_span


class DummyTokenizer:
    def __init__(self) -> None:
        self.special_tokens_map = {
            "vision_start": "<|vision_start|>",
            "vision_end": "<|vision_end|>",
        }
        self._token_to_id = {
            "<|vision_start|>": 1,
            "<|vision_end|>": 2,
        }
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def get_added_vocab(self) -> dict:
        return {}


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()


def _build_inputs() -> dict:
    input_ids = torch.tensor([[100, 1, 10, 11, 12, 13, 14, 15, 16, 17, 2, 200]])
    video_grid_thw = torch.tensor([[2, 2, 2]])
    return {"input_ids": input_ids, "video_grid_thw": video_grid_thw}


def main() -> None:
    inputs = _build_inputs()
    processor = DummyProcessor()
    visual_span = get_visual_token_span(inputs, processor)
    assert visual_span.start == 2 and visual_span.stop == 10

    frame_spans = get_frame_token_spans(inputs, processor)
    assert len(frame_spans) == 2
    assert frame_spans[0] == slice(2, 6)
    assert frame_spans[1] == slice(6, 10)

    span_slices = get_span_token_slice(((0, 1), (1, 2)), frame_spans)
    assert span_slices["spanA_slice"] == [slice(2, 6)]
    assert span_slices["spanB_slice"] == [slice(6, 10)]
    assert span_slices["union_slice"] == [slice(2, 10)]

    visual_frames = parse_slice("visual_frames:0:1", inputs["input_ids"].shape[1], inputs, processor)
    assert visual_frames == [slice(2, 6)]
    text_slice = parse_slice("text", inputs["input_ids"].shape[1], inputs, processor)
    for sl in text_slice:
        assert sl.stop <= visual_span.start or sl.start >= visual_span.stop


if __name__ == "__main__":
    main()
