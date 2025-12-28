from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class VisualSpanResult:
    """Resolved visual token span metadata."""

    span: slice
    num_tokens: int


def _flatten_input_ids(inputs: Dict) -> List[int]:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("inputs must include 'input_ids'.")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D (batch, seq). Got shape {tuple(input_ids.shape)}")
    return input_ids[0].tolist()


def _candidate_special_tokens() -> List[Tuple[str, str]]:
    return [
        ("vision_start", "<|vision_start|>"),
        ("vision_end", "<|vision_end|>"),
        ("image_start", "<|image_start|>"),
        ("image_end", "<|image_end|>"),
        ("video_start", "<|video_start|>"),
        ("video_end", "<|video_end|>"),
    ]


def _lookup_token_id(tokenizer, token_str: str) -> Optional[int]:
    if token_str in getattr(tokenizer, "special_tokens_map", {}).values():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        return token_id if token_id != tokenizer.unk_token_id else None
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    if token_id == tokenizer.unk_token_id:
        return None
    return token_id


def _find_special_token_span(input_ids: List[int], tokenizer) -> Optional[VisualSpanResult]:
    start_ids: List[int] = []
    end_ids: List[int] = []
    for name, token in _candidate_special_tokens():
        token_id = _lookup_token_id(tokenizer, token)
        if token_id is None:
            continue
        if name.endswith("start"):
            start_ids.append(token_id)
        else:
            end_ids.append(token_id)

    if not start_ids or not end_ids:
        return None

    start_idx = next((idx for idx, tok in enumerate(input_ids) if tok in start_ids), None)
    if start_idx is None:
        return None
    end_idx = next(
        (idx for idx in range(start_idx + 1, len(input_ids)) if input_ids[idx] in end_ids), None
    )
    if end_idx is None:
        return None
    if end_idx <= start_idx + 1:
        raise ValueError("Found vision start/end tokens but no visual tokens between them.")
    span = slice(start_idx + 1, end_idx)
    return VisualSpanResult(span=span, num_tokens=span.stop - span.start)


def _get_video_grid_thw(inputs: Dict) -> Tuple[int, int, int]:
    grid = inputs.get("video_grid_thw")
    if grid is None:
        raise ValueError("inputs must include 'video_grid_thw' for fallback visual token localization.")
    if grid.ndim == 2:
        grid = grid[0]
    if grid.numel() != 3:
        raise ValueError(f"video_grid_thw must have 3 values (t,h,w). Got {grid.tolist()}")
    t, h, w = [int(x) for x in grid.tolist()]
    return t, h, w


def _candidate_visual_pad_ids(tokenizer) -> List[int]:
    candidates: List[int] = []
    for token, token_id in getattr(tokenizer, "get_added_vocab", lambda: {})().items():
        if "image" in token or "video" in token or "vision" in token:
            if "pad" in token or "patch" in token:
                candidates.append(token_id)
    for token in getattr(tokenizer, "special_tokens_map", {}).values():
        if "pad" in token and ("image" in token or "video" in token or "vision" in token):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                candidates.append(token_id)
    return list(set(candidates))


def _find_pad_token_span(input_ids: List[int], num_tokens: int, tokenizer) -> VisualSpanResult:
    pad_ids = _candidate_visual_pad_ids(tokenizer)
    if not pad_ids:
        raise ValueError("Could not find visual pad token ids for fallback visual span lookup.")
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, tok in enumerate(input_ids):
        if tok in pad_ids:
            if start is None:
                start = idx
        else:
            if start is not None:
                segments.append((start, idx))
                start = None
    if start is not None:
        segments.append((start, len(input_ids)))

    for seg_start, seg_end in segments:
        if seg_end - seg_start == num_tokens:
            return VisualSpanResult(span=slice(seg_start, seg_end), num_tokens=num_tokens)

    raise ValueError(
        "Could not locate a contiguous visual token block matching the expected length. "
        "Ensure vision start/end tokens exist or provide valid video_grid_thw/pad tokens."
    )


def get_visual_token_span(inputs: Dict, processor) -> slice:
    """Return the slice covering only visual tokens.

    Uses vision start/end special tokens when available. Falls back to video_grid_thw and
    visual pad tokens to infer a contiguous visual token block.
    """

    input_ids = _flatten_input_ids(inputs)
    tokenizer = processor.tokenizer
    span_result = _find_special_token_span(input_ids, tokenizer)
    if span_result is not None:
        return span_result.span

    t, h, w = _get_video_grid_thw(inputs)
    num_tokens = t * h * w
    span_result = _find_pad_token_span(input_ids, num_tokens, tokenizer)
    return span_result.span


def get_frame_token_spans(inputs: Dict, processor) -> List[slice]:
    """Return a list of token spans, one per frame in the video."""

    visual_span = get_visual_token_span(inputs, processor)
    t, h, w = _get_video_grid_thw(inputs)
    tokens_per_frame = h * w
    total_tokens = tokens_per_frame * t
    if visual_span.stop - visual_span.start != total_tokens:
        raise ValueError(
            "Visual span length does not match video_grid_thw. "
            f"Expected {total_tokens} tokens, got {visual_span.stop - visual_span.start}."
        )
    spans = []
    for frame_idx in range(t):
        start = visual_span.start + frame_idx * tokens_per_frame
        end = start + tokens_per_frame
        spans.append(slice(start, end))
    return spans


def _merge_slices(slices: Iterable[slice]) -> List[slice]:
    sorted_slices = sorted(slices, key=lambda s: s.start)
    merged: List[slice] = []
    for sl in sorted_slices:
        if not merged:
            merged.append(sl)
            continue
        prev = merged[-1]
        if sl.start <= prev.stop:
            merged[-1] = slice(prev.start, max(prev.stop, sl.stop))
        else:
            merged.append(sl)
    return merged


def get_span_token_slice(
    frame_spans: Tuple[Tuple[int, int], Tuple[int, int]],
    frame_token_spans: Sequence[slice],
) -> Dict[str, List[slice]]:
    """Resolve token slices for two swapped frame spans.

    Returns a dict containing spanA_slice, spanB_slice, and union_slice (merged slices).
    """

    max_frames = len(frame_token_spans)
    span_slices: Dict[str, List[slice]] = {}
    for name, (start, end) in zip(("spanA", "spanB"), frame_spans):
        if start < 0 or end > max_frames or start >= end:
            raise ValueError(f"Invalid frame span {start}:{end} for {name} with {max_frames} frames.")
        span_slices[name] = list(frame_token_spans[start:end])

    union = _merge_slices(span_slices["spanA"] + span_slices["spanB"])
    span_slices["union"] = union
    return {
        "spanA_slice": span_slices["spanA"],
        "spanB_slice": span_slices["spanB"],
        "union_slice": span_slices["union"],
    }
