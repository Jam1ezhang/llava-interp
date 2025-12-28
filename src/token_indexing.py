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


def infer_T_eff(visual_len: int, video_grid_thw: Tuple[int, int, int]) -> int:
    """推断有效的帧数 T_eff，基于实际的视觉 token 长度。
    
    Args:
        visual_len: 实际的视觉 token 数量（从 special tokens 得到）
        video_grid_thw: (T, H, W) 元组，表示原始视频网格维度
        
    Returns:
        有效的帧数 T_eff
    """
    t, h, w = video_grid_thw
    tokens_per_frame = h * w
    if tokens_per_frame == 0:
        raise ValueError(f"Invalid video_grid_thw: tokens_per_frame (h*w) cannot be 0")
    t_eff = visual_len // tokens_per_frame
    if visual_len % tokens_per_frame != 0:
        raise ValueError(
            f"Visual token length {visual_len} is not divisible by tokens_per_frame {tokens_per_frame}. "
            f"This may indicate an inconsistent tokenization."
        )
    return t_eff


def get_frame_token_spans(inputs: Dict, processor) -> List[slice]:
    """Return a list of token spans, one per bin in the video.
    
    使用 T_eff 而不是 video_grid_thw 的 T 来分组，以处理下采样情况。
    """

    visual_span = get_visual_token_span(inputs, processor)
    visual_len = visual_span.stop - visual_span.start
    t, h, w = _get_video_grid_thw(inputs)
    
    # 使用 T_eff 而不是原始的 T
    t_eff = infer_T_eff(visual_len, (t, h, w))
    tokens_per_bin = visual_len // t_eff
    
    spans = []
    for bin_idx in range(t_eff):
        start = visual_span.start + bin_idx * tokens_per_bin
        end = start + tokens_per_bin
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
    raw_T: Optional[int] = None,
) -> Dict[str, List[slice]]:
    """Resolve token slices for two swapped frame spans.

    Args:
        frame_spans: ((startA, endA), (startB, endB)) 在原始帧空间的 span
        frame_token_spans: 基于 T_eff 的 token spans（每个 bin 一个 span）
        raw_T: 原始帧数（如果提供，用于将 raw 帧空间的 span 映射到 bins）
        
    Returns:
        dict containing spanA_slice, spanB_slice, and union_slice (merged slices).
        如果映射后两个 span 不可区分，会记录警告信息。
    """

    t_eff = len(frame_token_spans)
    span_slices: Dict[str, List[slice]] = {}
    warnings: List[str] = []
    
    # 如果提供了 raw_T，需要将 raw 帧空间的 span 映射到 T_eff 的 bins
    if raw_T is not None and raw_T > t_eff:
        # 计算每个 bin 包含多少原始帧
        frames_per_bin = raw_T / t_eff
        
        mapped_spans = []
        for name, (start, end) in zip(("spanA", "spanB"), frame_spans):
            if start < 0 or end > raw_T or start >= end:
                raise ValueError(
                    f"Invalid frame span {start}:{end} for {name} with {raw_T} raw frames."
                )
            
            # 映射到 bin 索引
            bin_start = int(start / frames_per_bin)
            bin_end = int((end - 1) / frames_per_bin) + 1  # 向上取整
            
            # 确保 bin 索引在有效范围内
            bin_start = max(0, min(bin_start, t_eff - 1))
            bin_end = max(bin_start + 1, min(bin_end, t_eff))
            
            mapped_spans.append((name, bin_start, bin_end))
        
        # 检查映射后是否可区分
        (nameA, binA_start, binA_end), (nameB, binB_start, binB_end) = mapped_spans
        if binA_start == binB_start and binA_end == binB_end:
            warnings.append(
                f"Warning: Spans {frame_spans[0]} and {frame_spans[1]} in raw frame space "
                f"map to the same bin range [{binA_start}:{binA_end}] after downsampling. "
                f"Skipping fine-grained distinction."
            )
            # 如果不可区分，返回空列表
            return {
                "spanA_slice": [],
                "spanB_slice": [],
                "union_slice": [],
                "warnings": warnings,
            }
        
        # 使用映射后的 bin 索引
        for name, bin_start, bin_end in mapped_spans:
            if bin_start < 0 or bin_end > t_eff or bin_start >= bin_end:
                raise ValueError(
                    f"Invalid mapped bin span {bin_start}:{bin_end} for {name} with {t_eff} bins."
                )
            span_slices[name] = list(frame_token_spans[bin_start:bin_end])
    else:
        # 直接使用 frame_spans，假设它们已经在 T_eff 空间
        max_frames = t_eff
        for name, (start, end) in zip(("spanA", "spanB"), frame_spans):
            if start < 0 or end > max_frames or start >= end:
                raise ValueError(
                    f"Invalid frame span {start}:{end} for {name} with {max_frames} frames."
                )
            span_slices[name] = list(frame_token_spans[start:end])

    union = _merge_slices(span_slices["spanA"] + span_slices["spanB"])
    span_slices["union"] = union
    
    result = {
        "spanA_slice": span_slices["spanA"],
        "spanB_slice": span_slices["spanB"],
        "union_slice": span_slices["union"],
    }
    if warnings:
        result["warnings"] = warnings
    
    return result
