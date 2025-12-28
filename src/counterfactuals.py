from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


def reverse_order(frames: Iterable[Image.Image]) -> List[Image.Image]:
    return list(reversed(list(frames)))


def local_swap(
    frames: Iterable[Image.Image],
    first_span: Tuple[int, int],
    second_span: Tuple[int, int],
) -> List[Image.Image]:
    frames_list = list(frames)
    first_start, first_end = first_span
    second_start, second_end = second_span
    if first_start < 0 or second_start < 0:
        raise ValueError("Span indices must be non-negative.")
    if first_end > second_start:
        raise ValueError("Spans must be non-overlapping and ordered.")
    if second_end > len(frames_list):
        raise ValueError("Span indices exceed frame count.")

    first_segment = frames_list[first_start:first_end]
    second_segment = frames_list[second_start:second_end]

    swapped = (
        frames_list[:first_start]
        + second_segment
        + frames_list[first_end:second_start]
        + first_segment
        + frames_list[second_end:]
    )
    return swapped


def motion_destroy(
    frames: Iterable[Image.Image],
    mode: str = "median",
) -> List[Image.Image]:
    frames_list = list(frames)
    if not frames_list:
        return []

    if mode == "first":
        reference = frames_list[0]
    elif mode == "median":
        stacked = np.stack([np.array(frame) for frame in frames_list], axis=0)
        median_frame = np.median(stacked, axis=0).astype(np.uint8)
        reference = Image.fromarray(median_frame)
    else:
        raise ValueError(f"Unsupported motion_destroy mode: {mode}")

    return [reference.copy() for _ in frames_list]


def motion_only(frames: Iterable[Image.Image]) -> List[Image.Image]:
    frames_list = list(frames)
    if len(frames_list) <= 1:
        return frames_list

    outputs: List[Image.Image] = [frames_list[0].copy()]
    prev = np.array(frames_list[0], dtype=np.int16)
    for frame in frames_list[1:]:
        current = np.array(frame, dtype=np.int16)
        diff = np.abs(current - prev).astype(np.uint8)
        outputs.append(Image.fromarray(diff))
        prev = current
    return outputs
