from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


def load_video_frames(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    num_frames: int = 8,
) -> List[Image.Image]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 1.0

    start_frame = 0
    end_frame = total_frames - 1

    if start_time is not None:
        start_frame = max(int(start_time * fps), 0)
    if end_time is not None:
        end_frame = min(int(end_time * fps), total_frames - 1)

    if end_frame < start_frame:
        end_frame = start_frame

    frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
    frames: List[Image.Image] = []

    for idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = capture.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    capture.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def format_multiple_choice(question: str, candidates: List[str]) -> str:
    option_lines = [f"{chr(65 + idx)}. {candidate}" for idx, candidate in enumerate(candidates)]
    options_block = "\n".join(option_lines)
    return f"{question}\nOptions:\n{options_block}\nAnswer:"
