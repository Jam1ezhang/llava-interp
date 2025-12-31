import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from PIL import Image

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))

from activation_patching import patch_inputs
from HookedLVLM import HookedLVLM


def _make_frames(num_frames: int, size: int = 224) -> List[Image.Image]:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    base = Image.fromarray(array)
    return [base.copy() for _ in range(num_frames)]


def run(model_id: str, device: str) -> None:
    model = HookedLVLM(model_id=model_id, device=device, quantize=False)
    model.model.eval()

    frames = _make_frames(4)
    prompt = "Which option is correct? A. Cat B. Dog C. Car D. Tree"

    inputs = model.adapter.prepare_inputs(frames, prompt)
    emb = model.get_text_model_in(frames, prompt)
    visual_span, frame_spans = model.adapter.locate_visual_spans(inputs, emb, num_frames=4)

    assert 0 <= visual_span.start < visual_span.stop <= emb.shape[1]
    assert len(frame_spans) == 4
    for span in frame_spans:
        assert visual_span.start <= span.start < span.stop <= visual_span.stop

    token_slice = [frame_spans[0]]
    with patch_inputs(model.text_model, emb, token_slice):
        with torch.no_grad():
            outputs = model.model(**inputs)
    assert outputs.logits is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.model_id, args.device)
