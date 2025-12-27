import argparse
import json
import os
import sys
from typing import Dict

from tqdm import tqdm

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))

from HookedLVLM import HookedLVLM
from VideoDatasets import VideoQADataset
from video_utils import format_multiple_choice, load_video_frames


def run_inference(
    annotations_path: str,
    video_root: str,
    output_path: str,
    model_id: str,
    device: str,
    num_frames: int,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    dataset = VideoQADataset(annotations_path=annotations_path, video_root=video_root, max_samples=limit)
    model = HookedLVLM(model_id=model_id, device=device, quantize=False)
    model.model.eval()

    results: Dict[str, Dict] = {}

    for sample in tqdm(dataset):
        question_id = sample.get("question_id") or sample["video_path"]
        video_frames = load_video_frames(
            sample["video_path"],
            start_time=sample.get("start"),
            end_time=sample.get("end"),
            num_frames=num_frames,
        )
        question = sample["question"]
        candidates = sample.get("candidates") or []
        prompt = format_multiple_choice(question, candidates) if candidates else question

        response = model.generate(
            video_frames,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        results[str(question_id)] = {
            "question": question,
            "candidates": candidates,
            "answer": sample.get("answer"),
            "response": response,
            "video_path": sample["video_path"],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2-VL/Qwen2.5-VL video inference.")
    parser.add_argument("--annotations", required=True, help="Path to the video QA json file.")
    parser.add_argument("--video_root", required=True, help="Root directory containing video files.")
    parser.add_argument("--output", required=True, help="Path to save inference results.")
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model ID for Qwen2-VL or Qwen2.5-VL.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample from each video.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens to generate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of samples.")

    args = parser.parse_args()
    run_inference(
        annotations_path=args.annotations,
        video_root=args.video_root,
        output_path=args.output,
        model_id=args.model_id,
        device=args.device,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
