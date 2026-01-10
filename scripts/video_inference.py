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
    stats_printed = False  # 标记是否已打印统计信息

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

        # # 只在第一次推理时打印 visual tokens 统计信息
        # if not stats_printed:
        #     inputs = model._prepare_inputs(video_frames, prompt)
        #     inputs_embeds = model.get_text_model_in(video_frames, prompt)
        #     print(inputs["input_ids"].tolist())
        #     visual_span, frame_spans = model.adapter.locate_visual_spans(
        #         inputs, inputs_embeds, num_frames=num_frames
        #     )
            
        #     # 计算每帧的 tokens 数量
        #     tokens_per_frame = []
        #     for span in frame_spans:
        #         tokens_per_frame.append(span.stop - span.start)
            
        #     total_visual_tokens = visual_span.stop - visual_span.start
        #     avg_tokens_per_frame = total_visual_tokens / num_frames
            
        #     # 打印统计信息（只打印第一行）
        #     print(f"Visual tokens: {total_visual_tokens} total, {avg_tokens_per_frame:.1f} per frame (frames: {num_frames}, per-frame tokens: {tokens_per_frame})")
        #     stats_printed = True

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
    parser.add_argument("--annotations", required=False,default="/home/user/gptdata/zym/codespace_interp/llava-interp/data/val_qa_situations_with_answer_frames_object_500.json", help="Path to the video QA json file.")
    parser.add_argument("--video_root", required=False,default="/home/user/gptdata/zym/codespace_interp/video-interp/playground/data/multiple_choice_qa/STAR/Charades_v1_480", help="Root directory containing video files.")
    parser.add_argument("--output", required=False,default="/home/user/gptdata/zym/codespace_interp/llava-interp/infer/results_ov.json", help="Path to save inference results.")
    parser.add_argument(
        "--model_id",
        default="/home/user/gptdata/zym/codespace_interp/llava-interp/ckpts/Qwen3-VL-8B-Instruct",
        help="Model ID for Qwen2-VL or Qwen2.5-VL.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to sample from each video.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens to generate.")
    parser.add_argument("--limit", type=int, default=2, help="Optional limit on the number of samples.")

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
