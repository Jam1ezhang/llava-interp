import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))

from activation_patching import PatchSpec, capture_layer_outputs, patch_inputs, patch_layer_outputs
from causal_metrics import logit_margin, normalized_use_score, parse_slice, pool_hidden_states
from counterfactuals import local_swap, motion_destroy, motion_only, reverse_order
from HookedLVLM import HookedLVLM
from VideoDatasets import VideoQADataset
from video_utils import format_multiple_choice, load_video_frames


def _label_candidates(candidates: List[str]) -> List[str]:
    return [chr(65 + idx) for idx in range(len(candidates))]


def _resolve_correct_index(answer: Optional[str], candidates: List[str]) -> Optional[int]:
    if answer is None:
        return None
    labels = _label_candidates(candidates)
    if answer in labels:
        return labels.index(answer)
    if answer in candidates:
        return candidates.index(answer)
    return None


def _score_option_logits(model: HookedLVLM, inputs: Dict, labels: List[str]) -> List[float]:
    with torch.no_grad():
        outputs = model.model(**inputs)
    logits = outputs.logits[:, -1, :].squeeze(0)
    tokenizer = model.processor.tokenizer
    scores: List[float] = []
    for label in labels:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Label '{label}' is not a single token for this tokenizer.")
        scores.append(logits[token_ids[0]].item())
    return scores


def _make_counterfactuals(
    frames: List,
    swap_spans: Tuple[Tuple[int, int], Tuple[int, int]],
    include_motion_only: bool,
) -> Dict[str, List]:
    counterfactuals = {
        "order_reverse": reverse_order(frames),
        "local_swap": local_swap(frames, *swap_spans),
        "motion_destroy": motion_destroy(frames, mode="median"),
    }
    if include_motion_only:
        counterfactuals["motion_only"] = motion_only(frames)
    return counterfactuals


def _collect_representations(
    model: HookedLVLM,
    frames: List,
    prompt: str,
    token_slice: slice,
    layers: Iterable[int],
) -> Dict[str, List[List[float]]]:
    inputs = model._prepare_inputs(frames, prompt)
    inputs_embeds = model.get_text_model_in(frames, prompt)
    reps: Dict[str, List[List[float]]] = {
        "inputs_mean": pool_hidden_states(inputs_embeds, token_slice).squeeze(0).tolist(),
        "layers_mean": [],
    }

    cache: Dict[int, torch.Tensor] = {}
    handles = capture_layer_outputs(model.text_model, layers, cache)
    with torch.no_grad():
        model.model(**inputs)
    for handle in handles:
        handle.remove()

    for layer in layers:
        reps["layers_mean"].append(pool_hidden_states(cache[layer], token_slice).squeeze(0).tolist())

    return reps


def run_causal_tracing(
    annotations_path: str,
    video_root: str,
    output_path: str,
    model_id: str,
    device: str,
    num_frames: int,
    swap_spans: Tuple[Tuple[int, int], Tuple[int, int]],
    token_slice_spec: str,
    layers: Optional[List[int]],
    include_motion_only: bool,
    limit: Optional[int],
    dump_representations: bool,
) -> None:
    dataset = VideoQADataset(annotations_path=annotations_path, video_root=video_root, max_samples=limit)
    model = HookedLVLM(model_id=model_id, device=device, quantize=False)
    model.model.eval()

    if layers is None:
        layers = list(range(len(model.text_model.model.layers)))

    results: Dict[str, Dict] = {}

    for sample in tqdm(dataset):
        question_id = sample.get("question_id") or sample["video_path"]
        frames = load_video_frames(
            sample["video_path"],
            start_time=sample.get("start"),
            end_time=sample.get("end"),
            num_frames=num_frames,
        )
        question = sample["question"]
        candidates = sample.get("candidates") or []
        if not candidates:
            results[str(question_id)] = {
                "question": question,
                "candidates": candidates,
                "answer": sample.get("answer"),
                "error": "No candidates provided for multiple-choice scoring.",
            }
            continue
        prompt = format_multiple_choice(question, candidates)
        labels = _label_candidates(candidates)
        correct_index = _resolve_correct_index(sample.get("answer"), candidates)

        inputs_clean = model._prepare_inputs(frames, prompt)
        token_slice = parse_slice(token_slice_spec, inputs_clean["input_ids"].shape[1])

        clean_logits = _score_option_logits(model, inputs_clean, labels)
        clean_margin = logit_margin(clean_logits, correct_index) if correct_index is not None else None

        counterfactuals = _make_counterfactuals(frames, swap_spans, include_motion_only)
        clean_inputs_embeds = model.get_text_model_in(frames, prompt)

        clean_cache: Dict[int, torch.Tensor] = {}
        handles = capture_layer_outputs(model.text_model, layers, clean_cache)
        with torch.no_grad():
            model.model(**inputs_clean)
        for handle in handles:
            handle.remove()

        sample_result: Dict[str, Dict] = {
            "question": question,
            "candidates": candidates,
            "answer": sample.get("answer"),
            "clean_logits": clean_logits,
            "clean_margin": clean_margin,
            "counterfactuals": {},
        }
        if dump_representations:
            sample_result["representations"] = _collect_representations(
                model,
                frames,
                prompt,
                token_slice,
                layers,
            )

        for name, cf_frames in counterfactuals.items():
            inputs_cf = model._prepare_inputs(cf_frames, prompt)
            cf_logits = _score_option_logits(model, inputs_cf, labels)
            cf_margin = logit_margin(cf_logits, correct_index) if correct_index is not None else None
            total_effect = None
            if clean_margin is not None and cf_margin is not None:
                total_effect = clean_margin - cf_margin

            patch_results: Dict[str, Optional[float]] = {}
            if total_effect is not None:
                with patch_inputs(model.text_model, clean_inputs_embeds, token_slice):
                    patched_logits = _score_option_logits(model, inputs_cf, labels)
                patched_margin = logit_margin(patched_logits, correct_index)
                recovered = patched_margin - cf_margin
                patch_results["inputs"] = normalized_use_score(recovered, total_effect)

                for layer in layers:
                    spec = PatchSpec(layer=layer, token_slice=token_slice)
                    with patch_layer_outputs(model.text_model, clean_cache, spec):
                        layer_logits = _score_option_logits(model, inputs_cf, labels)
                    layer_margin = logit_margin(layer_logits, correct_index)
                    recovered = layer_margin - cf_margin
                    patch_results[str(layer)] = normalized_use_score(recovered, total_effect)

            cf_entry: Dict[str, object] = {
                "logits": cf_logits,
                "margin": cf_margin,
                "total_effect": total_effect,
                "use_scores": patch_results,
            }
            if dump_representations:
                cf_entry["representations"] = _collect_representations(
                    model,
                    cf_frames,
                    prompt,
                    token_slice,
                    layers,
                )

            sample_result["counterfactuals"][name] = cf_entry

        results[str(question_id)] = sample_result

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual causal tracing for video QA.")
    parser.add_argument("--annotations", required=True, help="Path to the video QA json file.")
    parser.add_argument("--video_root", required=True, help="Root directory containing video files.")
    parser.add_argument("--output", required=True, help="Path to save tracing results.")
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model ID for Qwen2-VL or Qwen2.5-VL.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample from each video.")
    parser.add_argument(
        "--swap_spans",
        default="0:2,2:4",
        help="Two non-overlapping spans to swap, e.g. '0:2,2:4'.",
    )
    parser.add_argument(
        "--token_slice",
        default="all",
        help="Token slice spec for patching, e.g. 'all' or '0:128'.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to patch (default: all).",
    )
    parser.add_argument(
        "--include_motion_only",
        action="store_true",
        help="Include motion-only counterfactuals.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of samples.")
    parser.add_argument(
        "--dump_representations",
        action="store_true",
        help="Store pooled representations for Have metrics.",
    )

    args = parser.parse_args()
    span_parts = args.swap_spans.split(",")
    if len(span_parts) != 2:
        raise ValueError("swap_spans must contain two spans separated by a comma.")
    first = tuple(int(x) for x in span_parts[0].split(":"))
    second = tuple(int(x) for x in span_parts[1].split(":"))
    if len(first) != 2 or len(second) != 2:
        raise ValueError("Each swap span must be in the form start:end.")
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    run_causal_tracing(
        annotations_path=args.annotations,
        video_root=args.video_root,
        output_path=args.output,
        model_id=args.model_id,
        device=args.device,
        num_frames=args.num_frames,
        swap_spans=(first, second),
        token_slice_spec=args.token_slice,
        layers=layers,
        include_motion_only=args.include_motion_only,
        limit=args.limit,
        dump_representations=args.dump_representations,
    )


if __name__ == "__main__":
    main()
