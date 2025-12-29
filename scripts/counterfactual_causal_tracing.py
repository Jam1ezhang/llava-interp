import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))

from activation_patching import PatchSpec, capture_layer_outputs, patch_inputs, patch_layer_outputs
from causal_metrics import logit_margin, normalized_use_score, parse_slice, pool_hidden_states
from counterfactuals import local_swap, motion_destroy, motion_only, reverse_order
from HookedLVLM import HookedLVLM
from token_indexing import get_frame_token_spans, get_span_token_slice, get_visual_token_span
from VideoDatasets import VideoQADataset
from video_utils import format_multiple_choice, load_video_frames


def _label_candidates(candidates: List[str]) -> List[str]:
    return [chr(65 + idx) for idx in range(len(candidates))]


def _resolve_correct_index(
    answer: Optional[str], 
    candidates: List[str], 
    answer_number: Optional[int] = None
) -> Optional[int]:
    """解析正确答案的索引。
    
    Args:
        answer: 答案文本（可能是部分文本，如 "clothes"）
        candidates: 候选答案列表（完整文本，如 ["The closet/cabinet.", "The blanket.", "The clothes.", "The table."]）
        answer_number: 答案编号（从 0 开始），如果提供则直接使用
        
    Returns:
        正确答案在 candidates 中的索引，如果无法确定则返回 None
    """
    # 如果提供了 answer_number，直接使用（最可靠）
    if answer_number is not None and 0 <= answer_number < len(candidates):
        return answer_number
    
    if answer is None:
        return None
    
    # 去除首尾空格并转换为小写，用于匹配
    answer_normalized = answer.strip().lower()
    
    labels = _label_candidates(candidates)
    
    # 1. 检查是否是标签（A, B, C, D...）
    if answer.upper().strip() in labels:
        return labels.index(answer.upper().strip())
    
    # 2. 精确匹配候选答案
    if answer in candidates:
        return candidates.index(answer)
    
    # 3. 大小写不敏感的精确匹配
    for idx, candidate in enumerate(candidates):
        if candidate.strip().lower() == answer_normalized:
            return idx
    
    # 4. 部分匹配：检查 answer 是否是 candidate 的子串（或 candidate 包含 answer）
    # 去除标点符号和多余空格后匹配
    answer_clean = re.sub(r'[^\w\s]', '', answer_normalized)
    for idx, candidate in enumerate(candidates):
        candidate_clean = re.sub(r'[^\w\s]', '', candidate.strip().lower())
        # 检查 answer 是否在 candidate 中，或 candidate 是否在 answer 中
        if answer_clean in candidate_clean or candidate_clean in answer_clean:
            return idx
    
    # 5. 如果仍然无法匹配，返回 None
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
    include_local_swap: bool,
) -> Dict[str, List]:
    counterfactuals = {
        "order_reverse": reverse_order(frames),
        "motion_destroy": motion_destroy(frames, mode="median"),
    }
    if include_local_swap:
        counterfactuals["local_swap"] = local_swap(frames, *swap_spans)
    if include_motion_only:
        counterfactuals["motion_only"] = motion_only(frames)
    return counterfactuals


TokenSlice = Union[slice, Sequence[slice]]


def _frame_means(hidden_states: torch.Tensor, frame_spans: Sequence[slice]) -> List[List[float]]:
    return [pool_hidden_states(hidden_states, span).squeeze(0).tolist() for span in frame_spans]


def _collect_representations(
    model: HookedLVLM,
    inputs: Dict,
    inputs_embeds: torch.Tensor,
    token_slice: TokenSlice,
    layers: Iterable[int],
    dump_frame_reps: bool,
) -> Dict[str, object]:
    visual_span = get_visual_token_span(inputs, model.processor)
    frame_spans = get_frame_token_spans(inputs, model.processor)
    reps: Dict[str, object] = {
        "inputs_mean": pool_hidden_states(inputs_embeds, token_slice).squeeze(0).tolist(),
        "layers_mean": [],
        "visual_mean": pool_hidden_states(inputs_embeds, visual_span).squeeze(0).tolist(),
    }
    if dump_frame_reps:
        reps["frame_means"] = _frame_means(inputs_embeds, frame_spans)
        reps["layers_frame_means"] = []

    cache: Dict[int, torch.Tensor] = {}
    handles = capture_layer_outputs(model.text_model, layers, cache)
    with torch.no_grad():
        model.model(**inputs)
    for handle in handles:
        handle.remove()

    for layer in layers:
        reps["layers_mean"].append(pool_hidden_states(cache[layer], token_slice).squeeze(0).tolist())
        if dump_frame_reps:
            reps["layers_frame_means"].append(_frame_means(cache[layer], frame_spans))

    return reps


def _ensure_visual_only(token_slice: TokenSlice, visual_span: slice) -> None:
    slices = [token_slice] if isinstance(token_slice, slice) else list(token_slice)
    for sl in slices:
        if sl.start < visual_span.start or sl.stop > visual_span.stop:
            raise ValueError("token_slice includes text tokens; expected only visual token spans.")


def _save_vprime(
    inputs: Dict,
    inputs_embeds: torch.Tensor,
    processor,
    out_path: str,
    n_prime: int,
    d_prime: int,
) -> str:
    visual_span = get_visual_token_span(inputs, processor)
    v = inputs_embeds[:, visual_span, :].squeeze(0)
    vprime = F.interpolate(
        v.float().unsqueeze(0).unsqueeze(0),
        size=(n_prime, d_prime),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    torch.save(vprime.to(torch.float16).cpu(), out_path)
    return out_path


def _compute_use_scores(
    model: HookedLVLM,
    inputs_cf: Dict,
    labels: List[str],
    correct_index: int,
    cf_margin: float,
    total_effect: float,
    clean_inputs_embeds: torch.Tensor,
    clean_cache: Dict[int, torch.Tensor],
    token_slice: TokenSlice,
    layers: Iterable[int],
) -> Dict[str, Optional[float]]:
    patch_results: Dict[str, Optional[float]] = {}
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

    return patch_results


def run_causal_tracing(
    annotations_path: str,
    video_root: str,
    output_path: str,
    representations_output: Optional[str],
    model_id: str,
    device: str,
    num_frames: int,
    swap_spans: Tuple[Tuple[int, int], Tuple[int, int]],
    token_slice_spec: str,
    layers: Optional[List[int]],
    include_motion_only: bool,
    limit: Optional[int],
    dump_representations: bool,
    dump_frame_reps: bool,
    dump_vprime: bool,
    vprime_n: int,
    vprime_d: int,
    vprime_out_dir: Optional[str],
) -> None:
    dataset = VideoQADataset(annotations_path=annotations_path, video_root=video_root, max_samples=limit)
    model = HookedLVLM(model_id=model_id, device=device, quantize=False)
    model.model.eval()

    if layers is None:
        # 尝试多种方式访问 layers 属性
        text_model = model.text_model
        if hasattr(text_model, "model") and hasattr(text_model.model, "layers"):
            layers_attr = text_model.model.layers
        elif hasattr(text_model, "layers"):
            layers_attr = text_model.layers
        else:
            raise AttributeError(f"Could not find layers attribute in text_model of type {type(text_model)}")
        layers = list(range(len(layers_attr)))

    results: Dict[str, Dict] = {}
    representation_results: Optional[Dict[str, Dict]] = {} if representations_output else None

    resolved_vprime_dir: Optional[str] = None
    if dump_vprime:
        base_dir = os.path.dirname(output_path)
        resolved_vprime_dir = vprime_out_dir or os.path.join(base_dir, "vprime")
        os.makedirs(resolved_vprime_dir, exist_ok=True)

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
        correct_index = _resolve_correct_index(
            sample.get("answer"), 
            candidates,
            answer_number=sample.get("answer_number")
        )

        inputs_clean = model._prepare_inputs(frames, prompt)
        token_slice = parse_slice(
            token_slice_spec,
            inputs_clean["input_ids"].shape[1],
            inputs=inputs_clean,
            processor=model.processor,
        )
        visual_span = get_visual_token_span(inputs_clean, model.processor)
        if token_slice_spec.startswith("visual"):
            _ensure_visual_only(token_slice, visual_span)

        t_eff = len(get_frame_token_spans(inputs_clean, model.processor))
        include_local_swap = True
        if "qwen" in model_id.lower() and t_eff < num_frames:
            include_local_swap = False

        clean_logits = _score_option_logits(model, inputs_clean, labels)
        clean_margin = logit_margin(clean_logits, correct_index) if correct_index is not None else None

        counterfactuals = _make_counterfactuals(
            frames,
            swap_spans,
            include_motion_only,
            include_local_swap,
        )
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
        if dump_representations and not representations_output:
            representations = _collect_representations(
                model,
                inputs_clean,
                clean_inputs_embeds,
                token_slice,
                layers,
                dump_frame_reps,
            )
            if dump_vprime and resolved_vprime_dir is not None:
                vprime_path = os.path.join(resolved_vprime_dir, f"sample_{question_id}_clean_vprime.pt")
                representations["vprime_path"] = _save_vprime(
                    inputs_clean,
                    clean_inputs_embeds,
                    model.processor,
                    vprime_path,
                    vprime_n,
                    vprime_d,
                )
            sample_result["representations"] = representations
        if dump_representations and representation_results is not None:
            representations = _collect_representations(
                model,
                inputs_clean,
                clean_inputs_embeds,
                token_slice,
                layers,
                dump_frame_reps,
            )
            if dump_vprime and resolved_vprime_dir is not None:
                vprime_path = os.path.join(resolved_vprime_dir, f"sample_{question_id}_clean_vprime.pt")
                representations["vprime_path"] = _save_vprime(
                    inputs_clean,
                    clean_inputs_embeds,
                    model.processor,
                    vprime_path,
                    vprime_n,
                    vprime_d,
                )
            representation_results[str(question_id)] = {
                "representations": representations,
                "counterfactuals": {},
            }

        for name, cf_frames in counterfactuals.items():
            inputs_cf = model._prepare_inputs(cf_frames, prompt)
            cf_logits = _score_option_logits(model, inputs_cf, labels)
            cf_margin = logit_margin(cf_logits, correct_index) if correct_index is not None else None
            total_effect = None
            if clean_margin is not None and cf_margin is not None:
                total_effect = clean_margin - cf_margin

            patch_results: Dict[str, Optional[float]] = {}
            use_scores_by_span: Optional[Dict[str, Dict[str, Optional[float]]]] = None
            if total_effect is not None:
                patch_results = _compute_use_scores(
                    model,
                    inputs_cf,
                    labels,
                    correct_index,
                    cf_margin,
                    total_effect,
                    clean_inputs_embeds,
                    clean_cache,
                    token_slice,
                    layers,
                )
                if name == "local_swap":
                    frame_spans = get_frame_token_spans(inputs_clean, model.processor)
                    span_slices = get_span_token_slice(swap_spans, frame_spans, raw_T=num_frames)
                    # 处理警告信息
                    if "warnings" in span_slices:
                        for warning in span_slices["warnings"]:
                            print(f"[WARNING] {warning}")
                    text_slice = parse_slice(
                        "text",
                        inputs_clean["input_ids"].shape[1],
                        inputs=inputs_clean,
                        processor=model.processor,
                    )
                    use_scores_by_span = {
                        "spanA": _compute_use_scores(
                            model,
                            inputs_cf,
                            labels,
                            correct_index,
                            cf_margin,
                            total_effect,
                            clean_inputs_embeds,
                            clean_cache,
                            span_slices["spanA_slice"],
                            layers,
                        ),
                        "spanB": _compute_use_scores(
                            model,
                            inputs_cf,
                            labels,
                            correct_index,
                            cf_margin,
                            total_effect,
                            clean_inputs_embeds,
                            clean_cache,
                            span_slices["spanB_slice"],
                            layers,
                        ),
                        "union": _compute_use_scores(
                            model,
                            inputs_cf,
                            labels,
                            correct_index,
                            cf_margin,
                            total_effect,
                            clean_inputs_embeds,
                            clean_cache,
                            span_slices["union_slice"],
                            layers,
                        ),
                        "text_only": _compute_use_scores(
                            model,
                            inputs_cf,
                            labels,
                            correct_index,
                            cf_margin,
                            total_effect,
                            clean_inputs_embeds,
                            clean_cache,
                            text_slice,
                            layers,
                        ),
                    }

            cf_entry: Dict[str, object] = {
                "logits": cf_logits,
                "margin": cf_margin,
                "total_effect": total_effect,
                "use_scores": patch_results,
            }
            if use_scores_by_span is not None:
                cf_entry["use_scores_by_span"] = use_scores_by_span
            if dump_representations and not representations_output:
                cf_inputs_embeds = model.get_text_model_in(cf_frames, prompt)
                representations = _collect_representations(
                    model,
                    inputs_cf,
                    cf_inputs_embeds,
                    token_slice,
                    layers,
                    dump_frame_reps,
                )
                if dump_vprime and resolved_vprime_dir is not None:
                    suffix = name
                    if name == "local_swap":
                        suffix = f"{name}_{swap_spans[0][0]}-{swap_spans[0][1]}_{swap_spans[1][0]}-{swap_spans[1][1]}"
                    vprime_path = os.path.join(
                        resolved_vprime_dir,
                        f"sample_{question_id}_{suffix}_vprime.pt",
                    )
                    representations["vprime_path"] = _save_vprime(
                        inputs_cf,
                        cf_inputs_embeds,
                        model.processor,
                        vprime_path,
                        vprime_n,
                        vprime_d,
                    )
                cf_entry["representations"] = representations
            if dump_representations and representation_results is not None:
                cf_inputs_embeds = model.get_text_model_in(cf_frames, prompt)
                representations = _collect_representations(
                    model,
                    inputs_cf,
                    cf_inputs_embeds,
                    token_slice,
                    layers,
                    dump_frame_reps,
                )
                if dump_vprime and resolved_vprime_dir is not None:
                    suffix = name
                    if name == "local_swap":
                        suffix = f"{name}_{swap_spans[0][0]}-{swap_spans[0][1]}_{swap_spans[1][0]}-{swap_spans[1][1]}"
                    vprime_path = os.path.join(
                        resolved_vprime_dir,
                        f"sample_{question_id}_{suffix}_vprime.pt",
                    )
                    representations["vprime_path"] = _save_vprime(
                        inputs_cf,
                        cf_inputs_embeds,
                        model.processor,
                        vprime_path,
                        vprime_n,
                        vprime_d,
                    )
                representation_results[str(question_id)]["counterfactuals"][name] = {
                    "representations": representations
                }

            sample_result["counterfactuals"][name] = cf_entry

        results[str(question_id)] = sample_result

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if representation_results is not None and representations_output:
        rep_output_dir = os.path.dirname(representations_output)
        if rep_output_dir:
            os.makedirs(rep_output_dir, exist_ok=True)
        with open(representations_output, "w", encoding="utf-8") as f:
            json.dump(representation_results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual causal tracing for video QA.")
    parser.add_argument("--annotations", required=True, help="Path to the video QA json file.")
    parser.add_argument("--video_root", required=True, help="Root directory containing video files.")
    parser.add_argument("--output", required=True, help="Path to save tracing results.")
    parser.add_argument(
        "--representations_output",
        default=None,
        help="Optional path to save representations separately from tracing results.",
    )
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
        default="visual",
        help="Token slice spec for patching, e.g. 'visual', 'text', 'visual_frames:0:2', or '0:128'.",
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
    parser.add_argument(
        "--dump_frame_reps",
        action="store_true",
        help="Include per-frame representations (can be large).",
    )
    parser.add_argument(
        "--dump_vprime",
        action="store_true",
        help="Dump downsampled visual token matrices to .pt files.",
    )
    parser.add_argument("--vprime_n", type=int, default=128, help="Target token dimension for vprime.")
    parser.add_argument("--vprime_d", type=int, default=1024, help="Target hidden dimension for vprime.")
    parser.add_argument(
        "--vprime_out_dir",
        default=None,
        help="Directory to save vprime .pt files (default: output_dir/vprime).",
    )

    args = parser.parse_args()
    if args.representations_output and not args.dump_representations:
        raise ValueError("--representations_output requires --dump_representations.")
    if args.dump_vprime and not args.dump_representations:
        raise ValueError("--dump_vprime requires --dump_representations.")
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
        representations_output=args.representations_output,
        model_id=args.model_id,
        device=args.device,
        num_frames=args.num_frames,
        swap_spans=(first, second),
        token_slice_spec=args.token_slice,
        layers=layers,
        include_motion_only=args.include_motion_only,
        limit=args.limit,
        dump_representations=args.dump_representations,
        dump_frame_reps=args.dump_frame_reps,
        dump_vprime=args.dump_vprime,
        vprime_n=args.vprime_n,
        vprime_d=args.vprime_d,
        vprime_out_dir=args.vprime_out_dir,
    )


if __name__ == "__main__":
    main()
