from __future__ import annotations

import inspect
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from .base import VLMAdapter


class LlavaHFAdapter(VLMAdapter):
    def load(self) -> None:
        model_kwargs: Dict[str, object] = {
            "device_map": self.device,
            "torch_dtype": self.torch_dtype,
        }
        cache_dir = getattr(self, "cache_dir", None)
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir

        if self.quantize:
            quantize_type = getattr(self, "quantize_type", "fp16")
            if quantize_type == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["low_cpu_mem_usage"] = True
            elif quantize_type == "fp16":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["low_cpu_mem_usage"] = True
            elif quantize_type == "int8":
                model_kwargs["torch_dtype"] = torch.int8
                model_kwargs["low_cpu_mem_usage"] = True

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        except Exception:
            import transformers as hf_transformers

            candidates = [
                "LlavaNextForConditionalGeneration",
                "LlavaOnevisionForConditionalGeneration",
                "LlavaForConditionalGeneration",
                "LlavaNextVideoForConditionalGeneration",
            ]
            last_error = None
            for name in candidates:
                model_cls = getattr(hf_transformers, name, None)
                if model_cls is None:
                    continue
                try:
                    self.model = model_cls.from_pretrained(self.model_id, **model_kwargs)
                    break
                except Exception as exc:  # noqa: PERF203 - need to continue trying classes
                    last_error = exc
            else:
                raise last_error or ValueError(
                    f"Unsupported Llava model class for model_id={self.model_id!r}"
                )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        _ = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)

    def prepare_inputs(self, frames: List, prompt: str) -> Dict[str, torch.Tensor]:
        processor_signature = inspect.signature(self.processor.__call__)
        if "videos" in processor_signature.parameters:
            inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")
        else:
            if hasattr(self.processor, "apply_chat_template"):
                images = [{"type": "image"} for _ in frames]
                messages = [
                    {
                        "role": "user",
                        "content": [*images, {"type": "text", "text": prompt}],
                    }
                ]
                llava_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                img_placeholders = "<image>\n" * len(frames)
                llava_prompt = f"{img_placeholders}{prompt}"

            inputs = self.processor(text=llava_prompt, images=frames, return_tensors="pt")

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    inputs[key] = value.to(device=device, dtype=self.model.dtype)
                else:
                    inputs[key] = value.to(device=device)
        return inputs

    def get_text_model(self) -> nn.Module:
        return getattr(self.model, "language_model", None) or getattr(self.model, "model", self.model)

    def get_layers(self) -> Sequence[nn.Module]:
        text_model = self.get_text_model()
        if hasattr(text_model, "model") and hasattr(text_model.model, "layers"):
            return text_model.model.layers
        if hasattr(text_model, "layers"):
            return text_model.layers
        return []

    def locate_visual_spans(
        self,
        inputs: Dict[str, torch.Tensor],
        inputs_embeds: torch.Tensor,
        num_frames: int,
    ) -> Tuple[slice, List[slice]]:
        image_token_id = getattr(self.model.config, "image_token_index", None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if image_token_id is None and tokenizer is not None:
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is None:
            raise ValueError(f"Unable to resolve image_token_id for model_id={self.model_id!r}")

        input_ids = inputs["input_ids"][0]
        placeholders = (input_ids == image_token_id).nonzero(as_tuple=False).flatten().tolist()
        n_img = len(placeholders)
        if n_img == 0:
            raise ValueError(
                f"No <image> placeholders found in input_ids for model_id={self.model_id!r}."
            )

        l_ids = inputs["input_ids"].shape[1]
        l_emb = inputs_embeds.shape[1]
        diff = l_emb - l_ids
        if diff % n_img != 0:
            raise ValueError(
                "Unable to infer visual token expansion length. "
                f"model_id={self.model_id!r}, L_ids={l_ids}, L_emb={l_emb}, "
                f"n_img={n_img}, input_ids={input_ids.tolist()}."
            )
        k = 1 + diff // n_img

        frame_spans: List[slice] = []
        delta = 0
        for position in placeholders:
            start = position + delta
            end = start + k
            frame_spans.append(slice(start, end))
            delta += k - 1

        visual_span = slice(frame_spans[0].start, frame_spans[-1].stop)
        if n_img != num_frames:
            visual_len = visual_span.stop - visual_span.start
            tokens_per = visual_len // num_frames
            remainder = visual_len % num_frames
            approx_spans: List[slice] = []
            cursor = visual_span.start
            for idx in range(num_frames):
                extra = remainder if idx == num_frames - 1 else 0
                end = cursor + tokens_per + extra
                approx_spans.append(slice(cursor, end))
                cursor = end
            frame_spans = approx_spans
        return visual_span, frame_spans
