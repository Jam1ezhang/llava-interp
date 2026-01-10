from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from token_indexing import get_frame_token_spans, get_visual_token_span

from .base import VLMAdapter


class Qwen2VLAdapter(VLMAdapter):
    def load(self) -> None:
        model_kwargs: Dict[str, object] = {
            "device_map": self.device,
        }
        cache_dir = getattr(self, "cache_dir", None)
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir

        # 尝试启用 flash-attention
        use_flash_attention = getattr(self, "use_flash_attention", False)
        if use_flash_attention:
            try:
                import flash_attn
                # 加载配置并设置 flash attention
                config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
                if hasattr(config, "attn_implementation"):
                    config.attn_implementation = "flash_attention_2"
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif hasattr(config, "_attn_implementation"):
                    config._attn_implementation = "flash_attention_2"
                    model_kwargs["_attn_implementation"] = "flash_attention_2"
            except ImportError:
                print("Warning: flash-attn not installed, falling back to default attention.")
            except Exception as e:
                print(f"Warning: Failed to enable flash-attention: {e}, falling back to default attention.")

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

        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def prepare_inputs(self, frames: List, prompt: str) -> Dict[str, torch.Tensor]:
        if not hasattr(self.processor, "apply_chat_template"):
            text = f"USER: <video>\n{prompt} ASSISTANT:"
            inputs = self.processor(text=text, videos=[frames], return_tensors="pt")
            return inputs.to(self.model.device)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def get_text_model(self) -> nn.Module:
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        return getattr(self.model, "model", self.model)

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
        visual_span = get_visual_token_span(inputs, self.processor)
        frame_spans = get_frame_token_spans(inputs, self.processor)
        return visual_span, frame_spans
