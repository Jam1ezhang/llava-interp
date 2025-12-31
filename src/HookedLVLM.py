import os
from contextlib import contextmanager
from typing import Callable, List, Optional, Union

import torch
import yaml
from adapters.factory import make_adapter

from video_utils import load_video_frames

file_path = os.path.dirname(__file__)
config_file = os.path.join(file_path, "config.yaml")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

model_cache_dir = config["cache_dir"]
if model_cache_dir is None:
    model_cache_dir = os.path.join(file_path, "..", "models")


@contextmanager
def session_hook(model: torch.nn.Module, hook: Callable):
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class BlockAttentionHook:
    def __init__(self, indices_list):
        self.indices_list = indices_list

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs["hidden_states"]
        bsz, seq_len, _ = hidden_states.shape
        attention_mask = kwargs.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones(bsz, 1, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril(
                diagonal=0
            )
        else:
            attention_mask = attention_mask.clone()

        for i, j in self.indices_list:
            attention_mask[:, :, i, j] = False

        kwargs["attention_mask"] = attention_mask
        return args, kwargs


class HookedLVLM:
    """Hooked LVLM for video-enabled Qwen2-VL/Qwen2.5-VL models."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        hook_loc: str = "text_model_in",
        device: str = "cuda:0",
        quantize: bool = False,
        quantize_type: str = "fp16",
    ):
        torch_dtype = torch.float16
        if quantize and quantize_type == "int8":
            torch_dtype = torch.int8

        self.adapter = make_adapter(model_id, device, torch_dtype, quantize)
        self.adapter.cache_dir = model_cache_dir
        self.adapter.quantize_type = quantize_type
        self.adapter.load()

        self.model = self.adapter.model
        self.processor = self.adapter.processor
        self.hook_loc = hook_loc
        self.data = None
        self.text_model = self.adapter.get_text_model()

    @contextmanager
    def ablate_inputs(self, indices, replacement_tensor):
        def ablation_hook(module, args, kwargs):
            input_embeds = kwargs["inputs_embeds"]
            if input_embeds.shape[-2] == 1:
                return args, kwargs
            modified_input = input_embeds.clone()
            local_replacement_tensor = replacement_tensor.to(modified_input.dtype).to(modified_input.device)
            local_replacement_tensor = local_replacement_tensor.reshape(1, -1).expand(len(indices), -1)
            modified_input[:, indices, :] = local_replacement_tensor
            kwargs["inputs_embeds"] = modified_input
            return args, kwargs

        hook = self.text_model.register_forward_pre_hook(ablation_hook, with_kwargs=True)
        try:
            yield
        finally:
            hook.remove()

    @contextmanager
    def block_attention(self, attn_block_dict):
        hooks = []

        for layer, indices_list in attn_block_dict.items():
            hook = BlockAttentionHook(indices_list)
            h = self.text_model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
            hooks.append(h)

        has_error = False
        try:
            yield
        except Exception:
            import traceback

            traceback.print_exc()
            has_error = True
        finally:
            for h in hooks:
                h.remove()
            if has_error:
                raise Exception("An error occurred during the block_attention context")

    def prompt_hook(self, module, args, kwargs, output):
        self.data = kwargs["inputs_embeds"]
        return output

    def _prepare_inputs(
        self,
        video: Union[str, List],
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: int = 8,
    ):
        if isinstance(video, str):
            video_frames = load_video_frames(video, start_time=start_time, end_time=end_time, num_frames=num_frames)
        else:
            video_frames = video

        return self.adapter.prepare_inputs(video_frames, question)

    def forward(
        self,
        video: Union[str, List],
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: int = 8,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        inputs = self._prepare_inputs(
            video,
            question,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
        )

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        return outputs

    def generate(
        self,
        video: Union[str, List],
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: int = 8,
        max_new_tokens: int = 100,
        output_hidden_states: bool = False,
        do_sample: bool = True,
    ):
        inputs = self._prepare_inputs(
            video,
            question,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=True,
                do_sample=do_sample,
            )

        response_str = self.processor.batch_decode(
            output.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if output_hidden_states:
            return response_str, output.hidden_states

        return response_str

    def get_text_model_in(
        self,
        video: Union[str, List],
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: int = 8,
    ):
        inputs = self._prepare_inputs(
            video,
            question,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
        )

        if self.hook_loc == "text_model_in":
            with session_hook(self.text_model, self.prompt_hook):
                with torch.no_grad():
                    self.model(**inputs)
        else:
            raise ValueError(
                "Only 'text_model_in' support for hook location at the moment. "
                f"Got {self.hook_loc} instead."
            )

        return self.data
