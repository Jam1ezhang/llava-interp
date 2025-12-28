from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import torch


TokenSlice = Union[slice, Sequence[slice]]


@dataclass
class PatchSpec:
    layer: Optional[int]
    token_slice: TokenSlice


def _normalize_slices(token_slice: TokenSlice) -> Sequence[slice]:
    if isinstance(token_slice, slice):
        return [token_slice]
    return list(token_slice)


def _replace_hidden_states(
    hidden_states: torch.Tensor,
    replacement: torch.Tensor,
    token_slice: TokenSlice,
) -> torch.Tensor:
    patched = hidden_states.clone()
    for sl in _normalize_slices(token_slice):
        patched[:, sl, :] = replacement[:, sl, :]
    return patched


def _maybe_extract_hidden(output: object) -> Tuple[torch.Tensor, Tuple[object, ...]]:
    if isinstance(output, tuple):
        return output[0], output[1:]
    if isinstance(output, torch.Tensor):
        return output, ()
    raise TypeError(f"Unsupported hook output type: {type(output)}")


def capture_layer_outputs(
    model: torch.nn.Module,
    layers: Iterable[int],
    cache: Dict[int, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]

        def hook(module, args, kwargs, output, layer_idx=layer_idx):
            hidden, _ = _maybe_extract_hidden(output)
            cache[layer_idx] = hidden.detach()

        handles.append(layer.register_forward_hook(hook))

    return handles


@contextmanager
def patch_layer_outputs(
    model: torch.nn.Module,
    clean_cache: Dict[int, torch.Tensor],
    patch_spec: PatchSpec,
):
    layer_idx = patch_spec.layer
    if layer_idx is None:
        raise ValueError("PatchSpec.layer must be set for layer patching.")

    layer = model.model.layers[layer_idx]

    def hook(module, args, kwargs, output):
        hidden, rest = _maybe_extract_hidden(output)
        patched = _replace_hidden_states(hidden, clean_cache[layer_idx], patch_spec.token_slice)
        if rest:
            return (patched,) + rest
        return patched

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patch_inputs(
    model: torch.nn.Module,
    clean_inputs: torch.Tensor,
    token_slice: TokenSlice,
):
    def hook(module, args, kwargs):
        inputs_embeds = kwargs["inputs_embeds"]
        patched = inputs_embeds.clone()
        for sl in _normalize_slices(token_slice):
            patched[:, sl, :] = clean_inputs[:, sl, :]
        kwargs["inputs_embeds"] = patched
        return args, kwargs

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
