from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file
from torch import nn


NESTED_STATE_DICT_KEYS = (
    "state_dict",
    "model",
    "model_state_dict",
    "module",
    "network",
)


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)
    suffix = checkpoint_path.suffix.lower()

    if suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path), device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location=map_location)

    while isinstance(state_dict, dict):
        nested_state_dict = next(
            (
                state_dict[key]
                for key in NESTED_STATE_DICT_KEYS
                if key in state_dict and isinstance(state_dict[key], dict)
            ),
            None,
        )
        if nested_state_dict is None:
            break
        state_dict = nested_state_dict

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} does not contain a state_dict mapping."
        )

    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, str):
            normalized_state_dict[key.removeprefix("module.")] = value

    return normalized_state_dict


def _strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _matching_key_count(
    state_dict: dict[str, torch.Tensor],
    module_state_dict: dict[str, torch.Tensor],
) -> int:
    return sum(
        1
        for key, value in state_dict.items()
        if key in module_state_dict and module_state_dict[key].shape == value.shape
    )


def load_module_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    prefixes: Iterable[str] = (),
) -> tuple[torch.nn.modules.module._IncompatibleKeys, int]:
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    module_state_dict = module.state_dict()

    candidates = [state_dict]
    seen_prefixes = set()
    for prefix in prefixes:
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        stripped_state_dict = _strip_prefix(state_dict, prefix)
        if stripped_state_dict:
            candidates.append(stripped_state_dict)

    best_state_dict = max(
        candidates,
        key=lambda candidate: _matching_key_count(candidate, module_state_dict),
    )
    matched_keys = _matching_key_count(best_state_dict, module_state_dict)
    incompatible = module.load_state_dict(best_state_dict, strict=False)
    return incompatible, matched_keys
