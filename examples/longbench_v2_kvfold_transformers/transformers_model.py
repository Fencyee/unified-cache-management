#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Pure Transformers backend for Qwen3-32B LongBench experiments.

This module deliberately does not import vLLM or UCM.  It uses Accelerate's
Big Model Inference device map to place whole decoder layers on four visible
Ascend NPUs.  This is model parallelism for offline correctness evaluation;
it is not tensor parallelism and must not be used for vLLM TP4 performance
comparisons.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_POSITION_EMBEDDINGS = 131_072
DEFAULT_ORIGINAL_MAX_POSITION_EMBEDDINGS = 32_768
DEFAULT_YARN_FACTOR = 4.0
DEFAULT_ROPE_THETA = 1_000_000.0
EXPECTED_NPU_COUNT = 4
EXPECTED_NUM_HIDDEN_LAYERS = 64
EXPECTED_NUM_KEY_VALUE_HEADS = 8
EXPECTED_HEAD_DIM = 128


@dataclass(frozen=True)
class TransformersModelOptions:
    """Configuration for the pure Transformers Qwen3 backend."""

    model_path: str
    device_map: str | Mapping[str, Any] = "balanced"
    max_memory: Mapping[int | str, int | str] | None = None
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS
    original_max_position_embeddings: int = DEFAULT_ORIGINAL_MAX_POSITION_EMBEDDINGS
    yarn_factor: float = DEFAULT_YARN_FACTOR
    rope_theta: float = DEFAULT_ROPE_THETA
    local_files_only: bool = False
    trust_remote_code: bool = False
    allow_model_shape_mismatch: bool = False
    allow_npu_count_mismatch: bool = False
    allow_cpu_disk_offload: bool = False
    allow_device_map_mismatch: bool = False


@dataclass(frozen=True)
class GreedyGenerationResult:
    """Decoded output and token accounting from one greedy generation."""

    text: str
    input_tokens: int
    output_tokens: int
    output_token_ids: tuple[int, ...]


def _version_major(version: str) -> int:
    match = re.match(r"\s*(\d+)", version)
    if match is None:
        raise RuntimeError(f"cannot parse Transformers version: {version!r}")
    return int(match.group(1))


def _version_release(version: str) -> tuple[int, int]:
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"cannot parse Transformers version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _parse_max_memory(values: Sequence[str] | None) -> dict[int | str, str] | None:
    if not values:
        return None
    parsed: dict[int | str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"invalid --max-memory {value!r}; expected DEVICE=LIMIT"
            )
        device_text, limit = value.split("=", 1)
        device_text = device_text.strip()
        limit = limit.strip()
        if not device_text or not limit:
            raise argparse.ArgumentTypeError(
                f"invalid --max-memory {value!r}; expected DEVICE=LIMIT"
            )
        device: int | str
        try:
            device = int(device_text)
        except ValueError:
            device = device_text
        if device in parsed:
            raise argparse.ArgumentTypeError(
                f"duplicate --max-memory entry for device {device!r}"
            )
        parsed[device] = limit
    return parsed


def add_transformers_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the pure Transformers backend options to an argument parser."""

    group = parser.add_argument_group("pure Transformers Qwen3 backend")
    group.add_argument("--model", required=True, help="Local Qwen3-32B model path")
    group.add_argument(
        "--device-map",
        default="balanced",
        choices=("balanced", "auto", "balanced_low_0", "sequential"),
        help="Accelerate Big Model Inference placement policy",
    )
    group.add_argument(
        "--max-memory",
        action="append",
        default=[],
        metavar="DEVICE=LIMIT",
        help=(
            "Optional Accelerate memory limit; repeat for each device, for "
            "example --max-memory 0=48GiB"
        ),
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=DEFAULT_MAX_POSITION_EMBEDDINGS,
    )
    group.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=DEFAULT_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    )
    group.add_argument("--yarn-factor", type=float, default=DEFAULT_YARN_FACTOR)
    group.add_argument("--local-files-only", action="store_true")
    group.add_argument("--trust-remote-code", action="store_true")
    group.add_argument(
        "--allow-model-shape-mismatch",
        action="store_true",
        help="Allow a model other than the expected Qwen3-32B architecture",
    )
    group.add_argument(
        "--allow-npu-count-mismatch",
        action="store_true",
        help="Allow a visible NPU count other than four (at least one is required)",
    )
    group.add_argument(
        "--allow-cpu-disk-offload",
        action="store_true",
        help="Allow Accelerate to place model weights on CPU or disk",
    )
    group.add_argument(
        "--allow-device-map-mismatch",
        action="store_true",
        help="Allow the final model map not to cover all four logical NPUs",
    )


def options_from_namespace(args: argparse.Namespace) -> TransformersModelOptions:
    """Build :class:`TransformersModelOptions` from registered CLI options."""

    try:
        max_memory = _parse_max_memory(args.max_memory)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    return TransformersModelOptions(
        model_path=args.model,
        device_map=args.device_map,
        max_memory=max_memory,
        max_position_embeddings=args.max_position_embeddings,
        original_max_position_embeddings=args.original_max_position_embeddings,
        yarn_factor=args.yarn_factor,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        allow_model_shape_mismatch=args.allow_model_shape_mismatch,
        allow_npu_count_mismatch=args.allow_npu_count_mismatch,
        allow_cpu_disk_offload=args.allow_cpu_disk_offload,
        allow_device_map_mismatch=args.allow_device_map_mismatch,
    )


def build_dynamic_cache(config: Any | None = None) -> Any:
    """Construct ``DynamicCache`` across supported Transformers 4.x/5.x APIs.

    Recent versions accept the model config and use it to initialize hybrid or
    sliding-layer metadata.  Older versions expose a no-argument constructor.
    Signature inspection avoids hiding real constructor failures behind a
    broad ``TypeError`` retry.
    """

    try:
        from transformers.cache_utils import DynamicCache
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "DynamicCache is unavailable; Transformers >= 4.51 is required"
        ) from error

    parameters = inspect.signature(DynamicCache.__init__).parameters.values()
    accepts_config = any(
        parameter.name == "config" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_config and config is not None:
        return DynamicCache(config=config)
    return DynamicCache()


def apply_chat_template_non_thinking(
    tokenizer: Any,
    prompt_or_messages: str | Sequence[Mapping[str, Any]],
) -> str:
    """Render the Qwen3 chat template with thinking explicitly disabled."""

    if isinstance(prompt_or_messages, str):
        messages: list[Mapping[str, Any]] = [
            {"role": "user", "content": prompt_or_messages}
        ]
    else:
        messages = list(prompt_or_messages)
    if not messages:
        raise ValueError("at least one chat message is required")

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            "tokenizer.apply_chat_template(tokenize=False) did not return text"
        )
    return rendered


def encode_chat(
    tokenizer: Any,
    prompt_or_messages: str | Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    device: Any | None = None,
) -> dict[str, Any]:
    """Render and tokenize one non-thinking chat request.

    The function rejects overlong input instead of silently truncating it.
    LongBench uses middle truncation before this point, and applying a second,
    different truncation policy here would make paired results incomparable.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    text = apply_chat_template_non_thinking(tokenizer, prompt_or_messages)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if "input_ids" not in encoded:
        raise RuntimeError("tokenizer output does not contain input_ids")
    input_tokens = int(encoded["input_ids"].shape[-1])
    if input_tokens > max_tokens:
        raise ValueError(
            f"chat input has {input_tokens} tokens, exceeding limit {max_tokens}"
        )
    if device is None:
        return dict(encoded)
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def make_dynamic_cache(model: Any) -> Any:
    """Build a version-compatible empty DynamicCache for ``model``."""

    return build_dynamic_cache(getattr(model, "config", None))


def generate_greedy(
    model: Any,
    model_inputs: Mapping[str, Any],
    cache: Any,
    *,
    max_new_tokens: int,
    pad_token_id: int | None = None,
) -> Any:
    """Return full generated token sequences using deterministic decoding."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if "input_ids" not in model_inputs:
        raise ValueError("model_inputs must contain input_ids")
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("PyTorch is required for generation") from error

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "past_key_values": cache,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    with torch.inference_mode():
        return model.generate(
            **dict(model_inputs),
            **generation_kwargs,
        )


class TransformersQwen3Model:
    """Load and run Qwen3-32B with pure Transformers on four Ascend NPUs."""

    def __init__(self, options: TransformersModelOptions):
        self.options = options
        self.torch, self.transformers = self._import_runtime()
        self._validate_visible_npus()
        self.config = self._load_config()
        self._validate_model_shape(self.config)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self._validate_loaded_device_map()
        self.input_device = self.model.get_input_embeddings().weight.device

    @staticmethod
    def _import_runtime() -> tuple[Any, Any]:
        try:
            import accelerate  # noqa: F401
            import torch
            import torch_npu  # noqa: F401
            import transformers
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "pure Transformers inference requires torch, torch-npu, "
                "Transformers and Accelerate in the target environment"
            ) from error
        major, minor = _version_release(transformers.__version__)
        if major not in (4, 5) or (major == 4 and minor < 51):
            raise RuntimeError(
                "Transformers 4.51+ or 5.x is required, got "
                f"{transformers.__version__}"
            )
        return torch, transformers

    def _validate_visible_npus(self) -> None:
        torch = self.torch
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("no usable Ascend NPU is visible to torch-npu")
        count = int(torch.npu.device_count())
        if count < 1:
            raise RuntimeError("torch-npu reports zero visible NPUs")
        if count != EXPECTED_NPU_COUNT and not self.options.allow_npu_count_mismatch:
            raise RuntimeError(
                f"expected exactly {EXPECTED_NPU_COUNT} visible NPUs, got {count}; "
                "set ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 or explicitly use "
                "--allow-npu-count-mismatch"
            )
        self.visible_npu_count = count

    def _load_config(self) -> Any:
        config = self.transformers.AutoConfig.from_pretrained(
            self.options.model_path,
            trust_remote_code=self.options.trust_remote_code,
            local_files_only=self.options.local_files_only,
        )
        yarn = {
            "rope_type": "yarn",
            "factor": float(self.options.yarn_factor),
            "original_max_position_embeddings": int(
                self.options.original_max_position_embeddings
            ),
        }
        config.max_position_embeddings = int(self.options.max_position_embeddings)
        # Transformers 5 uses rope_parameters.  Transformers 4 uses
        # rope_scaling; some late 4.x releases expose both names as aliases.
        if _version_major(self.transformers.__version__) >= 5:
            yarn["rope_theta"] = float(self.options.rope_theta)
            config.rope_parameters = yarn
            if hasattr(config, "standardize_rope_params"):
                config.standardize_rope_params()
            if hasattr(config, "validate_rope"):
                config.validate_rope()
        else:
            config.rope_scaling = yarn
        # Keep this attribute explicit for Transformers 4.x RoPE code.
        config.rope_theta = float(self.options.rope_theta)
        return config

    def _validate_model_shape(self, config: Any) -> None:
        actual = {
            "model_type": getattr(config, "model_type", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "head_dim": getattr(config, "head_dim", None),
        }
        expected = {
            "model_type": "qwen3",
            "num_hidden_layers": EXPECTED_NUM_HIDDEN_LAYERS,
            "num_key_value_heads": EXPECTED_NUM_KEY_VALUE_HEADS,
            "head_dim": EXPECTED_HEAD_DIM,
        }
        mismatches = [
            f"{name}={actual[name]!r} (expected {expected_value!r})"
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        ]
        if mismatches and not self.options.allow_model_shape_mismatch:
            raise RuntimeError(
                "model is not the expected Qwen3-32B configuration: "
                + ", ".join(mismatches)
                + "; explicitly use --allow-model-shape-mismatch to continue"
            )

    def _load_tokenizer(self) -> Any:
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            self.options.model_path,
            trust_remote_code=self.options.trust_remote_code,
            local_files_only=self.options.local_files_only,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> Any:
        model_kwargs: dict[str, Any] = {
            "config": self.config,
            "device_map": self.options.device_map,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.options.trust_remote_code,
            "local_files_only": self.options.local_files_only,
        }
        if self.options.max_memory is not None:
            model_kwargs["max_memory"] = dict(self.options.max_memory)
        if _version_major(self.transformers.__version__) >= 5:
            model_kwargs["dtype"] = self.torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = self.torch.bfloat16

        model = self.transformers.AutoModelForCausalLM.from_pretrained(
            self.options.model_path,
            **model_kwargs,
        )
        return model.eval()

    def _device_identity(self, value: Any) -> tuple[str, int | None]:
        if isinstance(value, int):
            return "npu", value
        if isinstance(value, str) and value == "disk":
            return "disk", None
        try:
            device = self.torch.device(value)
        except (RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"unrecognized hf_device_map value: {value!r}"
            ) from error
        return device.type, device.index

    def _validate_loaded_device_map(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if not isinstance(device_map, Mapping) or not device_map:
            raise RuntimeError("loaded model does not expose a non-empty hf_device_map")

        used_npus: set[int] = set()
        offloaded: set[str] = set()
        unexpected: set[str] = set()
        for value in device_map.values():
            device_type, device_index = self._device_identity(value)
            if device_type == "npu":
                used_npus.add(0 if device_index is None else device_index)
            elif device_type in ("cpu", "disk"):
                offloaded.add(device_type)
            else:
                unexpected.add(str(value))

        if offloaded and not self.options.allow_cpu_disk_offload:
            raise RuntimeError(
                "Accelerate offloaded model weights to "
                f"{sorted(offloaded)}; reduce per-device weight pressure or "
                "explicitly use --allow-cpu-disk-offload"
            )
        if unexpected:
            raise RuntimeError(
                f"hf_device_map contains non-NPU devices: {sorted(unexpected)}"
            )

        expected_npus = set(range(EXPECTED_NPU_COUNT))
        if used_npus != expected_npus and not self.options.allow_device_map_mismatch:
            raise RuntimeError(
                f"expected hf_device_map to cover NPUs {sorted(expected_npus)}, "
                f"got {sorted(used_npus)}; use device_map='balanced' or "
                "explicitly use --allow-device-map-mismatch"
            )
        self.used_npus = frozenset(used_npus)
        LOGGER.info(
            "Loaded Qwen3 model with Transformers %s on logical NPUs %s",
            self.transformers.__version__,
            sorted(used_npus),
        )

    def prepare_inputs(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Render non-thinking chat input, tokenize it and move it to NPU."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        max_input_tokens = self.options.max_position_embeddings - max_new_tokens
        return encode_chat(
            self.tokenizer,
            prompt_or_messages,
            max_tokens=max_input_tokens,
            device=self.input_device,
        )

    def generate_greedy(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = 128,
    ) -> GreedyGenerationResult:
        """Generate one deterministic non-thinking answer with DynamicCache."""

        inputs = self.prepare_inputs(prompt_or_messages, max_new_tokens=max_new_tokens)
        input_tokens = int(inputs["input_ids"].shape[-1])
        cache = make_dynamic_cache(self.model)
        sequences = generate_greedy(
            self.model,
            inputs,
            cache,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if sequences.ndim != 2 or sequences.shape[0] != 1:
            raise RuntimeError(
                f"expected one generated sequence, got shape {tuple(sequences.shape)}"
            )
        generated = sequences[0, input_tokens:].detach().to("cpu")
        token_ids = tuple(int(token_id) for token_id in generated.tolist())
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return GreedyGenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=len(token_ids),
            output_token_ids=token_ids,
        )


def load_qwen3(
    options: TransformersModelOptions | argparse.Namespace | str | None = None,
    **option_overrides: Any,
) -> TransformersQwen3Model:
    """Load Qwen3 from an options object, CLI namespace, path, or kwargs."""

    if isinstance(options, argparse.Namespace):
        if option_overrides:
            raise ValueError("CLI namespace cannot be combined with option overrides")
        resolved = options_from_namespace(options)
    elif isinstance(options, TransformersModelOptions):
        if option_overrides:
            raise ValueError("options object cannot be combined with option overrides")
        resolved = options
    elif isinstance(options, str):
        resolved = TransformersModelOptions(
            model_path=options,
            **option_overrides,
        )
    elif options is None:
        resolved = TransformersModelOptions(**option_overrides)
    else:
        raise TypeError(
            "options must be TransformersModelOptions, argparse.Namespace, "
            "a model path, or None"
        )
    return TransformersQwen3Model(resolved)


__all__ = [
    "GreedyGenerationResult",
    "TransformersModelOptions",
    "TransformersQwen3Model",
    "add_transformers_model_arguments",
    "apply_chat_template_non_thinking",
    "build_dynamic_cache",
    "encode_chat",
    "generate_greedy",
    "load_qwen3",
    "make_dynamic_cache",
    "options_from_namespace",
]
