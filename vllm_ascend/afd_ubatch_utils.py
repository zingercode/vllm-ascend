# SPDX-License-Identifier: Apache-2.0

from typing import Any


DEFAULT_AFD_DECODE_UBATCH_TOKEN_THRESHOLD = 8
SUPPORTED_AFD_NUM_UBATCHES = (1, 3)


def _get_afd_extra_config(vllm_config: Any) -> dict[str, Any]:
    afd_config = getattr(vllm_config, "afd_config", None)
    if afd_config is None:
        return {}

    extra_config = getattr(afd_config, "afd_extra_config", None) or {}
    if not isinstance(extra_config, dict):
        raise TypeError("afd_extra_config must be a JSON object")
    return extra_config


def _get_positive_int(
    extra_config: dict[str, Any], key: str, default: int
) -> int:
    raw_value = extra_config.get(key, default)
    if isinstance(raw_value, bool):
        raise ValueError(f"afd_extra_config.{key} must be a positive integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"afd_extra_config.{key} must be a positive integer, got {raw_value!r}"
        ) from exc
    if value < 1 or str(value) != str(raw_value):
        raise ValueError(
            f"afd_extra_config.{key} must be a positive integer, got {raw_value!r}"
        )
    return value


def get_afd_num_ubatches(vllm_config: Any) -> int:
    """Return the AFD-private stage count; one means disabled."""
    num_ubatches = _get_positive_int(
        _get_afd_extra_config(vllm_config), "num_ubatches", 1
    )
    if num_ubatches not in SUPPORTED_AFD_NUM_UBATCHES:
        raise ValueError(
            "afd_extra_config.num_ubatches must be 1 or 3, "
            f"got {num_ubatches}"
        )
    return num_ubatches


def get_afd_decode_ubatch_token_threshold(vllm_config: Any) -> int:
    return _get_positive_int(
        _get_afd_extra_config(vllm_config),
        "decode_ubatch_token_threshold",
        DEFAULT_AFD_DECODE_UBATCH_TOKEN_THRESHOLD,
    )


def is_afd_ubatching_enabled(vllm_config: Any) -> bool:
    return get_afd_num_ubatches(vllm_config) > 1


def validate_afd_ubatching_mode(vllm_config: Any) -> None:
    """Reject graph execution for the eager-only AFD U3 pipeline."""
    if not is_afd_ubatching_enabled(vllm_config):
        return

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not getattr(model_config, "enforce_eager", False):
        raise ValueError(
            "AFD UBatch3 stream overlap currently requires enforce_eager=True"
        )

    compilation_config = getattr(vllm_config, "compilation_config", None)
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    if cudagraph_mode is not None and getattr(cudagraph_mode, "name", "NONE") != "NONE":
        raise ValueError(
            "AFD UBatch3 stream overlap does not support ACLGraph execution"
        )
