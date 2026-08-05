# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.afd_ubatch_utils import (
    get_afd_num_ubatches,
    validate_afd_ubatching_mode,
)


def make_config(
    num_ubatches=1,
    *,
    enforce_eager=True,
    cudagraph_mode="NONE",
):
    return SimpleNamespace(
        afd_config=SimpleNamespace(
            afd_extra_config={"num_ubatches": num_ubatches}
        ),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=cudagraph_mode)
        ),
    )


@pytest.mark.parametrize("num_ubatches", [1, 3])
def test_supported_afd_ubatch_counts(num_ubatches):
    assert get_afd_num_ubatches(make_config(num_ubatches)) == num_ubatches


@pytest.mark.parametrize("num_ubatches", [2, 4])
def test_rejects_unsupported_afd_ubatch_counts(num_ubatches):
    with pytest.raises(ValueError, match="must be 1 or 3"):
        get_afd_num_ubatches(make_config(num_ubatches))


def test_ubatch3_requires_eager_execution():
    with pytest.raises(ValueError, match="enforce_eager=True"):
        validate_afd_ubatching_mode(
            make_config(3, enforce_eager=False)
        )


def test_ubatch3_rejects_aclgraph():
    with pytest.raises(ValueError, match="does not support ACLGraph"):
        validate_afd_ubatching_mode(
            make_config(3, cudagraph_mode="FULL")
        )
