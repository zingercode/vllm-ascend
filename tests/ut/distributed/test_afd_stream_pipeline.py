# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.distributed.NPUP2PAFDConnector import (
    _AFD_DP_METADATA_MESSAGE,
    _AFD_MESSAGE_TYPE_KEY,
    _AFD_PROFILE_CONTROL_ACK_MESSAGE,
    _AFD_PROFILE_CONTROL_MESSAGE,
    NPUP2PAFDConnector,
    _decode_afd_p2p_message,
    _decode_profile_control_ack,
    _select_ubatch_group,
)


def test_select_ubatch_group_validates_index():
    groups = [object(), object(), object()]
    assert _select_ubatch_group(groups, 2) is groups[2]

    try:
        _select_ubatch_group(groups, 3)
    except RuntimeError as exc:
        assert "outside the configured range" in str(exc)
    else:
        raise AssertionError("out-of-range ubatch index was accepted")


def test_decode_afd_p2p_message_preserves_metadata_compatibility():
    metadata = {0: object()}
    mode = object()

    message_type, payload = _decode_afd_p2p_message(
        (metadata, False, True, mode)
    )

    assert message_type == _AFD_DP_METADATA_MESSAGE
    assert payload == (metadata, False, True, mode)


def test_profile_control_message_round_trip():
    connector = object.__new__(NPUP2PAFDConnector)
    connector.dst_list = [7]
    connector._send_p2p_object_to_ffn = MagicMock()
    connector._recv_p2p_object = MagicMock(
        return_value={
            _AFD_MESSAGE_TYPE_KEY: _AFD_PROFILE_CONTROL_ACK_MESSAGE,
            "is_start": True,
            "success": True,
            "error": None,
        }
    )

    connector.send_profile_control(True, "capture-42")

    envelope = connector._send_p2p_object_to_ffn.call_args.args[0]
    assert envelope == {
        _AFD_MESSAGE_TYPE_KEY: _AFD_PROFILE_CONTROL_MESSAGE,
        "is_start": True,
        "profile_prefix": "capture-42",
    }
    message_type, payload = _decode_afd_p2p_message(envelope)
    assert message_type == _AFD_PROFILE_CONTROL_MESSAGE
    assert payload == (True, "capture-42")
    connector._recv_p2p_object.assert_called_once_with(7)


def test_profile_control_rejects_malformed_action():
    with pytest.raises(RuntimeError, match="is_start must be a bool"):
        _decode_afd_p2p_message(
            {
                _AFD_MESSAGE_TYPE_KEY: _AFD_PROFILE_CONTROL_MESSAGE,
                "is_start": "yes",
                "profile_prefix": None,
            }
        )


def test_profile_control_ack_propagates_ffn_failure():
    with pytest.raises(RuntimeError, match="FFN profiler command failed"):
        _decode_profile_control_ack(
            {
                _AFD_MESSAGE_TYPE_KEY: _AFD_PROFILE_CONTROL_ACK_MESSAGE,
                "is_start": False,
                "success": False,
                "error": "export failed",
            },
            expected_is_start=False,
        )


def test_allocate_streamed_a2f_buffers_uses_dp_metadata_shape():
    connector = object.__new__(NPUP2PAFDConnector)
    connector.dp_metadata_list = {}
    connector.update_state_from_dp_metadata({
        2: SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor([5], dtype=torch.int32)
        )
    })
    connector.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(hc_mult=4, hidden_size=16),
        ),
    )
    hidden_buffer = MagicMock()
    input_ids_buffer = MagicMock()

    with patch(
        "torch.empty", side_effect=[hidden_buffer, input_ids_buffer]
    ) as empty:
        result = connector.allocate_streamed_a2f_buffers(2)

    assert result == (hidden_buffer, input_ids_buffer)
    assert empty.call_args_list == [
        call((5, 4, 16), dtype=torch.bfloat16, device="npu"),
        call(5, dtype=torch.int32, device="npu"),
    ]


def test_streamed_attn_send_waits_then_enqueues_tensor_sends():
    connector = object.__new__(NPUP2PAFDConnector)
    stream = MagicMock(name="send_stream")
    connector.a2f_send_stream = stream
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        ranks=[10, 11],
        device_group=MagicMock(name="device_group"),
    )
    connector.a2e_groups = [group, group, group]
    hidden_states = MagicMock(name="hidden_states")
    input_ids = MagicMock(name="input_ids")
    hidden_states.contiguous.return_value = hidden_states
    input_ids.to.return_value = input_ids
    input_ids.contiguous.return_value = input_ids
    wait_event = MagicMock(name="compute_done")
    done_event = MagicMock(name="send_done")

    with (
        patch("torch.npu.stream", return_value=nullcontext()),
        patch("torch.distributed.send") as send,
    ):
        result = connector.send_attn_output_streamed(
            hidden_states,
            input_ids,
            ubatch_idx=1,
            wait_event=wait_event,
            done_event=done_event,
        )

    assert result is done_event
    wait_event.wait.assert_called_once_with(stream)
    assert send.call_args_list == [
        call(hidden_states, dst=11, group=group.device_group),
        call(input_ids, dst=11, group=group.device_group),
    ]
    hidden_states.record_stream.assert_called_once_with(stream)
    input_ids.to.assert_called_once_with(dtype=torch.int32)
    input_ids.record_stream.assert_called_once_with(stream)
    done_event.record.assert_called_once_with(stream)
