# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta
from typing import Any, Optional

import torch
import pickle
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group
from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase,
                                                         AFDConnectorFactory,
                                                         AFDConnectorMetadata)
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
from vllm.distributed.parallel_state import (
    GroupCoordinator, TensorMetadata, _split_tensor_dict,
    get_world_group, init_afd_process_group, init_model_parallel_group)
from vllm.config import VllmConfig, CUDAGraphMode, CompilationMode
from vllm.sequence import IntermediateTensors
from vllm.logger import logger
from vllm.forward_context import get_forward_context
from vllm_ascend.afd_ubatch_utils import (
    get_afd_num_ubatches,
    validate_afd_ubatching_mode,
)
from vllm_ascend.distributed.metadata import NPUP2PAFDConnectorMetadata


_AFD_DP_METADATA_MESSAGE = "dp_metadata"
_AFD_PROFILE_CONTROL_MESSAGE = "profile_control"
_AFD_PROFILE_CONTROL_ACK_MESSAGE = "profile_control_ack"
_AFD_MESSAGE_TYPE_KEY = "__afd_message_type__"


def _select_ubatch_group(
    groups: list[GroupCoordinator], ubatch_idx: int
) -> GroupCoordinator:
    if not 0 <= ubatch_idx < len(groups):
        raise RuntimeError(
            f"ubatch index {ubatch_idx} is outside the configured range "
            f"[0, {len(groups)})"
        )
    return groups[ubatch_idx]


def _decode_afd_p2p_message(obj):
    """Decode one object received on the low-rate AFD control channel."""
    if (
        isinstance(obj, dict)
        and obj.get(_AFD_MESSAGE_TYPE_KEY) == _AFD_PROFILE_CONTROL_MESSAGE
    ):
        is_start = obj.get("is_start")
        profile_prefix = obj.get("profile_prefix")
        if type(is_start) is not bool:
            raise RuntimeError("AFD profile control is_start must be a bool")
        if profile_prefix is not None and not isinstance(profile_prefix, str):
            raise RuntimeError(
                "AFD profile control profile_prefix must be a string or None"
            )
        return _AFD_PROFILE_CONTROL_MESSAGE, (is_start, profile_prefix)

    if not isinstance(obj, tuple):
        raise RuntimeError(
            f"unsupported AFD p2p message type: {type(obj).__name__}"
        )
    if len(obj) == 4:
        data, is_graph_capturing, is_warmup, cudagraph_mode = obj
    elif len(obj) == 2:
        # Backward compatibility with the original metadata tuple.
        data, is_graph_capturing = obj
        is_warmup = False
        cudagraph_mode = CUDAGraphMode.NONE
    else:
        raise RuntimeError(
            f"unsupported AFD metadata tuple length: {len(obj)}"
        )
    return _AFD_DP_METADATA_MESSAGE, (
        data,
        is_graph_capturing,
        is_warmup,
        cudagraph_mode,
    )


def _decode_profile_control_ack(obj, expected_is_start: bool) -> None:
    if (
        not isinstance(obj, dict)
        or obj.get(_AFD_MESSAGE_TYPE_KEY) != _AFD_PROFILE_CONTROL_ACK_MESSAGE
    ):
        raise RuntimeError("invalid AFD profile control acknowledgement")
    if obj.get("is_start") is not expected_is_start:
        raise RuntimeError("AFD profile control acknowledgement action mismatch")
    if obj.get("success") is not True:
        error = obj.get("error") or "unknown FFN profiler error"
        raise RuntimeError(f"paired AFD FFN profiler command failed: {error}")


class DefaultProcessGroupSwitcher:
    """Context manager that temporarily swaps the default process group.

    Used so that ``init_model_parallel_group`` creates the a2e / e2a
    sub-groups on top of the AFD process group instead of the world group.
    """

    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)


class NPUP2PAFDConnector(AFDConnectorBase):
    def __init__(self,
                 rank: int,
                 local_rank: int,
                 config: "VllmConfig",
                 ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        # Cache afd_config for use in init_afd_connector and downstream
        # consumers (e.g. quant_mode exposure for the FFN runner).
        self.afd_config = config.afd_config
        self.backend = "hccl"
        self.attn_size = 0
        self.ffn_size = 0
        self.use_aclgraph = self._use_aclgraph()
        self.dst_list = []
        # dp_metadata_list cache populated by update_state_from_dp_metadata
        self.dp_metadata_list: dict = {}
        # Expose quant_mode from AFDConfig so the FFN runner can decide
        # whether to pass dynamic_scales into compute_ffn_output.
        self.quant_mode = config.afd_config.quant_mode if config.afd_config else 0
        self.role = config.afd_config.afd_role if config.afd_config else None

    def _use_aclgraph(self) -> bool:
        return self.config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and \
               self.config.compilation_config.mode == CompilationMode.VLLM_COMPILE and \
               not self.config.model_config.enforce_eager

    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        # Idempotent guard: NPUFFNModelRunner.__init__ already invokes this
        # once to populate attn_size / ffn_size, and worker.py
        # start_ffn_server_loop invokes initialize_afd_connector() again.
        # Without this guard the second call hits
        # _patched_new_process_group_helper with a duplicate group_name
        # ("afd" / "p2p") and raises ValueError.
        if self._initialized:
            logger.info("NPUP2PAFDConnector already initialized, skipping")
            return
        assert self.config.afd_config is not None, "AFD config is not set"
        self.backend = torch.distributed.get_backend(get_world_group().device_group)
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        self.role = role
        self.attn_size, self.ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        assert self.attn_size == self.ffn_size, "Attention size and FFN size must be the same"
        self.min_size = self.attn_size
        world_rank = self.rank if role == "attention" else self.rank + self.attn_size
        # p2p_rank: all FFN [0, ffn_size), the first min_size Attention ranks
        # use [ffn_size, ffn_size + min_size)
        self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        self.rank = world_rank

        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, "
            f"world_rank = {world_rank}, backend = {self.backend}")

        afd_host = self.config.afd_config.afd_host
        afd_port = self.config.afd_config.afd_port

        self.num_ubatches = get_afd_num_ubatches(self.config)
        validate_afd_ubatching_mode(self.config)
        afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{afd_host}:{afd_port}",
            world_size=self.ffn_size + self.attn_size,
            rank=self.rank,
            group_name="afd",
        )

        ffn_ranks = [i for i in range(self.ffn_size, self.ffn_size + self.attn_size)]
        attn_ranks = [i for i in range(self.attn_size)]

        # All FFN ranks and the first min_size Attention ranks participate in
        # p2p communication.
        # All FFN:        world_rank in [0, ffn_size)
        # First min_size Attention: world_rank in [ffn_size, ffn_size + min_size)
        import datetime
        timeout = datetime.timedelta(seconds=30000)
        if self.is_vaild_rank_for_inequal_AF(self.rank):
            self.p2p_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{afd_host}:{afd_port + 1}",
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timeout  # TODO(yxj):use timeout set
            )

        if self.is_attn_top_min_size_rank(self.rank):
            local_attn_rank = self.rank
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            # Keep stage 0 byte-for-byte compatible with the proven
            # single-stage path, including its registry names and creation
            # order. Additional stages get independent communicators.
            self.a2e_group = init_model_parallel_group(
                sub_group_ranks,
                self.local_rank,
                backend=self.backend,
                group_name="a2e",
            )
            self.e2a_group = init_model_parallel_group(
                sub_group_ranks,
                self.local_rank,
                backend=self.backend,
                group_name="e2a",
            )
            self.a2e_groups = [self.a2e_group]
            self.e2a_groups = [self.e2a_group]
            for idx in range(1, self.num_ubatches):
                self.a2e_groups.append(
                    init_model_parallel_group(
                        sub_group_ranks,
                        self.local_rank,
                        backend=self.backend,
                        group_name=f"a2e_ubatch_{idx}",
                    )
                )
                self.e2a_groups.append(
                    init_model_parallel_group(
                        sub_group_ranks,
                        self.local_rank,
                        backend=self.backend,
                        group_name=f"e2a_ubatch_{idx}",
                    )
                )
        logger.info(
            "AFD communication groups initialized: ubatches=%d ranks=%s",
            self.num_ubatches,
            self.a2e_group.ranks,
        )
        if self.num_ubatches > 1 and self.role == "attention":
            self._initialize_attention_stream_pipeline()
        logger.info("p2p connector initialized")

        self._initialized = True

    def _initialize_attention_stream_pipeline(self) -> None:
        num_layers = self.config.model_config.hf_config.num_hidden_layers
        self.a2f_send_stream = torch.npu.Stream()
        self.f2a_recv_stream = torch.npu.Stream()
        self.attn_compute_events = [
            [torch.npu.Event() for _ in range(self.num_ubatches)]
            for _ in range(num_layers)
        ]
        self.attn_send_events = [
            [torch.npu.Event() for _ in range(self.num_ubatches)]
            for _ in range(num_layers)
        ]
        self.attn_recv_events = [
            [torch.npu.Event() for _ in range(self.num_ubatches)]
            for _ in range(num_layers)
        ]

    def get_attention_pipeline_events(
        self, layer_idx: int, ubatch_idx: int
    ) -> tuple[torch.npu.Event, torch.npu.Event, torch.npu.Event]:
        if self.num_ubatches == 1 or self.role != "attention":
            raise RuntimeError("Attention stream events require AFD UBatch3")
        return (
            self.attn_compute_events[layer_idx][ubatch_idx],
            self.attn_send_events[layer_idx][ubatch_idx],
            self.attn_recv_events[layer_idx][ubatch_idx],
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def _send_tensor_dict_async(
            self,
            tensor_dict: dict[str, torch.Tensor],
            dst: int,
            process_group: GroupCoordinator,
    ) -> list:
        """Asynchronously send a tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary to send
            dst: Destination rank (local rank)
            process_group: The process group to use for communication

        Returns:
            List of work objects that can be used to wait for operation completion
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        process_group.send_object(metadata_list, dst=dst)
        work_list = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip empty tensors
                continue
            num = torch.distributed.send(tensor, dst=process_group.ranks[dst], group=process_group.device_group)
            work_list.append(num)
        return work_list

    def _recv_tensor_dict_async(
            self,
            src: int,
            process_group: GroupCoordinator,
    ) -> tuple[dict[str, torch.Tensor | Any], list]:
        """Asynchronously receive a tensor dictionary.

        Args:
            src: Source rank (local rank)
            process_group: The process group to use for communication
            p2p_group: The process group to use for communication

        Returns:
            tuple: (tensor_dict, work_list) - tensor dictionary and work object list
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []

        assert src < process_group.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = process_group.recv_object(src=src)

        tensor_dict: dict[str, Any] = {}
        work_list = []
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                # Create empty tensor from metadata
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)

                if tensor.numel() == 0:
                    # Skip empty tensors
                    tensor_dict[key] = tensor
                    continue
                work = torch.distributed.recv(tensor, src=process_group.ranks[src], group=process_group.device_group)
                work_list.append(work)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict, work_list

    @staticmethod
    def _wait_event(stream: torch.npu.Stream, event: Optional[torch.npu.Event]) -> None:
        if event is not None:
            event.wait(stream)

    @staticmethod
    def _record_stream(tensor: torch.Tensor, stream: torch.npu.Stream) -> None:
        tensor.record_stream(stream)

    def _get_ubatch_num_tokens(self, ubatch_idx: int) -> int:
        dp_metadata = self.dp_metadata_list.get(ubatch_idx)
        if dp_metadata is None:
            raise RuntimeError(
                f"Missing AFD DP metadata for ubatch {ubatch_idx}"
            )
        num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu
        dp_rank = self.config.parallel_config.data_parallel_rank
        return int(num_tokens_across_dp[dp_rank].item())

    def allocate_streamed_a2f_buffers(
        self, ubatch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate the fixed-shape A2F payload described by DP metadata."""
        num_tokens = self._get_ubatch_num_tokens(ubatch_idx)
        hf_config = self.config.model_config.hf_config
        hidden_states = torch.empty(
            (num_tokens, hf_config.hc_mult, hf_config.hidden_size),
            dtype=self.config.model_config.dtype,
            device="npu",
        )
        # ``NPUModelRunner.input_ids`` uses int32.  The streamed path does not
        # exchange TensorMetadata, so both P2P endpoints must use the same
        # explicit wire dtype.  Receiving the int32 payload into an int64
        # tensor doubles the expected byte count and corrupts the token IDs.
        input_ids = torch.empty(num_tokens, dtype=torch.int32, device="npu")
        return hidden_states, input_ids

    def send_attn_output_streamed(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        ubatch_idx: int,
        wait_event: torch.npu.Event,
        done_event: torch.npu.Event,
    ) -> torch.npu.Event:
        """Enqueue A2F tensors on the Attention send stream."""
        if input_ids is None:
            raise RuntimeError("AFD UBatch3 A2F send requires input_ids")
        group = _select_ubatch_group(self.a2e_groups, ubatch_idx)
        dst = (group.rank_in_group + 1) % group.world_size
        with torch.npu.stream(self.a2f_send_stream):
            self._wait_event(self.a2f_send_stream, wait_event)
            send_hidden_states = hidden_states.contiguous()
            # Keep the wire format stable even if a caller supplies int64 IDs.
            send_input_ids = input_ids.to(dtype=torch.int32).contiguous()
            self._record_stream(send_hidden_states, self.a2f_send_stream)
            self._record_stream(send_input_ids, self.a2f_send_stream)
            torch.distributed.send(
                send_hidden_states,
                dst=group.ranks[dst],
                group=group.device_group,
            )
            torch.distributed.send(
                send_input_ids,
                dst=group.ranks[dst],
                group=group.device_group,
            )
            done_event.record(self.a2f_send_stream)
        return done_event

    def recv_ffn_output_streamed(
        self,
        hidden_states: torch.Tensor,
        ubatch_idx: int,
        wait_event: torch.npu.Event,
        done_event: torch.npu.Event,
    ) -> tuple[torch.Tensor, torch.npu.Event]:
        """Enqueue F2A receive without waiting on the host."""
        group = _select_ubatch_group(self.e2a_groups, ubatch_idx)
        src = (group.rank_in_group - 1) % group.world_size
        recv_hidden_states = torch.empty_like(hidden_states)
        with torch.npu.stream(self.f2a_recv_stream):
            self._wait_event(self.f2a_recv_stream, wait_event)
            self._record_stream(recv_hidden_states, self.f2a_recv_stream)
            torch.distributed.recv(
                recv_hidden_states,
                src=group.ranks[src],
                group=group.device_group,
            )
            done_event.record(self.f2a_recv_stream)
        return recv_hidden_states, done_event

    def recv_attn_output_streamed(
        self,
        ubatch_idx: int,
        recv_stream: torch.npu.Stream,
        wait_event: Optional[torch.npu.Event],
        done_event: torch.npu.Event,
        metadata: Optional[AFDConnectorMetadata] = None,
    ) -> tuple[AFDRecvOutput, torch.npu.Event]:
        """Enqueue the FFN-side A2F receive using precomputed shapes."""
        group = _select_ubatch_group(self.a2e_groups, ubatch_idx)
        src = (group.rank_in_group - 1) % group.world_size
        hidden_states, input_ids = self.allocate_streamed_a2f_buffers(ubatch_idx)
        with torch.npu.stream(recv_stream):
            self._wait_event(recv_stream, wait_event)
            self._record_stream(hidden_states, recv_stream)
            self._record_stream(input_ids, recv_stream)
            torch.distributed.recv(
                hidden_states,
                src=group.ranks[src],
                group=group.device_group,
            )
            torch.distributed.recv(
                input_ids,
                src=group.ranks[src],
                group=group.device_group,
            )
            done_event.record(recv_stream)
        get_forward_context().input_ids = input_ids
        return AFDRecvOutput(hidden_states=hidden_states, metadata=metadata), done_event

    def send_ffn_output_streamed(
        self,
        hidden_states: torch.Tensor,
        ubatch_idx: int,
        send_stream: torch.npu.Stream,
        wait_event: torch.npu.Event,
        done_event: torch.npu.Event,
    ) -> torch.npu.Event:
        """Enqueue F2A output on the FFN send stream."""
        group = _select_ubatch_group(self.e2a_groups, ubatch_idx)
        dst = (group.rank_in_group + 1) % group.world_size
        with torch.npu.stream(send_stream):
            self._wait_event(send_stream, wait_event)
            send_hidden_states = hidden_states.contiguous()
            self._record_stream(send_hidden_states, send_stream)
            torch.distributed.send(
                send_hidden_states,
                dst=group.ranks[dst],
                group=group.device_group,
            )
            done_event.record(send_stream)
        return done_event

    def configure_metadata(self, metadata: Optional["AFDConnectorMetadata"],
                           **kwargs) -> None:
        if metadata is not None and metadata.connector_data is None:
            metadata.connector_data = NPUP2PAFDConnectorMetadata()

    def send_attn_output(self,
                         hidden_states: torch.Tensor,
                         metadata: Optional[AFDConnectorMetadata] = None,
                         **kwargs) -> Any:
        """
        This method will be called by the ATTN side.


        * To send the intermediate tensors generated by ATTN instances to FFN.
        """
        ubatch_idx = getattr(get_forward_context(), "ubatch_idx", 0)
        a2e_group = _select_ubatch_group(self.a2e_groups, ubatch_idx)
        intermediate_tensors = self.create_intermediate_tensors(
            backend=self.backend,
            hidden_states=hidden_states,
            **kwargs
        )
        try:
            dst = (a2e_group.rank_in_group + 1) % a2e_group.world_size
            self._send_tensor_dict_async(
                intermediate_tensors.tensors,
                dst=dst,
                process_group=a2e_group,
            )
            return hidden_states
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_attn_output(
            self,
            metadata: Optional[AFDConnectorMetadata] = None,
            **kwargs
    ) -> Any:
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        a2e_group = _select_ubatch_group(self.a2e_groups, ubatch_idx)

        src = (a2e_group.rank_in_group - 1) % a2e_group.world_size
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=a2e_group,
        )

        if self.backend == "hccl":
            recv_input_ids = intermediate_tensors["input_ids"]
            if recv_input_ids is not None:
                get_forward_context().input_ids = recv_input_ids
            return AFDRecvOutput(
                hidden_states=intermediate_tensors["hidden_states"],
                metadata=metadata,
            )
        else:
            return AFDRecvOutput(
                hidden_states=intermediate_tensors["hidden_states"],
                metadata=metadata
            )

    def create_recv_metadata(self, **kwargs):
        return None

    def update_metadata(self, metadata, recv_output):
        pass

    # -------------------------------------------------------------------------
    #                                attn <- ffn
    # -------------------------------------------------------------------------
    def send_ffn_output(
            self,
            hidden_states: torch.Tensor,
            metadata: Optional[AFDConnectorMetadata] = None,
            **kwargs
    ) -> None:
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        e2a_group = _select_ubatch_group(self.e2a_groups, ubatch_idx)
        intermediate_tensors = IntermediateTensors(
            {
                "hidden_states": hidden_states,
            }
        )
        dst = (e2a_group.rank_in_group + 1) % e2a_group.world_size
        self._send_tensor_dict_async(
            intermediate_tensors.tensors,
            dst=dst,
            process_group=e2a_group,
        )

    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None,
                        metadata: Optional[AFDConnectorMetadata] = None) -> torch.Tensor:
        ubatch_idx = getattr(get_forward_context(), "ubatch_idx", 0)
        e2a_group = _select_ubatch_group(self.e2a_groups, ubatch_idx)
        # Use e2a_group for expert/ffn -> attention communication
        src = (e2a_group.rank_in_group - 1) % e2a_group.world_size
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=e2a_group,
        )
        recv_hs = intermediate_tensors["hidden_states"]
        return recv_hs

    def create_intermediate_tensors(self, backend, hidden_states, **kwargs):
        """Factory method for creating intermediate tensor objects.

        Gate computation is always on the FFN side, so only ``hidden_states``
        and ``input_ids`` (needed for tid2eid mapping) are transferred.
        """
        base_tensors = {"hidden_states": hidden_states}

        if backend == "hccl":
            base_tensors["input_ids"] = kwargs.get('input_ids')

        return IntermediateTensors(base_tensors)

    def current_stream_synchronize(self, backend):
        if backend == "hccl":
            torch.npu.current_stream().synchronize()
        else:
            torch.cuda.current_stream().synchronize()

    def compute_moe(self, experts, hidden_states, **kwargs):
        """Delegate to ``afd_ffn_compute`` on the experts module.

        Gate computation is always on the FFN side, so routing tensors
        (``router_logits`` / ``topk_weights`` / ``topk_ids``) are not
        transferred from the attention side and are computed locally by
        ``afd_ffn_compute`` which reuses the non-AFD MoE path.
        """
        return experts.afd_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            router_logits=kwargs.get('router_logits'),
            topk_weights=kwargs.get('topk_weights'),
            topk_ids=kwargs.get('topk_ids'),
            row_idx=kwargs.get('row_idx'))

    def is_vaild_rank_for_inequal_AF(self, rank):
        # Only support ffn rank < attn rank
        return (self.ffn_size <= rank < self.ffn_size + self.min_size) or rank < self.ffn_size

    def is_attn_top_min_size_rank(self, rank):
        # Only support ffn rank < attn rank
        return rank < self.min_size

    def send_is_ubatch(self, data):
        for dst in self.dst_list:
            object_bytes = pickle.dumps(data)
            object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)

            object_tensor_npu = torch.empty(object_tensor_cpu.shape,
                                            dtype=torch.uint8,
                                            device="npu")
            object_tensor_npu.copy_(object_tensor_cpu)

            size_tensor = torch.tensor([object_tensor_cpu.numel()],
                                        dtype=torch.long,
                                        device="npu")

            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor_npu, dst=dst, group=self.p2p_pg)

    def recv_is_ubatch(self):
        src = self.p2p_rank % self.min_size + self.ffn_size

        size_tensor = torch.empty(1, dtype=torch.long, device="npu")
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        object_tensor_npu = torch.empty(size_tensor.item(), dtype=torch.uint8, device="npu")
        rank_object = torch.distributed.recv(object_tensor_npu, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, "Received object sender rank does not match the size sender rank."

        object_tensor_cpu = object_tensor_npu.cpu()
        data = pickle.loads(object_tensor_cpu.numpy().tobytes())
        return data

    # -------------------------------------------------------------------------
    #                       dp_metadata_list exchange
    #    (ported from v0.13 CAMP2PAFDConnector, used by start_ffn_server_loop)
    # -------------------------------------------------------------------------
    def _send_p2p_object(self, data, dst: int) -> None:
        object_bytes = pickle.dumps(data)
        object_tensor_cpu = torch.frombuffer(
            bytearray(object_bytes), dtype=torch.uint8
        )
        object_tensor_npu = torch.empty(
            object_tensor_cpu.shape, dtype=torch.uint8, device="npu"
        )
        object_tensor_npu.copy_(object_tensor_cpu)
        size_tensor = torch.tensor(
            [object_tensor_cpu.numel()], dtype=torch.long, device="npu"
        )
        torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
        torch.distributed.send(
            object_tensor_npu, dst=dst, group=self.p2p_pg
        )

    def _send_p2p_object_to_ffn(self, data) -> None:
        for dst in self.dst_list:
            self._send_p2p_object(data, dst)

    def _recv_p2p_object(self, src: int):
        size_tensor = torch.empty(1, dtype=torch.long, device="npu")
        rank_size = torch.distributed.recv(
            size_tensor, src=src, group=self.p2p_pg
        )
        object_tensor_npu = torch.empty(
            size_tensor.item(), dtype=torch.uint8, device="npu"
        )
        rank_object = torch.distributed.recv(
            object_tensor_npu, src=src, group=self.p2p_pg
        )
        if rank_object != rank_size:
            raise RuntimeError(
                "Received object sender rank does not match the size sender rank."
            )
        return pickle.loads(object_tensor_npu.cpu().numpy().tobytes())

    def _recv_p2p_object_from_attn(self):
        src = self.p2p_rank % self.min_size + self.ffn_size
        return self._recv_p2p_object(src)

    def send_dp_metadata_list(
        self,
        data,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ):
        """Send dp_metadata_list to the corresponding FFN ranks.

        Args:
            data: dp_metadata_list dict
            is_graph_capturing: whether in graph capture stage
            is_warmup: whether in warmup stage
            cudagraph_mode: cudagraph mode
        """
        send_data = (data, is_graph_capturing, is_warmup, cudagraph_mode)

        # p2p_pg uses HCCL, so the serialized object is copied to NPU before
        # the two size/payload sends.
        self._send_p2p_object_to_ffn(send_data)

    def send_profile_control(
        self,
        is_start: bool,
        profile_prefix: str | None = None,
    ) -> None:
        """Send an out-of-band profiler command to paired FFN workers.

        This message is sent only for /start_profile and /stop_profile. It is
        deliberately kept off the per-layer tensor communication hot path.
        """
        envelope = {
            _AFD_MESSAGE_TYPE_KEY: _AFD_PROFILE_CONTROL_MESSAGE,
            "is_start": is_start,
            "profile_prefix": profile_prefix,
        }
        self._send_p2p_object_to_ffn(envelope)
        # FFN sends the acknowledgement only after profiler.start()/stop()
        # completes. In particular, profile-stop does not return to the HTTP
        # caller before the FFN trace has been exported.
        for src in self.dst_list:
            _decode_profile_control_ack(
                self._recv_p2p_object(src),
                expected_is_start=is_start,
            )

    def send_profile_control_ack(
        self,
        is_start: bool,
        success: bool,
        error: str | None = None,
    ) -> None:
        dst = self.p2p_rank % self.min_size + self.ffn_size
        self._send_p2p_object(
            {
                _AFD_MESSAGE_TYPE_KEY: _AFD_PROFILE_CONTROL_ACK_MESSAGE,
                "is_start": is_start,
                "success": success,
                "error": error,
            },
            dst,
        )

    def recv_afd_message(self):
        """Receive either normal DP metadata or an AFD control message."""
        return _decode_afd_p2p_message(self._recv_p2p_object_from_attn())

    def recv_dp_metadata_list(self):
        """Receive dp_metadata_list.

        Returns:
            tuple: (data, is_graph_capturing, is_warmup, cudagraph_mode)
        """
        message_type, payload = self.recv_afd_message()
        if message_type != _AFD_DP_METADATA_MESSAGE:
            raise RuntimeError(
                f"received {message_type} while waiting for DP metadata"
            )
        return payload

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict,
        is_graph_capturing: bool = False,
    ):
        """Update connector state from received dp_metadata_list.

        Args:
            dp_metadata_list: dp_metadata_list dict
            is_graph_capturing: whether in graph capture stage
        """
        self.dp_metadata_list = dp_metadata_list
