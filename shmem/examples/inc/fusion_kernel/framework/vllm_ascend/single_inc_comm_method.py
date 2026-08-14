"""vLLM-Ascend 0.19.1 MoECommMethod adapter for the unified INC op.

This is the framework-side adapter, not the device extension.  Importing it is
safe before the extension is loaded; execution fails explicitly when the
``inc_fusion::moe`` Torch operator has not been registered.
"""

from __future__ import annotations

from collections.abc import Callable
import os

import torch
import torch.distributed as dist

from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AlltoAllCommImpl,
    FusedExpertsResult,
)
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType

from inc_moe_runtime import BackendMode
from torch_fusion_runtime import PreparedTorchExecutor


_MODE_ID = {
    BackendMode.SERIAL_SHMEM: 1,
    BackendMode.SERIAL_INC: 2,
    BackendMode.FUSED_SHMEM: 3,
    BackendMode.FUSED_INC: 4,
}

_DEBUG_SYNC = os.environ.get("INC_FUSION_DEBUG_SYNC", "").lower()
_TRACE_CALLS = os.environ.get("INC_FUSION_TRACE_CALLS", "") == "1"
_TRACE_DEVICE_LAYER = int(os.environ.get("INC_FUSION_TRACE_DEVICE_LAYER", "-1"))
_REPEAT_DEVICE_LAYER = int(
    os.environ.get("INC_FUSION_REPEAT_DEVICE_LAYER", "-1")
)


class SingleIncMoECommMethod(AlltoAllCommImpl):
    """Execute one of the four attribution modes through one Torch op.

    Inheriting AlltoAllCommImpl only reuses vLLM's TP-aware prepare/finalize
    behavior.  Its token dispatcher is not invoked: dispatch, FFN and combine
    are all owned by ``inc_fusion::moe``.
    """

    def __init__(
        self,
        moe_config,
        mode: BackendMode,
        prepared_weight_resolver: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        prepared_executor_resolver: Callable[[int], PreparedTorchExecutor]
        | None = None,
    ):
        if not mode.is_factorial:
            raise ValueError("native_vllm must use vLLM's native comm method")
        self.mode = mode
        # vLLM-Ascend 0.19.1 transposes and may format-cast its MoE weights.
        # The current fusion kernel consumes its own ND layout.  Requiring an
        # explicit setup-time resolver prevents a future registered op from
        # silently interpreting vLLM's native/NZ storage with the wrong layout.
        self._prepared_weight_resolver = prepared_weight_resolver
        self._prepared_executor_resolver = prepared_executor_resolver
        self._debug_layer_ids: dict[tuple[int, int], int] = {}
        self._debug_layer_calls: dict[int, int] = {}
        self._native_reference = (
            AlltoAllCommImpl(moe_config)
            if os.environ.get("INC_FUSION_DEBUG_NATIVE_COMPARE") == "1"
            else None
        )
        super().__init__(moe_config)

    def _uses_inc_global_output(self) -> bool:
        return (
            self.mode in (BackendMode.SERIAL_INC, BackendMode.FUSED_INC)
            and self.moe_config.experts_per_token >= 2
        )

    def _require_global_output_prepare_contract(self) -> None:
        prepare_finalize = self.prepare_finalize
        if (
            getattr(prepare_finalize, "enable_shared_expert_dp", False)
            or getattr(prepare_finalize, "replace_allreduce", False)
        ):
            raise NotImplementedError(
                "INC global-output fanout currently requires the normal "
                "TP token-slice prepare path; shared-expert-DP and "
                "replace-allreduce require a separate output contract"
            )

    def _prepared_global_rows(self) -> int:
        """Rows in the tensor before All2All finalization unpads it."""
        prepare_finalize = self.prepare_finalize
        num_tokens = int(prepare_finalize.num_tokens)
        tp_size = int(prepare_finalize.tp_size)
        # PrepareAndFinalizeWithAll2All pads only the small-token case to one
        # token per TP rank.  Otherwise tensor_split preserves the original
        # total even when its per-rank slices are uneven.
        return max(num_tokens, tp_size)

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if not self._uses_inc_global_output():
            return super().finalize(
                hidden_states, reduce_results, padded_hidden_states_shape
            )
        del reduce_results
        self._require_global_output_prepare_contract()
        expected_rows = self._prepared_global_rows()
        if hidden_states.shape[0] != expected_rows:
            raise RuntimeError(
                "INC global output disagrees with the prepared TP token "
                f"shape: got {hidden_states.shape[0]}, expected "
                f"{expected_rows}"
            )
        num_tokens = int(self.prepare_finalize.num_tokens)
        return hidden_states[:num_tokens]

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ) -> FusedExpertsResult:
        if fused_experts_input.quant.quant_type is not QuantType.NONE:
            raise NotImplementedError(
                "the first single-INC qualification is BF16 unquantized; "
                "quantized baselines must not be mixed into it"
            )
        x = fused_experts_input.hidden_states
        if x.dtype is not torch.bfloat16:
            raise TypeError(f"single-INC currently requires BF16, got {x.dtype}")
        activation = fused_experts_input.activation
        activation_name = getattr(activation, "name", activation)
        if str(activation_name).lower() != "silu":
            raise NotImplementedError(
                f"single-INC currently requires silu/SwiGLU, got "
                f"{fused_experts_input.activation}"
            )
        w13 = fused_experts_input.weights.w1
        w2 = fused_experts_input.weights.w2
        if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
            raise TypeError("single-INC requires one local BF16 w13/w2 tensor")
        if self._prepared_weight_resolver is None:
            raise RuntimeError(
                "single-INC prepared weight resolver is missing; weight layout "
                "must be converted/cached during model setup, never in forward"
            )
        source_weight_key = (int(w13.data_ptr()), int(w2.data_ptr()))
        trace_layer = self._debug_layer_ids.setdefault(
            source_weight_key, len(self._debug_layer_ids)
        )
        trace_call = self._debug_layer_calls.get(trace_layer, 0)
        if _TRACE_CALLS:
            print(
                f"INC_FUSION_CALL rank={dist.get_rank()} layer={trace_layer} "
                f"call={trace_call} tokens={x.shape[0]} stage=enter",
                flush=True,
            )
        reference_out = None
        if self._native_reference is not None:
            debug_layer = self._debug_layer_ids.setdefault(
                source_weight_key, len(self._debug_layer_ids)
            )
            selected_layer = int(
                os.environ.get("INC_FUSION_DEBUG_LAYER", "-1")
            )
            if selected_layer < 0 or debug_layer == selected_layer:
                reference_out = self._native_reference.fused_experts(
                    fused_experts_input
                ).routed_out
        w13, w2 = self._prepared_weight_resolver(w13, w2)
        if w13.ndim != 3 or w2.ndim != 3:
            raise RuntimeError("prepared single-INC weights must be rank-3")
        hidden = x.shape[-1]
        fusion_layout = (
            w13.shape[2] == hidden
            and w2.shape[1] == hidden
            and w13.shape[1] == 2 * w2.shape[2]
        )
        row_major_b_layout = (
            w13.shape[1] == hidden
            and w2.shape[2] == hidden
            and w13.shape[2] == 2 * w2.shape[1]
        )
        if not (fusion_layout or row_major_b_layout):
            raise RuntimeError(
                "prepared weights must be w13[E,2I,H], w2[E,H,I] or "
                "w13[E,H,2I], w2[E,I,H]"
            )

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]
        topk_weights = fused_experts_input.topk_weights
        expert_map = fused_experts_input.routing.expert_map
        if self._prepared_executor_resolver is None:
            raise RuntimeError(
                "single-INC prepared executor resolver is missing; route, "
                "output and native executor must be prepared during setup"
            )
        prepared_executor = self._prepared_executor_resolver(x.shape[0])
        if not prepared_executor.accepts(x.shape[0]):
            raise RuntimeError("prepared executor does not accept this batch")
        prepared_route = prepared_executor.route
        topk_ids, topk_weights = prepared_route.pack(
            topk_ids, topk_weights, prepared_executor.handle,
            prepared_executor.uses_inc_route_relay,
        )
        if _DEBUG_SYNC in ("route", "all"):
            torch.npu.synchronize()
        if _TRACE_CALLS:
            print(
                f"INC_FUSION_CALL rank={dist.get_rank()} layer={trace_layer} "
                f"call={trace_call} tokens={x.shape[0]} stage=route_ready",
                flush=True,
            )
        if prepared_executor.returns_global_output:
            self._require_global_output_prepare_contract()
            out = prepared_executor.output_for(
                x, global_output_rows=self._prepared_global_rows()
            )
        else:
            out = prepared_executor.output_for(x)

        namespace = getattr(torch.ops, "inc_fusion", None)
        op = getattr(namespace, "moe", None) if namespace is not None else None
        if op is None:
            raise RuntimeError(
                "inc_fusion::moe is not registered; load the device extension "
                "before constructing the vLLM engine"
            )
        out = op(
            prepared_executor.handle,
            x,
            w13,
            w2,
            topk_ids,
            topk_weights,
            expert_map,
            prepared_route.global_counts,
            prepared_route.dispatch_rows,
            prepared_route.assignments,
            prepared_route.active_token_counts,
            prepared_route.group_lists,
            prepared_route.waves,
            prepared_route.status,
            out,
            _MODE_ID[self.mode],
        )
        if trace_layer == _REPEAT_DEVICE_LAYER:
            torch.npu.synchronize()
            first = out.clone()
            out = op(
                prepared_executor.handle,
                x,
                w13,
                w2,
                topk_ids,
                topk_weights,
                expert_map,
                prepared_route.global_counts,
                prepared_route.dispatch_rows,
                prepared_route.assignments,
                prepared_route.active_token_counts,
                prepared_route.group_lists,
                prepared_route.waves,
                prepared_route.status,
                out,
                _MODE_ID[self.mode],
            )
            torch.npu.synchronize()
            repeat_error = float(
                (first.float() - out.float()).abs().max().item()
            )
            print(
                f"INC_FUSION_REPEAT_DEVICE rank={dist.get_rank()} "
                f"layer={trace_layer} call={trace_call} "
                f"first_checksum={float(first.float().sum().item()):.9g} "
                f"second_checksum={float(out.float().sum().item()):.9g} "
                f"max_error={repeat_error:.9g}",
                flush=True,
            )
        if trace_layer == _TRACE_DEVICE_LAYER:
            trace_values = torch.ops.inc_fusion_native.worker_debug_traces(
                prepared_route.expert_owner, prepared_executor.handle
            )
            records = []
            for begin in range(0, len(trace_values), 9):
                raw = list(trace_values[begin : begin + 9])
                start, end = raw[2], raw[3]
                checkpoints = [
                    value - start if value > start else 0
                    for value in raw[4:9]
                ]
                records.append(
                    {
                        "role": raw[0],
                        "lane": raw[1],
                        "cycles": end - start if end > start else 0,
                        "checkpoints": checkpoints,
                    }
                )
            print(
                f"INC_FUSION_DEVICE_TRACE rank={dist.get_rank()} "
                f"layer={trace_layer} call={trace_call} records={records}",
                flush=True,
            )
        if _DEBUG_SYNC in ("moe", "all"):
            torch.npu.synchronize()
            if _TRACE_CALLS:
                slot_values = torch.ops.inc_fusion_native.worker_debug_slots(
                    prepared_route.expert_owner, prepared_executor.handle
                )
                print(
                    f"INC_FUSION_SLOTS rank={dist.get_rank()} "
                    f"layer={trace_layer} call={trace_call} "
                    f"values={list(slot_values)}",
                    flush=True,
                )
        if _TRACE_CALLS:
            print(
                f"INC_FUSION_CALL rank={dist.get_rank()} layer={trace_layer} "
                f"call={trace_call} tokens={x.shape[0]} stage=moe_enqueued",
                flush=True,
            )
        if os.environ.get("INC_FUSION_DEBUG_STATS") == "1":
            layer = self._debug_layer_ids.setdefault(
                source_weight_key, len(self._debug_layer_ids)
            )
            call = self._debug_layer_calls.get(layer, 0)
            self._debug_layer_calls[layer] = call + 1
            selected_layer = int(os.environ.get("INC_FUSION_DEBUG_LAYER", "-1"))
            if selected_layer < 0 or layer == selected_layer:
                input_checksum = float(x.float().sum().item())
                input_absmax = float(x.float().abs().max().item())
                ids_checksum = int(topk_ids.to(torch.int64).sum().item())
                weights_checksum = float(topk_weights.float().sum().item())
                checksum = float(out.float().sum().item())
                absmax = float(out.float().abs().max().item())
                reference_text = ""
                if reference_out is not None:
                    comparison_out = out
                    if prepared_executor.returns_global_output:
                        rank = int(self.prepare_finalize.tp_rank)
                        local_rows = x.shape[0]
                        # The source-rank-major global output uses compact
                        # tensor_split slices.  Sum the deterministic split
                        # lengths to locate this rank without device sync.
                        global_rows = self._prepared_global_rows()
                        base, extra = divmod(
                            global_rows,
                            int(self.prepare_finalize.tp_size),
                        )
                        offset = rank * base + min(rank, extra)
                        comparison_out = out[offset : offset + local_rows]
                    reference_checksum = float(
                        reference_out.float().sum().item()
                    )
                    max_error = float(
                        (comparison_out.float() - reference_out.float())
                        .abs()
                        .max()
                        .item()
                    )
                    reference_text = (
                        f" reference_checksum={reference_checksum:.9g}"
                        f" reference_max_error={max_error:.9g}"
                    )
                print(
                    f"INC_FUSION_LAYER_STATS rank={dist.get_rank()} "
                    f"layer={layer} call={call} tokens={out.shape[0]} "
                    f"input_checksum={input_checksum:.9g} "
                    f"input_absmax={input_absmax:.9g} "
                    f"ids_checksum={ids_checksum} "
                    f"weights_checksum={weights_checksum:.9g} "
                    f"output_checksum={checksum:.9g} "
                    f"output_absmax={absmax:.9g}{reference_text}",
                    flush=True,
                )
        elif _TRACE_CALLS:
            self._debug_layer_calls[trace_layer] = trace_call + 1
        expected_shape = (
            (self._prepared_global_rows(), x.shape[1])
            if prepared_executor.returns_global_output
            else x.shape
        )
        if (
            out.shape != expected_shape
            or out.dtype != x.dtype
            or out.device != x.device
        ):
            raise RuntimeError(
                "inc_fusion::moe returned a tensor with incompatible "
                "shape/dtype/device"
            )
        return FusedExpertsResult(routed_out=out)


__all__ = ["SingleIncMoECommMethod"]
