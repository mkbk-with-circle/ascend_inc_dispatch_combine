"""Register the graph-visible, allocation-free ``inc_fusion::moe`` op.

The compiled bridge owns ``inc_fusion_native::worker_moe_out``.  This module
adds a stable schema plus fake implementation for vLLM/torch.compile.  Output
HBM and the native executor are both prepared during engine setup.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.utils.torch_utils import direct_register_custom_op


_INC_FUSION_LIBRARY: Library | None = None
_REGISTERED = False


def _native_op():
    namespace = getattr(torch.ops, "inc_fusion_native", None)
    op = getattr(namespace, "worker_moe_out", None) \
        if namespace is not None else None
    if op is None:
        raise RuntimeError(
            "inc_fusion_native::worker_moe_out is missing; load the compiled "
            "single-INC runtime bridge before registering inc_fusion::moe"
        )
    return op


def _moe(
    executor_handle: int,
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor | None,
    global_counts: torch.Tensor,
    dispatch_rows: torch.Tensor,
    assignments: torch.Tensor,
    active_token_counts: torch.Tensor,
    group_lists: torch.Tensor,
    waves: torch.Tensor,
    route_status: torch.Tensor,
    out: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    # Router tensors which do not enter the device ABI remain arguments of the
    # graph-visible op so torch.compile cannot move the prepared route outside
    # this request.  The native hot op consumes only the packed protocol.
    del topk_ids, topk_weights, expert_map, global_counts, route_status
    _native_op()(
        executor_handle, x, w13, w2, dispatch_rows, assignments, waves,
        active_token_counts, group_lists, out, mode
    )
    return out


def _moe_fake(
    executor_handle: int,
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor | None,
    global_counts: torch.Tensor,
    dispatch_rows: torch.Tensor,
    assignments: torch.Tensor,
    active_token_counts: torch.Tensor,
    group_lists: torch.Tensor,
    waves: torch.Tensor,
    route_status: torch.Tensor,
    out: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    del (
        executor_handle,
        w13, w2, topk_ids, topk_weights, expert_map, global_counts,
        dispatch_rows, assignments, active_token_counts, group_lists, waves,
        route_status, mode
    )
    if (
        out.ndim != x.ndim
        or out.shape[1:] != x.shape[1:]
        or out.shape[0] < x.shape[0]
        or out.dtype != x.dtype
        or out.device != x.device
    ):
        raise RuntimeError(
            "prepared output must preserve dtype/device/hidden dimensions "
            "and contain at least the local input rows"
        )
    return out


def register_inc_fusion_op() -> None:
    global _INC_FUSION_LIBRARY, _REGISTERED
    if _REGISTERED:
        return
    _native_op()  # fail before publishing a graph-visible but unusable op
    library = Library("inc_fusion", "FRAGMENT")
    direct_register_custom_op(
        op_name="moe",
        op_func=_moe,
        fake_impl=_moe_fake,
        mutates_args=["out"],
        target_lib=library,
        dispatch_key="PrivateUse1",
    )
    _INC_FUSION_LIBRARY = library  # operator lifetime follows the Library
    _REGISTERED = True


__all__ = ["register_inc_fusion_op"]
