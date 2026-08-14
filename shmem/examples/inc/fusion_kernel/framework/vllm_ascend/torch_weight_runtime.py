"""Setup-time weight conversion/cache for the fusion kernel's BF16 ND ABI."""

from __future__ import annotations

import torch
import torch_npu

from inc_moe_runtime import WeightLayout


_ACL_FORMAT_ND = 2
_ACL_FORMAT_FRACTAL_NZ = 29


class PreparedWeightCache:
    """Own a persistent fusion-layout pair; ``resolve`` never transforms."""

    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        layout: WeightLayout,
        local_experts: int,
        hidden: int,
        intermediate: int,
    ) -> None:
        if not isinstance(layout, WeightLayout):
            raise TypeError("layout must be an explicit WeightLayout")
        if local_experts <= 0 or hidden <= 0 or intermediate <= 0:
            raise ValueError("weight dimensions must be positive")
        for tensor, name in ((w13, "w13"), (w2, "w2")):
            if tensor.device.type != "npu" or tensor.dtype is not torch.bfloat16:
                raise TypeError(f"{name} must be a BF16 NPU tensor")
            if tensor.ndim != 3:
                raise ValueError(f"{name} must be rank-3")
        if w13.device != w2.device:
            raise ValueError("w13 and w2 must be on the same NPU")

        expected = (
            (local_experts, 2 * intermediate, hidden),
            (local_experts, hidden, intermediate),
        )
        transposed = (
            (local_experts, hidden, 2 * intermediate),
            (local_experts, intermediate, hidden),
        )
        wanted = expected if layout is WeightLayout.FUSION_ND else transposed
        if tuple(w13.shape) != wanted[0] or tuple(w2.shape) != wanted[1]:
            raise ValueError(
                f"weight shapes disagree with {layout.value}: "
                f"{tuple(w13.shape)}, {tuple(w2.shape)}"
            )

        def to_nd(tensor: torch.Tensor, name: str) -> torch.Tensor:
            storage_format = int(torch_npu.get_npu_format(tensor))
            if storage_format == _ACL_FORMAT_ND:
                return tensor
            if storage_format == _ACL_FORMAT_FRACTAL_NZ:
                return torch_npu.npu_format_cast(tensor, _ACL_FORMAT_ND)
            raise RuntimeError(
                f"unsupported {name} NPU storage format {storage_format}; "
                "only ND(2) and FRACTAL_NZ(29) have an explicit setup path"
            )

        nd_w13 = to_nd(w13, "w13")
        nd_w2 = to_nd(w2, "w2")
        self.w13 = nd_w13.contiguous()
        self.w2 = nd_w2.contiguous()
        if int(torch_npu.get_npu_format(self.w13)) != _ACL_FORMAT_ND or \
                int(torch_npu.get_npu_format(self.w2)) != _ACL_FORMAT_ND:
            raise RuntimeError("prepared fusion weights are not ND")
        if tuple(self.w13.shape) != wanted[0] or tuple(
            self.w2.shape
        ) != wanted[1]:
            raise RuntimeError("prepared fusion weight shape invariant failed")
        self._source_w13_ptr = int(w13.data_ptr())
        self._source_w2_ptr = int(w2.data_ptr())
        self._device = w13.device

    def resolve(
        self, w13: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if w13.device != self._device or w2.device != self._device or \
                int(w13.data_ptr()) != self._source_w13_ptr or \
                int(w2.data_ptr()) != self._source_w2_ptr:
            raise RuntimeError(
                "vLLM replaced or moved a prepared MoE weight; rebuild the "
                "setup-time fusion weight cache"
            )
        return self.w13, self.w2


__all__ = ["PreparedWeightCache"]
