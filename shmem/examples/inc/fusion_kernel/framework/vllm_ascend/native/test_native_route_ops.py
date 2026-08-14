"""Single-NPU smoke test for the stream-aware native route operators."""

from __future__ import annotations

import os
import struct

import torch
import torch_npu  # noqa: F401


def _status_words(status: torch.Tensor) -> tuple[int, ...]:
    raw = bytes(status.cpu().tolist())
    return struct.unpack_from("<8I", raw)


def main() -> None:
    library = os.environ["INC_FUSION_TORCH_LIBRARY"]
    torch.ops.load_library(library)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    ids = torch.tensor(
        [[0, 1], [2, 3], [1, 2], [0, 3], [3, 1]],
        dtype=torch.int64,
        device=device,
    )
    weights = torch.tensor(
        [[0.4, 0.6], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.25, 0.75]],
        dtype=torch.float32,
        device=device,
    )
    counts = torch.empty((3, 5), dtype=torch.int32, device=device)
    status = torch.empty(64, dtype=torch.uint8, device=device)
    torch.ops.inc_fusion_native.route_count_out(ids, counts, status, 2)
    torch.npu.synchronize()
    expected_counts = torch.tensor(
        [[1, 1, 1, 1, 5], [1, 1, 1, 1, 0], [0, 1, 0, 1, 0]],
        dtype=torch.int32,
    )
    assert torch.equal(counts.cpu(), expected_counts)
    assert _status_words(status)[0] == 0

    global_counts = torch.stack((counts, counts), dim=0).contiguous()
    owner = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    local = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    dispatch_rows = torch.empty(10 * 32, dtype=torch.uint8, device=device)
    assignments = torch.empty(10 * 32, dtype=torch.uint8, device=device)
    group_lists = torch.empty((3, 2), dtype=torch.int64, device=device)
    waves = torch.empty(3 * 64, dtype=torch.uint8, device=device)
    scratch = torch.empty(8, dtype=torch.int32, device=device)
    torch.ops.inc_fusion_native.route_pack_out(
        ids,
        weights,
        global_counts,
        owner,
        local,
        dispatch_rows,
        assignments,
        group_lists,
        waves,
        scratch,
        status,
        0,
        256,
        2,
        2,
        3,
        2,
    )
    torch.npu.synchronize()
    words = _status_words(status)
    assert words[0] == 0, words
    assert words[4:7] == (8, 10, 3), words
    expected_groups = torch.tensor(
        [[2, 4], [2, 4], [0, 2]], dtype=torch.int64
    )
    assert torch.equal(group_lists.cpu(), expected_groups)

    first_row = struct.unpack_from("<6IQ", bytes(dispatch_rows.cpu().tolist()))
    assert first_row == (0, 0, 0, 0, 2, 0, 0), first_row
    print("native Torch route_count_out/route_pack_out smoke test passed")


if __name__ == "__main__":
    main()
