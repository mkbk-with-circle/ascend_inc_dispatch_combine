/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PCIE_NIC_MATCHER_H
#define PCIE_NIC_MATCHER_H

#include <cstddef>
#include <cstring>

namespace shm {
namespace topo {

inline const char* get_path_basename(const char* path)
{
    if (path == nullptr) {
        return "";
    }
    const char* base = strrchr(path, '/');
    return (base == nullptr) ? path : base + 1;
}

// Python is_pix_topo uses pci_path[0:-18] to strip the last 18 chars, then
// falls back to lspci Serial comparison for bridges with the same hardware.
constexpr size_t PCI_PATH_TAIL_LEN = 18;

// Full implementation in pcie_nic.cpp (needs popen for lspci)
bool is_pix_topo(const char* path_a, const char* path_b);

} // namespace topo
} // namespace shm

#endif // PCIE_NIC_MATCHER_H
