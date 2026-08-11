#ifndef INC_DC_UB_TILE_AICORE_H
#define INC_DC_UB_TILE_AICORE_H

#include "inc_dc_platform_capabilities.h"

// Host computes the effective tile; kernels must not clamp it.

constexpr int INC_AIV_UB_BUDGET_BYTES_AICORE =
    static_cast<int>(inc::dc::kIncDcAivUbBudgetBytes);
constexpr int INC_AIV_UB_RESERVE_BYTES_AICORE = 0;

__aicore__ inline int IncUbAlignUpAicore(int x, int align)
{
    return (x + align - 1) / align * align;
}

__aicore__ inline int IncScalarFp16RowBytesAicore(int tile_elems)
{
    return IncUbAlignUpAicore(tile_elems * static_cast<int>(sizeof(uint16_t)), 32);
}

__aicore__ inline int IncScalarUbBytesRequiredAicore(int tile_elems, int n_contrib)
{
    return IncScalarFp16RowBytesAicore(tile_elems) * (n_contrib + 1);
}

#endif
