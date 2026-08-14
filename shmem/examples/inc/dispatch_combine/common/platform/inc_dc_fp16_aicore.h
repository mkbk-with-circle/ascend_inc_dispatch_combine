#ifndef INC_DC_FP16_AICORE_H
#define INC_DC_FP16_AICORE_H

#include "kernel_operator.h"

// Match inc_dc_fp16_host.h (software IEEE path), not a hardware half cast.
__aicore__ inline float IncLdexpf(float x, int exp)
{
    if (exp > 0) {
        for (int i = 0; i < exp; ++i) {
            x *= 2.f;
        }
    } else {
        for (int i = 0; i < -exp; ++i) {
            x *= 0.5f;
        }
    }
    return x;
}

__aicore__ inline float IncFrexpf(float x, int &exp)
{
    exp = 0;
    if (x == 0.f) {
        return 0.f;
    }
    while (x >= 1.f) {
        x *= 0.5f;
        ++exp;
    }
    while (x < 0.5f) {
        x *= 2.f;
        --exp;
    }
    return x;
}

__aicore__ inline float IncAicoreInf()
{
    return 1.f / 0.f;
}

__aicore__ inline uint32_t IncFloatToU32(float x)
{
    return static_cast<uint32_t>(static_cast<int>(x));
}

__aicore__ inline float IncHalfBitsToFp32(uint16_t v)
{
    const uint32_t sign = (v >> 15) & 1u;
    const uint32_t exp = (v >> 10) & 0x1fu;
    const uint32_t mant = v & 0x3ffu;
    if (exp == 0) {
        float val = static_cast<float>(static_cast<int>(mant)) / 1024.f;
        val = IncLdexpf(val, -14);
        return sign ? -val : val;
    }
    if (exp == 31u) {
        const float inf = IncAicoreInf();
        return sign ? -inf : inf;
    }
    float val = 1.f + static_cast<float>(static_cast<int>(mant)) / 1024.f;
    val = IncLdexpf(val, static_cast<int>(exp) - 15);
    return sign ? -val : val;
}

__aicore__ inline uint16_t IncFp32ToHalfBits(float x)
{
    if (x != x) {
        return 0x7e00;
    }
    const uint32_t sign = x < 0.f ? 1u : 0u;
    x = x < 0.f ? -x : x;
    if (x == 0.f) {
        return static_cast<uint16_t>(sign << 15);
    }
    int exp = 0;
    float mant = IncFrexpf(x, exp);
    exp += 14;
    if (exp <= 0) {
        const uint32_t m = IncFloatToU32(mant * IncLdexpf(1.f, 24) + 0.5f);
        return static_cast<uint16_t>((sign << 15) | (m >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    }
    uint32_t out_exp = static_cast<uint32_t>(exp);
    uint32_t m = IncFloatToU32((mant - 0.5f) * 2048.f + 0.5f);
    uint32_t out_m = m & 0x3ffu;
    if (m >= 1024u) {
        out_m = 0;
        ++out_exp;
        if (out_exp >= 31u) {
            return static_cast<uint16_t>((sign << 15) | 0x7c00u);
        }
    }
    return static_cast<uint16_t>((sign << 15) | (out_exp << 10) | out_m);
}

__aicore__ inline float IncGmHalfToFp32(__gm__ uint16_t *p)
{
    return IncHalfBitsToFp32(*p);
}

__aicore__ inline void IncFp32ToGmHalf(float x, __gm__ uint16_t *p)
{
    *p = IncFp32ToHalfBits(x);
}

#endif
