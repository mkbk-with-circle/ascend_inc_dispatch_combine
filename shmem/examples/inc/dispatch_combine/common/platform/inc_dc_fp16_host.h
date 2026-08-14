#ifndef INC_DC_FP16_HOST_H
#define INC_DC_FP16_HOST_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace inc {

// IEEE-754 binary16 <-> binary32 (host reference). RNE on float->half.
inline float Fp16BitsToFloat(uint16_t bits)
{
    const uint32_t sign = (bits >> 15) & 1u;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.f : 0.f;
        }
        return (sign ? -1.f : 1.f) * static_cast<float>(mant) / 1024.f * std::ldexp(1.f, -14);
    }
    if (exp == 31u) {
        if (mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    float val = std::ldexp(1.f + static_cast<float>(mant) / 1024.f, static_cast<int>(exp) - 15);
    return sign ? -val : val;
}

inline uint16_t FloatToFp16Bits(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 31) & 1u;
    uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;

    if (exp == 0xffu) {
        if (mant == 0) {
            return static_cast<uint16_t>((sign << 15) | 0x7c00u);
        }
        return static_cast<uint16_t>((sign << 15) | 0x7e00u);
    }

    if ((bits & 0x7fffffffu) == 0) {
        return static_cast<uint16_t>(sign << 15);
    }

    if (exp == 0) {
        while ((mant & 0x00800000u) == 0) {
            mant <<= 1;
            --exp;
        }
        mant &= 0x7fffffu;
        ++exp;
    }

    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
    uint32_t new_mant = mant >> 13;
    const uint32_t round_bit = (mant >> 12) & 1u;
    const uint32_t sticky = (mant & 0xfffu) != 0;

    if (round_bit && (sticky || (new_mant & 1u))) {
        ++new_mant;
        if (new_mant == 0x400u) {
            new_mant = 0;
            ++new_exp;
        }
    }

    if (new_exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    }

    if (new_exp <= 0) {
        if (new_exp < -10) {
            return static_cast<uint16_t>(sign << 15);
        }
        mant = (mant & 0x7fffffu) | 0x800000u;
        const int32_t shift = 1 - new_exp;
        uint32_t sub = mant >> static_cast<uint32_t>(shift + 13);
        const uint32_t rb = (mant >> static_cast<uint32_t>(shift + 12)) & 1u;
        const uint32_t st = (mant & ((1u << static_cast<uint32_t>(shift + 12)) - 1u)) != 0;
        if (rb && (st || (sub & 1u))) {
            ++sub;
        }
        return static_cast<uint16_t>((sign << 15) | (sub & 0x3ffu));
    }

    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(new_exp) << 10) | (new_mant & 0x3ffu));
}

} // namespace inc

#endif
