/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_RMA_UB2GM_DEFINITIONS_H
#define SIMT_RMA_UB2GM_DEFINITIONS_H

#if defined(USE_SIMT)

#include <cstdint>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "acl/acl.h"

#define TYPED_FUNCTION_DEF(FUNC) \
    FUNC(half)                   \
    FUNC(float)                  \
    FUNC(int8)                   \
    FUNC(int16)                  \
    FUNC(int32)                  \
    FUNC(int64)                  \
    FUNC(uint8)                  \
    FUNC(uint16)                 \
    FUNC(uint32)                 \
    FUNC(uint64)                 \
    FUNC(char)                   \
    FUNC(schar)                  \
    FUNC(bfloat16)

#define TYPE_FUNCTION_TEMPLATE(TYPE)                                             \
    extern void test_ub2gm_aclshmem_##TYPE##_put(                                \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_put_warp(                          \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_put_block(                         \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmem_##TYPE##_put_nbi(                            \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_put_nbi_warp(                      \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_put_nbi_block(                     \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmem_##TYPE##_get(                                \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_get_warp(                          \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_get_block(                         \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmem_##TYPE##_get_nbi(                            \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_get_nbi_warp(                      \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_##TYPE##_get_nbi_block(                     \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);

TYPED_FUNCTION_DEF(TYPE_FUNCTION_TEMPLATE);

#undef TYPE_FUNCTION_TEMPLATE
#undef TYPED_FUNCTION_DEF

#define SIZED_FUNCTION_DEF(FUNC) \
    FUNC(8)                      \
    FUNC(16)                     \
    FUNC(32)                     \
    FUNC(64)                     \
    FUNC(128)

#define SIZED_FUNCTION_TEMPLATE(SIZE)                                                                                  \
    extern void test_ub2gm_aclshmem_put##SIZE(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_put##SIZE##_warp(                                                                 \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmemx_put##SIZE##_block(                                                                \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmem_put##SIZE##_nbi(                                                                   \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmemx_put##SIZE##_nbi_warp(                                                             \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmemx_put##SIZE##_nbi_block(                                                            \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmem_get##SIZE(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_ub2gm_aclshmemx_get##SIZE##_warp(                                                                 \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmemx_get##SIZE##_block(                                                                \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmem_get##SIZE##_nbi(                                                                   \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmemx_get##SIZE##_nbi_warp(                                                             \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                                       \
    extern void test_ub2gm_aclshmemx_get##SIZE##_nbi_block(                                                            \
        aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);

SIZED_FUNCTION_DEF(SIZED_FUNCTION_TEMPLATE);

#undef SIZED_FUNCTION_TEMPLATE
#undef SIZED_FUNCTION_DEF

extern void test_ub2gm_aclshmem_putmem(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_putmem_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_putmem_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmem_putmem_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_putmem_nbi_warp(
    aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_putmem_nbi_block(
    aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmem_getmem(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_getmem_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_getmem_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmem_getmem_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_getmem_nbi_warp(
    aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_ub2gm_aclshmemx_getmem_nbi_block(
    aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);

using FunctionInterface = void(aclrtStream, void*, void*, size_t, int32_t);

enum class TypeName { HALF, FLOAT, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, UCHAR, SCHAR, BFLOAT16 };

inline std::ostream& operator<<(std::ostream& os, const TypeName& type)
{
    switch (type) {
        case TypeName::HALF:
            os << "HALF";
            break;
        case TypeName::FLOAT:
            os << "FLOAT";
            break;
        case TypeName::INT8:
            os << "INT8";
            break;
        case TypeName::INT16:
            os << "INT16";
            break;
        case TypeName::INT32:
            os << "INT32";
            break;
        case TypeName::INT64:
            os << "INT64";
            break;
        case TypeName::UINT8:
            os << "UINT8";
            break;
        case TypeName::UINT16:
            os << "UINT16";
            break;
        case TypeName::UINT32:
            os << "UINT32";
            break;
        case TypeName::UINT64:
            os << "UINT64";
            break;
        case TypeName::UCHAR:
            os << "UCHAR";
            break;
        case TypeName::SCHAR:
            os << "SCHAR";
            break;
        case TypeName::BFLOAT16:
            os << "BFLOAT16";
            break;
    }
    return os;
}

enum class Bits { BITS8, BITS16, BITS32, BITS64, BITS128 };

inline std::ostream& operator<<(std::ostream& os, const Bits& bits)
{
    switch (bits) {
        case Bits::BITS8:
            os << "BITS8";
            break;
        case Bits::BITS16:
            os << "BITS16";
            break;
        case Bits::BITS32:
            os << "BITS32";
            break;
        case Bits::BITS64:
            os << "BITS64";
            break;
        case Bits::BITS128:
            os << "BITS128";
            break;
    }
    return os;
}

enum class OpType { PUT, GET };

inline std::ostream& operator<<(std::ostream& os, const OpType& op)
{
    switch (op) {
        case OpType::PUT:
            os << "PUT";
            break;
        case OpType::GET:
            os << "GET";
            break;
    }
    return os;
}

enum class Scope { THREAD, WARP, BLOCK };

inline std::ostream& operator<<(std::ostream& os, const Scope& scope)
{
    switch (scope) {
        case Scope::THREAD:
            os << "THREAD";
            break;
        case Scope::WARP:
            os << "WARP";
            break;
        case Scope::BLOCK:
            os << "BLOCK";
            break;
    }
    return os;
}

enum class IOMode { BLOCKING, NON_BLOCKING };

inline std::ostream& operator<<(std::ostream& os, const IOMode& mode)
{
    switch (mode) {
        case IOMode::BLOCKING:
            os << "BLOCKING";
            break;
        case IOMode::NON_BLOCKING:
            os << "NON_BLOCKING";
            break;
    }
    return os;
}

enum class SigType { Typed, Sized, Memory };

inline std::ostream& operator<<(std::ostream& os, const SigType& sig)
{
    switch (sig) {
        case SigType::Typed:
            os << "Typed";
            break;
        case SigType::Sized:
            os << "Sized";
            break;
        case SigType::Memory:
            os << "Memory";
            break;
    }
    return os;
}

template <SigType sig, typename... Ts>
class Sig {};

template <>
class Sig<SigType::Typed, OpType, TypeName, IOMode, Scope> {
public:
    OpType op;
    TypeName name;
    IOMode io_mode;
    Scope scope;

    Sig(OpType op, TypeName name, IOMode io_mode, Scope scope) : op(op), name(name), io_mode(io_mode), scope(scope) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Typed, OpType, TypeName, IOMode, Scope>& sig)
{
    os << "Sig<Typed>[" << sig.op << ", " << sig.name << ", " << sig.io_mode << ", " << sig.scope << "]";
    return os;
}

template <>
class Sig<SigType::Sized, OpType, Bits, IOMode, Scope> {
public:
    OpType op;
    Bits bits;
    IOMode io_mode;
    Scope scope;

    Sig(OpType op, Bits bits, IOMode io_mode, Scope scope) : op(op), bits(bits), io_mode(io_mode), scope(scope) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Sized, OpType, Bits, IOMode, Scope>& sig)
{
    os << "Sig<Sized>[" << sig.op << ", " << sig.bits << ", " << sig.io_mode << ", " << sig.scope << "]";
    return os;
}

template <>
class Sig<SigType::Memory, OpType, IOMode, Scope> {
public:
    OpType op;
    IOMode io_mode;
    Scope scope;

    Sig(OpType op, IOMode io_mode, Scope scope) : op(op), io_mode(io_mode), scope(scope) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Memory, OpType, IOMode, Scope>& sig)
{
    os << "Sig<Memory>[" << sig.op << ", " << sig.io_mode << ", " << sig.scope << "]";
    return os;
}

using TypedSig = Sig<SigType::Typed, OpType, TypeName, IOMode, Scope>;
using SizedSig = Sig<SigType::Sized, OpType, Bits, IOMode, Scope>;
using MemorySig = Sig<SigType::Memory, OpType, IOMode, Scope>;
using Sigs = std::variant<TypedSig, SizedSig, MemorySig>;

constexpr std::pair<TypeName, size_t> TypeSizeMap[] = {
    {TypeName::HALF, sizeof(int16_t)},        {TypeName::FLOAT, sizeof(float)},
    {TypeName::INT8, sizeof(int8_t)},         {TypeName::INT16, sizeof(int16_t)},
    {TypeName::INT32, sizeof(int32_t)},       {TypeName::INT64, sizeof(int64_t)},
    {TypeName::UINT8, sizeof(uint8_t)},       {TypeName::UINT16, sizeof(uint16_t)},
    {TypeName::UINT32, sizeof(uint32_t)},     {TypeName::UINT64, sizeof(uint64_t)},
    {TypeName::UCHAR, sizeof(unsigned char)}, {TypeName::SCHAR, sizeof(signed char)},
    {TypeName::BFLOAT16, sizeof(int16_t)}};

inline size_t get_type_size(TypeName type)
{
    for (const auto& pair : TypeSizeMap) {
        if (pair.first == type) {
            return pair.second;
        }
    }
    return 0;
}

constexpr std::pair<Bits, size_t> BitsSizeMap[] = {
    {Bits::BITS8, 1}, {Bits::BITS16, 2}, {Bits::BITS32, 4}, {Bits::BITS64, 8}, {Bits::BITS128, 16}};

inline size_t get_bits_size(Bits bits)
{
    for (const auto& pair : BitsSizeMap) {
        if (pair.first == bits) {
            return pair.second;
        }
    }
    return 0;
}

using TestCaseEntry = std::tuple<FunctionInterface*, Sigs>;

#define TYPED_ENTRIES(TYPE, TYPE_ENUM)                                                        \
    std::make_tuple(                                                                          \
        test_ub2gm_aclshmem_##TYPE##_put,                                                     \
        TypedSig(OpType::PUT, TypeName::TYPE_ENUM, IOMode::BLOCKING, Scope::THREAD)),         \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_put_warp,                                           \
            TypedSig(OpType::PUT, TypeName::TYPE_ENUM, IOMode::BLOCKING, Scope::WARP)),       \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_put_block,                                          \
            TypedSig(OpType::PUT, TypeName::TYPE_ENUM, IOMode::BLOCKING, Scope::BLOCK)),      \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmem_##TYPE##_put_nbi,                                             \
            TypedSig(OpType::PUT, TypeName::TYPE_ENUM, IOMode::NON_BLOCKING, Scope::THREAD)), \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_put_nbi_warp,                                       \
            TypedSig(OpType::PUT, TypeName::TYPE_ENUM, IOMode::NON_BLOCKING, Scope::WARP)),   \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_put_nbi_block,                                      \
            TypedSig(OpType::PUT, TypeName::TYPE_ENUM, IOMode::NON_BLOCKING, Scope::BLOCK)),  \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmem_##TYPE##_get,                                                 \
            TypedSig(OpType::GET, TypeName::TYPE_ENUM, IOMode::BLOCKING, Scope::THREAD)),     \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_get_warp,                                           \
            TypedSig(OpType::GET, TypeName::TYPE_ENUM, IOMode::BLOCKING, Scope::WARP)),       \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_get_block,                                          \
            TypedSig(OpType::GET, TypeName::TYPE_ENUM, IOMode::BLOCKING, Scope::BLOCK)),      \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmem_##TYPE##_get_nbi,                                             \
            TypedSig(OpType::GET, TypeName::TYPE_ENUM, IOMode::NON_BLOCKING, Scope::THREAD)), \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_get_nbi_warp,                                       \
            TypedSig(OpType::GET, TypeName::TYPE_ENUM, IOMode::NON_BLOCKING, Scope::WARP)),   \
        std::make_tuple(                                                                      \
            test_ub2gm_aclshmemx_##TYPE##_get_nbi_block,                                      \
            TypedSig(OpType::GET, TypeName::TYPE_ENUM, IOMode::NON_BLOCKING, Scope::BLOCK)),

#define SIZED_ENTRIES(SIZE, BITS_ENUM)                                                                               \
    std::make_tuple(                                                                                                 \
        test_ub2gm_aclshmem_put##SIZE, SizedSig(OpType::PUT, Bits::BITS_ENUM, IOMode::BLOCKING, Scope::THREAD)),     \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_put##SIZE##_warp,                                                                   \
            SizedSig(OpType::PUT, Bits::BITS_ENUM, IOMode::BLOCKING, Scope::WARP)),                                  \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_put##SIZE##_block,                                                                  \
            SizedSig(OpType::PUT, Bits::BITS_ENUM, IOMode::BLOCKING, Scope::BLOCK)),                                 \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmem_put##SIZE##_nbi,                                                                     \
            SizedSig(OpType::PUT, Bits::BITS_ENUM, IOMode::NON_BLOCKING, Scope::THREAD)),                            \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_put##SIZE##_nbi_warp,                                                               \
            SizedSig(OpType::PUT, Bits::BITS_ENUM, IOMode::NON_BLOCKING, Scope::WARP)),                              \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_put##SIZE##_nbi_block,                                                              \
            SizedSig(OpType::PUT, Bits::BITS_ENUM, IOMode::NON_BLOCKING, Scope::BLOCK)),                             \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmem_get##SIZE, SizedSig(OpType::GET, Bits::BITS_ENUM, IOMode::BLOCKING, Scope::THREAD)), \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_get##SIZE##_warp,                                                                   \
            SizedSig(OpType::GET, Bits::BITS_ENUM, IOMode::BLOCKING, Scope::WARP)),                                  \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_get##SIZE##_block,                                                                  \
            SizedSig(OpType::GET, Bits::BITS_ENUM, IOMode::BLOCKING, Scope::BLOCK)),                                 \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmem_get##SIZE##_nbi,                                                                     \
            SizedSig(OpType::GET, Bits::BITS_ENUM, IOMode::NON_BLOCKING, Scope::THREAD)),                            \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_get##SIZE##_nbi_warp,                                                               \
            SizedSig(OpType::GET, Bits::BITS_ENUM, IOMode::NON_BLOCKING, Scope::WARP)),                              \
        std::make_tuple(                                                                                             \
            test_ub2gm_aclshmemx_get##SIZE##_nbi_block,                                                              \
            SizedSig(OpType::GET, Bits::BITS_ENUM, IOMode::NON_BLOCKING, Scope::BLOCK)),

#define MEMORY_ENTRIES()                                                                                              \
    std::make_tuple(test_ub2gm_aclshmem_putmem, MemorySig(OpType::PUT, IOMode::BLOCKING, Scope::THREAD)),             \
        std::make_tuple(test_ub2gm_aclshmemx_putmem_warp, MemorySig(OpType::PUT, IOMode::BLOCKING, Scope::WARP)),     \
        std::make_tuple(test_ub2gm_aclshmemx_putmem_block, MemorySig(OpType::PUT, IOMode::BLOCKING, Scope::BLOCK)),   \
        std::make_tuple(test_ub2gm_aclshmem_putmem_nbi, MemorySig(OpType::PUT, IOMode::NON_BLOCKING, Scope::THREAD)), \
        std::make_tuple(                                                                                              \
            test_ub2gm_aclshmemx_putmem_nbi_warp, MemorySig(OpType::PUT, IOMode::NON_BLOCKING, Scope::WARP)),         \
        std::make_tuple(                                                                                              \
            test_ub2gm_aclshmemx_putmem_nbi_block, MemorySig(OpType::PUT, IOMode::NON_BLOCKING, Scope::BLOCK)),       \
        std::make_tuple(test_ub2gm_aclshmem_getmem, MemorySig(OpType::GET, IOMode::BLOCKING, Scope::THREAD)),         \
        std::make_tuple(test_ub2gm_aclshmemx_getmem_warp, MemorySig(OpType::GET, IOMode::BLOCKING, Scope::WARP)),     \
        std::make_tuple(test_ub2gm_aclshmemx_getmem_block, MemorySig(OpType::GET, IOMode::BLOCKING, Scope::BLOCK)),   \
        std::make_tuple(test_ub2gm_aclshmem_getmem_nbi, MemorySig(OpType::GET, IOMode::NON_BLOCKING, Scope::THREAD)), \
        std::make_tuple(                                                                                              \
            test_ub2gm_aclshmemx_getmem_nbi_warp, MemorySig(OpType::GET, IOMode::NON_BLOCKING, Scope::WARP)),         \
        std::make_tuple(                                                                                              \
            test_ub2gm_aclshmemx_getmem_nbi_block, MemorySig(OpType::GET, IOMode::NON_BLOCKING, Scope::BLOCK)),

static std::vector<TestCaseEntry> Entries = {
    TYPED_ENTRIES(half, HALF) TYPED_ENTRIES(float, FLOAT) TYPED_ENTRIES(int8, INT8) TYPED_ENTRIES(int16, INT16)
        TYPED_ENTRIES(int32, INT32) TYPED_ENTRIES(int64, INT64) TYPED_ENTRIES(uint8, UINT8)
            TYPED_ENTRIES(uint16, UINT16) TYPED_ENTRIES(uint32, UINT32) TYPED_ENTRIES(uint64, UINT64)
                TYPED_ENTRIES(char, UCHAR) TYPED_ENTRIES(schar, SCHAR) TYPED_ENTRIES(bfloat16, BFLOAT16)
                    SIZED_ENTRIES(8, BITS8) SIZED_ENTRIES(16, BITS16) SIZED_ENTRIES(32, BITS32)
                        SIZED_ENTRIES(64, BITS64) SIZED_ENTRIES(128, BITS128) MEMORY_ENTRIES()};

#undef MEMORY_ENTRIES
#undef SIZED_ENTRIES
#undef TYPED_ENTRIES

#endif // defined(USE_SIMT)

#endif // SIMT_RMA_UB2GM_DEFINITIONS_H
