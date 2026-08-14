/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_RMA_PERFTEST_ARGPARSER
#define SIMT_RMA_PERFTEST_ARGPARSER

#include <iostream>
#include <string>
#include <optional>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

enum class OpType { Get, Put, None };
enum class VfType { Simt, Simd };

inline OpType str2op(const std::string& s) {
    if (s == "get") {
        return OpType::Get;
    } else if (s == "put") {
        return OpType::Put;
    } else if (s == "none") {
        return OpType::None;
    } else {
        throw std::invalid_argument("Invalid operation type");
    }
}

inline std::string to_string(OpType op) {
    switch (op) {
        case OpType::Get:
            return "get";
        case OpType::Put:
            return "put";
        case OpType::None:
            return "none";
        default:
            throw std::invalid_argument("Invalid operation type");
    }
}

inline std::string to_string(VfType vf) {
    switch (vf) {
        case VfType::Simt:
            return "simt";
        case VfType::Simd:
            return "simd";
        default:
            throw std::invalid_argument("Invalid vectorization type");
    }
}

inline int32_t datatype_to_datasize(const std::string& s) {
    if (s == "int8" || s == "uint8" || s == "char") {
        return 8;
    } else if (s == "int16" || s == "uint16") {
        return 16;
    } else if (s == "float" || s == "int32" || s == "uint32") {
        return 32;
    } else if (s == "int64" || s == "uint64") {
        return 64;
    } else {
        throw std::invalid_argument("Invalid data type: " + s);
    }
}

struct Config {
    int npes = 2;            // --pes : number of PEs (must be 2 for this test)
    int mype = -1;           // --pe-id : this process's PE rank (0=ACTIVE, 1=PASSIVE)
    int gnpus = 2;           // --gnpus : number of NPUs on this node
    int first_pe = 0;        // --fpe : first PE id. Kept only for CLI compatibility with the
                             //         other shmem_perftest samples; unused here (rank comes from
                             //         --pe-id, device from mype % gnpus + first_npu).
    int first_npu = 0;       // --fnpu : first NPU id; device = mype % gnpus + first_npu
    std::string ipport = "tcp://127.0.0.1:8760";  // --ipport : bootstrap ip:port
    int block_size_min = 32;  // --block-size / --block-range : kernel grid (cores) sweep lower bound
    int block_size_max = 32;  // --block-range : kernel grid (cores) sweep upper bound
    std::vector<int> block_sizes;  // resolved core counts to test; filled from --block-list, or
                                   // expanded from [block_size_min, block_size_max] when no list given.
    int bytes_in_exp_min = 3;
    int bytes_in_exp_max = 20;
    int loop_count = 1000;
    int ub_size = 16;

    std::optional<OpType> req_op_type;   // --test-type : put/get/none
    std::optional<int32_t> req_datasize; // --datatype  : converted to DATA_SIZE bits

    friend std::ostream& operator<<(std::ostream& os, const Config& c) {
        os << "================ Config ================\n"
           << std::left << std::setw(20) << "npes:" << c.npes << "\n"
           << std::setw(20) << "mype:" << c.mype << "\n"
           << std::setw(20) << "gnpus:" << c.gnpus << "\n"
           << std::setw(20) << "first_pe:" << c.first_pe << "\n"
           << std::setw(20) << "first_npu:" << c.first_npu << "\n"
           << std::setw(20) << "ipport:" << c.ipport << "\n"
           << std::setw(20) << "block_size_min:" << c.block_size_min << "\n"
           << std::setw(20) << "block_size_max:" << c.block_size_max << "\n"
           << std::setw(20) << "bytes-in-exp-min:" << c.bytes_in_exp_min << "\n"
           << std::setw(20) << "bytes-in-exp-max:" << c.bytes_in_exp_max << "\n"
           << std::setw(20) << "loop_count:" << c.loop_count << "\n"
           << std::setw(20) << "ub_size:" << c.ub_size << "\n"
           << "========================================";
        return os;
    }
};

// ---------- 使用说明 ----------
inline void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --pe-id <int> [options]\n"
              << "Options:\n"
              << "  --pes <int>              Number of PEs, must be 2. Default: 2\n"
              << "  --pe-id <int>            This process's PE rank (0=ACTIVE, 1=PASSIVE). Required\n"
              << "  --gnpus <int>            Number of NPUs on this node. Default: 2\n"
              << "  --fpe <int>              First PE id. Kept for CLI compatibility; unused. Default: 0\n"
              << "  --fnpu <int>             First NPU id; device = pe_id % gnpus + fnpu. Default: 0\n"
              << "  --ipport <ip:port>       Bootstrap address. Default: tcp://127.0.0.1:8760\n"
              << "  --test-type <put|get|none>  Must match the compile-time OP_TYPE, else error.\n"
              << "  --datatype <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>\n"
              << "                           Must match the compile-time DATA_SIZE (in bits), else error.\n"
              << "  --block-size <int>       Kernel grid (cores). Default: 32\n"
              << "  --block-size-min <int>   Cores sweep lower bound. Default: 32\n"
              << "  --block-size-max <int>   Cores sweep upper bound. Default: 32\n"
              << "  --block-list <b1,b2,...> Explicit core counts to test, comma-separated\n"
              << "                           (e.g. 2,4,8). Overrides --block-size/--block-range.\n"
              << "  --bytes-in-exp-min <int> Default: 3\n"
              << "  --bytes-in-exp-max <int> Default: 20\n"
              << "  --loop-count <int>       Default: 1000\n"
              << "  --ub-size <int>          Default: 16\n"
              << "  --help                   Show this message\n";
}

// ---------- 参数校验 ----------
// 各项约束均来源于 main.cpp 中对 Config 字段的实际使用方式：
//  - pes  : 本测试固定两卡 Active/Passive 模型，必须为 2。
//  - pe-id: 作为 PE rank，且固定两卡模型只区分 ACTIVE_PE(0) / PASSIVE_PE(1)，
//            因此只能取 0 或 1。实际设备号由 pe_id % gnpus + fnpu 推导。
//  - gnpus / fnpu : 须为非负，gnpus 须为正。
//  - block_size_* : kernel 的 grid 大小（核数）扫描区间，与家族 -b/--block-range
//            对齐；二者须为正且 min <= max。上限（per_core_bytes * block 不超过
//            对称堆容量）由调用方自行保证，这里不再校验。
//  - bytes-in-exp-* : per_invocation_bytes = 1 << exp，指数需落在
//            [3, 20]，且 min <= max。上界 20 对应
//            per_invocation_bytes = 1MB = per_core_bytes（每核物理缓冲）。
//  - loop-count 在 [1, 10000)（loops 是求平均时的除数）。
//  - ub_size 为 UB 大小(KB)，须为正。
inline bool validate_config(const Config& c) {
    if (c.npes != 2) {
        std::cerr << "Error: --pes must be 2 (this test uses a fixed 2-card "
                  << "Active/Passive model), got " << c.npes << ".\n";
        return false;
    }
    if (c.mype != 0 && c.mype != 1) {
        std::cerr << "Error: --pe-id must be 0 (ACTIVE_PE) or 1 (PASSIVE_PE), got "
                  << c.mype << ".\n";
        return false;
    }
    if (c.gnpus < 1) {
        std::cerr << "Error: --gnpus must be >= 1, got " << c.gnpus << ".\n";
        return false;
    }
    // first_pe is not consumed by the test (see Config); still range-check it so a
    // bogus value passed for CLI compatibility fails loudly rather than silently.
    if (c.first_pe < 0) {
        std::cerr << "Error: --fpe must be >= 0, got " << c.first_pe << ".\n";
        return false;
    }
    if (c.first_npu < 0) {
        std::cerr << "Error: --fnpu must be >= 0, got " << c.first_npu << ".\n";
        return false;
    }
    if (c.block_size_min < 1) {
        std::cerr << "Error: --block-size-min must be >= 1, got " << c.block_size_min << ".\n";
        return false;
    }
    if (c.block_size_max < 1) {
        std::cerr << "Error: --block-size-max must be >= 1, got " << c.block_size_max << ".\n";
        return false;
    }
    if (c.block_size_min > c.block_size_max) {
        std::cerr << "Error: --block-size-min (" << c.block_size_min
                  << ") must not exceed --block-size-max (" << c.block_size_max << ").\n";
        return false;
    }
    if (c.bytes_in_exp_min < 3 || c.bytes_in_exp_min > 20) {
        std::cerr << "Error: --bytes-in-exp-min must be in [3, 20], got "
                  << c.bytes_in_exp_min << ".\n";
        return false;
    }
    if (c.bytes_in_exp_max < 3 || c.bytes_in_exp_max > 20) {
        std::cerr << "Error: --bytes-in-exp-max must be in [3, 20], got "
                  << c.bytes_in_exp_max << ".\n";
        return false;
    }
    if (c.bytes_in_exp_min > c.bytes_in_exp_max) {
        std::cerr << "Error: --bytes-in-exp-min (" << c.bytes_in_exp_min
                  << ") must not exceed --bytes-in-exp-max (" << c.bytes_in_exp_max << ").\n";
        return false;
    }
    if (c.loop_count < 1 || c.loop_count >= 10000) {
        std::cerr << "Error: --loop-count must be in [1, 10000), got " << c.loop_count << ".\n";
        return false;
    }
    if (c.ub_size < 1) {
        std::cerr << "Error: --ub-size must be >= 1 (KB), got " << c.ub_size << ".\n";
        return false;
    }
    return true;
}

// ---------- 参数解析 ----------
std::optional<Config> parse_args(int argc, char* argv[]) {
    Config conf;
    bool pe_id_provided = false;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return std::nullopt;
            }

            if (i + 1 < argc) {
                i += 1;
                std::string val = argv[i];

                if (arg == "--pes") {
                    conf.npes = std::stoi(val);
                } else if (arg == "--pe-id") {
                    conf.mype = std::stoi(val);
                    pe_id_provided = true;
                } else if (arg == "--gnpus") {
                    conf.gnpus = std::stoi(val);
                } else if (arg == "--fpe") {
                    conf.first_pe = std::stoi(val);   // accepted for compatibility; unused
                } else if (arg == "--fnpu") {
                    conf.first_npu = std::stoi(val);
                } else if (arg == "--ipport") {
                    conf.ipport = val;
                } else if (arg == "--test-type") {
                    conf.req_op_type = str2op(val);
                } else if (arg == "--datatype") {
                    conf.req_datasize = datatype_to_datasize(val);
                } else if (arg == "--block-size") {
                    conf.block_size_min = std::stoi(val);
                    conf.block_size_max = conf.block_size_min;
                } else if (arg == "--block-size-min") {
                    conf.block_size_min = std::stoi(val);
                } else if (arg == "--block-size-max") {
                    conf.block_size_max = std::stoi(val);
                } else if (arg == "--block-list") {
                    conf.block_sizes.clear();
                    size_t start = 0;
                    while (start < val.size()) {
                        size_t comma = val.find(',', start);
                        std::string tok = val.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                        if (!tok.empty()) {
                            conf.block_sizes.push_back(std::stoi(tok));
                        }
                        if (comma == std::string::npos) {
                            break;
                        }
                        start = comma + 1;
                    }
                } else if (arg == "--bytes-in-exp-min") {
                    conf.bytes_in_exp_min = std::stoi(val);
                } else if (arg == "--bytes-in-exp-max") {
                    conf.bytes_in_exp_max = std::stoi(val);
                } else if (arg == "--loop-count") {
                    conf.loop_count = std::stoi(val);
                } else if (arg == "--ub-size") {
                    conf.ub_size = std::stoi(val);
                } else {
                    std::cerr << "Unknown argument: " << arg << "\n";
                    return std::nullopt;
                }
            } else {
                std::cerr << "Argument " << arg << " missing value.\n";
                return std::nullopt;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return std::nullopt;
    }

    if (!pe_id_provided) {
        std::cerr << "Error: --pe-id is required.\n";
        return std::nullopt;
    }

    // Resolve the set of core counts to test. --block-list (if given) wins over
    // --block-size/--block-range, mirroring the other shmem_perftest samples.
    // block_size_min/max are kept consistent with the resolved set because they
    // drive the symmetric-memory allocation upper bound and the CSV file name.
    if (!conf.block_sizes.empty()) {
        int lo = conf.block_sizes.front();
        int hi = conf.block_sizes.front();
        for (int b : conf.block_sizes) {
            if (b < 1) {
                std::cerr << "Error: --block-list values must be >= 1, got " << b << ".\n";
                return std::nullopt;
            }
            lo = std::min(lo, b);
            hi = std::max(hi, b);
        }
        conf.block_size_min = lo;
        conf.block_size_max = hi;
    }

    if (!validate_config(conf)) {
        return std::nullopt;
    }

    // No explicit list: expand the validated [min, max] range into block_sizes.
    if (conf.block_sizes.empty()) {
        for (int b = conf.block_size_min; b <= conf.block_size_max; ++b) {
            conf.block_sizes.push_back(b);
        }
    }

    return conf;
}


template <int32_t data_size>
struct TraitTypeFromDataSize {  };

// Specializations for specific data sizes
template<>
struct TraitTypeFromDataSize<8> { using type = int8_t; };

template<>
struct TraitTypeFromDataSize<16> { using type = int16_t; };

template<>
struct TraitTypeFromDataSize<32> { using type = int32_t; };

template<>
struct TraitTypeFromDataSize<64> { using type = int64_t; };

#endif
