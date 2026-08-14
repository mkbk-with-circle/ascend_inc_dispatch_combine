/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iomanip>
#include <getopt.h>
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"

int g_npus = 8;
const char *ipport;
static const char *fuc_data_type;
static const char *fuc_test_type;
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

static aclshmem_prof_pe_t *out_profs;

extern "C" void launch_udma_perf_kernel(uint32_t block_dim, void *stream, uint8_t *dst_gva, uint8_t *src_gva,
                                         uint8_t *sig_gva, int elements, int32_t frame_id, int test_mode,
                                         int data_type, int ub_size_kb, int64_t prof_pe_val, int loop_count,
                                         uint64_t signal_base, int metric, int batch);

static perftest::udma_mode_t get_udma_mode(const char *test_type_str)
{
    if (strcmp(test_type_str, "put") == 0) return perftest::TEST_MODE_UDMA_PUT;
    if (strcmp(test_type_str, "bi_put") == 0) return perftest::TEST_MODE_UDMA_BI_PUT;
    if (strcmp(test_type_str, "get") == 0) return perftest::TEST_MODE_UDMA_GET;
    if (strcmp(test_type_str, "bi_get") == 0) return perftest::TEST_MODE_UDMA_BI_GET;
    if (strcmp(test_type_str, "put_signal") == 0) return perftest::TEST_MODE_UDMA_PUT_SIGNAL;
    return perftest::TEST_MODE_UDMA_PUT;
}

static perftest::perf_metric_t get_perf_metric(const char *metric_str)
{
    if (strcmp(metric_str, "lat") == 0) return perftest::PERF_METRIC_LAT;
    return perftest::PERF_METRIC_BW;
}

template <typename T>
int test_udma_perf_test_impl(int pe_id, int n_pes, uint64_t local_mem_size,
                              int min_exponent, int max_exponent, int loop_count,
                              perftest::udma_mode_t test_mode, perftest::perf_data_type_t data_type_enum,
                              int prof_pe, int ub_size_kb, perftest::perf_metric_t metric, int batch,
                              std::vector<std::vector<std::string>> &csv_data)
{
    int32_t device_id = (pe_id % g_npus + f_npu);
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    void *sig_ptr = nullptr;
    if (test_mode == perftest::TEST_MODE_UDMA_PUT_SIGNAL) {
        sig_ptr = aclshmem_malloc(n_pes * sizeof(uint64_t));
        std::vector<uint64_t> sig_init(n_pes, 0);
        aclrtMemcpy(sig_ptr, n_pes * sizeof(uint64_t), sig_init.data(),
                    n_pes * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    constexpr uint32_t block_dim = 1;
    int frame_id = 0;
    for (int exponent = min_exponent; exponent <= max_exponent; exponent++) {
        int datasize = std::pow(2, exponent);
        std::cout << "pe: " << pe_id << " size: " << datasize << " frame_id: " << frame_id << std::endl;

        void *dst_ptr = aclshmem_malloc(datasize);
        void *src_ptr = aclshmem_malloc(datasize);

        int trans_size = datasize / sizeof(T);

        std::vector<T> src_input(trans_size, 0);
        std::vector<T> dst_input(trans_size, 0);
        for (int i = 0; i < trans_size; i++) {
            src_input[i] = (T)(pe_id + 10);
            dst_input[i] = (T)(pe_id + 100);
        }
        status = aclrtMemcpy(src_ptr, datasize, src_input.data(), datasize, ACL_MEMCPY_HOST_TO_DEVICE);
        status = aclrtMemcpy(dst_ptr, datasize, dst_input.data(), datasize, ACL_MEMCPY_HOST_TO_DEVICE);

        uint64_t signal_base = static_cast<uint64_t>(frame_id) * 1000000ULL + 1;
        launch_udma_perf_kernel(block_dim, stream, (uint8_t *)dst_ptr, (uint8_t *)src_ptr,
                                 (uint8_t *)sig_ptr, trans_size, frame_id, static_cast<int>(test_mode),
                                 static_cast<int>(data_type_enum), ub_size_kb, prof_pe, loop_count, signal_base,
                                 static_cast<int>(metric), batch);
        status = aclrtSynchronizeStream(stream);

        bool verify_success = true;
        auto compare_values = [&](T *p1, T *p2, size_t count, const char *l1, const char *l2) -> bool {
            for (size_t i = 0; i < count; i++) {
                if (p1[i] != p2[i]) {
                    std::cout << "  [ERROR] Mismatch at index " << i << ": " << l1 << "=" << (double)p1[i]
                              << ", " << l2 << "=" << (double)p2[i] << std::endl;
                    return false;
                }
            }
            return true;
        };

        std::vector<T> dst_host(trans_size, 0);
        std::vector<T> src_host(trans_size, 0);
        status = aclrtMemcpy(dst_host.data(), datasize, dst_ptr, datasize, ACL_MEMCPY_DEVICE_TO_HOST);
        status = aclrtMemcpy(src_host.data(), datasize, src_ptr, datasize, ACL_MEMCPY_DEVICE_TO_HOST);

        int peer_pe = (pe_id + 1) % n_pes;
        if (test_mode == perftest::TEST_MODE_UDMA_PUT) {
            std::cout << "\n[Verification] put: checking..." << std::endl;
            if (pe_id != prof_pe) {
                T expected_val = static_cast<T>(prof_pe + 10);
                if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                    verify_success = false;
                }
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_GET) {
            std::cout << "\n[Verification] get: checking..." << std::endl;
            if (pe_id == prof_pe) {
                T expected_val = static_cast<T>(peer_pe + 10);
                if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                    verify_success = false;
                }
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_BI_PUT) {
            std::cout << "\n[Verification] bi_put: checking..." << std::endl;
            T expected_val = static_cast<T>(peer_pe + 10);
            if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                verify_success = false;
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_BI_GET) {
            std::cout << "\n[Verification] bi_get: checking..." << std::endl;
            T expected_val = static_cast<T>(peer_pe + 10);
            if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                verify_success = false;
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_PUT_SIGNAL) {
            std::cout << "\n[Verification] put_signal: checking data + signal..." << std::endl;
            if (pe_id != prof_pe) {
                T expected_val = static_cast<T>(prof_pe + 10);
                if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                    verify_success = false;
                }
                std::vector<uint64_t> sig_init(n_pes, 0);
                aclrtMemcpy(sig_init.data(), n_pes * sizeof(uint64_t), sig_ptr,
                            n_pes * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
                uint64_t expected_sig =
                    signal_base + static_cast<uint64_t>(perftest::PERFTEST_WARMUP_ITERS + loop_count - 1);
                if (sig_init[prof_pe] != expected_sig) {
                    std::cout << "  [ERROR] signal slot[" << prof_pe << "]=" << sig_init[prof_pe]
                              << ", expected=" << expected_sig << std::endl;
                    verify_success = false;
                }
            }
        }

        if (verify_success) {
            std::cout << "[Verification] SUCCESS" << std::endl;
        } else {
            std::cout << "[Verification] FAILED" << std::endl;
        }

        aclshmemx_get_prof(&out_profs, false);
        collect_prof_data_to_csv_v2(out_profs, frame_id, datasize, 1, g_npus, ub_size_kb, loop_count,
                                    metric == perftest::PERF_METRIC_BW, csv_data);

        aclshmem_free(dst_ptr);
        aclshmem_free(src_ptr);
        status = aclrtSynchronizeStream(stream);

        frame_id++;
        if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
            std::cerr << "warning: frame_id reached limit " << ACLSHMEM_CYCLE_PROF_FRAME_CNT << std::endl;
            break;
        }
    }
    aclshmemx_get_prof(nullptr, true);

    if (sig_ptr != nullptr) {
        aclshmem_free(sig_ptr);
    }
    status = aclshmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int n_pes = 2;
    int pe_id = 0;
    ipport = "tcp://127.0.0.1:8768";
    g_npus = 2;
    f_pe = 0;
    f_npu = 4;
    const char *test_type = "put";
    fuc_data_type = "float";
    int input_block_size = 1;
    int min_exponent = 3;
    int max_exponent = 17;
    int loop_count = 1000;
    int ub_size_kb = 16;
    const char *metric_str = "bw";
    int batch = 0;  // 0 / unset => default async (== loop_count); 1 => sync; N => quiet every N submits

    static struct option long_options[] = {
        {"pes", required_argument, 0, 0},
        {"pe-id", required_argument, 0, 0},
        {"ipport", required_argument, 0, 0},
        {"gnpus", required_argument, 0, 0},
        {"fpe", required_argument, 0, 0},
        {"fnpu", required_argument, 0, 0},
        {"test-type", required_argument, 0, 't'},
        {"datatype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'},
        {"block-range", required_argument, 0, 0},
        {"exponent", required_argument, 0, 'e'},
        {"exponent-range", required_argument, 0, 0},
        {"loop-count", required_argument, 0, 0},
        {"ub-size", required_argument, 0, 0},
        {"metric", required_argument, 0, 0},
        {"batch", required_argument, 0, 0},
        {0, 0, 0, 0}};

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:b:e:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't': test_type = optarg; break;
            case 'd': fuc_data_type = optarg; break;
            case 'b': input_block_size = std::atoi(optarg); break;
            case 'e': min_exponent = max_exponent = std::atoi(optarg); break;
            case 0:
                if (strcmp(long_options[option_index].name, "pes") == 0) n_pes = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "pe-id") == 0) pe_id = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "ipport") == 0) ipport = optarg;
                else if (strcmp(long_options[option_index].name, "gnpus") == 0) g_npus = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "fpe") == 0) f_pe = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "fnpu") == 0) f_npu = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "block-range") == 0) {
                    input_block_size = std::atoi(optarg);
                    if (optind < argc) {
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "exponent-range") == 0) {
                    min_exponent = std::atoi(optarg);
                    if (optind < argc) {
                        max_exponent = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "loop-count") == 0) loop_count = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "ub-size") == 0) ub_size_kb = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "metric") == 0) metric_str = optarg;
                else if (strcmp(long_options[option_index].name, "batch") == 0) batch = std::atoi(optarg);
                break;
            default:
                std::cerr << "Unknown argument" << std::endl;
                return 1;
        }
    }

    if (input_block_size != 1) {
        std::cerr << "WARN: UDMA perftest forces block_size=1 (got " << input_block_size << ")" << std::endl;
    }

    if (strcmp(metric_str, "bw") != 0 && strcmp(metric_str, "lat") != 0) {
        std::cerr << "Error: --metric must be 'bw' or 'lat' (got '" << metric_str << "')" << std::endl;
        return 1;
    }
    perftest::perf_metric_t metric = get_perf_metric(metric_str);
    if (metric == perftest::PERF_METRIC_LAT && strcmp(test_type, "put") != 0) {
        std::cerr << "Error: --metric lat is only supported with -t put (got '" << test_type << "')" << std::endl;
        return 1;
    }

    if (batch < 0) {
        std::cerr << "Error: --batch must be >= 0 (got " << batch << ")" << std::endl;
        return 1;
    }
    if (loop_count < 1) {
        std::cerr << "Error: --loop-count must be >= 1 (got " << loop_count << ")" << std::endl;
        return 1;
    }
    if (batch == 0 || batch > loop_count) {
        batch = loop_count;  // default: full async, single trailing quiet
    }

    std::cout << "[INFO] udma_perftest start, pe=" << pe_id << ", t=" << test_type
              << ", d=" << fuc_data_type << ", exp=" << min_exponent << "-" << max_exponent
              << ", loop=" << loop_count << ", ub=" << ub_size_kb << "KB"
              << ", metric=" << metric_str << ", batch=" << batch << std::endl;

    fuc_test_type = test_type;
    perftest::udma_mode_t test_mode = get_udma_mode(test_type);
    perftest::perf_data_type_t data_type_enum = get_data_type(fuc_data_type);

    uint64_t max_datasize = (1ULL << max_exponent);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    const uint64_t ONE_GB = 1024UL * 1024UL * 1024;
    const uint64_t MAX_GB = 40;
    if (max_datasize * 2 > local_mem_size) {
        uint64_t gb_needed = (max_datasize * 2 + ONE_GB - 1) / ONE_GB;
        if (gb_needed > MAX_GB) {
            std::cerr << "Error: required mem exceeds 40GB" << std::endl;
            return 1;
        }
        local_mem_size = gb_needed * ONE_GB;
    }

    const char *prof_pe_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    int prof_pe = (prof_pe_env != nullptr) ? std::atoi(prof_pe_env) : 0;

    std::vector<std::vector<std::string>> csv_data = {
        {"DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s(1000)", "Bandwidth/GiB/s(1024)", "CoreMaxTime/us", "SingleCoreTime/us"}};

    int status = 0;
#define UDMA_TEST_IMPL_OP(type) \
    status = test_udma_perf_test_impl<type>(pe_id, n_pes, local_mem_size, min_exponent, max_exponent, \
                                              loop_count, test_mode, data_type_enum, prof_pe, ub_size_kb, \
                                              metric, batch, csv_data)
    DISPATCH_BY_TYPE(fuc_data_type, UDMA_TEST_IMPL_OP);
#undef UDMA_TEST_IMPL_OP

    if (pe_id == prof_pe) {
        std::string csv_filename = "output/udma_" + std::string(metric_str) + "_" +
                                    std::string(fuc_test_type) + "_" +
                                    std::string(fuc_data_type) + "_" + int_to_string(prof_pe) + ".csv";
        write_csv(csv_filename, csv_data);
    }

    std::cout << "[SUCCESS] udma_perftest done in pe " << pe_id << std::endl;
    return 0;
}
