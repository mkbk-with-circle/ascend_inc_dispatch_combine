/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCS_SIO_LINK_PERF_UTILS_H
#define HCCS_SIO_LINK_PERF_UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "shmem.h"
#include "host_device/shmem_common_types.h"

// NOTE: cycle2us mapping must stay in sync with src/host/utils/prof/prof_util.cpp:prof_data_print
inline int64_t get_cycle2us()
{
    const char *soc_name = aclrtGetSocName();
    int64_t cycle2us = 50;
    if (soc_name != nullptr) {
        std::string soc_str(soc_name);
        if (soc_str.find("Ascend950") != std::string::npos) {
            cycle2us = 1000;
        }
    }
    return cycle2us;
}

static void print_perf_result(aclshmem_prof_pe_t *out_profs, int32_t frame_id,
                                uint32_t block_dim, size_t transfer_bytes, size_t per_block_bytes,
                                const char *tag, int64_t cycle2us,
                                std::vector<std::vector<std::string>> &csv_data)
{
    double max_core_time_us = 0.0;

    double sum_core_time_us = 0.0;
    int valid_block_count = 0;
    std::vector<double> core_times;
    std::cout << "============================================================" << std::endl;
    std::cout << "[" << tag << "] Perf Result: B/PerBlock=" << per_block_bytes
              << ", DataSize/B=" << transfer_bytes
              << ", blocks=" << block_dim << std::endl;
    std::cout << "BlockID   FrameID   B/PerBlock     Cycles         Count          AvgTime(us)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (uint32_t b = 0; b < block_dim; b++) {
        uint64_t total_cycles = 0;
        uint64_t total_count = 0;
        total_cycles = out_profs->block_prof[b].cycles[frame_id];
        total_count = out_profs->block_prof[b].ccount[frame_id];
        double avg_time_us = (total_count > 0)
            ? static_cast<double>(total_cycles) / static_cast<double>(total_count) / cycle2us
            : 0.0;
        std::cout << std::left << std::setw(10) << b
                  << std::setw(10) << frame_id
                  << std::setw(15) << per_block_bytes
                  << std::setw(15) << total_cycles
                  << std::setw(15) << total_count
                  << std::fixed << std::setprecision(3) << avg_time_us << std::endl;
        if (avg_time_us > max_core_time_us) {
            max_core_time_us = avg_time_us;
        }
        if (avg_time_us > 0.0) {
            sum_core_time_us += avg_time_us;
            valid_block_count++;
        }
        core_times.push_back(avg_time_us);
    }
    std::cout << "------------------------------------------------------------" << std::endl;

    double avg_core_time_us = (valid_block_count > 0) ? sum_core_time_us / valid_block_count : 0.0;

    double bandwidth = 0.0;
    if (max_core_time_us > 0.0) {
        bandwidth = static_cast<double>(transfer_bytes) / max_core_time_us * 1e6 / (1024.0 * 1024.0 * 1024.0);
    }
    std::cout << "[" << tag << "] MaxCoreTime=" << std::fixed << std::setprecision(3)
              << max_core_time_us << " us" << std::endl;
    std::cout << "[" << tag << "] AvgCoreTime=" << std::fixed << std::setprecision(3)
              << avg_core_time_us << " us" << std::endl;
    std::cout << "[" << tag << "] Bandwidth=" << std::fixed << std::setprecision(4)
              << bandwidth << " GB/s" << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<std::string> row;
    row.push_back(std::to_string(per_block_bytes));
    row.push_back(std::to_string(transfer_bytes));
    row.push_back(std::to_string(block_dim));
    row.push_back(std::to_string(HCCS_SIO_UB_SIZE_KB));
    row.push_back(std::to_string(bandwidth));
    row.push_back(std::to_string(max_core_time_us));
    row.push_back(std::to_string(avg_core_time_us));
    for (size_t i = 0; i < core_times.size(); i++) {
        row.push_back(std::to_string(core_times[i]));
    }
    csv_data.push_back(row);
}

static void write_perf_csv(const std::string &filename, const std::vector<std::vector<std::string>> &data)
{
    std::string dir;
    size_t pos = filename.find_last_of("/");
    if (pos != std::string::npos) {
        dir = filename.substr(0, pos);
    }
    if (!dir.empty()) {
        std::error_code ec;
        if (!std::filesystem::create_directories(dir, ec) && ec) {
            std::cerr << "Failed to create directory: " << dir << ", error: " << ec.message() << std::endl;
            return;
        }
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); i++) {
            ofs << row[i];
            if (i + 1 < row.size()) ofs << ",";
        }
        ofs << "\n";
    }
    ofs.close();
    std::cout << "Performance CSV written to: " << filename << std::endl;
}

#endif
