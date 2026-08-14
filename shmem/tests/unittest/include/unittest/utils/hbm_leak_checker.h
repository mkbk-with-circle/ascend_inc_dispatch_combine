/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HBM_LEAK_CHECKER_H
#define HBM_LEAK_CHECKER_H

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

class HbmLeakChecker {
public:
  static HbmLeakChecker &Instance() {
    static HbmLeakChecker instance;
    return instance;
  }

  void SetCardRange(int first_npu, int gnpu_num) {
    first_npu_ = first_npu;
    gnpu_num_ = gnpu_num;
  }

  bool CheckBefore() {
    std::cout << "==========================================" << std::endl;
    std::cout << "HBM Memory Check - BEFORE UT" << std::endl;
    std::cout << "==========================================" << std::endl;

    bool all_sampled = true;
    before_hbm_.clear();
    for (int card_id = first_npu_; card_id < first_npu_ + gnpu_num_;
         ++card_id) {
      int hbm_used = GetHbmUsage(card_id);
      if (hbm_used >= 0) {
        before_hbm_[card_id] = hbm_used;
        std::cout << "Card " << card_id << ": HBM Used = " << hbm_used << " MB"
                  << std::endl;
      } else {
        all_sampled = false;
        std::cerr << "Error: Failed to get HBM usage for card " << card_id
                  << std::endl;
      }
    }
    if (!all_sampled) {
      std::cout << "WARNING: Failed to sample HBM baseline for one or more "
                << "cards. Leak check may be incomplete." << std::endl;
    }
    std::cout << "==========================================" << std::endl;
    return all_sampled;
  }

  bool CheckAfter() {
    std::cout << "Waiting 30 seconds for cleanup..." << std::endl;
    sleep(30);

    std::cout << "==========================================" << std::endl;
    std::cout << "HBM Memory Check - AFTER UT" << std::endl;
    std::cout << "==========================================" << std::endl;

    bool has_leak = false;
    bool has_check_warning = false;
    const int threshold = 5;

    for (int card_id = first_npu_; card_id < first_npu_ + gnpu_num_;
         ++card_id) {
      int hbm_used = GetHbmUsage(card_id);
      if (hbm_used < 0) {
        has_check_warning = true;
        std::cerr << "Error: Failed to get HBM usage for card " << card_id
                  << std::endl;
        continue;
      }

      auto it = before_hbm_.find(card_id);
      if (it == before_hbm_.end()) {
        has_check_warning = true;
        std::cout << "Card " << card_id << ": HBM Used = " << hbm_used
                  << " MB (no baseline - check skipped)" << std::endl;
        continue;
      }

      int before_hbm = it->second;
      int diff = hbm_used - before_hbm;
      int abs_diff = std::abs(diff);

      std::cout << "Card " << card_id << ":" << std::endl;
      std::cout << "  Before: " << before_hbm << " MB" << std::endl;
      std::cout << "  After:  " << hbm_used << " MB" << std::endl;
      std::cout << "  Diff:   " << diff << " MB" << std::endl;

      if (abs_diff > threshold) {
        if (diff > 0) {
          std::cout << "  Status: ERROR: Memory leak detected! (exceeds "
                    << threshold << " MB threshold)" << std::endl;
          has_leak = true;
        } else {
          std::cout << "  Status: Memory freed (" << (0 - diff)
                    << " MB released)" << std::endl;
        }
      } else {
        std::cout << "  Status: No leak detected (within " << threshold
                  << " MB tolerance)" << std::endl;
      }
      std::cout << std::endl;
    }

    std::cout << "==========================================" << std::endl;
    if (has_check_warning) {
      std::cout << "WARNING: HBM check incomplete - baseline unavailable or "
                << "sampling failed for one or more cards." << std::endl;
      std::cout << std::endl;
      std::cout << "Possible causes:" << std::endl;
      std::cout << "  1. npu-smi tool is not available or failed to run."
                << std::endl;
      std::cout << "     Please verify: npu-smi info" << std::endl;
      std::cout << "  2. CheckBefore() was not called or also failed."
                << std::endl;
      std::cout << std::endl;
    }
    if (has_leak) {
      std::cout << "ERROR: Memory leak detected on one or more cards!"
                << std::endl;
      std::cout << std::endl;
      std::cout << "Possible causes:" << std::endl;
      std::cout << "  1. Other programs are using the NPU cards concurrently."
                << std::endl;
      std::cout << "     Please check if other processes are using the cards."
                << std::endl;
      std::cout << "  2. Memory leak in the code under test." << std::endl;
      std::cout << "     Please review the code for potential memory leaks."
                << std::endl;
    } else {
      std::cout << "Overall: No memory leak detected" << std::endl;
    }
    std::cout << "==========================================" << std::endl;

    return !has_leak;
  }

private:
  HbmLeakChecker() : first_npu_(0), gnpu_num_(1) {}
  HbmLeakChecker(const HbmLeakChecker &) = delete;
  HbmLeakChecker &operator=(const HbmLeakChecker &) = delete;

  int GetHbmUsage(int card_id) {
    std::string cmd = "npu-smi info 2>/dev/null";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
      return -1;
    }

    std::array<char, 512> buffer;
    std::vector<std::string> lines;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
      lines.push_back(std::string(buffer.data()));
    }
    pclose(pipe);

    int chip_index = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
      const std::string &line = lines[i];
      if (line.empty() || line[0] != '|') {
        continue;
      }

      if (!IsHbmDataLine(line)) {
        continue;
      }

      if (chip_index == card_id) {
        int hbm_mb = ParseHbmFromLine(line);
        if (hbm_mb >= 0) {
          return hbm_mb;
        }
        return -1;
      }

      ++chip_index;
    }

    return -1;
  }

  int ParseNpuIdFromLine(const std::string &line) {
    if (line.empty() || line[0] != '|') {
      return -1;
    }

    size_t first_pipe = line.find('|', 1);
    if (first_pipe == std::string::npos) {
      return -1;
    }

    std::string first_field = line.substr(1, first_pipe - 1);
    size_t start = first_field.find_first_not_of(" \t");
    if (start == std::string::npos) {
      return -1;
    }

    size_t end = first_field.find_first_of(" \t", start);
    if (end == std::string::npos) {
      end = first_field.length();
    }

    std::string num_str = first_field.substr(start, end - start);
    if (!num_str.empty() && std::isdigit(num_str[0])) {
      return std::atoi(num_str.c_str());
    }
    return -1;
  }

  bool IsHbmDataLine(const std::string &line) {
    return line.find("0000:") != std::string::npos;
  }

  int ParsePhyIdFromHbmLine(const std::string& line) {
        size_t first_pipe = line.find('|');
        if (first_pipe == std::string::npos) {
            return -1;
        }
        
        size_t second_pipe = line.find('|', first_pipe + 1);
        if (second_pipe == std::string::npos) {
            return -1;
        }
        
        size_t third_pipe = line.find('|', second_pipe + 1);
        if (third_pipe == std::string::npos) {
            return -1;
        }
        
        std::string second_field = line.substr(second_pipe + 1, third_pipe - second_pipe - 1);
        
        size_t start = second_field.find_first_not_of(" \t");
        if (start == std::string::npos) {
            return -1;
        }
        
        size_t end = second_field.find_first_of(" \t", start);
        if (end == std::string::npos) {
            end = second_field.length();
        }
        
        std::string num_str = second_field.substr(start, end - start);
        
        for (char c : num_str) {
            if (!std::isdigit(c)) {
                return -1;
            }
        }
        
        if (!num_str.empty()) {
            return std::atoi(num_str.c_str());
        }
        return -1;
    }

  int ParseHbmFromLine(const std::string &line) {
    size_t last_pipe = line.rfind('|');
    if (last_pipe == std::string::npos || last_pipe == 0) {
      return -1;
    }

    size_t second_last_pipe = line.rfind('|', last_pipe - 1);
    if (second_last_pipe == std::string::npos) {
      return -1;
    }

    std::string last_field =
        line.substr(second_last_pipe + 1, last_pipe - second_last_pipe - 1);

    size_t last_slash = last_field.rfind('/');
    if (last_slash == std::string::npos) {
      return -1;
    }

    std::string before_slash = last_field.substr(0, last_slash);

    std::string hbm_used_str;
    bool found_digit = false;
    for (int i = static_cast<int>(before_slash.length()) - 1; i >= 0; --i) {
      char c = before_slash[i];
      if (std::isdigit(c)) {
        hbm_used_str = c + hbm_used_str;
        found_digit = true;
      } else if (found_digit && (c == ' ' || c == '\t')) {
        break;
      }
    }

    if (hbm_used_str.empty()) {
      return -1;
    }

    return std::atoi(hbm_used_str.c_str());
  }

  int ParseCardIdFromLine(const std::string &line) {
    if (line.empty() || line[0] != '|') {
      return -1;
    }

    size_t first_pipe = line.find('|', 1);
    if (first_pipe == std::string::npos) {
      return -1;
    }

    std::string first_field = line.substr(1, first_pipe - 1);
    size_t start = first_field.find_first_not_of(" \t");
    if (start == std::string::npos) {
      return -1;
    }

    size_t end = first_field.find_first_of(" \t", start);
    if (end == std::string::npos) {
      end = first_field.length();
    }

    std::string num_str = first_field.substr(start, end - start);
    if (!num_str.empty() && std::isdigit(num_str[0])) {
      return std::atoi(num_str.c_str());
    }
    return -1;
  }

  int ParseHbmValue(const std::string &str) {
    std::string cleaned;
    for (char c : str) {
      if (std::isdigit(c) || c == '.' || c == 'K' || c == 'M' || c == 'G' ||
          c == 'k' || c == 'm' || c == 'g') {
        cleaned += c;
      }
    }

    if (cleaned.empty())
      return -1;

    double value = 0;
    char unit = 'M';

    std::istringstream iss(cleaned);
    if (!(iss >> value)) {
      return -1;
    }

    std::string remaining;
    if (iss >> remaining) {
      if (!remaining.empty()) {
        unit = std::toupper(remaining[0]);
      }
    } else {
      size_t alpha_pos = cleaned.find_first_of("KMGkmg");
      if (alpha_pos != std::string::npos) {
        unit = std::toupper(cleaned[alpha_pos]);
        cleaned = cleaned.substr(0, alpha_pos);
        value = std::atof(cleaned.c_str());
      }
    }

    switch (unit) {
    case 'K':
      return static_cast<int>(value / 1024);
    case 'M':
      return static_cast<int>(value);
    case 'G':
      return static_cast<int>(value * 1024);
    default:
      return static_cast<int>(value);
    }
  }

  int first_npu_;
  int gnpu_num_;
  std::map<int, int> before_hbm_;
};

#endif // HBM_LEAK_CHECKER_H