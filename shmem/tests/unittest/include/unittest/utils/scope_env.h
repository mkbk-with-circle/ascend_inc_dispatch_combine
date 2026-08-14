/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCOPE_ENV_H
#define SCOPE_ENV_H

#include <unistd.h>
#include <gtest/gtest.h>

// 临时环境变量设置，析构时自动还原原值（不存在则 unset）
class ScopedEnv {
public:
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

    ScopedEnv(const char *name, const char *value) : name_(name) {
        const char *ord = std::getenv(name);
        if (ord != nullptr) {
            ord_exist_ = true;
            ord_value_ = ord;
        }
        setenv(name, value, 1);
    }
    ~ScopedEnv() {
        if (ord_exist_) {
            setenv(name_.c_str(), ord_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }
private:
    std::string name_;
    std::string ord_value_; // 原本存在的环境变量值
    bool ord_exist_ = false;
};

#endif  // SCOPE_ENV_H