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
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <dirent.h>
#include <cstring>
#include <functional>
#include <mutex>
#include "shmemi_file_util.h"
#include "dl_comm_def.h"
#include "utils/shmemi_logger.h"

const std::string DRIVER_VER_V5_BEGIN = "V100R001C10B001"; // hdk26.0.0
const std::string DRIVER_VER_V5_END = "V100R001C14B999";
const std::string DRIVER_VER_V4 = "V100R001C23SPC005B219"; // hdk25.5.0
const std::string DRIVER_VER_V3 = "V100R001C21B035";
const std::string DRIVER_VER_V2 = "V100R001C19SPC109B220";
const std::string DRIVER_VER_V1 = "V100R001C18B100";

HybmGvaVersion checkVer = HYBM_GVA_UNKNOWN;
static std::string cachedCannVersion;
static std::once_flag cannVersionOnce;

HybmGvaVersion HybmGetGvaVersion()
{
    return checkVer;
}

static std::vector<int32_t> ParseVersion(const std::string &ver)
{
    std::vector<int32_t> parts;
    std::string tmp;
    for (size_t i = 0; i <= ver.length(); ++i) {
        if (i == ver.length() || ver[i] == '.') {
            if (!tmp.empty()) {
                try {
                    parts.push_back(std::stoi(tmp));
                } catch (...) {
                    parts.push_back(0);
                }
            }
            tmp.clear();
        } else if (std::isdigit(ver[i])) {
            tmp += ver[i];
        } else {
            break;
        }
    }
    return parts;
}

static bool VersionGreaterOrEqual(const std::string &currentVer, const std::string &requiredVer)
{
    auto currentParts = ParseVersion(currentVer);
    auto requiredParts = ParseVersion(requiredVer);

    if (currentParts.empty() || requiredParts.empty()) {
        return false;
    }

    size_t compareLen = std::min(currentParts.size(), requiredParts.size());
    for (size_t i = 0; i < compareLen; ++i) {
        if (currentParts[i] > requiredParts[i]) {
            return true;
        }
        if (currentParts[i] < requiredParts[i]) {
            return false;
        }
    }
    return true;
}

static void QueryCannVersion()
{
    std::string cannHomePath = std::getenv("ASCEND_HOME_PATH") ? std::getenv("ASCEND_HOME_PATH") : "";
    if (cannHomePath.empty()) {
        SHM_LOG_WARN("ASCEND_HOME_PATH environment variable not set");
        return;
    }

    std::string cannVersion;
    std::string searchPath = cannHomePath;
    std::string infoFileName = "ascend_toolkit_install.info";

    std::function<void(const std::string&)> searchDir = [&](const std::string &path) {
        DIR *dir = opendir(path.c_str());
        if (dir == nullptr) {
            return;
        }
        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            std::string fullPath = path + "/" + entry->d_name;
            if (entry->d_type == DT_DIR) {
                searchDir(fullPath);
            } else if (strcmp(entry->d_name, infoFileName.c_str()) == 0) {
                std::ifstream infile(fullPath, std::ifstream::in);
                if (infile.is_open()) {
                    std::string line;
                    while (getline(infile, line)) {
                        auto found = line.find("version=");
                        if (found == 0) {
                            cannVersion = line.substr(8);
                            break;
                        }
                    }
                    infile.close();
                }
            }
            if (!cannVersion.empty()) {
                break;
            }
        }
        closedir(dir);
    };

    searchDir(searchPath);
    if (cannVersion.empty()) {
        SHM_LOG_WARN("Cannot find ascend_toolkit_install.info or version field in ASCEND_HOME_PATH");
    } else {
        SHM_LOG_INFO("CANN version: " << cannVersion);
    }
    cachedCannVersion = cannVersion;
}

std::string GetCannVersion()
{
    std::call_once(cannVersionOnce, QueryCannVersion);
    return cachedCannVersion;
}

bool CannVersionCheck(const std::string &requiredVer)
{
    std::string currentVer = GetCannVersion();
    if (currentVer.empty()) {
        SHM_LOG_WARN("CANN version check skipped: version not found");
        return true;
    }
    return VersionGreaterOrEqual(currentVer, requiredVer);
}

// ============================================================
// HCOMM 控制面版本检测（通过 version.info 中的 timestamp 字段）
// ============================================================
static uint64_t cachedHcommTimestamp = 0;
static std::once_flag hcommTimestampOnce;

static void QueryHcommTimestamp()
{
    std::string cannHomePath = std::getenv("ASCEND_HOME_PATH") ? std::getenv("ASCEND_HOME_PATH") : "";
    if (cannHomePath.empty()) {
        SHM_LOG_WARN("ASCEND_HOME_PATH environment variable not set, default to hcomm V2");
        return;
    }

    std::string versionFilePath = cannHomePath + "/share/info/hcomm/version.info";
    std::ifstream infile(versionFilePath, std::ifstream::in);
    if (!infile.is_open()) {
        SHM_LOG_WARN("Cannot open hcomm version.info: " << versionFilePath << ", default to V2");
        return;
    }

    std::string line;
    while (getline(infile, line)) {
        auto found = line.find("timestamp=");
        if (found != 0) {
            continue;
        }
        // 格式: timestamp=20260707_000327035
        std::string tsStr = line.substr(std::string("timestamp=").length());
        // 提取 YYYYMMDD 部分 (前8位) + 小时分钟 (第9位开始的4位)
        // "20260707_000327035" -> 去掉下划线，取前12位: "202607070003"
        tsStr.erase(std::remove(tsStr.begin(), tsStr.end(), '_'), tsStr.end());
        if (tsStr.length() >= 12) {
            tsStr = tsStr.substr(0, 12);  // YYYYMMDDHHMM
            try {
                cachedHcommTimestamp = std::stoull(tsStr);
            } catch (...) {
                SHM_LOG_WARN("Failed to parse hcomm timestamp: " << tsStr << ", default to V2");
            }
        }
        break;
    }
    infile.close();

    if (cachedHcommTimestamp == 0) {
        SHM_LOG_WARN("No valid timestamp in hcomm version.info, default to V2");
    } else {
        SHM_LOG_INFO("HCOMM version timestamp: " << cachedHcommTimestamp);
    }
}

uint64_t GetHcommTimestampVersion()
{
    std::call_once(hcommTimestampOnce, QueryHcommTimestamp);
    return cachedHcommTimestamp;
}

bool IsHcommV2()
{
    uint64_t ts = GetHcommTimestampVersion();
    if (ts == 0) {
        return true;  // 读取失败，默认使用新版 V2
    }
    // 2026-07-07 00:00 = 202607070000
    constexpr uint64_t HCOMM_V2_CUTOFF = 202607070000ULL;
    return ts >= HCOMM_V2_CUTOFF;
}

std::string GetDriverVersionPath(const std::string &driverEnvStr, const std::string &keyStr)
{
    std::string driverVersionPath;
    std::string tempPath;
    for (uint32_t i = 0; i < driverEnvStr.length(); ++i) {
        if (driverEnvStr[i] != ':') {
            tempPath += driverEnvStr[i];
        }
        if (driverEnvStr[i] == ':' || i == driverEnvStr.length() - 1) {
            if (!shm::utils::FileUtil::Realpath(tempPath)) {
                tempPath.clear();
                continue;
            }
            auto found = tempPath.find(keyStr);
            if (found == std::string::npos) {
                tempPath.clear();
                continue;
            }
            if (tempPath.length() <= found + keyStr.length() || tempPath[found + keyStr.length()] == '/') {
                driverVersionPath = tempPath.substr(0, found);
                break;
            }
            tempPath.clear();
        }
    }
    return driverVersionPath;
}

std::string LoadDriverVersionInfoFile(const std::string &realName, const std::string &keyStr)
{
    std::string driverVersion;
    char realFile[shm::utils::FileUtil::GetSafePathMax()] = {0};
    if (shm::utils::FileUtil::IsSymlink(realName) || realpath(realName.c_str(), realFile) == nullptr) {
        SHM_LOG_WARN("driver version path is not a valid real path");
        return "";
    }

    std::ifstream infile(realFile, std::ifstream::in);
    if (!infile.is_open()) {
        SHM_LOG_WARN("driver version file does not exist");
        return "";
    }

    std::string line;
    int32_t maxRows = 100;
    while (getline(infile, line)) {
        --maxRows;
        if (maxRows < 0) {
            SHM_LOG_WARN("driver version file content is too long.");
            infile.close();
            return "";
        }
        auto found = line.find(keyStr);
        if (found == 0) {
            uint32_t len = line.length() - keyStr.length();
            driverVersion = line.substr(keyStr.length(), len);
            break;
        }
    }
    infile.close();
    return driverVersion;
}

std::string CastDriverVersion(const std::string &driverEnv)
{
    std::string driverVersionPath = GetDriverVersionPath(driverEnv, "/driver/lib64");
    if (!driverVersionPath.empty()) {
        driverVersionPath += "/driver/version.info";
        std::string driverVersion = LoadDriverVersionInfoFile(driverVersionPath, "Innerversion=");
        return driverVersion;
    } else {
        SHM_LOG_WARN("cannot found version file in :" << driverEnv << ", try local default path.");
        driverVersionPath = "/usr/local/Ascend/driver/version.info";  // try default path
        std::string driverVersion = LoadDriverVersionInfoFile(driverVersionPath, "Innerversion=");
        return driverVersion;
    }
}

int32_t GetValueFromVersion(const std::string &ver, std::string key)
{
    int32_t val = 0;
    auto found = ver.find(key);
    if (found == std::string::npos) {
        return -1;
    }

    std::string tmp;
    found += 1;
    while (found < ver.length() && std::isdigit(ver[found])) {
        tmp += ver[found];
        ++found;
    }

    try {
        val = std::stoi(tmp);
    } catch (...) {
        val = -1;
    }
    return val;
}

static bool DriverVersionCheck(const std::string &ver)
{
    auto libPath = std::getenv("LD_LIBRARY_PATH");
    if (libPath == nullptr) {
        SHM_LOG_ERROR("check driver version failed, Environment LD_LIBRARY_PATH not set.");
        return false;
    }

    std::string readVer = CastDriverVersion(libPath);
    if (readVer.empty()) {
        SHM_LOG_WARN("Read version is empty; inner package skipped, defaulting to the latest version.");
        return true;
    }

    int32_t baseVal = GetValueFromVersion(ver, "V");
    int32_t readVal = GetValueFromVersion(readVer, "V");
    if (baseVal == -1 || readVal == -1 || baseVal != readVal) {
        SHM_LOG_INFO("driver version mismatch detected, V Version not equal");
        return false;
    }

    baseVal = GetValueFromVersion(ver, "R");
    readVal = GetValueFromVersion(readVer, "R");
    if (baseVal == -1 || readVal == -1 || baseVal != readVal) {
        SHM_LOG_INFO("driver version mismatch detected, R Release not equal");
        return false;
    }

    baseVal = GetValueFromVersion(ver, "C");
    readVal = GetValueFromVersion(readVer, "C");
    if (baseVal == -1 || readVal == -1 || readVal < baseVal) {
        SHM_LOG_INFO("driver version mismatch detected, C Customer is too low");
        return false;
    }
    if (readVal > baseVal) {
        return true;
    }

    baseVal = GetValueFromVersion(ver, "SPC");
    readVal = GetValueFromVersion(readVer, "SPC");
    if (baseVal != -1) {
        if (readVal == -1 || readVal < baseVal) {
            SHM_LOG_DEBUG("Driver version mismatch, base=" << baseVal << ", read=" << readVal);
            return false;
        }
        if (readVal > baseVal) {
            return true;
        }
    } else {
        if (readVal != -1) {
            return true;
        }
    }

    baseVal = GetValueFromVersion(ver, "B");
    readVal = GetValueFromVersion(readVer, "B");
    if (baseVal == -1 || readVal == -1 || readVal < baseVal) {
        SHM_LOG_INFO("driver version mismatch detected, B Build is too low");
        return false;
    }
    return true;
}

int32_t HalGvaPrecheck(void)
{
    if (!DriverVersionCheck(DRIVER_VER_V5_END) && DriverVersionCheck(DRIVER_VER_V5_BEGIN)) {
        SHM_LOG_INFO("Driver version V5 found, compatible with V4");
        checkVer = HYBM_GVA_V4;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V4)) {
        SHM_LOG_INFO("Driver version V4 found");
        checkVer = HYBM_GVA_V4;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V3)) {
        SHM_LOG_INFO("Driver version V3 found");
        checkVer = HYBM_GVA_V3;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V2)) {
        SHM_LOG_INFO("Driver version V2 found");
        checkVer = HYBM_GVA_V2;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V1)) {
        SHM_LOG_INFO("Driver version V1 found");
        checkVer = HYBM_GVA_V1;
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_ERROR("Failed to determine driver version");
    return ACLSHMEM_INNER_ERROR;
}
