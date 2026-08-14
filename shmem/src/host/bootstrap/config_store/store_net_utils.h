/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEMFABRIC_NET_UTIL_H
#define MEMFABRIC_NET_UTIL_H

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fstream>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>

#include "shmemi_logger.h"
#include "shmemi_host_def.h"

namespace shm {
namespace utils {

constexpr int MIN_PORT = 1024;
constexpr int MAX_PORT = 65536;
constexpr int MAX_ATTEMPTS = 1000;
constexpr int MAX_IFCONFIG_LENGTH = 23;
constexpr int MAX_IP = 256;
constexpr int DEFAULT_IFNAME_LNEGTH = 4;

// Invalid IPv4 address prefixes
constexpr uint32_t IPV4_APIPA_PREFIX = 0xA9FE;  // 169.254.x.x (APIPA/Link-local)
constexpr uint32_t IPV4_LOOPBACK_PREFIX = 0x7F; // 127.x.x.x (Loopback)

// Resolve `name` (literal IP or hostname) to sockaddr_storage.
// `family` is AF_INET / AF_INET6 / AF_UNSPEC.
// On success, fills `out` and returns ACLSHMEM_SUCCESS.
//
// AF_UNSPEC: iterates all getaddrinfo results, prefers IPv4 for deterministic
// dual-stack resolution, and logs every candidate so operators can diagnose
// mismatches.  If consistent family selection across PEs is critical, use IP
// literals or an explicit family (AF_INET / AF_INET6) instead of AF_UNSPEC.
inline int32_t ResolveAddress(const std::string& name, int family, sockaddr_storage& out)
{
    constexpr size_t maxHostnameLen = 253;
    if (name.size() > maxHostnameLen) {
        SHM_LOG_ERROR("Hostname too long: " << name.size() << " exceeds max " << maxHostnameLen);
        return ACLSHMEM_INVALID_PARAM;
    }

    addrinfo hints{};
    hints.ai_family = family;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* raw = nullptr;
    int gai = getaddrinfo(name.c_str(), nullptr, &hints, &raw);
    if (gai != 0) {
        SHM_LOG_ERROR("getaddrinfo failed for '" << name << "': " << gai_strerror(gai));
        return ACLSHMEM_INVALID_PARAM;
    }
    if (raw == nullptr) {
        SHM_LOG_ERROR("getaddrinfo returned null for '" << name << "'");
        return ACLSHMEM_INVALID_PARAM;
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> guard(raw, freeaddrinfo);
    if (family != AF_UNSPEC && raw->ai_family != family) {
        SHM_LOG_ERROR(
            "getaddrinfo for '" << name << "' returned address family " << raw->ai_family << ", expected " << family);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (family == AF_UNSPEC) {
        // Iterate all results: prefer IPv4 for deterministic dual-stack behaviour,
        // log every candidate so operators can diagnose resolution mismatches.
        addrinfo* preferred = nullptr;
        for (addrinfo* rp = raw; rp != nullptr; rp = rp->ai_next) {
            char ipBuf[INET6_ADDRSTRLEN] = {0};
            if (rp->ai_family == AF_INET) {
                auto* sin = reinterpret_cast<struct sockaddr_in*>(rp->ai_addr);
                inet_ntop(AF_INET, &sin->sin_addr, ipBuf, sizeof(ipBuf));
                SHM_LOG_INFO("getaddrinfo '" << name << "' candidate: IPv4 " << ipBuf);
                if (preferred == nullptr) {
                    preferred = rp; // prefer first IPv4
                }
            } else if (rp->ai_family == AF_INET6) {
                auto* sin6 = reinterpret_cast<struct sockaddr_in6*>(rp->ai_addr);
                inet_ntop(AF_INET6, &sin6->sin6_addr, ipBuf, sizeof(ipBuf));
                SHM_LOG_INFO("getaddrinfo '" << name << "' candidate: IPv6 " << ipBuf);
            } else {
                SHM_LOG_INFO("getaddrinfo '" << name << "' candidate: family " << rp->ai_family);
            }
        }
        if (preferred != nullptr) {
            std::memcpy(&out, preferred->ai_addr, preferred->ai_addrlen);
        } else {
            std::memcpy(&out, raw->ai_addr, raw->ai_addrlen);
        }
    } else {
        std::memcpy(&out, raw->ai_addr, raw->ai_addrlen);
    }
    return ACLSHMEM_SUCCESS;
}

inline int32_t shmem_get_uid_magic(shmemx_bootstrap_uid_state_t* innerUId)
{
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) {
        SHM_LOG_ERROR("open random failed");
        return ACLSHMEM_INNER_ERROR;
    }

    urandom.read(reinterpret_cast<char*>(&innerUId->magic), sizeof(innerUId->magic));
    if (urandom.fail()) {
        SHM_LOG_ERROR("read random failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("init magic id to " << innerUId->magic);
    return ACLSHMEM_SUCCESS;
}

inline int32_t bind_tcp_port_v4(int& sockfd, int port, shmemx_bootstrap_uid_state_t* innerUId, char* ip_str)
{
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v4 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v4, sizeof(on_v4)) == 0) {
            innerUId->addr.addr.addr4.sin_port = htons(port);
            sockaddr* cur_addr = reinterpret_cast<sockaddr*>(&innerUId->addr.addr.addr4);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr4)) == 0) {
                SHM_LOG_INFO("bind ipv4 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket opt fail:" << errno << "," << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket fail:" << errno << ", " << ip_str << ":" << port);
    }
    return -1;
}

inline int32_t bind_tcp_port_v6(int& sockfd, int port, shmemx_bootstrap_uid_state_t* innerUId, char* ip_str)
{
    sockfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v6 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v6, sizeof(on_v6)) == 0) {
            innerUId->addr.addr.addr6.sin6_port = htons(port);
            sockaddr* cur_addr = reinterpret_cast<sockaddr*>(&innerUId->addr.addr.addr6);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr6)) == 0) {
                SHM_LOG_INFO("bind ipv6 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket6 fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket6 opt fail:" << errno << "," << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket6 fail:" << errno << "," << ip_str << ":" << port);
    }
    return -1;
}

inline int32_t aclshmemi_get_port_magic(shmemx_bootstrap_uid_state_t* innerUId, char* ip_str)
{
    static std::random_device rd;
    const int min_port = MIN_PORT;
    const int max_port = MAX_PORT;
    const int max_attempts = MAX_ATTEMPTS;
    const int offset_bit = 32;
    uint64_t seed = 1;
    seed |= static_cast<uint64_t>(getpid()) << offset_bit;
    seed |= static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFF);
    static std::mt19937_64 gen(seed);
    std::uniform_int_distribution<> dis(min_port, max_port);

    int sockfd = -1;
    int32_t ret;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = dis(gen);
        if (innerUId->addr.type == ADDR_IPv4) {
            ret = bind_tcp_port_v4(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        } else {
            ret = bind_tcp_port_v6(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        }
    }
    SHM_LOG_ERROR("Not find a available tcp port");
    return -1;
}

inline int32_t aclshmemi_using_env_port(aclshmemi_bootstrap_uid_state_t* innerUId, char* ip_str, uint16_t envPort)
{
    if (envPort < MIN_PORT) { // envPort > MAX_PORT always false
        SHM_LOG_ERROR("env port is invalid. " << envPort);
        return ACLSHMEM_INVALID_PARAM;
    }

    int sockfd = -1;
    int32_t ret;
    if (innerUId->addr.type == ADDR_IPv4) {
        ret = bind_tcp_port_v4(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    } else {
        ret = bind_tcp_port_v6(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    }
    SHM_LOG_ERROR("init with env port fialed " << envPort << ", ret=" << ret);
    return ret;
}

// Probe usable address families on interface matching ifaName (prefix match).
// Prefer IPv4 when both are available; select IPv6 only when IPv4 is absent.
// Logs which families are available. Returns ACLSHMEM_SUCCESS and sets sockType, or error if none.
inline int32_t aclshmemi_detect_ifa_family(const char* ifaName, sa_family_t& sockType)
{
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno);
        return ACLSHMEM_INVALID_PARAM;
    }

    bool hasV4 = false;
    bool hasV6 = false;
    const size_t ifaLen = strlen(ifaName);
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr || ifa->ifa_name == nullptr) {
            continue;
        }
        if (strncmp(ifa->ifa_name, ifaName, ifaLen) != 0) {
            continue;
        }
        if ((ifa->ifa_flags & IFF_LOOPBACK) || !(ifa->ifa_flags & IFF_RUNNING) || !(ifa->ifa_flags & IFF_UP)) {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET) {
            hasV4 = true;
        } else if (ifa->ifa_addr->sa_family == AF_INET6) {
            auto* sa6 = reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr);
            if (!IN6_IS_ADDR_LINKLOCAL(&sa6->sin6_addr)) {
                hasV6 = true;
            }
        }
    }
    freeifaddrs(ifaddr);

    SHM_LOG_INFO(
        "SHMEM_UID_SOCK_IFNAME interface \"" << ifaName << "\" available families: IPv4=" << (hasV4 ? "yes" : "no")
                                             << ", IPv6=" << (hasV6 ? "yes" : "no"));

    if (hasV4) {
        sockType = AF_INET;
        SHM_LOG_INFO(
            "SHMEM_UID_SOCK_IFNAME interface \"" << ifaName << "\" selected family: IPv4"
                                                 << (hasV6 ? " (IPv6 also available, IPv4 preferred)" : ""));
        return ACLSHMEM_SUCCESS;
    }
    if (hasV6) {
        sockType = AF_INET6;
        SHM_LOG_INFO("SHMEM_UID_SOCK_IFNAME interface \"" << ifaName << "\" selected family: IPv6");
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_ERROR(
        "SHMEM_UID_SOCK_IFNAME interface \"" << ifaName
                                             << "\" has no usable IPv4/IPv6 address (need UP+RUNNING, non-loopback; "
                                                "IPv6 excludes link-local)");
    return ACLSHMEM_INVALID_VALUE;
}

inline int32_t parse_interface_with_type(const char* ipInfo, char* IP, sa_family_t& sockType, bool& flag)
{
    const char* delim = ":";
    const char* sep = strchr(ipInfo, delim[0]);
    if (sep == nullptr) {
        // No address family suffix: probe interface and prefer IPv4 over IPv6.
        size_t nameLen = strlen(ipInfo);
        if (nameLen == 0 || nameLen >= MAX_IFCONFIG_LENGTH - 1) {
            SHM_LOG_ERROR("Invalid interface name length: " << nameLen);
            return ACLSHMEM_INVALID_VALUE;
        }
        std::copy_n(ipInfo, nameLen, IP);
        IP[nameLen] = '\0';
        flag = true;
        SHM_LOG_INFO(
            "SHMEM_UID_SOCK_IFNAME has no family suffix (\""
            << ipInfo << "\"), auto-detecting usable address family. Specify explicitly as \"" << ipInfo
            << ":inet4\" or \"" << ipInfo << ":inet6\" if needed.");
        return aclshmemi_detect_ifa_family(IP, sockType);
    }

    size_t leftLen = sep - ipInfo;
    if (leftLen == 0 || leftLen >= MAX_IFCONFIG_LENGTH - 1) {
        SHM_LOG_ERROR("Invalid interface name length: " << leftLen);
        return ACLSHMEM_INVALID_VALUE;
    }
    std::copy_n(ipInfo, leftLen, IP);
    IP[leftLen] = '\0';
    if (strcmp(sep + 1, "inet6") == 0) {
        sockType = AF_INET6;
    } else if (strcmp(sep + 1, "inet4") == 0) {
        sockType = AF_INET;
    } else {
        flag = true;
        SHM_LOG_WARN(
            "SHMEM_UID_SOCK_IFNAME has invalid family suffix (\""
            << (sep + 1) << "\"), expected \"inet4\" or \"inet6\". Auto-detecting usable address family.");
        return aclshmemi_detect_ifa_family(IP, sockType);
    }
    flag = true;
    SHM_LOG_INFO("Parse ipInfo success: ifaPrefix=" << IP << ", sockType=" << (sockType == AF_INET ? "IPv4" : "IPv6"));
    return ACLSHMEM_SUCCESS;
}

inline int32_t aclshmemi_auto_get_ip(struct sockaddr* ifaAddr, char* local, sa_family_t& sockType)
{
    sa_family_t prevFamily = sockType;
    sockType = ifaAddr->sa_family;
    if (prevFamily != sockType) {
        SHM_LOG_INFO(
            "auto-select: address family switched from " << (prevFamily == AF_INET ? "AF_INET" : "AF_INET6") << " to "
                                                         << (sockType == AF_INET ? "AF_INET" : "AF_INET6"));
    }
    if (sockType == AF_INET) {
        auto localIp = reinterpret_cast<struct sockaddr_in*>(ifaAddr)->sin_addr;
        if (inet_ntop(sockType, &localIp, local, MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv4 to string failed. ");
            return ACLSHMEM_INVALID_PARAM;
        }
        return ACLSHMEM_SUCCESS;
    } else if (sockType == AF_INET6) {
        auto localIp = reinterpret_cast<struct sockaddr_in6*>(ifaAddr)->sin6_addr;
        if (inet_ntop(sockType, &localIp, local, MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv6 to string failed. ");
            return ACLSHMEM_INVALID_PARAM;
        }
        return ACLSHMEM_SUCCESS;
    }
    return ACLSHMEM_INVALID_PARAM;
}

inline bool aclshmemi_is_valid_ip(struct sockaddr* ifaAddr)
{
    if (ifaAddr == nullptr) {
        return false;
    }

    if (ifaAddr->sa_family == AF_INET) {
        auto* addr = reinterpret_cast<struct sockaddr_in*>(ifaAddr);
        uint32_t ip = ntohl(addr->sin_addr.s_addr);

        // Check for invalid IPv4 addresses
        if (ip == 0) {
            SHM_LOG_DEBUG("invalid IPv4: 0.0.0.0");
            return false;
        }
        if ((ip >> 16) == IPV4_APIPA_PREFIX) {
            SHM_LOG_DEBUG("invalid IPv4: APIPA address 169.254.x.x");
            return false;
        }
        if ((ip >> 24) == IPV4_LOOPBACK_PREFIX) {
            SHM_LOG_DEBUG("invalid IPv4: loopback address 127.x.x.x");
            return false;
        }
        return true;
    } else if (ifaAddr->sa_family == AF_INET6) {
        auto* addr6 = reinterpret_cast<struct sockaddr_in6*>(ifaAddr);

        // Check for invalid IPv6 addresses
        if (IN6_IS_ADDR_UNSPECIFIED(&addr6->sin6_addr)) {
            SHM_LOG_DEBUG("invalid IPv6: unspecified address ::");
            return false;
        }
        if (IN6_IS_ADDR_LOOPBACK(&addr6->sin6_addr)) {
            SHM_LOG_DEBUG("invalid IPv6: loopback address ::1");
            return false;
        }
        // Check for link-local address (fe80::/10)
        if (IN6_IS_ADDR_LINKLOCAL(&addr6->sin6_addr)) {
            SHM_LOG_DEBUG("invalid IPv6: link-local address fe80::/10");
            return false;
        }
        return true;
    }

    return false;
}

inline bool aclshmemi_check_ifa(
    struct ifaddrs* ifa, sa_family_t sockType, bool flag, char* ifaName, size_t ifaLen, bool autoSelect = false)
{
    if (ifa->ifa_addr == nullptr || ifa->ifa_netmask == nullptr || ifa->ifa_name == nullptr) {
        SHM_LOG_DEBUG("loop ifa_addr/ifa_netmask/ifa_name is nullptr");
        return false;
    }

    // socket type match and input env ifa valid
    if (ifa->ifa_addr->sa_family != sockType && flag) {
        SHM_LOG_DEBUG("sa family is not match, get " << ifa->ifa_addr->sa_family << ", expect " << sockType);
        return false;
    }
    if (autoSelect && !flag) {
        SHM_LOG_DEBUG("auto-select mode: accepting any address family, current=" << ifa->ifa_addr->sa_family);
    }

    // prefix match with input ifa name (skip when autoSelect)
    if (!autoSelect && strncmp(ifa->ifa_name, ifaName, ifaLen) != 0) {
        SHM_LOG_DEBUG("ifa name prefix un-match, get " << ifa->ifa_name << ", expect " << ifaName);
        return false;
    }

    // auto-select: filter out virtual and unusable interfaces
    if (autoSelect) {
        std::string ifNameStr(ifa->ifa_name);
        // Exclude loopback, docker, veth, bridge and other virtual interfaces
        if (ifNameStr.find("lo") == 0 ||     // loopback
            ifNameStr.find("docker") == 0 || // docker bridge
            ifNameStr.find("veth") == 0 ||   // virtual ethernet
            ifNameStr.find("br-") == 0 ||    // bridge
            ifNameStr.find("virbr") == 0 ||  // virtual bridge
            ifNameStr.find("tun") == 0 ||    // tunnel
            ifNameStr.find("tap") == 0) {    // tap device
            SHM_LOG_DEBUG("skip virtual interface: " << ifa->ifa_name);
            return false;
        }
    }

    // ignore ifa which is down or loopback or not running
    if ((ifa->ifa_flags & IFF_LOOPBACK) || !(ifa->ifa_flags & IFF_RUNNING) || !(ifa->ifa_flags & IFF_UP)) {
        SHM_LOG_DEBUG("ifa flag un-match, flag=" << ifa->ifa_flags);
        return false;
    }

    // auto-select: validate IP address
    if (autoSelect && !aclshmemi_is_valid_ip(ifa->ifa_addr)) {
        SHM_LOG_DEBUG("invalid IP address on interface: " << ifa->ifa_name);
        return false;
    }

    if (sockType == AF_INET6) {
        struct sockaddr_in6* sa6 = reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr);
        if (IN6_IS_ADDR_LINKLOCAL(&sa6->sin6_addr)) {
            SHM_LOG_DEBUG("ifa is scope link addr " << ifaName);
            return false;
        }
    }
    return true;
}

inline int32_t aclshmemi_get_ip_from_ifa(char* local, sa_family_t& sockType, const char* ipInfo)
{
    struct ifaddrs* ifaddr;
    char ifaName[MAX_IFCONFIG_LENGTH] = {0}; // Initialize to zero
    sockType = AF_INET;
    bool flag = false;
    bool autoSelect = false;

    // Treat nullptr or empty string as auto-select
    if (ipInfo == nullptr || ipInfo[0] == '\0') {
        // Auto-select network interface: skip virtual interfaces and use the first valid one
        autoSelect = true;
        SHM_LOG_INFO("auto-select network interface (skip lo/docker/veth/br-/virbr/tun/tap)");
    } else if (parse_interface_with_type(ipInfo, ifaName, sockType, flag) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("IP size set in SHMEM_CONF_STORE_MASTER_IF format has wrong length");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno);
        return ACLSHMEM_INVALID_PARAM;
    }
    int32_t result = ACLSHMEM_INVALID_PARAM;
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!aclshmemi_check_ifa(ifa, sockType, flag, ifaName, strlen(ifaName), autoSelect)) {
            continue;
        }

        bool ipObtained = false;
        if (sockType == AF_INET && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr)->sin_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv4 to string failed. ");
                continue;
            }
            ipObtained = true;
        } else if (sockType == AF_INET6 && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr)->sin6_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv6 to string failed. ");
                continue;
            }
            ipObtained = true;
        } else {
            auto ret = aclshmemi_auto_get_ip(ifa->ifa_addr, local, sockType);
            if (ret != ACLSHMEM_SUCCESS) {
                continue;
            }
            ipObtained = true;
        }

        if (ipObtained) {
            if (autoSelect) {
                SHM_LOG_INFO(
                    "auto-selected interface: " << ifa->ifa_name << ", IP: " << local
                                                << ", family: " << (sockType == AF_INET ? "AF_INET" : "AF_INET6"));
            }
            result = ACLSHMEM_SUCCESS;
            break;
        }
    }
    freeifaddrs(ifaddr);
    return result;
}

inline bool aclshmemi_parse_port(const std::string& portStr, uint16_t& port)
{
    constexpr int maxPort = 65535;
    try {
        size_t parsedLen = 0;
        int portInt = std::stoi(portStr, &parsedLen);
        if (parsedLen != portStr.size()) {
            SHM_LOG_ERROR("Invalid port format: " << portStr);
            return false;
        }
        if (portInt < 0 || portInt > maxPort) {
            SHM_LOG_ERROR("Invalid port value: " << portInt << ", must be in range [0, " << maxPort << "]");
            return false;
        }
        port = static_cast<uint16_t>(portInt);
        return true;
    } catch (const std::exception& e) {
        SHM_LOG_ERROR("Invalid port string: " << e.what());
        return false;
    }
}

inline int32_t aclshmemi_get_ip_from_env(char* ip, uint16_t& port, sa_family_t& sockType, const char* ipPort)
{
    if (ipPort == nullptr) {
        return ACLSHMEM_INVALID_PARAM;
    }
    SHM_LOG_DEBUG("get env SHMEM_UID_SESSION_ID value:" << ipPort);
    std::string ipPortStr = ipPort;

    if (ipPort[0] == '[') {
        sockType = AF_INET6;
        size_t found = ipPortStr.find_last_of(']');
        if (found == std::string::npos) {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid: no closing ']'");
            return ACLSHMEM_INVALID_PARAM;
        }
        if (found + 1 >= ipPortStr.size() || ipPortStr[found + 1] != ':') {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid: missing ':' after ']'");
            return ACLSHMEM_INVALID_PARAM;
        }
        std::string ipStr = ipPortStr.substr(1, found - 1);
        std::string portStr = ipPortStr.substr(found + 2);

        if (ipStr.size() >= MAX_IP) {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid: ipStr too long");
            return ACLSHMEM_INVALID_PARAM;
        }

        std::snprintf(ip, MAX_IP, "%s", ipStr.c_str());

        if (!aclshmemi_parse_port(portStr, port)) {
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        size_t found = ipPortStr.find_last_of(':');
        if (found == std::string::npos || ipPortStr.length() - found <= 1) {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
            return ACLSHMEM_INVALID_PARAM;
        }
        std::string ipStr = ipPortStr.substr(0, found);
        std::string portStr = ipPortStr.substr(found + 1);

        if (ipStr.size() >= MAX_IP) {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid: ipStr too long");
            return ACLSHMEM_INVALID_PARAM;
        }

        char addr_buf[sizeof(struct in6_addr)] = {0};
        if (inet_pton(AF_INET6, ipStr.c_str(), addr_buf) == 1) {
            sockType = AF_INET6;
        } else if (inet_pton(AF_INET, ipStr.c_str(), addr_buf) == 1) {
            sockType = AF_INET;
        } else {
            sockaddr_storage resolved{};
            if (ResolveAddress(ipStr, AF_UNSPEC, resolved) != ACLSHMEM_SUCCESS) {
                return ACLSHMEM_INVALID_PARAM;
            }
            sockType = resolved.ss_family;
        }

        std::snprintf(ip, MAX_IP, "%s", ipStr.c_str());

        if (!aclshmemi_parse_port(portStr, port)) {
            return ACLSHMEM_INVALID_PARAM;
        }
    }
    return ACLSHMEM_SUCCESS;
}

inline int32_t aclshmemi_set_ip_info(
    aclshmemx_uniqueid_t* uid, sa_family_t& sock_type, char* pta_env_ip, uint16_t pta_env_port, bool is_from_ifa)
{
    // init default uid
    SHM_ASSERT_RETURN(uid != nullptr, ACLSHMEM_INVALID_PARAM);
    aclshmemx_uniqueid_t default_uid{};
    default_uid.version = ACLSHMEM_UNIQUEID_VERSION;
    *uid = default_uid;
    shmemx_bootstrap_uid_state_t* innerUID = reinterpret_cast<shmemx_bootstrap_uid_state_t*>(uid);
    if (sock_type == AF_INET) {
        innerUID->addr.addr.addr4.sin_family = AF_INET;
        if (inet_pton(AF_INET, pta_env_ip, &(innerUID->addr.addr.addr4.sin_addr)) <= 0) {
            sockaddr_storage resolved{};
            if (ResolveAddress(pta_env_ip, AF_INET, resolved) != ACLSHMEM_SUCCESS) {
                return ACLSHMEM_NOT_INITED;
            }
            auto* sin = reinterpret_cast<const struct sockaddr_in*>(&resolved);
            innerUID->addr.addr.addr4.sin_addr = sin->sin_addr;
        }
        innerUID->addr.type = ADDR_IPv4;
    } else if (sock_type == AF_INET6) {
        innerUID->addr.addr.addr6.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, pta_env_ip, &(innerUID->addr.addr.addr6.sin6_addr)) <= 0) {
            sockaddr_storage resolved{};
            if (ResolveAddress(pta_env_ip, AF_INET6, resolved) != ACLSHMEM_SUCCESS) {
                return ACLSHMEM_NOT_INITED;
            }
            auto* sin6 = reinterpret_cast<const struct sockaddr_in6*>(&resolved);
            innerUID->addr.addr.addr6.sin6_addr = sin6->sin6_addr;
        }
        innerUID->addr.type = ADDR_IPv6;
    } else {
        SHM_LOG_ERROR("IP Type is not IPv4 or IPv6");
        return ACLSHMEM_INVALID_PARAM;
    }

    // fill ip port as part of uid
    if (is_from_ifa) {
        int32_t ret = aclshmemi_get_port_magic(innerUID, pta_env_ip);
        if (ret != 0) {
            SHM_LOG_ERROR("get available port failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        int32_t ret = aclshmemi_using_env_port(innerUID, pta_env_ip, pta_env_port);
        if (ret != 0) {
            SHM_LOG_ERROR("using env port failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    SHM_LOG_INFO("gen unique id success.");
    return ACLSHMEM_SUCCESS;
}

} // namespace utils
} // namespace shm

#endif
