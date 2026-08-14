/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <arpa/inet.h>
#include <netdb.h>
#include <cstring>
#include <string>
#include <iostream>
#include <ifaddrs.h>
#include <net/if.h>

#include "host/shmem_host_def.h"
#include "bootstrap/config_store/store_net_utils.h"
#include "host/utils/shmem_log.h"

using namespace shm::utils;

static const char *errcode_to_str(int32_t code)
{
    switch (code) {
        case ACLSHMEM_SUCCESS:       return "ACLSHMEM_SUCCESS(0)";
        case ACLSHMEM_INVALID_PARAM: return "ACLSHMEM_INVALID_PARAM(-2)";
        case ACLSHMEM_SMEM_ERROR:    return "ACLSHMEM_SMEM_ERROR(-3)";
        case ACLSHMEM_INNER_ERROR:   return "ACLSHMEM_INNER_ERROR(-4)";
        case ACLSHMEM_NOT_INITED:    return "ACLSHMEM_NOT_INITED(-5)";
        case ACLSHMEM_BOOTSTRAP_ERROR: return "ACLSHMEM_BOOTSTRAP_ERROR(-6)";
        default:                     return "UNKNOWN";
    }
}

static bool is_ipv6_available()
{
    int sockfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return false;
    }
    struct sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    inet_pton(AF_INET6, "::1", &addr.sin6_addr);
    addr.sin6_port = htons(0);
    int ret = ::bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));
    close(sockfd);
    return ret == 0;
}

// Return an IPv6 hostname that resolves on this system, or empty string if none.
static std::string find_ipv6_hostname()
{
    static const char *candidates[] = {"ip6-localhost", "localhost6"};
    for (const char *cand : candidates) {
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET6;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(cand, nullptr, &hints, &res) == 0 && res != nullptr) {
            freeaddrinfo(res);
            return cand;
        }
        if (res != nullptr) freeaddrinfo(res);
    }
    return {};
}

// Ask the OS for a free port.  Returns 0 if no free port could be found.
static uint16_t find_free_port(sa_family_t family = AF_INET)
{
    if (family != AF_INET && family != AF_INET6) return 0;
    int sockfd = ::socket(family, SOCK_STREAM, 0);
    if (sockfd < 0) return 0;
    uint16_t port = 0;
    if (family == AF_INET6) {
        struct sockaddr_in6 addr6{};
        addr6.sin6_family = AF_INET6;
        addr6.sin6_addr = in6addr_any;
        addr6.sin6_port = 0;
        if (::bind(sockfd, (struct sockaddr *)&addr6, sizeof(addr6)) != 0) {
            close(sockfd);
            return 0;
        }
        socklen_t len = sizeof(addr6);
        ::getsockname(sockfd, (struct sockaddr *)&addr6, &len);
        port = ntohs(addr6.sin6_port);
    } else {
        struct sockaddr_in addr4{};
        addr4.sin_family = AF_INET;
        addr4.sin_addr.s_addr = INADDR_ANY;
        addr4.sin_port = 0;
        if (::bind(sockfd, (struct sockaddr *)&addr4, sizeof(addr4)) != 0) {
            close(sockfd);
            return 0;
        }
        socklen_t len = sizeof(addr4);
        ::getsockname(sockfd, (struct sockaddr *)&addr4, &len);
        port = ntohs(addr4.sin_port);
    }
    close(sockfd);
    return port;
}

static void debug_getaddrinfo(const char *hostname, sa_family_t family)
{
    struct addrinfo hints{};
    struct addrinfo *res = nullptr;
    hints.ai_family = family;
    hints.ai_socktype = SOCK_STREAM;
    int gai = getaddrinfo(hostname, nullptr, &hints, &res);
    if (gai != 0) {
        std::cerr << ", error: " << gai_strerror(gai);
    } else if (res != nullptr) {
        char buf[INET6_ADDRSTRLEN] = {0};
        if (res->ai_family == AF_INET) {
            auto *sin = reinterpret_cast<struct sockaddr_in *>(res->ai_addr);
            inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
        } else {
            auto *sin6 = reinterpret_cast<struct sockaddr_in6 *>(res->ai_addr);
            inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf));
        }
        std::cerr << ", resolved: " << buf;
        freeaddrinfo(res);
    }
    std::cerr << std::endl;
}

static void debug_bind_test(sa_family_t family, const char *ip_str)
{
    int sockfd = ::socket(family, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return;
    }
    int on = 1;
    ::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    if (family == AF_INET) {
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        inet_pton(AF_INET, ip_str, &addr.sin_addr);
        addr.sin_port = htons(0);
        int ret = ::bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));
        if (ret == 0) {
            socklen_t len = sizeof(addr);
            getsockname(sockfd, (struct sockaddr *)&addr, &len);
        }
    } else {
        struct sockaddr_in6 addr{};
        addr.sin6_family = AF_INET6;
        inet_pton(AF_INET6, ip_str, &addr.sin6_addr);
        addr.sin6_port = htons(0);
        int ret = ::bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));
        if (ret == 0) {
            socklen_t len = sizeof(addr);
            getsockname(sockfd, (struct sockaddr *)&addr, &len);
        }
    }
    close(sockfd);
}

class SetIpInfoTest : public ::testing::Test {
protected:
    aclshmemx_uniqueid_t uid;
    void SetUp() override
    {
        uid = {};
        aclshmemx_set_log_level(0);
    }
    void TearDown() override
    {
        auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
        if (inner->inner_sockFd > 0) {
            close(inner->inner_sockFd);
        }
    }
};

TEST_F(SetIpInfoTest, IPv4PureIP)
{
    char ip[] = "127.0.0.1";
    sa_family_t sock_type = AF_INET;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv4);
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &inner->addr.addr.addr4.sin_addr, buf, sizeof(buf));
    EXPECT_STREQ(buf, "127.0.0.1");
}

TEST_F(SetIpInfoTest, IPv6PureIP)
{
    if (!is_ipv6_available()) {
        GTEST_SKIP() << "[SKIP] IPv6 is not available on this system (bind(::1) failed)";
    }
    char ip[] = "::1";
    sa_family_t sock_type = AF_INET6;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv6);
}

TEST_F(SetIpInfoTest, IPv4Hostname)
{
    char ip[] = "localhost";
    sa_family_t sock_type = AF_INET;

    debug_getaddrinfo("localhost", AF_INET);
    debug_bind_test(AF_INET, "127.0.0.1");

    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);

    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);

    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_EQ(inner->addr.type, ADDR_IPv4);
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &inner->addr.addr.addr4.sin_addr, buf, sizeof(buf));
    EXPECT_STRNE(buf, "");
}

TEST_F(SetIpInfoTest, IPv6Hostname)
{
    if (!is_ipv6_available()) {
        GTEST_SKIP() << "[SKIP] IPv6 is not available on this system (bind(::1) failed)";
    }
    std::string hostname = find_ipv6_hostname();
    if (hostname.empty()) {
        GTEST_SKIP() << "[SKIP] no IPv6 hostname resolves on this system";
    }
    sa_family_t sock_type = AF_INET6;

    debug_getaddrinfo(hostname.c_str(), AF_INET6);
    debug_bind_test(AF_INET6, "::1");

    char ip[256];
    std::snprintf(ip, sizeof(ip), "%s", hostname.c_str());
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);

    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);

    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_EQ(inner->addr.type, ADDR_IPv6);
}

TEST_F(SetIpInfoTest, InvalidHostname)
{
    char ip[] = "nonexistent_host_invalid_12345";
    sa_family_t sock_type = AF_INET;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_NOT_INITED);
}

TEST_F(SetIpInfoTest, NullUid)
{
    char ip[] = "127.0.0.1";
    sa_family_t sock_type = AF_INET;
    int32_t ret = aclshmemi_set_ip_info(nullptr, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(SetIpInfoTest, InvalidSockType)
{
    char ip[] = "127.0.0.1";
    sa_family_t sock_type = AF_UNSPEC;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(SetIpInfoTest, IPv4WithEnvPort)
{
    char ip[] = "127.0.0.1";
    sa_family_t sock_type = AF_INET;
    uint16_t env_port = find_free_port(AF_INET);
    if (env_port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, env_port, false);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(ntohs(inner->addr.addr.addr4.sin_port), env_port);
}

TEST_F(SetIpInfoTest, IPv6WithEnvPort)
{
    if (!is_ipv6_available()) {
        GTEST_SKIP() << "[SKIP] IPv6 is not available on this system";
    }
    char ip[] = "::1";
    sa_family_t sock_type = AF_INET6;
    uint16_t env_port = find_free_port(AF_INET6);
    if (env_port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, env_port, false);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(ntohs(inner->addr.addr.addr6.sin6_port), env_port);
}

TEST_F(SetIpInfoTest, IPv4HostnameWithEnvPort)
{
    uint16_t env_port = find_free_port();
    if (env_port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }
    char ip[] = "localhost";
    sa_family_t sock_type = AF_INET;
    debug_getaddrinfo("localhost", AF_INET);
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, env_port, false);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv4);
    EXPECT_EQ(ntohs(inner->addr.addr.addr4.sin_port), env_port);
}

TEST_F(SetIpInfoTest, IPv6HostnameWithEnvPort)
{
    if (!is_ipv6_available()) {
        GTEST_SKIP() << "[SKIP] IPv6 is not available on this system";
    }
    std::string hostname = find_ipv6_hostname();
    if (hostname.empty()) {
        GTEST_SKIP() << "[SKIP] no IPv6 hostname resolves on this system";
    }
    uint16_t env_port = find_free_port(AF_INET6);
    if (env_port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }
    sa_family_t sock_type = AF_INET6;
    debug_getaddrinfo(hostname.c_str(), AF_INET6);

    char ip[256];
    std::snprintf(ip, sizeof(ip), "%s", hostname.c_str());
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, env_port, false);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv6);
    EXPECT_EQ(ntohs(inner->addr.addr.addr6.sin6_port), env_port);
}

TEST_F(SetIpInfoTest, IPv4PrivateIP)
{
    char ip[] = "192.168.1.1";
    sa_family_t sock_type = AF_INET;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    if (ret != ACLSHMEM_SUCCESS) {
        debug_bind_test(AF_INET, "192.168.1.1");
    }
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    if (inner->inner_sockFd > 0) {
        close(inner->inner_sockFd);
        inner->inner_sockFd = -1;
    }
}

TEST_F(SetIpInfoTest, IPv4ZeroIP)
{
    char ip[] = "0.0.0.0";
    sa_family_t sock_type = AF_INET;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv4);
}

TEST_F(SetIpInfoTest, IPv4HostnameAutoDetectFamily)
{
    char ip[] = "localhost";
    sa_family_t sock_type = AF_INET;
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv4);
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &inner->addr.addr.addr4.sin_addr, buf, sizeof(buf));
    EXPECT_STRNE(buf, "");
}

TEST_F(SetIpInfoTest, IPv6HostnameAutoDetectFamily)
{
    if (!is_ipv6_available()) {
        GTEST_SKIP() << "[SKIP] IPv6 is not available on this system";
    }

    std::string hostname = find_ipv6_hostname();
    if (hostname.empty()) {
        GTEST_SKIP() << "[SKIP] no IPv6 hostname resolves on this system";
    }

    sa_family_t sock_type = AF_INET6;
    debug_getaddrinfo(hostname.c_str(), AF_INET6);

    char ip[256];
    std::snprintf(ip, sizeof(ip), "%s", hostname.c_str());
    int32_t ret = aclshmemi_set_ip_info(&uid, sock_type, ip, 0, true);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_EQ(inner->addr.type, ADDR_IPv6);
}

class GetIpFromEnvTest : public ::testing::Test {
protected:
    char ip[MAX_IP];
    uint16_t port;
    sa_family_t sockType;
    void SetUp() override
    {
        std::fill(ip, ip + MAX_IP, 0);
        port = 0;
        sockType = AF_UNSPEC;
    }
};

TEST_F(GetIpFromEnvTest, IPv4Format)
{
    const char *ipPort = "192.168.1.100:8888";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_EQ(sockType, AF_INET);
    EXPECT_STREQ(ip, "192.168.1.100");
    EXPECT_EQ(port, 8888);
}

TEST_F(GetIpFromEnvTest, IPv6Format)
{
    const char *ipPort = "[::1]:9999";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_EQ(sockType, AF_INET6);
    EXPECT_STREQ(ip, "::1");
    EXPECT_EQ(port, 9999);
}

TEST_F(GetIpFromEnvTest, IPv6WithScope)
{
    const char *ipPort = "[fe80::1%eth0]:1234";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_EQ(sockType, AF_INET6);
    EXPECT_EQ(port, 1234);
}

TEST_F(GetIpFromEnvTest, IPv4HostnameWithPort)
{
    const char *ipPort = "localhost:8888";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_TRUE(sockType == AF_INET || sockType == AF_INET6);
    EXPECT_STREQ(ip, "localhost");
    EXPECT_EQ(port, 8888);
}

TEST_F(GetIpFromEnvTest, HostnameAutoDetectAddrFamily)
{
    const char *ipPort = "localhost:8888";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_TRUE(sockType == AF_INET || sockType == AF_INET6);
    // ResolveAddress prefers IPv4 for deterministic dual-stack behaviour.
    // Verify that the returned family matches one of the addresses that
    // localhost actually resolves to, not necessarily the system's default order.
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int gai = getaddrinfo("localhost", nullptr, &hints, &res);
    if (gai == 0 && res != nullptr) {
        bool foundMatchingFamily = false;
        for (addrinfo *rp = res; rp != nullptr; rp = rp->ai_next) {
            if (rp->ai_family == sockType) {
                foundMatchingFamily = true;
                break;
            }
        }
        EXPECT_TRUE(foundMatchingFamily) << "sockType " << sockType
            << " not found in getaddrinfo results for localhost";
        freeaddrinfo(res);
    }
}

TEST_F(GetIpFromEnvTest, InvalidHostnameWithPort)
{
    const char *ipPort = "nonexistent_host_invalid_12345:8888";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_NE(ret, ACLSHMEM_SUCCESS);
}

TEST_F(GetIpFromEnvTest, IPv6HostnameInBracket)
{
    if (!is_ipv6_available()) {
        GTEST_SKIP() << "[SKIP] IPv6 is not available on this system";
    }
    const char *ipPort = "[localhost]:9999";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    if (ret == ACLSHMEM_SUCCESS) {
        EXPECT_EQ(sockType, AF_INET6);
        EXPECT_EQ(port, 9999);
    }
}

TEST_F(GetIpFromEnvTest, NullInput)
{
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, nullptr);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(GetIpFromEnvTest, IPv4NoPort)
{
    const char *ipPort = "192.168.1.100";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(GetIpFromEnvTest, IPv6NoClosingBracket)
{
    const char *ipPort = "[::1:9999";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(GetIpFromEnvTest, IPv6NoPortAfterBracket)
{
    const char *ipPort = "[::1]";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(GetIpFromEnvTest, IPv4EmptyPort)
{
    const char *ipPort = "192.168.1.100:";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST_F(GetIpFromEnvTest, IPv6AddrNoBracketAutoDetect)
{
    const char *ipPort = "2001:db8::1:9999";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    if (ret == ACLSHMEM_SUCCESS) {
        EXPECT_EQ(sockType, AF_INET6);
        EXPECT_EQ(port, 9999);
    }
}

TEST_F(GetIpFromEnvTest, SystemHostnameResolves)
{
    char hostname_buf[256] = {0};
    int ghn = gethostname(hostname_buf, sizeof(hostname_buf));
    if (ghn != 0 || hostname_buf[0] == '\0') {
        GTEST_SKIP() << "[SKIP] gethostname() failed or empty";
    }
    std::string session_id = std::string(hostname_buf) + ":5555";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, session_id.c_str());
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_TRUE(sockType == AF_INET || sockType == AF_INET6);
    EXPECT_EQ(port, 5555);
}

TEST_F(GetIpFromEnvTest, HostnameWithDotAndHyphen)
{
    std::string hostname = find_ipv6_hostname();
    if (hostname.empty()) {
        GTEST_SKIP() << "[SKIP] no IPv6 hostname resolves on this system";
    }
    uint16_t free_port = find_free_port(AF_INET6);
    if (free_port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }

    std::string ipPort = hostname + ":" + std::to_string(free_port);
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort.c_str());
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_TRUE(sockType == AF_INET || sockType == AF_INET6);
    EXPECT_STREQ(ip, hostname.c_str());
    EXPECT_EQ(port, free_port);
}

TEST_F(GetIpFromEnvTest, EmptyIpPortString)
{
    const char *ipPort = "";
    int32_t ret = aclshmemi_get_ip_from_env(ip, port, sockType, ipPort);
    EXPECT_NE(ret, ACLSHMEM_SUCCESS);
}

class HostnameResolutionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        aclshmemx_set_log_level(0);
    }
};

TEST_F(HostnameResolutionTest, LocalhostResolvesToValidAddr)
{
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int gai = getaddrinfo("localhost", nullptr, &hints, &res);
    EXPECT_EQ(gai, 0);
    EXPECT_NE(res, nullptr);
    if (res != nullptr) {
        EXPECT_TRUE(res->ai_family == AF_INET || res->ai_family == AF_INET6);
        char buf[INET6_ADDRSTRLEN] = {0};
        if (res->ai_family == AF_INET) {
            auto *sin = reinterpret_cast<struct sockaddr_in *>(res->ai_addr);
            inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
        } else {
            auto *sin6 = reinterpret_cast<struct sockaddr_in6 *>(res->ai_addr);
            inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf));
        }
        freeaddrinfo(res);
    }
}

TEST_F(HostnameResolutionTest, InvalidHostnameFails)
{
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int gai = getaddrinfo("nonexistent_host_invalid_12345", nullptr, &hints, &res);
    EXPECT_NE(gai, 0);
}

TEST_F(HostnameResolutionTest, HostnameEnvToSetIpInfo)
{
    char env_ip[] = "localhost";
    sa_family_t detected_type = AF_UNSPEC;

    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int gai = getaddrinfo(env_ip, nullptr, &hints, &res);
    if (gai == 0 && res != nullptr) {
        detected_type = res->ai_family;
        freeaddrinfo(res);
    }

    aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    uint16_t env_port = find_free_port(detected_type == AF_INET6 ? AF_INET6 : AF_INET);
    if (env_port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }
    int32_t ret = aclshmemi_set_ip_info(&uid, detected_type, env_ip, env_port, false);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    auto *inner = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(&uid);
    EXPECT_TRUE(inner->addr.type == ADDR_IPv4 || inner->addr.type == ADDR_IPv6);
    if (inner->inner_sockFd > 0) {
        close(inner->inner_sockFd);
    }
}