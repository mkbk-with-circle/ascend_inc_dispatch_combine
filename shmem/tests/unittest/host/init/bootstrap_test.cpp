#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include "shmemi_host_common.h"

// simple test to verify bootstrap SO loading behavior

// helper that tries to init bootstrap with given flags
static int try_bootstrap(int flags) {
    aclshmemx_init_attr_t attr;
    memset(&attr, 0, sizeof(attr));
    attr.my_pe = 0;
    attr.n_pes = 1;
    attr.option_attr.sockFd = -1;
    attr.option_attr.shm_init_timeout = 10;
    attr.option_attr.control_operation_timeout = 10;
    attr.ip_port[0] = '\0';
    return aclshmemi_bootstrap_init(flags, &attr);
}

static uint16_t find_free_port() {
    int sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return 0;
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    if (::bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(sockfd);
        return 0;
    }
    socklen_t len = sizeof(addr);
    ::getsockname(sockfd, (struct sockaddr *)&addr, &len);
    close(sockfd);
    return ntohs(addr.sin_port);
}

TEST(BootstrapTest, LoadConfigStoreSuccess) {
    // ensure SO file exists in current directory
    FILE *f = fopen("aclshmem_bootstrap_config_store.so", "r");
    if (!f) {
        GTEST_SKIP() << "config_store SO not present, skipping";
    } else {
        fclose(f);
    }
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    EXPECT_EQ(rc, ACLSHMEM_SUCCESS);
    aclshmemi_bootstrap_finalize();
}

TEST(BootstrapTest, MissingSoFails) {
    // temporarily rename so to simulate missing
    const char *name = "aclshmem_bootstrap_config_store.so";
    if (access(name, F_OK) == 0) {
        rename(name, "tmp.so");
    }
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    EXPECT_EQ(rc, ACLSHMEM_INVALID_PARAM);
    // restore
    if (access("tmp.so", F_OK) == 0) {
        rename("tmp.so", name);
    }
}

TEST(BootstrapTest, ModeSwitching) {
    // should handle MPI and DEFAULT modes
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_MPI);
    EXPECT_NE(rc, ACLSHMEM_SUCCESS); // MPI so usually not present
    rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    // rc either success or invalid_value depending on so presence
    aclshmemi_bootstrap_finalize();
}

// repeated initialization exercises both DEFAULT and UNIQUEID paths
TEST(BootstrapTest, RepeatInitDefaultThenUnique) {
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    if (rc == ACLSHMEM_SUCCESS) {
        aclshmemi_bootstrap_finalize();
    }

    rc = try_bootstrap(ACLSHMEMX_INIT_WITH_UNIQUEID);
    EXPECT_TRUE(rc == ACLSHMEM_SUCCESS || rc == ACLSHMEM_INVALID_PARAM ||
                rc == ACLSHMEM_INVALID_VALUE);
    if (rc == ACLSHMEM_SUCCESS) {
        aclshmemi_bootstrap_finalize();
    }
}

TEST(BootstrapTest, InitWithIpPortHostname) {
    aclshmemx_init_attr_t attr;
    memset(&attr, 0, sizeof(attr));
    attr.my_pe = 0;
    attr.n_pes = 1;
    attr.option_attr.sockFd = -1;
    attr.option_attr.shm_init_timeout = 10;
    attr.option_attr.control_operation_timeout = 10;

    int test_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (test_sock < 0) {
        GTEST_SKIP() << "Cannot create test socket";
    }
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = 0;
    bind(test_sock, (struct sockaddr *)&addr, sizeof(addr));
    socklen_t len = sizeof(addr);
    getsockname(test_sock, (struct sockaddr *)&addr, &len);
    int port = ntohs(addr.sin_port);
    close(test_sock);

    std::string ipport = "tcp://localhost:" + std::to_string(port);
    strncpy(attr.ip_port, ipport.c_str(), sizeof(attr.ip_port) - 1);
    attr.ip_port[sizeof(attr.ip_port) - 1] = '\0';

    FILE *f = fopen("aclshmem_bootstrap_config_store.so", "r");
    int rc = aclshmemi_bootstrap_init(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (f) {
        fclose(f);
        EXPECT_NE(rc, ACLSHMEM_INVALID_PARAM);
        if (rc == ACLSHMEM_SUCCESS) {
            aclshmemi_bootstrap_finalize();
        }
    } else {
        EXPECT_NE(rc, ACLSHMEM_INVALID_PARAM);
    }
}

TEST(BootstrapTest, InitWithIpPortInvalidHostname) {
    aclshmemx_init_attr_t attr;
    memset(&attr, 0, sizeof(attr));
    attr.my_pe = 0;
    attr.n_pes = 1;
    attr.option_attr.sockFd = -1;
    attr.option_attr.shm_init_timeout = 10;
    attr.option_attr.control_operation_timeout = 10;
    strncpy(attr.ip_port, "tcp://nonexistent_host_invalid_12345:9999", sizeof(attr.ip_port) - 1);
    attr.ip_port[sizeof(attr.ip_port) - 1] = '\0';

    int rc = aclshmemi_bootstrap_init(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    EXPECT_EQ(rc, ACLSHMEM_INVALID_PARAM);
}

TEST(BootstrapTest, InitWithIpPortIPv6Hostname) {
    aclshmemx_init_attr_t attr;
    memset(&attr, 0, sizeof(attr));
    attr.my_pe = 0;
    attr.n_pes = 1;
    attr.option_attr.sockFd = -1;
    attr.option_attr.shm_init_timeout = 10;
    attr.option_attr.control_operation_timeout = 10;

    // Auto-detect an IPv6 hostname on this system
    const char *candidates[] = {"ip6-localhost", "localhost"};
    const char *hostname = nullptr;
    for (const char *cand : candidates) {
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET6;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(cand, nullptr, &hints, &res) == 0 && res != nullptr) {
            freeaddrinfo(res);
            hostname = cand;
            break;
        }
        if (res != nullptr) freeaddrinfo(res);
    }
    if (hostname == nullptr) {
        GTEST_SKIP() << "[SKIP] no IPv6 hostname resolves on this system";
    }

    uint16_t port = find_free_port();
    if (port == 0) {
        GTEST_SKIP() << "[SKIP] could not find a free port";
    }

    std::string ipport = std::string("tcp://") + hostname + ":" + std::to_string(port);
    strncpy(attr.ip_port, ipport.c_str(), sizeof(attr.ip_port) - 1);
    attr.ip_port[sizeof(attr.ip_port) - 1] = '\0';

    int rc = aclshmemi_bootstrap_init(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    FILE *f = fopen("aclshmem_bootstrap_config_store.so", "r");
    if (f) {
        fclose(f);
        EXPECT_NE(rc, ACLSHMEM_INVALID_PARAM);
    }
    // Always finalize if init succeeded, regardless of .so presence,
    // otherwise subsequent tests will hang on leaked bootstrap state.
    if (rc == ACLSHMEM_SUCCESS) {
        aclshmemi_bootstrap_finalize();
    }
}

