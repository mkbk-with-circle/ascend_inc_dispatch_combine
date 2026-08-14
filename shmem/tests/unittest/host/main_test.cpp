/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <csignal>
#include <iostream>
#include <algorithm>
#include <array>
#include <cerrno>
#include <unistd.h>
#include <vector>
#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "unittest/utils/hbm_leak_checker.h"

int test_global_ranks;
int test_gnpu_num;
char test_global_ipport[ACLSHMEM_MAX_IP_PORT_LEN];
int test_first_rank;
int test_first_npu;
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;
int32_t test_set_attr(
    int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char* ip_port, aclshmemx_init_attr_t* attributes)
{
    SHM_ASSERT_RETURN(local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_pes <= ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_pe < ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            SHM_LOG_ERROR("my_pe:" << my_pe << " ip_port is nullptr!");
            return ACLSHMEM_INVALID_VALUE;
        }
    } else {
        SHM_LOG_WARN("init with my_pe:" << my_pe << " ip_port is nullptr!");
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void*>(&default_flag_uid);
    aclshmemx_uniqueid_t* uid_args = (aclshmemx_uniqueid_t*)(attributes->comm_args);

    return ACLSHMEM_SUCCESS;
}

void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: " << n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);

    aclshmemx_init_attr_t attributes;

    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
}

void test_cross_init(int pe_id, int n_pes, uint64_t local_mem_size, aclrtStream* st)
{
    *st = nullptr;
    int status = 0;
    if (n_pes != (n_pes & (~(static_cast<unsigned int>(n_pes) - 1)))) {
        std::cout << "[TEST] input pe_size: " << n_pes << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, test_global_ipport, &attributes);
    attributes.option_attr.data_op_engine_type =
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
}

void test_multi_instance_init(int pe_id, int n_pes, uint64_t local_mem_size, aclrtStream* st, int inst_count)
{
    *st = nullptr;
    int status = 0;
    if (n_pes != (n_pes & (~(static_cast<unsigned int>(n_pes) - 1)))) {
        std::cout << "[TEST] input pe_size: " << n_pes << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);

    // 创建inst_count个shmem实例，用于UT内切换
    for (int i = 0; i < inst_count; i++) {
        aclshmemx_init_attr_t attributes;
        const char* ipport = "tcp://127.0.0.1:0";
        test_set_attr(pe_id, n_pes, local_mem_size, ipport, &attributes);
        attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE);

        // 起始ID为1
        uint64_t INSTANCE_ID = (1 + i);
        attributes.instance_id = INSTANCE_ID;
        attributes.comm_args = nullptr;
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

        EXPECT_EQ(status, 0);
    }

    *st = stream;
}

int32_t test_rdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: " << n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);
    aclshmemx_init_attr_t attributes;

    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
    return status;
}

int32_t test_sdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: " << n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);
    aclshmemx_init_attr_t attributes;

    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
    return status;
}

int32_t test_udma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: " << n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);
    aclshmemx_init_attr_t attributes;

    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
    return status;
}

void test_finalize(aclrtStream stream, int device_id)
{
    int status = aclshmem_finalize();
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclrtDestroyStream(stream), 0);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_multi_instance_finalize(aclrtStream stream, int device_id, int inst_count)
{
    // 多实例销毁
    for (int i = 0; i < inst_count; i++) {
        uint64_t INSTANCE_ID = (1 + i);
        int status = aclshmemx_finalize(INSTANCE_ID);
        EXPECT_EQ(status, 0);
    }

    EXPECT_EQ(aclrtDestroyStream(stream), 0);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count)
{
    pid_t pids[process_count];
    int status[process_count];
    for (int i = 0; i < process_count; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            func(i + test_first_rank, test_global_ranks, local_mem_size);
            raise(::testing::Test::HasFailure() ? SIGUSR2 : SIGUSR1);
        }
    }
    for (int i = 0; i < process_count; ++i) {
        waitpid(pids[i], &status[i], 0);
        if (WIFSIGNALED(status[i])) {
            int sig = WTERMSIG(status[i]);
            if (sig != SIGUSR1) {
                FAIL();
            }
        } else {
            FAIL();
        }
    }
}

static ssize_t read_full(int fd, void* buf, size_t count)
{
    size_t total = 0;
    while (total < count) {
        ssize_t n = read(fd, static_cast<char*>(buf) + total, count - total);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            return static_cast<ssize_t>(total);
        }
        total += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(total);
}

static ssize_t write_full(int fd, const void* buf, size_t count)
{
    size_t total = 0;
    while (total < count) {
        ssize_t n = write(fd, static_cast<const char*>(buf) + total, count - total);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        total += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(total);
}

void test_mutil_task_with_uid(
    std::function<void(int, int, uint64_t, aclshmemx_uniqueid_t&)> func, uint64_t local_mem_size, int process_count)
{
    if (process_count == 0) {
        return;
    }

    pid_t pids[process_count];
    int status[process_count];

    std::vector<std::array<int, 2>> dist_pipes(process_count);
    std::array<int, 2> uid_pipe;
    if (pipe(uid_pipe.data()) != 0) {
        ADD_FAILURE() << "pipe(uid_pipe) failed: " << strerror(errno);
        return;
    }

    for (int i = 0; i < process_count; ++i) {
        if (pipe(dist_pipes[i].data()) != 0) {
            ADD_FAILURE() << "pipe(dist_pipes[" << i << "]) failed: " << strerror(errno);
            close(uid_pipe[0]);
            close(uid_pipe[1]);
            for (int k = 0; k < i; ++k) {
                close(dist_pipes[k][0]);
                close(dist_pipes[k][1]);
                kill(pids[k], SIGKILL);
                waitpid(pids[k], &status[k], 0);
            }
            return;
        }
        pids[i] = fork();
        if (pids[i] < 0) {
            ADD_FAILURE() << "fork[" << i << "] failed: " << strerror(errno);
            close(uid_pipe[0]);
            close(uid_pipe[1]);
            for (int k = 0; k <= i; ++k) {
                close(dist_pipes[k][0]);
                close(dist_pipes[k][1]);
            }
            for (int k = 0; k < i; ++k) {
                kill(pids[k], SIGKILL);
                waitpid(pids[k], &status[k], 0);
            }
            return;
        }
        if (pids[i] == 0) {
            close(uid_pipe[0]);
            for (int j = 0; j < process_count; ++j) {
                close(dist_pipes[j][1]);
                if (j != i) {
                    close(dist_pipes[j][0]);
                }
            }

            aclshmemx_uniqueid_t uid{};
            if (i == 0) {
                setenv("SHMEM_UID_SESSION_ID", "127.0.0.1:8666", 1);
                int ret = aclshmemx_get_uniqueid(&uid);
                if (ret != ACLSHMEM_SUCCESS || uid.version != ACLSHMEM_UNIQUEID_VERSION) {
                    ret = (ret != ACLSHMEM_SUCCESS) ? ret : ACLSHMEM_INVALID_VALUE;
                }
                if (write_full(uid_pipe[1], &ret, sizeof(ret)) != sizeof(ret) ||
                    write_full(uid_pipe[1], &uid, sizeof(uid)) != sizeof(uid)) {
                    close(uid_pipe[1]);
                    raise(SIGUSR2);
                }
                close(uid_pipe[1]);
                func(i + test_first_rank, test_global_ranks, local_mem_size, uid);
                raise(::testing::Test::HasFailure() ? SIGUSR2 : SIGUSR1);
            }
            close(uid_pipe[1]);

            if (read_full(dist_pipes[i][0], &uid, sizeof(uid)) != sizeof(uid)) {
                close(dist_pipes[i][0]);
                raise(SIGUSR2);
            }
            close(dist_pipes[i][0]);

            func(i + test_first_rank, test_global_ranks, local_mem_size, uid);
            raise(::testing::Test::HasFailure() ? SIGUSR2 : SIGUSR1);
        }
    }

    close(uid_pipe[1]);
    close(dist_pipes[0][1]);
    for (int i = 0; i < process_count; ++i) {
        close(dist_pipes[i][0]);
    }

    aclshmemx_uniqueid_t uid{};
    int get_uid_ret = ACLSHMEM_SUCCESS;
    if (read_full(uid_pipe[0], &get_uid_ret, sizeof(get_uid_ret)) != sizeof(get_uid_ret) ||
        read_full(uid_pipe[0], &uid, sizeof(uid)) != sizeof(uid)) {
        ADD_FAILURE() << "read uid_pipe failed: " << strerror(errno);
        close(uid_pipe[0]);
        for (int i = 1; i < process_count; ++i) {
            close(dist_pipes[i][1]);
        }
        for (int i = 0; i < process_count; ++i) {
            kill(pids[i], SIGKILL);
            waitpid(pids[i], &status[i], 0);
        }
        return;
    }
    close(uid_pipe[0]);

    if (get_uid_ret != ACLSHMEM_SUCCESS) {
        ADD_FAILURE() << "aclshmemx_get_uniqueid failed in rank 0, ret=" << get_uid_ret;
        for (int i = 1; i < process_count; ++i) {
            close(dist_pipes[i][1]);
        }
        for (int i = 0; i < process_count; ++i) {
            kill(pids[i], SIGKILL);
            waitpid(pids[i], &status[i], 0);
        }
        return;
    }

    for (int i = 1; i < process_count; ++i) {
        if (write_full(dist_pipes[i][1], &uid, sizeof(uid)) != sizeof(uid)) {
            ADD_FAILURE() << "write dist_pipes[" << i << "] failed: " << strerror(errno);
            close(dist_pipes[i][1]);
            for (int k = i + 1; k < process_count; ++k) {
                close(dist_pipes[k][1]);
            }
            for (int k = 0; k < process_count; ++k) {
                kill(pids[k], SIGKILL);
            }
            for (int k = 0; k < process_count; ++k) {
                waitpid(pids[k], &status[k], 0);
            }
            return;
        }
        close(dist_pipes[i][1]);
    }

    for (int i = 0; i < process_count; ++i) {
        pid_t pid_ret = waitpid(pids[i], &status[i], 0);
        if (pid_ret == -1) {
            ADD_FAILURE() << "waitpid[" << i << "] failed: " << strerror(errno);
            continue;
        }
        if (WIFSIGNALED(status[i])) {
            int sig = WTERMSIG(status[i]);
            if (sig != SIGUSR1) {
                FAIL();
            }
        } else {
            FAIL();
        }
    }
}

// UID init/finalize loop helper.  In each iteration rank 0 calls
// aclshmemx_get_uniqueid() to obtain a fresh UID (with a fresh socket fd
// and magic), then distributes it to the other ranks through
// pre-created pipes.  All ranks call func(), barrier, then finalize.
void test_mutil_task_uid_loop(
    std::function<void(int, int, uint64_t, aclshmemx_uniqueid_t&)> func, uint64_t local_mem_size, int process_count,
    int loop_count)
{
    if (process_count == 0) {
        return;
    }

    const char* session_id = "127.0.0.1:8666";

    pid_t pids[process_count];
    int status[process_count];

    // pipes for rank-0 → other-ranks UID distribution (intra-iteration)
    std::vector<std::array<int, 2>> uid_to_rank(process_count);
    for (int i = 1; i < process_count; ++i) {
        if (pipe(uid_to_rank[i].data()) != 0) {
            ADD_FAILURE() << "pipe(uid_to_rank[" << i << "]) failed: " << strerror(errno);
            return;
        }
    }

    for (int i = 0; i < process_count; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            ADD_FAILURE() << "fork[" << i << "] failed: " << strerror(errno);
            for (int k = 0; k < i; ++k) {
                kill(pids[k], SIGKILL);
                waitpid(pids[k], &status[k], 0);
            }
            for (int j = 1; j < process_count; ++j) {
                close(uid_to_rank[j][0]);
                close(uid_to_rank[j][1]);
            }
            return;
        }
        if (pids[i] == 0) {
            // --- child ---
            if (i == 0) {
                setenv("SHMEM_UID_SESSION_ID", session_id, 1);
                for (int j = 1; j < process_count; ++j) {
                    close(uid_to_rank[j][0]);
                }
            } else {
                for (int j = 1; j < process_count; ++j) {
                    if (j == i) {
                        close(uid_to_rank[j][1]);
                    } else {
                        close(uid_to_rank[j][0]);
                        close(uid_to_rank[j][1]);
                    }
                }
            }

            int32_t device_id = i % test_gnpu_num + test_first_npu;
            aclInit(nullptr);
            aclrtSetDevice(device_id);
            aclshmemx_set_conf_store_tls(false, nullptr, 0);

            for (int loop = 0; loop < loop_count; ++loop) {
                aclshmemx_uniqueid_t uid{};
                if (i == 0) {
                    int ret = aclshmemx_get_uniqueid(&uid);
                    if (ret != ACLSHMEM_SUCCESS || uid.version != ACLSHMEM_UNIQUEID_VERSION) {
                        raise(SIGUSR2);
                    }
                    for (int j = 1; j < process_count; ++j) {
                        if (write_full(uid_to_rank[j][1], &uid, sizeof(uid)) != sizeof(uid)) {
                            raise(SIGUSR2);
                        }
                    }
                } else {
                    if (read_full(uid_to_rank[i][0], &uid, sizeof(uid)) != sizeof(uid)) {
                        raise(SIGUSR2);
                    }
                }
                func(i + test_first_rank, test_global_ranks, local_mem_size, uid);
            }

            aclrtResetDevice(device_id);
            aclFinalize();

            if (i == 0) {
                for (int j = 1; j < process_count; ++j) {
                    close(uid_to_rank[j][1]);
                }
            } else {
                close(uid_to_rank[i][0]);
            }
            raise(::testing::Test::HasFailure() ? SIGUSR2 : SIGUSR1);
        }
    }

    for (int j = 1; j < process_count; ++j) {
        close(uid_to_rank[j][0]);
        close(uid_to_rank[j][1]);
    }

    for (int i = 0; i < process_count; ++i) {
        pid_t pid_ret = waitpid(pids[i], &status[i], 0);
        if (pid_ret == -1) {
            ADD_FAILURE() << "waitpid[" << i << "] failed: " << strerror(errno);
            continue;
        }
        if (WIFSIGNALED(status[i])) {
            int sig = WTERMSIG(status[i]);
            if (sig != SIGUSR1) {
                FAIL();
            }
        } else {
            FAIL();
        }
    }
}

int main(int argc, char** argv)
{
    int idx = 1;
    test_global_ranks = std::atoi(argv[idx++]);
    std::copy_n(argv[idx++], ACLSHMEM_MAX_IP_PORT_LEN - 1, test_global_ipport);
    test_global_ipport[ACLSHMEM_MAX_IP_PORT_LEN - 1] = '\0';
    test_gnpu_num = std::atoi(argv[idx++]);
    test_first_rank = std::atoi(argv[idx++]);
    test_first_npu = std::atoi(argv[idx++]);

    HbmLeakChecker::Instance().SetCardRange(test_first_npu, test_gnpu_num);
    if (!HbmLeakChecker::Instance().CheckBefore()) {
        std::cout << "WARNING: HBM baseline sampling failed, leak detection "
                  << "may be incomplete." << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    bool no_leak = HbmLeakChecker::Instance().CheckAfter();
    if (!no_leak && result == 0) {
        return 1;
    }
    return result;
}
