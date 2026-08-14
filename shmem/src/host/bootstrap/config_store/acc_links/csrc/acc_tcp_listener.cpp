/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acc_tcp_listener.h"

#include <net/if.h>
#include <netdb.h>
#include <sys/time.h>

#include "acc_common_util.h"
#include "store_net_utils.h"

namespace shm {
namespace acc {

bool AccTcpListener::PrepareSockAddr(mf_sockaddr& addr) noexcept
{
    if (addr.type == IpV4) {
        addr.ip.ipv4.sin_family = AF_INET;
        addr.ip.ipv4.sin_port = htons(listenPort_);
        if (inet_pton(AF_INET, listenIp_.c_str(), &addr.ip.ipv4.sin_addr) != 1) {
            sockaddr_storage resolved{};
            if (shm::utils::ResolveAddress(listenIp_, AF_INET, resolved) != ACLSHMEM_SUCCESS) {
                LOG_ERROR("Failed to resolve hostname " << listenIp_ << " to IPv4 address"
                    << ", refusing to fallback to INADDR_ANY");
                return false;
            }
            auto *sin = reinterpret_cast<const struct sockaddr_in *>(&resolved);
            addr.ip.ipv4.sin_addr = sin->sin_addr;
        }
    } else if (addr.type == IpV6) {
        addr.ip.ipv6.sin6_family = AF_INET6;
        addr.ip.ipv6.sin6_port = htons(listenPort_);
        if (inet_pton(AF_INET6, listenIp_.c_str(), &addr.ip.ipv6.sin6_addr) != 1) {
            sockaddr_storage resolved{};
            if (shm::utils::ResolveAddress(listenIp_, AF_INET6, resolved) != ACLSHMEM_SUCCESS) {
                LOG_ERROR("Failed to resolve hostname " << listenIp_ << " to IPv6 address"
                    << ", refusing to fallback to in6addr_any");
                return false;
            }
            auto *sin6 = reinterpret_cast<const struct sockaddr_in6 *>(&resolved);
            addr.ip.ipv6.sin6_addr = sin6->sin6_addr;
        }
    }
    return true;
}


Result AccTcpListener::CreateSocketForStrat(mf_sockaddr &addr, int &tmpFD) noexcept
{
    if (AccCommonUtil::IsValidIPv4(listenIp_)) {
        tmpFD = ::socket(AF_INET, SOCK_STREAM, 0);
        addr.type = IpV4;
    } else if (AccCommonUtil::IsValidIPv6(listenIp_)) {
        tmpFD = ::socket(AF_INET6, SOCK_STREAM, 0);
        addr.type = IpV6;
    } else {
        sockaddr_storage resolved{};
        if (shm::utils::ResolveAddress(listenIp_, AF_UNSPEC, resolved) == ACLSHMEM_SUCCESS) {
            if (resolved.ss_family == AF_INET6) {
                tmpFD = ::socket(AF_INET6, SOCK_STREAM, 0);
                addr.type = IpV6;
            } else {
                tmpFD = ::socket(AF_INET, SOCK_STREAM, 0);
                addr.type = IpV4;
            }
        } else {
            tmpFD = ::socket(AF_INET, SOCK_STREAM, 0);
            addr.type = IpV4;
        }
    }
    if (tmpFD < 0) {
        char buffer[256];
        auto ret = strerror_r(errno, buffer, sizeof(buffer));
        LOG_ERROR("Failed to create listen socket, error " << buffer <<
            ", please check if running of fd limit:" << ret);
        return ACC_ERROR;
    }
    ipType_ = addr.type;
    return ACC_OK;
}

Result AccTcpListener::BindAndListenSocket(int tmpFD, mf_sockaddr &addr) noexcept
{
    int result_bind = -1;
    if (addr.type == IpV4) {
        result_bind = ::bind(tmpFD, reinterpret_cast<struct sockaddr *>(&addr.ip.ipv4), sizeof(addr.ip.ipv4));
    } else if (addr.type == IpV6) {
        result_bind = ::bind(tmpFD, reinterpret_cast<struct sockaddr *>(&addr.ip.ipv6), sizeof(addr.ip.ipv6));
    }
    if (result_bind < 0 || ::listen(tmpFD, 200L) < 0) {
        auto errorNum = errno;
        SafeCloseFd(tmpFD);
        if (errorNum == EADDRINUSE) {
            LOG_INFO("address in use for bind listen on " << NameAndPort());
            return ACC_LINK_ADDRESS_IN_USE;
        }
        LOG_ERROR("Failed to bind or listen on " << NameAndPort() << " as errno " << strerror(errorNum));
        return ACC_ERROR;
    }
    return ACC_OK;
}

Result AccTcpListener::Start() noexcept
{
    if (started_) {
        LOG_INFO("AccTcpListener at " << NameAndPort() << " already started");
        return ACC_OK;
    }

    if (connHandler_ == nullptr) {
        LOG_ERROR("Invalid connection handler");
        return ACC_INVALID_PARAM;
    }

    /* create socket */
    mf_sockaddr addr {};
    auto tmpFD {-1};
    if (CreateSocketForStrat(addr, tmpFD) != ACC_OK) {
        return ACC_ERROR;
    }

    LOG_INFO("create socket success tmp_fd = " << tmpFD << " listen fd = " << listenFd_);
    /* assign address */
    if (!PrepareSockAddr(addr)) {
        SafeCloseFd(tmpFD);
        return ACC_ERROR;
    }
    int result_bind = -1;
    /* set option, bind and listen */
    if (listenFd_ > 0) {
        LOG_INFO("use already applied listen fd=" << listenFd_ << " close tmp_fd=" << tmpFD);
        SafeCloseFd(tmpFD);
        tmpFD = listenFd_;  // use already applied fd
        result_bind = 0;
    } else {
        if (reusePort_) {
            int flags = 1;
            if (::setsockopt(tmpFD, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<void *>(&flags), sizeof(flags)) < 0) {
                SafeCloseFd(tmpFD);
                LOG_ERROR("Failed to set reuse port of " << NameAndPort() << " as " << strerror(errno));
                return ACC_ERROR;
            }
        }
        if (addr.type == IpV4) {
            result_bind = ::bind(tmpFD, reinterpret_cast<struct sockaddr *>(&addr.ip.ipv4), sizeof(addr.ip.ipv4));
        } else if (addr.type == IpV6) {
            result_bind = ::bind(tmpFD, reinterpret_cast<struct sockaddr *>(&addr.ip.ipv6), sizeof(addr.ip.ipv6));
        }
        LOG_INFO("bind ip port success" << tmpFD);
    }
    if (result_bind < 0 || ::listen(tmpFD, 200L) < 0) {
        auto errorNum = errno;
        SafeCloseFd(tmpFD);
        if (errorNum == EADDRINUSE) {
            LOG_ERROR("address in use for bind listen on " << NameAndPort());
            listenFd_ = -1;
            return ACC_LINK_ADDRESS_IN_USE;
        }
        LOG_ERROR("Failed to bind or listen on " << NameAndPort() << " as errno " << strerror(errorNum) << ", please check fd:" << listenFd_);
        listenFd_ = -1;
        return ACC_ERROR;
    }

    auto ret = StartAcceptThread();
    if (ret != ACC_OK) {
        SafeCloseFd(tmpFD);
        listenFd_ = -1;
        return ret;
    }

    listenFd_ = tmpFD;

    while (!threadStarted_.load()) {
        usleep(100L);
    }

    started_ = true;
    return ACC_OK;
}

Result AccTcpListener::StartAcceptThread() noexcept
{
    threadStarted_.store(false);

    try {
        acceptThread_ = std::thread([this]() {
            this->RunInThread();
        });
    } catch (const std::system_error& e) {
        LOG_ERROR("Failed to create accept thread: " << e.what());
        return ACC_ERROR;
    } catch (...) {
        LOG_ERROR("Unknown error creating accept thread");
        return ACC_ERROR;
    }

    std::string thrName = "AccListener";
    if (pthread_setname_np(acceptThread_.native_handle(), thrName.c_str()) != 0) {
        LOG_WARN("Failed to set thread name of oob tcp server");
    }

    return ACC_OK;
}

void AccTcpListener::Stop(bool afterFork) noexcept
{
    if (!started_) {
        return;
    }

    needStop_ = true;
    if (acceptThread_.joinable()) {
        if (afterFork) {
            acceptThread_.detach();
        } else {
            acceptThread_.join();
        }
    }
    SafeCloseFd(listenFd_, !afterFork);
    started_ = false;
}

void AccTcpListener::RunInThread() noexcept
{
    LOG_INFO("Acc listener accept thread for " << NameAndPort() << " start ...");
    threadStarted_.store(true);

    while (!needStop_) {
        try {
            struct pollfd pollEventFd = {};
            pollEventFd.fd = listenFd_;
            pollEventFd.events = POLLIN;
            pollEventFd.revents = 0;

            int rc = poll(&pollEventFd, 1, 500L);
            if (rc < 0 && errno != EINTR) {
                LOG_ERROR("Get poll event failed  , errno " << strerror(errno));
                break;
            } else if (needStop_) {
                LOG_WARN("Acc listener accept thread get stop signal, will exit...");
                break;
            } else if (rc == 0) {
                continue;
            }

            mf_sockaddr addressIn {};
            auto fd {-1};
            if (ipType_ == IpV6) {
                socklen_t len = sizeof(sockaddr_in6);
                fd = ::accept(listenFd_, reinterpret_cast<struct sockaddr *>(&addressIn.ip.ipv6), &len);
            } else if (ipType_ == IpV4) {
                socklen_t len = sizeof(sockaddr_in);
                fd = ::accept(listenFd_, reinterpret_cast<struct sockaddr *>(&addressIn.ip.ipv4), &len);
            }
            if (fd < 0) {
                LOG_WARN("Failed to accept on new socket with " << strerror(errno) << ", ignore and continue");
                continue;
            }

            int flags = 1;
            setsockopt(fd, SOL_TCP, TCP_NODELAY, &flags, sizeof(flags));

            struct timeval timeout = {ACC_LINK_RECV_TIMEOUT, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

            ProcessNewConnection(fd, addressIn);
        } catch (std::exception &ex) {
            LOG_WARN("Got exception in AccTcpListener::RunInThread, exception " << ex.what() <<
                ", ignore and continue");
        } catch (...) {
            LOG_WARN("Got unknown error in AccTcpListener::RunInThread, ignore and continue");
        }
    }

    LOG_INFO("Working thread for AccTcpStore listener at " << NameAndPort() << " exiting");
}

void AccTcpListener::FormatIPAddressAndPort(mf_sockaddr addressIn, std::string &ipPort) noexcept
{
    if (ipType_ == IpV6) {
        char ipStr[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &(addressIn.ip.ipv6.sin6_addr), ipStr, INET6_ADDRSTRLEN);
        ipPort = ipStr;
        ipPort += ":";
        ipPort += std::to_string(ntohs(addressIn.ip.ipv6.sin6_port));
    } else if (ipType_ == IpV4) {
        ipPort = inet_ntoa(addressIn.ip.ipv4.sin_addr);
        ipPort += ":";
        ipPort += std::to_string(ntohs(addressIn.ip.ipv4.sin_port));
    }
}

void AccTcpListener::ProcessNewConnection(int fd, mf_sockaddr addressIn) noexcept
{
    std::string ipPort {};
    FormatIPAddressAndPort(addressIn, ipPort);

    /* receive header */
    AccConnReq req;
    auto received = ::recv(fd, &req, sizeof(req), 0);
    if (received != sizeof(req)) {
        LOG_ERROR("Failed to read header from the socket connected from " << ipPort);
        SafeCloseFd(fd);
        return;
    }

    SSL *ssl = nullptr;
    if (enableTls_) {
        auto ret = AccTcpSslHelper::NewSslLink(true, fd, sslCtx_, ssl);
        if (ret != ACC_OK) {
            LOG_ERROR("Failed to new connection ssl link");
            SafeCloseFd(fd);
            return ;
        }
    }

    LOG_INFO("Connected from " << ipPort << " successfully, ssl " << (enableTls_ ? "enable" : "disable"));
    auto newLink = AccMakeRef<AccTcpLinkComplexDefault>(fd, ipPort, AccTcpLinkDefault::NewId(), ssl);
    if (newLink == nullptr) {
        LOG_ERROR("Failed to create listener tcp link object, probably out of memory");
        if (ssl != nullptr) {
            if (AccCommonUtil::SslShutdownHelper(ssl) != ACC_OK) {
                LOG_ERROR("shut down ssl failed!");
            }
            OpenSslApiWrapper::SslFree(ssl);
            ssl = nullptr;
        }
        SafeCloseFd(fd);
        return;
    }

    // tmpLink作为智能指针 异常分支返回时会自动析构释放资源
    auto result = connHandler_(req, newLink.Get());
    if (result != ACC_OK) {
        return;
    }

    AccConnResp resp;
    resp.result = 0;
    auto sent = newLink->BlockSend(reinterpret_cast<void *>(&resp), sizeof(resp));
    if (sent != ACC_OK) {
        LOG_WARN("Failed to connect response to " << ipPort);
    }
}
}
}