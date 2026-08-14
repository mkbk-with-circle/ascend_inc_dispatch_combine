# SHMEM Env Vars

## 初始化相关

使用unique id的接口初始化时，需要手动配置环境变量SHMEM_UID_SESSION_ID或者SHMEM_UID_SOCK_IFNAME，同时配置时只读SHMEM_UID_SESSION_ID。指定SHMEM_UID_SESSION_ID时需保证ip可连通，port空闲。指定SHMEM_UID_SOCK_IFNAME时需保证网口指定的网络协议地址存在。

* `SHMEM_UID_SESSION_ID`:直接指定PE 0的监听socket的ip和端口，支持格式：
  - **IPv4字面量**：`ip:port`，如 `SHMEM_UID_SESSION_ID=192.168.1.100:1234`
  - **IPv6字面量**：`[ip]:port`，如 `SHMEM_UID_SESSION_ID=[::1]:886`
  - **主机名**：`hostname:port`，如 `SHMEM_UID_SESSION_ID=my-server:5555`（主机名通过系统名称解析服务（DNS、/etc/hosts 等）解析为实际 IP 地址）
SHMEM_UID_SESSION_ID配置示例：

```bash
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
SHMEM_UID_SESSION_ID=localhost:8888
```

* `SHMEM_UID_SOCK_IFNAME`:指定PE 0的监听socket的网口名和网络层协议。支持以下两种格式：
  - **网口名+协议**：`<ifname>:<inet4|inet6>`，显式指定地址族，如 `SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4` 取 IPv4、`SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6` 取 IPv6
  - **仅网口名**：`<ifname>`，不指定协议族时自动探测该网口可用地址族（日志会打印可用的 IPv4/IPv6）；两者都可用时优先 IPv4，仅 IPv6 可用时选 IPv6，都不可用则报错。如 `SHMEM_UID_SOCK_IFNAME=eth0`
SHMEM_UID_SOCK_IFNAME配置示例：

```bash
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  取ipv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  取ipv6
SHMEM_UID_SOCK_IFNAME=eth0          自动探测可用协议（优先ipv4）
```

以上两个环境变量均未配置时自动搜索可用网口（IPv4/IPv6均可，跳过lo/docker/veth/br-/virbr/tun/tap等虚拟网口）。

### RDMA场景

使能RDMA场景下，配置TC和SL

* `HCCL_RDMA_TC`: 复用HCCL的环境变量，配置RDMA网卡的traffic class。该环境变量的取值范围为[0, 255]，且需要配置为4的整数倍，默认值: 132。[参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0089.html)
* `HCCL_RDMA_SL`: 复用HCCL的环境变量，配置RDMA网卡的service level。该值需要和网卡配置的PFC优先级保持一致，若配置不一致可能导致性能劣化。该环境变量需要配置为整数，取值范围为[0, 7], 默认值: 4。[参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0090.html)

## 多实例相关

由于每个实例都有独立的bootstrap，每个bootstrap构建时需要提供一个可用端口

> **注意**：default 模式下每个初始化实例独占端口，端口被占用期间不支持再次初始化。多个并发实例需使用不同端口，单实例重复初始化场景请确保前一次 `finalize` 已释放端口后再重新初始化，避免端口冲突。

* `SHMEM_INSTANCE_PORT_RANGE`:直接指定可用的端口范围。
SHMEM_INSTANCE_PORT_RANGE配置示例：
export SHMEM_INSTANCE_PORT_RANGE=1024:2047

## 日志相关

日志相关环境变量及详细介绍见[SHMEM日志](../debug/log_debug.md)。

## Profiling相关

SHMEM提供Profiling打点工具，通过采集系统时钟周期数并转换为实际时间，精准量化不同Block（计算核）、不同Frame（埋点 ID）下的MTE搬运性能，详细介绍请参考[在示例中使用Profiling工具](../debug/profiling.md)。

* `SHMEM_CYCLE_PROF_PE`: 用于设置需要进行Profiling采集的pe，pe_id设置范围[0，PEs-1]，需要取消采集请`unset SHMEM_CYCLE_PROF_PE`。
