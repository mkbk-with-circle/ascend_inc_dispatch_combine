# 硬件配置摘要

> 采集时间：2026-06-23（`hostname-mspqn.foreman.pxe`）  
> 数据来源：本机命令行（`lscpu` / `npu-smi` / `lspci` / `sysfs` / `ethtool` / `hccn_tool` 等）

---

## 1. 系统概览

| 项目 | 值 |
|------|-----|
| 主机名 | `hostname-mspqn.foreman.pxe` |
| OS | openEuler 22.03 (LTS-SP4) |
| 内核 | Linux 5.10.0-216.0.0.115.oe2203sp4.aarch64 |
| 架构 | aarch64 (Little Endian) |
| CPU 厂商 | HiSilicon（Kunpeng，implementer `0x48`，part `0xd02`） |
| 逻辑 CPU | **640**（4 socket × 80 core × 2 thread） |
| CPU 频率 | 400–2900 MHz |
| 主机内存 | **~2.0 TiB**（`MemTotal` ≈ 2111187796 kB） |
| NUMA 节点 | **8**（每节点约 252 GiB DRAM） |

### CPU 缓存

| 层级 | 容量 |
|------|------|
| L1d / L1i | 各 20 MiB（320 instances） |
| L2 | 400 MiB |
| L3 | 560 MiB |

---

## 2. NPU（Ascend 910）

### 2.1 规模与型号

| 项目 | 值 |
|------|-----|
| NPU 卡数 | **8**（NPU ID 0–7） |
| 每卡芯片数 | **2** × Ascend910 + 1 × Mcu |
| 逻辑 PE 总数 | **16**（Phy-ID 0–15） |
| 板卡型号 | `IT22HMDA_4_S`（Huawei） |
| PCI Device ID | `0x19E5:0xD803` |
| 驱动 / npu-smi | **25.5.2**（`ascendhal` 7.35.23） |
| 健康状态 | 全部 **OK** |

### 2.2 显存（HBM）

| 项目 | 值 |
|------|-----|
| 每 die HBM 容量 | **65536 MiB（64 GiB）** |
| 整机 HBM 总量 | **16 die × 64 GiB ≈ 1 TiB** |
| DDR | 0（本机无板载 DDR 显存） |

`npu-smi` 可监控 **HBM Bandwidth Usage Rate**；空闲时利用率接近 0%。

### 2.3 片间互联（HCCS）—— INC / HCCL 关键带宽路径

本机 **16 PE 同节点通信主要走 HCCS**，而非 RoCE 网卡。

**拓扑（`npu-smi info -t topo`）**

- 同卡两 die：**SIO**（例如 Phy-ID0 ↔ Phy-ID1）
- 跨卡 die：**HCCS_SW**（经 HCCS 交换机）
- 16×16 矩阵中，非对角元素大多为 `HCCS_SW`，同卡对为 `SIO`

**链路状态（`npu-smi info -t hccs -i 0 -c 0`，其余卡类似）**

| 项目 | 观测值 |
|------|--------|
| Health | OK |
| Lane mode | 每链路 **4 lane**（共 7 条活跃链路） |
| Link lane | 全部 `1111`（lane 均 up） |
| Link speed | 每条 **224**（npu-smi 内部单位，表示额定 HCCS 链路速率档位） |
| Retry / Error | 0 |

**HCCS 带宽探测（`npu-smi info -t hccs-bw -i 0 -c 0 -time 1`）**

空闲时各链路 tx/rx ≈ 0 GB/s；需在跑 HCCL / SHMEM 负载时用 `-time` 参数实测。

> **参考规格（厂商公开资料，非本机实测）**：Ascend 910 单芯片 HCCS 互联标称约 **200+ GB/s 级**（视链路数与方向而定）。INC 样机 9-rank / 16-rank 测试应以此为主带宽预算，而非主机 PCIe。

### 2.4 主机 PCIe（NPU 设备节点）

`/sys/bus/pci/devices/0000:9d:00.0` 等 NPU 功能节点：

| 项目 | 值 |
|------|-----|
| Max link | 2.5 GT/s PCIe × **1** |
| Current link | 2.5 GT/s PCIe × **1** |

此为 OS 可见的 **PCIe 管理/侧带功能**（理论约 ~250 MB/s 量级），**不是** HBM 数据面的 HCCS 带宽。Host↔Device 大数据搬运通常经 Ascend 专有 IO 路径，不能据此估算 SHMEM RMA 吞吐上限。

### 2.5 PE 映射（`npu-smi info -m`）

| NPU ID | Chip ID | Phy-ID | PCIe BDF（示例） |
|--------|---------|--------|------------------|
| 0 | 0 / 1 | 0 / 1 | 9d:00.0 / 9f:00.0 |
| 1 | 0 / 1 | 2 / 3 | 99:00.0 / 9b:00.0 |
| 2 | 0 / 1 | 4 / 5 | 95:00.0 / 97:00.0 |
| 3 | 0 / 1 | 6 / 7 | 91:00.0 / 93:00.0 |
| 4 | 0 / 1 | 8 / 9 | 8d:00.0 / 8f:00.0 |
| 5 | 0 / 1 | 10 / 11 | 89:00.0 / 8b:00.0 |
| 6 | 0 / 1 | 12 / 13 | 85:00.0 / 87:00.0 |
| 7 | 0 / 1 | 14 / 15 | 81:00.0 / 83:00.0 |

### 2.6 片外集群网络（HCCN / RoCE）

| 项目 | 值 |
|------|-----|
| `hccn_tool -i N -link -g` | 全部 **DOWN** |
| `hccn_tool -i 0 -bandwidth -g` | TX/RX ≈ 0 MB/s |
| `net health` | Init |

**结论**：当前为 **单机 16 PE 池化** 环境，无跨机 HCCS/RoCE 链路 UP；INC T1（9-rank）与 T2（16-rank）均在节点内完成。

---

## 3. 主机网络

| 接口 | 状态 | 速率 | 驱动 | PCIe |
|------|------|------|------|------|
| `enp196s0f0` | **UP**（`199.108.7.2/18`） | **25000 Mb/s**（25 GbE） | `hinic3` | Gen4 ×8（max 32 GT/s） |
| `enp196s0f1` | DOWN | — | hinic3 | 同 PF 另一端口 |
| `enp162s0f0/1` | DOWN | 支持 10GbaseKR | hinic3 | `a2:00.0/1` |

网卡芯片：Huawei `19e5:0222`。  
**与 INC 关系**：当前 SHMEM/HCCL 样机不依赖该网口；仅管理/部署流量。

---

## 4. 存储

| 设备 | 容量 | 类型 | PCIe 链路 |
|------|------|------|-----------|
| `nvme0n1`–`nvme3n1` | 各 **3.5 TiB** | Huawei HWE72P453T8L007N | **Gen5 32 GT/s ×4**（每盘理论单向 ~15 GB/s 量级） |
| `sda` / `sdb` | 各 447 GiB | HWE74ST3480L007N（SAS） | 经 HiSilicon SAS 3.0 HBA |
| 根分区 | 3.5 TiB LVM | `nvme2n1p3` → `vg_sda-lv_root` | — |

---

## 5. 软件栈（与性能相关）

| 组件 | 版本 / 路径 |
|------|-------------|
| CANN | **9.0.0**（`/usr/local/Ascend/cann-9.0.0`） |
| 驱动包 | 25.5.2 |
| 其他 CANN | `cann-8.5.0`（并存） |
| INC 默认 `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-9.0.0` |

---

## 6. INC 样机带宽相关要点

| 场景 | 主要带宽路径 | 本机特征 |
|------|--------------|----------|
| T1 9-rank（8W+1S） | 同节点 HCCS P2P（`aclshmem_putmem` / signal） | 16 PE 全互联，HCCS 交换机矩阵 |
| T2 16-rank（8W+8S） | 同上 | 占满 16 Phy-ID |
| Host 预置 staging / verify | PCIe/NVMe + DRAM | 4× Gen5 NVMe，DRAM ~2 TiB |
| 跨机扩展 | RoCE / HCCN | **未启用**（link DOWN） |

### 实测建议

1. **HCCS 有效带宽**：在跑 `inc_s06` / `allgather` 时用 `npu-smi info -t hccs-bw -i <card> -c <chip> -time <秒>` 采样。  
2. **HBM 带宽**：`npu-smi info -t usages` 中的 `HBM Bandwidth Usage Rate`。  
3. **勿用** NPU sysfs PCIe `2.5 GT/s ×1` 估算 device RMA 吞吐。

---

## 7. 采集命令备忘

```bash
# CPU / 内存
lscpu && free -h

# NPU 概览
npu-smi info
npu-smi info -m          # PE 映射
npu-smi info -t topo -i 0
npu-smi info -t hccs -i 0 -c 0
npu-smi info -t hccs-bw -i 0 -c 0 -time 5

# 片外网络（当前 DOWN）
hccn_tool -i 0 -link -g
hccn_tool -i 0 -bandwidth -g

# PCIe 链路（sysfs）
cat /sys/bus/pci/devices/0000:9d:00.0/{current,max}_link_{speed,width}
cat /sys/class/nvme/nvme0/device/current_link_speed

# 主机网卡
ethtool enp196s0f0
```
