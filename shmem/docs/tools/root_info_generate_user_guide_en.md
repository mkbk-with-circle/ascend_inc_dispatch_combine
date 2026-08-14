# Root Info Generate Tool User Guide

## Overview

**root_info_generate** is a tool that shows the topology address information about Ascend 950 NPUs. It helps you stay on top of networking structures and device configurations.

**Notes**:
- **Support**: Currently, this tool is only supported for Ascend 950.
- **Automatic**: The tool automatically generates root information by calling related APIs when SHMEM initializes. You do not need to manually run the tool.
- **Auxiliary**: This tool helps you view and understand information such as network topologies and EID addresses, making development and commissioning easier.
- **Optional**: When the SHMEM is used properly, you do not need to manually run this tool.

## Background Information

### Introduction

Ascend 950 uses the Unified Bus (UB) for hyperplane networking. Various topological networking configurations are employed across different product modalities. This module is designed to identify the endpoint addresses of each edge within diverse topologies.

#### Networking

The networking mainly uses two types of networking: mesh and CLOS. For more information, see the paper at https://arxiv.org/abs/2503.20377.

##### Mesh

Each NPU is directly connected to every other NPU through a dedicated physical link. As a result, each pair of NPUs has its own independent pair of communication addresses.

**Example**: On a single NPU board with 8 NPUs, there are `8 × 7/2 = 28` physical links, corresponding to 28 communication address pairs.

**Features**:
- Direct point-to-point links, no switch chip required
- Highest communication quality with the lowest latency
- Address count: `N × (N – 1)/2` address pairs (where `N` is the number of NPUs)

##### CLOS

Any two NPUs communicate through forwarding by a switch chip. As a result, each NPU only requires a single address.

**Multi-plane design**: In typical networking, CLOS networks are often divided into multiple planes for reliability. Each plane has its own address.

**Example**: In a liquid-cooled POD, an NPU uses two independent logical ports to connect to two separate network planes.

**Features**:
- Forwarding through switch chips
- Number of addresses equals the number of planes
- Traffic load balancing and redundancy enabled

##### Network Planning

#### Network Layers

In Ascend 950, networks are divided into multiple layers based on communication quality and scope.

| Network Layer| Description|
|:-------|:-----------|
| **Layer 0**| Communication quality is the highest and latency is the lowest. This layer typically uses mesh networking, including full-mesh networks within a single NPU board and chassis-level networks in POD configurations.|
| **Layer 1**| Communication quality is the second-highest with medium latency. This layer uses CLOS networking, covering a larger communication scope at the supernode level while still within scale-up boundaries.|
| **Layer 2**| Communication quality is the lowest. This layer also uses CLOS networking, spanning the entire cluster, and is primarily implemented with RoCE or UBoE scale-out networks.|

**Comparison**:

- **Layer 0**: intra-board/intra-chassis communication, with the highest quality
- **Layer 1**: intra-supernode communication, with medium quality
- **Layer 2**: inter-cluster communication, with the lowest quality but the widest scope

#### Network Addresses

| Network Layer| Addressing Plan|
|:-------|:-----------|
| **Layer 0**| Mesh networking dominates, so multiple communication address pairs exist. In topology information, each port address on each NPU is represented. **Address type: EID**|
| **Layer 1**| Addresses are assigned per network plane. In multi-plane networking, the number of addresses equals the number of planes. Collective communication distributes traffic across planes. **Address type: EID**|
| **Layer 2**| The addressing plan is the same as that for layer 1. **Address type: IP address**|

**Address types**:

- **Extended ID (EID)**: extended identifier, used for UB network communication at layer 0 and layer 1.
- **IP address**: used for scale-out network communication (such as RoCE and UBoE) at layer 2.

#### EID Address

An EID address is 16 bytes (128 bits) long and is represented as a 32-character hexadecimal string in the JSON output.

```
Example: "000000000000006000100000dfdf008b"
Length: 32 hexadecimal characters (16 bytes)
```

**EID structure** (for reference):
- Includes a subnet_prefix (subnet prefix).
- Includes an interface_id (interface identifier).
- Used for endpoint addressing in UB networks.

## Functions

- **Topology query**: You can obtain the complete topology address information about a specified NPU.
- **Buffer detection**: The buffer size required for topology data is obtained automatically.
- **File path query**: The path to a topology configuration file is provided.
- **JSON output**: Topology information is displayed in a structured JSON format.

## Prerequisites

### System Requirements
- **SoC type**: Ascend 950
- **Platform**: Linux
- **Dependent libraries**: libdcmi.so driver library and c_sec security library

### Compilation Requirements
- CMake 3.16+
- C++17 compiler
- The Ascend driver has been installed.

## Installation Description

### Source Code Compilation

```bash
# Clone the code repository.
git clone <repository_address>
cd shmem

# Build with the Ascend 950 SoC.
bash scripts/build.sh -soc_type Ascend950

# Tool location:
# - After build (available immediately): build/bin/root_info_generate
# - After installation (full installation): install/shmem/bin/root_info_generate
```

### Installation Verification

The tool can be used immediately after build:
```bash
# Check the tool generated after the build.
ls -l build/bin/root_info_generate

# Run the tool (after the build).
./build/bin/root_info_generate <physical_ID>

# Alternatively, use the installed tool (after full build and installation).
./install/shmem/bin/root_info_generate <physical_ID>
```

### Packaging and Installation

After full build and installation, an installation package is generated:
```
install/aarch64/SHMEM_1.0.0_linux-aarch64.run
```

## Instructions

### Basic Syntax

```bash
./root_info_generate <physical_ID>
```

**Parameter description**:
- `physical_ID`: physical ID (an integer ranging from 0 to 63) of an NPU

### Examples

#### Example 1: Querying the NPU Whose ID Is 3

```bash
./root_info_generate 3
```

**Expected output**:
```
Generating root info for NPU with physical ID: 3
Required buffer size: 2048 bytes
topo_addr_info_get succeeded, actual size: 1329 bytes
Rank info:
{"version": "2.0","topo_file_path": "/usr/local/Ascend/driver/topo/950/atlas_850_1.json","rank_count": 1,"rank_list": [{"device_id": 3,"local_id": 3,"level_list": [{"net_layer": 0,"net_instance_id": "sp-1_srv65535","net_type": "MESH","net_attr": "","rank_addr_list": [{"addr_type": "EID","addr": "000000000000006000100000dfdf008b","plane_id": "plane_1","ports": ["1/0"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00cb","plane_id": "plane_1","ports": ["1/8"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00c3","plane_id": "plane_1","ports": ["1/7"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00ab","plane_id": "plane_1","ports": ["1/4"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00a3","plane_id": "plane_1","ports": ["1/3"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf009b","plane_id": "plane_1","ports": ["1/2"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf0093","plane_id": "plane_1","ports": ["1/1"]}]}, {"net_layer": 1,"net_instance_id": "superpod_-1","net_type": "CLOS","net_attr": "","rank_addr_list": [{"addr_type": "EID","addr": "000000000000004000100000dfdf00df","plane_id": "plane_1","ports": ["1/5","1/6"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf005f","plane_id": "plane_0","ports": ["0/4","0/5","0/6","0/7"]}]}]}]}
Topology file path: /usr/local/Ascend/driver/topo/950/atlas_850_1.json
Root info generation completed successfully
```

#### Example 2: Querying Other NPUs

```bash
./root_info_generate 0
./root_info_generate 5
```

#### Example 3: Querying an NPU Using an Invalid Physical ID

```bash
./root_info_generate 100
```

**Output error**:
```
Generating root info for NPU with physical ID: 100
Error: topo_addr_info_get_size failed with ret=-1
```

## Output Interpretation

### Output Structure for Successful Execution

After the tool is successfully executed, the following information is displayed:

1. **Buffer size**
   ```
   Required buffer size: XXX bytes
   ```

2. **Actual data size**
   ```
   topo_addr_info_get succeeded, actual size: XXX bytes
   ```

3. **JSON topology data** (compact format. You are advised to use jq to format the data.)
   - The output is a JSON string that contains topology address information.
   - You can use `jq '.'` to format the output to make it easier to read.

4. **Topology file path**
   ```
   Topology file path: /usr/local/Ascend/driver/topo/950/atlas_850_1.json
   ```

### Error Handling

If the tool fails to be executed, an error message is displayed:
```
Error: topo_addr_info_get_size failed with ret=-1
```

Common error causes:
- Physical ID out of range (valid range 0–63)
- Driver not initialized or device inaccessible
- Insufficient device permissions

## Common Problems and Solutions

### Problem 1: The Tool Cannot Be Found

**Symptom**:
```bash
./root_info_generate: command not found
```

**Solution**:
1. Build with soc_type set to Ascend 950
2. Verify the installation location: `build/bin/` or `install/shmem/bin/`
3. Check the build log to ensure that the tool has been generated.

### Problem 2: Missing Library Dependency

**Symptom**:
```bash
error while loading shared libraries: libshmem_rootinfo.so: cannot open shared object file
```

**Solution**:
```bash
# Set a library path.
export LD_LIBRARY_PATH=install/shmem/lib:$LD_LIBRARY_PATH

# Alternatively, copy the library to a standard location.
cp install/shmem/lib/*.so /usr/local/lib/
ldconfig
```

### Problem 3: The Driver Is Not Initialized

**Symptom**:
```bash
Error: topo_addr_info_get_size failed with ret=-1
```

**Solution**:
1. Check the Ascend driver installation: `ls /usr/local/Ascend/`
2. Initialize the driver by running a driver initialization script.
3. Verify NPUs: `ls /dev/davinci*`

### Problem 4: Permission Denied

**Symptom**:
```bash
Error: cannot access /dev/davinci0
```

**Solution**:
```bash
# Check permissions for devices.
ls -l /dev/davinci*

# Add a user to a corresponding group.
sudo usermod -a -G ascend <username>

# Alternatively, run the tool with appropriate permissions.
sudo ./root_info_generate 0
```

### Problem 5: Invalid Physical ID

**Symptom**:
```bash
Error: get_mainboard_id returned null for phy_id=100
```

**Solution**:
- Use a valid physical ID ranging from 0 to `ACLSHMEMI_MAX_NPU_COUNT` – 1.
- Check available NPUs: `ls /dev/davinci*`

## Support and Debugging

### Getting Help

1. **Check logs**: Carefully review error messages.
2. **Verify installation**: Ensure that all dependencies have been installed.
3. **Test the driver**: Verify the Ascend driver functionality.
4. **Contact support**: Submit problems with detailed logs.

### Debug Mode

Enable detailed logging for debugging:
```bash
# Set up the debugging environment (if available).
export SHMEM_LOG_LEVEL=DEBUG
./root_info_generate 0
```

### Log Analysis

Check system logs to obtain additional information:
```bash
# Check driver logs.
dmesg | grep -i ascend

# Check application logs.
journalctl -u ascend-driver
```
