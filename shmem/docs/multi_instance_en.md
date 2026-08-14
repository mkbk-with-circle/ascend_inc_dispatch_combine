# Multi-instance Support

## Overview

The SHMEM multi-instance capability allows multiple independent SHMEM instances to be created within a single process. Each instance has its own communicator, memory space, and synchronization resources. The multi-instance capability is applicable to the following scenarios:

- **Coexistence of multiple communicators**: Multiple task domains in a single process require independent SHMEM contexts.
- **Resource isolation**: SHMEM resources for different tasks/modules are isolated to avoid mutual interference.
- **Elastic scaling**: Creation and release of different instances do not affect each other.
- **Dynamic instance management**: Instances are created or released by `instance_id` at runtime.

## Core Concepts

### Instance
An SHMEM instance contains a complete set of initialization states, typically including:
- Independent communicator (`Team World`)
- Independent shared memory heap (`Heap`)
- Independent synchronization resource (`Sync Pool/Counter`)
- Independent runtime snapshots of State, Bootstrap, and MemoryManager

### Instance Identifier (Instance ID)
The current repository uses `instance_id` (`uint64_t`) to identify instances. The instance lifecycle is managed by the initialization attributes and `finalize` API.

### Context Switching
SHMEM uses the **global variable swap** to switch instances. During the switching, the global state of the current instance is written back first, and then the global state of the target instance is loaded.

## Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                     Single process                     │
│  ┌───────────────────────────────────────────────────┐  │
│ │              SHMEM runtime (global)                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│  │  │  Instance 0 │  │  Instance 1 │  │  Instance N│ │  │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌──────┐ │  │  │
│  │  │  │ Team  │  │  │  │ Team  │  │  │  │ Team │ │  │  │
│  │  │  │ Heap  │  │  │  │ Heap  │  │  │  │ Heap │ │  │  │
│  │  │  │ Sync  │  │  │  │ Sync  │  │  │  │ Sync │ │  │  │
│  │  │  │ State │  │  │  │ State │  │  │  │ State│ │  │  │
│  │  │  └───────┘  │  │  └───────┘  │  │  └──────┘ │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Design Point| Current Implementation| Description|
|---|---|---|
| Instance isolation| Independent `aclshmem_context` snapshot| Each instance stores independent `state`, `host_state`, `boot_handle` and `memory_manager`.|
| Context management| Global variable swap| Switching through `aclshmemx_instance_ctx_set`|
| Instance identifier| `instance_id` (`uint64_t`) | Located in `aclshmemx_init_attr_t` and used for `aclshmem_finalize(instance_id)`|
| Device-side route| `instance_id << 1` | The kernel accesses the `state` of the corresponding instance through the hard-coded address.|

## Quick Start (Current Repository APIs)

### 1. Creating an Instance

```cpp
#include "shmem.h"

aclshmemx_init_attr_t attr;
attr.my_pe = pe_id;
attr.n_pes = pe_size;
attr.local_mem_size = 1024UL * 1024UL * 1024UL;
attr.ip_port = "tcp://x.x.x.x:0"        // Set the port to 0. A port will be allocated later.
attr.comm_args = nullptr;               // default mode
attr.instance_id = 2;

int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

### 2. Switching to the Context of the Target Instance

```cpp
aclshmemx_instance_ctx_set(2);

// Subsequent SHMEM APIs apply to instance 2.
void *ptr = aclshmem_malloc(1024);
aclshmem_free(ptr);
```

### 3. Obtaining the Current Instance

```cpp
aclshmem_instance_ctx *ctx = aclshmemx_instance_ctx_get();
// Currently, it is mainly used to read the ID. For details about other capabilities, see the previous section.
uint64_t cur_id = (ctx == nullptr) ? 0 : ctx->id;
```

### 4. Releasing the Instance

```cpp
aclshmem_finalize(2);
```
You can also refer to the `examples/multi_instance` directory.

## Specifications and Restrictions (Code Implementation)

| Project| Specifications|
|---|---|
| Type of `instance_id`| `uint64_t` |
| Default instance| `instance_id = 0` |
| Upper limit of PE| `ACLSHMEM_MAX_PES = 16384` |
| Upper limit of local memory| `ACLSHMEM_MAX_LOCAL_SIZE = 40GB` |
| Team behavior of non-0 instances| During initialization, only the world team slots are retained. `ACLSHMEM_NOT_SUPPORTED` is returned for `team split`, `translate`, and `get_config`.|
| Maximum length of `ip_port`| `ACLSHMEM_MAX_IP_PORT_LEN = 64` (including the ending `\0`)|
| Default multi-instance port range| Depending on `SHMEM_INSTANCE_PORT_RANGE=start:end`|
| Default port allocation rule| The input port must be `0`. The actual port is = `start_port + instance_id`.|
| Default available instance ID range| `instance_id <= end_port - start_port` (including the boundary)|

## Environment Variables and Running Requirements

### Kernel-side Usage Requirements
- During the build, add the `MULTI_INSTANCE` macro, similar to `add_compile_definitions(MULTI_INSTANCE)`.
- When multiple instances are used, `aclshmemx_instance_ctx_set(id)` must be placed at the beginning of the operator to specify the current instance.
- Inter-instance switchover can be performed during operator execution.

### Default Multi-instance Mode
- `SHMEM_INSTANCE_PORT_RANGE` must be set in the format of `start:end`.
- The port in `ip_port` in the initialization attribute must be `0`, for example, `(attr.ip_port = "tcp://x.x.x.x:0")`.

### Unique ID Mode
- Follow the existing `SHMEM_UID_SESSION_ID` and `SHMEM_UID_SOCK_IFNAME` rules.

## Precautions

- If instances with the same `instance_id` are being created, a WARN log is generated and returned immediately (the instance will not be created repeatedly).
- `aclshmem_finalize(instance_id)` switches to the target instance and then releases the instance to avoid incorrect release.
- The instance context-related API path is protected by a mutex lock.

## Comparison with the Single-Instance Mode

| Feature| Single-Instance Mode| Multi-instance Mode|
|---|---|---|
| Initialization identifier| `instance_id=0`| `instance_id>0` |
| Resource isolation| Global sharing| Isolation by instance|
| Context switch| Not required| `aclshmemx_instance_ctx_set` required|
| Lifecycle management| `aclshmem_finalize()` | `aclshmem_finalize(instance_id)` |
| Advanced team operations| Supported| Supported only when `instance_id=0`|

## Related Code and Documents

- Host initialization and multi-instance implementation: `src/host/init/shmem_init.cpp`
- Device multi-instance state routing: `src/device/shmemi_device_common.hpp`
- Team multi-instance restriction logic: `src/host/team/shmem_team.cpp`
- Example: `examples/multi_instance`

---

Last updated: 2026-03-25
