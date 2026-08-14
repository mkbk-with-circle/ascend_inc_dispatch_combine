# Organizational Structure of Code
## SHMEM Organizational Structure
```
├── 3rdparty // Dependent third-party libraries
├── docs     // Documents
├── examples // Usage examples
├── include  # Header files
├── scripts // Related scripts
├── src      // Source Code
├── tests    // Test cases
```

## include
```
The header files in the `include` directory are organized as follows:
include/
├── shmem.h                                 // All SHMEM external APIs
├── device/                                 // Device-side header files
│ ├── shmem_def.h                          // Public standard APIs defined on the device side
│   ├── ub2gm/                              // Data plane low-level API ub2gm driven by AI Core, extended API + x
│   │   ├── shmem_device_rma.h              // AI Core ub2gm RMA API
│   │   └── engine/                         // Low-level API ub2gm directly driven by AI Core, extended API + x
│   │       └── shmem_device_mte.h          // AI Core directly-driven MTE API
│   ├── gm2gm/                              // Device-side data plane high-level and low-level API gm2gm driven by AI Core, extended API + x
│   │   ├── shmem_device_rma.h              // AI Core high-level RMA API
│   │   ├── shmem_device_amo.h              // AI Core high-level atomic memory operation API
│   │   ├── shmem_device_so.h               // AI Core high-level signal operation API
│   │   ├── shmem_device_cc.h               // AI Core high-level collective communication API
│   │   ├── shmem_device_p2p_sync.h         // AI Core high-level P2P synchronization API
│   │   ├── shmem_device_mo.h               // AI Core high-level memory ordering API
│   │   └── engine/                         // Low-level API gm2gm directly driven by AI Core, extended API + x
│   │       ├── shmem_device_rdma.h         // AI Core directly-driven RDMA API
│   │       ├── shmem_device_sdma.h         // AI Core directly-driven SDMA API
│   │       └── shmem_device_mte.h          // AI Core directly-driven MTE API
│   └── team/                               // Device-side team management APIs
│       └── shmem_device_team.h             // Device-side team management APIs
├── host_device/                            // Shared directory of the host and devices
│       └── shmem_common_types.h            // Common data structures
└── host/                                   // Host-side header files
    ├── shmem_host_def.h                    // APIs defined on the host side
    ├── init/                               // Host-side initialization APIs
    │   └── shmem_host_init.h               // Host initialization API
    ├── team/                               // Host-side team management APIs
    │   └── shmem_host_team.h               // Host team management API
    ├── mem/                                // Host-side memory management APIs
    │   └── shmem_host_heap.h               // Host memory management API
    ├── data_plane/                         // Host-side data plane API driven by CPU
    │   ├── shmem_host_rma.h                // High-level RMA API on the host
    │   ├── shmem_device_so.h               // High-level signal operation API on the host
    │   ├── shmem_host_cc.h                 // High-level collective communication API on the host
    │   └── shmem_host_p2p_sync.h           // Host-side high-level P2P synchronization API
    └── utils/                              // Host-side DFX APIs
        └── shmem_log.h                    // dfx-log API on the host
```
## src
```
└── src
    ├── device             // Device-side API implementation
    └── host
        ├─bootstrap      // Bootstrap API implementation
        ├─hybm            // hybrid memory
        ├─init           // Host-side initialization API implementation
        ├─mem            // Host-side memory management API implementation
        ├─python_wrapper  // Python API encapsulation
        ├─sync            // Synchronization API implementation
        ├─team           // Host-side team management API implementation
        └─transport       // Link setup-related content
```
## examples
```
└── examples
    ├── allgather                             // AllGather communication operator sample
    ├── allgather_matmul                       // AllGather+MatMul fused operator sample
    ├── allgather_matmul_padding               // Sample of AllGather+MatMul fused operator with padding
    ├── allgather_matmul_with_gather_result    // Sample of AllGather+MatMul fused operator with the gather result retained
    ├── dispatch_gmm_combine                   // Sample of GMM dispatch and result combination
    ├── dynamic_tiling                        // Sample of dynamic tiling implementation
    ├── kv_shuffle                            // KV cache shuffle sample
    ├── matmul_allreduce                      // Sample of MatMul+AllReduce fused operator
    ├── matmul_reduce_scatter                  // Sample of MatMul+ReduceScatter fused operator
    ├── matmul_reduce_scatter_padding         // Sample of MatMul+ReduceScatter fused operator with padding
    ├── rdma_demo                             // RDMA implementation sample
    ├── rdma_handlewait_test                  // Sample for testing RDMA handle wait
    ├── rdma_perftest                         // RDMA performance test sample
    └── sdma                                  // SDMA implementation sample
```
## tests
```
└── tests
    ├── examples
    └── unittest
        ├── init  // Unit test of initialization APIs
        ├── mem   // Unit test of memory management APIs
        ├── sync  // Unit test of synchronization management APIs
        └── team  // Unit test of team management APIs
```
## docs
```
└── docs
    ├── api                          // shmem api
    ├── debug                       // SHMEM debugging
    ├── deployment                  // SO file deployment and usage guide
    ├── doxygen
    ├── example                      // Samples
    ├── images                      // Images
    ├── SECURITY.md                 // Security statement
    ├── code_organization.md         // Project organization structure (this file)
    ├── compilation_build_guide.md   // Related scripts
    ├── conf.py                      // conf file
    ├── index.rst                    // Directory tree
    ├── multi_instance.md           // Multi-instance support
    ├── principles.md               // SHMEM principles
    ├── quickstart.md                // Quick start
```

## scripts
Stores related scripts.
[Functions and Usage of Scripts](compilation_build_guide_en.md)
