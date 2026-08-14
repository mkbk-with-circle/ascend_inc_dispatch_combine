## Environment Requirements
See the environment requirements in [rdma_demo](../../rdma_demo/README_en.md).

## Instructions
1. Build in the `shmem/` directory.
```bash
bash scripts/build.sh -enable_rdma -examples
```
2. Run the `bash run.sh` command in the `examples/rdma_handlewait_test/use_handlewait` directory or run the following commands in the `shmem/` directory:
- For single-server dual-device, run the following commands:
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/use_handlewait 2 0 tcp://127.0.0.1:8765 2 0 0 & # PE 0
    ./build/bin/use_handlewait 2 1 tcp://127.0.0.1:8765 2 0 0 & # PE 1
    ```
    > Note: \<shmem-root-directory\> is the root directory of the SHMEM project.
- For cross-server dual-device, run the following commands:

    Assume that the IP address of server A is `ip1` and that of server B is `ip2`.
    Run the following commands on server A:
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/use_handlewait 2 0 tcp://ip1:8765 1 0 0 # PE 0
    ```
    At the same time, run the following commands on server B:
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/use_handlewait 2 1 tcp://ip1:8765 1 1 0 # PE 1
    ```
    > Note: \<shmem-root-directory\> is the root directory of the SHMEM project.

3. Parameters in the command line
```bash
    ./use_handlewait <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```
- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization. The format is `tcp://<IP address>:<port number>`. To perform a cross-server test, set the IP address to the IP address of the host where PE0 is located.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used to run this sample on the current server.
