Instructions:

1. Build in the `shmem/` directory:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2. Run in the `shmem/` directory:
```bash
export PROJECT_ROOT=<shmem-root-directory>
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
export SHMEM_UID_SESSION_ID=127.0.0.1:8899
./build/bin/udma_atomic_add 2 0 tcp://127.0.0.1:8899 2 0 0 & # PE 0
./build/bin/udma_atomic_add 2 1 tcp://127.0.0.1:8899 2 0 0 & # PE 1
```

3. Parameters in the command line
```bash
./udma_atomic_add <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```

- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization. The format is tcp://`<IP_address>:<port_number>`.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used to run this sample on the current server.
