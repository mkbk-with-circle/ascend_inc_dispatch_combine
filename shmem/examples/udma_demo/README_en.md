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

bash examples/udma_demo/run.sh 0 # AllGather test
bash examples/udma_demo/run.sh 1 # put signal test
```
By default, the script launches with 8 devices on a single server, sequentially starting `PE 0` to `PE 7`, and waits for all processes to exit.

3. Configure the parameters in the script command.
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type]
```

- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization. The format is tcp://`<IP_address>:<port_number>`.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used to run this sample on the current server.
- test_type: test type (optional). Value 0 (default) indicates that the all-gather test is performed. Value 1 indicates that the put signal test is performed.
