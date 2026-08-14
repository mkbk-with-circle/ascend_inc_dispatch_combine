# SHMEM Env Vars

## Initialization Related
When the unique ID API is used for initialization, you need to manually configure the environment variable `SHMEM_UID_SESSION_ID` or `SHMEM_UID_SOCK_IFNAME`. If both are configured, only `SHMEM_UID_SESSION_ID` will be read. When specifying `SHMEM_UID_SESSION_ID`, ensure that the IP address is reachable and the port is idle. When specifying `SHMEM_UID_SOCK_IFNAME`, ensure that the network protocol address specified by the network port exists.

* `SHMEM_UID_SESSION_ID`: Directly specifies the IP address and port number of the listening socket of PE 0.
SHMEM_UID_SESSION_ID configuration example:
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886

* `SHMEM_UID_SOCK_IFNAME`: Specifies the network port name and network layer protocol of the listening socket of PE 0.
SHMEM_UID_SOCK_IFNAME configuration example:
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4 (IPv4)
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6 (IPv6)

## Multi-instance Related
Since each instance has an independent bootstrap, an available port must be provided when each bootstrap is built.

* `SHMEM_INSTANCE_PORT_RANGE`: Specifies the available port range.
SHMEM_INSTANCE_PORT_RANGE configuration example:
export SHMEM_INSTANCE_PORT_RANGE=1024:2047

## Log Related

For details about log-related environment variables, see [SHMEM Logs](../debug/log_debug_en.md).

## Profiling Related
SHMEM provides a profile data tracing tool. By collecting the number of system clock cycles and converting it into actual time, the tool accurately quantifies the MTE movement performance in different blocks (compute cores) and frames (trace point IDs). For details, see [Using the Profiler to Collect Profile Data in a Sample Project](../debug/profiling_en.md).
* `SHMEM_CYCLE_PROF_PE`: Sets PEs on which profiling is to be performed. The value range of `pe_id` is [0, PEs-1]. To cancel profiling, run the `unset SHMEM_CYCLE_PROF_PE` command.
