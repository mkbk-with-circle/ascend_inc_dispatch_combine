This example is primarily designed to simulate instance creation and operator execution across multiple instances. It performs several rounds of instance creation and release, and in each instance the AllGather operator is executed once with precision verification.

Note: The execution of this example depends on multiple ports. By default, the script reserves ports in the range 1024–2047. You can customize this range by setting the environment variable `SHMEM_INSTANCE_PORT_RANGE` in `run.sh`.

Instructions:

1. Build in the `shmem/` directory:
```
bash scripts/build.sh -examples
```
2. Run the demo in the `shmem/examples/multi_instance` directory:
```
# Complete AllGather under PEs and verify precision.
# PEs : [4, 8]
# TYPEs : [int, int32_t, float16_t, bfloat16_t]
bash run.sh -pes ${PEs} -type ${TYPEs}
```
