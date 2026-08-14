Instructions:

1. Build in the `shmem/` directory.
```
bash scripts/build.sh -examples
```
2. Run the demo in the `shmem/examples/allgather` directory:
```
# Complete AllGather under PEs and verify precision. The performance data is output in the `result.csv` file.
# PEs : [2, 4, 8]
# TYPEs : [int, int32_t, float16_t, bfloat16_t]
bash run.sh -pes ${PEs} -type ${TYPEs}
```
