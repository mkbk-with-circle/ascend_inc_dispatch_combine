<!-- 中文 / Chinese -->
# 样例介绍

使用方式:

1. 在shmem/目录编译:

    - A2/A3 平台:

    ```bash
    bash scripts/build.sh -examples
    ```

    - Ascend950 平台:

    ```bash
    bash scripts/build.sh -soc_type Ascend950 -examples
    ```

2. 在shmem/examples/allgather目录执行demo:

    ```bash
    # 完成PEs卡下的allgather同时验证精度，性能数据会输出在result.csv中。
    # PEs : [2, 4, 8]
    # TYPEs : [int32_t, float16_t, bfloat16_t]
    bash run.sh -pes ${PEs} -type ${TYPEs}
    ```

<!-- English -->
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
