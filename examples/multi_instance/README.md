<!-- 中文 / Chinese -->
# 样例介绍

本example主要为了模拟多实例下的实例创建和算子执行流程，会进行数次不同实例的创建以及释放，同时每个实例都会运行一次allgather算子，并验证精度。

注意：本example执行依赖多个端口，默认执行脚本会将预留端口设置为1024-2047，可在run.sh中的SHMEM_INSTANCE_PORT_RANGE环境变量中自定义。

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

2. 在shmem/examples/multi_instance目录执行demo:

    ```bash
    # 完成PEs卡下的allgather同时验证精度。
    # PEs : [4, 8]
    # TYPEs : [int32_t]
    bash run.sh -pes ${PEs} -type ${TYPEs}
    ```

<!-- English -->
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
