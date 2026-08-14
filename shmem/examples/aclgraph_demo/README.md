# 样例介绍

示例场景:

aclGraph图结构如下：
![image.png](https://raw.gitcode.com/user-images/assets/8546182/ad5e3cc9-ae42-40d0-a665-14acd664a0e7/image.png)
需要注意的是为了适配aclGraph，需要将allgather算子的magic入参从`int`值换成Device侧的地址`__gm__ int *`这样支持在aclGraph图（model）循环调用中修改。
将第一个add的输出作为第一个allgather的输入，之后将两个allgather的输出作为第二个add的输入。将其作为aclGraph的图（model），第一次循环进行图的捕获，后续循环重放捕获完成的图。通过每次循环第二个add的输出是否符合预期，来判断allgather在图中功能是否正常。

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

2. 在shmem/examples/aclgraph_demo目录执行demo:

    ```bash
    # 完成PEs卡下的aclgraph(add + allgather + allgather + add)同时验证每次精度。
    # PEs : [2, 4, 8]
    bash run.sh -pes ${PEs}
    ```
