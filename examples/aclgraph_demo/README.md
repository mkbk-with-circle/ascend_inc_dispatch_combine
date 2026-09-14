<!-- 中文 / Chinese -->
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

<!-- English -->
Example Scenario

The ACLGraph structure is as follows:
![image.png](https://raw.gitcode.com/user-images/assets/8546182/ad5e3cc9-ae42-40d0-a665-14acd664a0e7/image.png 'image.png')
To adapt to ACLGraph, the magic input parameter of the `allGather` operator needs to be changed from the `int` value to a device address (`__gm__ int *`). In this way, the parameter can be modified in the looped calls of the ACLGraph (model).
Use the output of the first `add` as the input to the first `allGather`. Then, feed the outputs of the two `allGather` operators into the second `add`. Treat this sequence as an aclGraph model: the first loop performs graph capture, and subsequent loops replay the captured graph. By checking whether the output of the second `add` in each loop matches the expected result, you can determine whether the `allGather` operators function correctly within the ACLGraph.

Instructions:

1. Build in the `shmem/` directory:
```
bash scripts/build.sh -examples
```
2. Run the demo in the `shmem/examples/aclgraph_demo` directory:
```
# Complete ACLGraph (add + allGather + allGather + add) under PEs and verify precision in each iteration.
# PEs : [2, 4, 8]
bash run.sh -pes ${PEs}
```
