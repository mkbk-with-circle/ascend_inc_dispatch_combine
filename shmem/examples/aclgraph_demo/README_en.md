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
