# nb-borrow Pull V2 environment

当前验证环境：

- 16 × Ascend 910B2C，64 GiB HBM/卡；
- 两个独立 8 卡 HCCS 平面；
- CANN 9.1.0-beta.3；
- W2：Worker 0/1，INC 2；
- W4：Worker 0--3，INC 4。

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
cmake -S . -B /tmp/shmem-final-clean-build-20260913 \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-final-clean-build-20260913 -j8
```

设备测试前必须确认整机无其他 NPU 进程，且 W+1 个 rank 位于同一 HCCS 平面。
当前代码、测试命令和构建指纹见
[当前报告](report/nb-borrow/pull_v2_current_20260913/README.md)。

其他机器配置文件只提供部署参数模板，不携带当前性能结论。
