# INC 代码与数据同步清单（2026-08-10）

## 应同步

- `examples/inc/dispatch_combine/`：当前单 INC Dispatch/Combine 实现、API、测试和脚本。
- `examples/inc/fusion_kernel/`：ABI 13 Fusion kernel、Torch bridge、vLLM adapter、测试与运行手册。
- `examples/CMakeLists.txt`：上述 target 的构建入口。
- `docs/inc/`：当前设计、API、硬件 profile、最新资格化结果和发布候选。

同步后的 Single-INC 公开入口只有
`examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp`，唯一学习示例为
`common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp`。旧 Easy/Inference
示例和 SingleInc async/request/query/cancel/plan/stats 接口不属于当前交付面。

最新 Fusion 资格化入口是
`docs/inc/report/nb-borrow/fusion_kernel_qualified_path_20260811/README.md`；扩展 sweep 位于
`fusion_kernel_release_20260810/`，清理后的发布验证见
`release_validation_20260811/`。

## 不应同步

- `/workspace/inc-runtime` 或宿主 `.borrow/inc-vllm-0191` 下的 build、control、launcher log。
- `.git/broken-metadata-20260810/`、Codex attachment、临时 patch、NPU profile 和 core dump。
- `__pycache__`、`*.pyc`、`*.orig`、`._*`、PID/READY/stdout/stderr 文件。

## Git 交付结构

当前同步分支为 `inc-single-fusion-logical-20260811`，从
`origin/master@7965bdd0bc9c9c9b270e7508c3c86c65caa7969a` 建立，按开发逻辑拆分为：

1. `227879d`：单 INC Dispatch+Combine 的初始交付基础；
2. `6bfb81b`：ABI 13 fusion kernel；
3. `57e4475`：vLLM-Ascend 接入；
4. `9343813`：CMake 构建接线；
5. `95f4d7d`：硬件 profile 与 gate；
6. `9bf33eb`：架构、协议与 API 文档；
7. `118e44c`：最新资格化和性能数据。

分支相对主线只修改 `examples/CMakeLists.txt`，并新增 `examples/inc/` 与 `docs/inc/`；
不包含构建目录、二进制演示稿、参考模型源码或运行日志。原始单体快照 `780334b` 只用于
核对资格化源码内容，不再作为推荐同步单位。

## 发布前门禁

```bash
# Python 适配层要求 Python >= 3.10；nb 主机默认 3.9，须在隔离容器的 3.11 中运行。
sudo docker exec -e PYTHONDONTWRITEBYTECODE=1 -w /workspace/shmem \
  montyyin_inc_vllm_0191 \
  python -m unittest discover -s \
  examples/inc/fusion_kernel/framework/vllm_ascend -p 'test_*.py'

# C++/plan tests（容器；追加 build lib，保留镜像原有 LD_LIBRARY_PATH）。
for test_bin in \
  inc_fusion_plan_tests inc_fusion_api_tests inc_fusion_compute_tests \
  inc_fusion_route_pack_tests inc_fusion_benchmark_tests; do
  sudo docker exec montyyin_inc_vllm_0191 bash -lc \
    "export LD_LIBRARY_PATH=/workspace/inc-runtime/build/shmem-cann851/lib:\
/workspace/inc-runtime/build/torch-bridge-cann851:\$LD_LIBRARY_PATH; \
exec /workspace/inc-runtime/build/shmem-cann851/bin/$test_bin"
done
```

NPU smoke 前必须再次确认 `npu-smi info` 中目标平面无其他进程。当前生产接入仍有硬阻塞：
与 native 相同 prompt 的最终 token 不一致；这个事实已经记录在最新报告中，不得在同步时删除。
