# INC 代码与数据同步清单（2026-08-10）

## 应同步

- `examples/inc/dispatch_combine/`：当前多 INC 与单 INC Dispatch/Combine 实现、API、测试和脚本。
- `examples/inc/fusion_kernel/`：ABI 13 Fusion kernel、Torch bridge、vLLM adapter、测试与运行手册。
- `examples/CMakeLists.txt`：上述 target 的构建入口。
- `docs/inc/`：当前设计、API、硬件 profile、历史正式结果和 2026-08-10 发布候选。

最新 Fusion 数据入口是
`docs/inc/report/nb-borrow/fusion_kernel_release_20260810/README.md`；2026-08-09 的完整四路径
矩阵必须一并保留，不能用新结果覆盖。

## 不应同步

- `/workspace/inc-runtime` 或宿主 `.borrow/inc-vllm-0191` 下的 build、control、launcher log。
- `.git/broken-metadata-20260810/`、Codex attachment、临时 patch、NPU profile 和 core dump。
- `__pycache__`、`*.pyc`、`*.orig`、`._*`、PID/READY/stdout/stderr 文件。

## Git 注意事项

当前工作区原本引用的本地 commit object 已丢失。本轮已将 Git HEAD 修复到
`origin/master@7965bdd0bc9c9c9b270e7508c3c86c65caa7969a`，分支名为
`inc-single-fusion-release-20260810`；工作区实现相对该远端基线会显示大量新增/修改。
不要执行 `git reset --hard`，也不要不审查地 `git add -A`。

建议在同步前只暂存明确范围：

```bash
git add examples/CMakeLists.txt \
  examples/inc/dispatch_combine \
  examples/inc/fusion_kernel \
  docs/inc
git diff --cached --check
git status --short
```

确认 staged diff 没有带入构建目录和大日志后，再由维护者提交/推送。由于原本本地基线不可恢复，
最终远端合并应按目录审查，而不是把当前全树差异当作一次普通小 patch。

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
