# nb-borrow 单 INC vLLM-Ascend 复现实验手册

本文记录 ABI 13 发布候选的隔离环境、构建、验证和正式跑数方法。当前冻结路径为 BF16
`fused_inc`；vLLM 默认 BF16 ND 的连续权重由 kernel 直接按 row-major B 消费，不在
forward 中复制或转置。仓库报告只保留当前 ABI 13 数据。

## 1. 环境与拓扑

- 主机：`nb-borrow`，16 × Ascend 910B2C，两组 HCCS 平面 `0..7`、`8..15`。
- 平面 A：W2 为 worker NPU `0,1` + INC NPU `2`；W4 为 worker NPU `0,1,2,3` + INC
  NPU `4`。平面 B 可独立使用物理 `8..12`。
- Fusion SHMEM PE 映射为 worker `0..W-1`，INC PE 为 `W`；这是 runner profile，
  kernel/API 没有写死物理卡号。
- 容器：平面 A 为 `montyyin_inc_vllm_0191`，平面 B 为
  `montyyin_inc_vllm_0191_planeb`。第二容器内仍使用局部 device ID `0..4`；不要把物理
  `8..12` 直接传给 `torch.npu.set_device`。
- 镜像：`quay.io/ascend/vllm-ascend:v0.19.1rc1`。
- 镜像 digest：
  `quay.io/ascend/vllm-ascend@sha256:66fd1ee885ffa696e79b1cd6034d4d6a4b1bec121b3c1cec9b596ad298362caa`。
- image ID：`sha256:7caca37dfb8e4eb4fe1428dd283b516179e354a69b72cadae7fc4409e10654f3`。
- CANN 8.5.1，Python 3.11.14，PyTorch 2.9.0，torch-npu 2.9.0，vLLM 0.19.1。
- 模型：`/models/Qwen3-30B-A3B`，只读挂载。
- 源码：宿主机当前仓库只读挂载到 `/workspace/shmem`；构建、Python stage、控制文件和
  日志全部写入独立目录 `/workspace/inc-runtime`，不会修改镜像或宿主已有 Python 环境。

任何 NPU 操作前都要在宿主机执行：

```bash
npu-smi info
```

必须确认 16 张卡均无计算进程，而不只是容器可见的 0..4。不要复用别人的容器、端口或
control 目录。nb 上不做 8W+1INC 的等效链路结论或正式 gate。

## 2. 重建隔离容器

已有容器直接启动即可：

```bash
sudo docker start montyyin_inc_vllm_0191
sudo docker exec -it montyyin_inc_vllm_0191 bash
```

若容器不存在，先创建专属宿主目录，再用相同镜像和挂载重建。下面的 `SRC` 应替换为
源码仓库绝对路径；不要把源码以可写方式挂进容器。

```bash
mkdir -p /export/home/yinjinrun.montyyin/.borrow/inc-vllm-0191

sudo docker run -d --name montyyin_inc_vllm_0191 \
  --runtime ascend --privileged --network host --ipc host \
  --cgroupns host --security-opt label=disable --shm-size 64m \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --log-opt max-size=500m --log-opt max-file=5 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4 \
  -e SOC_VERSION=ascend910b1 -e TASK_QUEUE_ENABLE=1 -e OMP_NUM_THREADS=1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 \
  --device /dev/davinci3 --device /dev/davinci4 \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v SRC:/workspace/shmem:ro \
  -v /export/home/models:/models:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /var/log/npu:/var/log/npu \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /export/home/yinjinrun.montyyin/.borrow/inc-vllm-0191:/workspace/inc-runtime \
  quay.io/ascend/vllm-ascend@sha256:66fd1ee885ffa696e79b1cd6034d4d6a4b1bec121b3c1cec9b596ad298362caa \
  bash -lc 'exec sleep infinity'
```

## 3. 构建与运行环境

进入容器后设置：

```bash
export INC_RUNTIME=/workspace/inc-runtime
export INC_SHMEM_BUILD=$INC_RUNTIME/build/shmem-cann851
export INC_TORCH_BRIDGE=$INC_RUNTIME/build/torch-bridge-cann851
export PYTHONPATH=$INC_RUNTIME/stage/python:/workspace/shmem/examples/inc/fusion_kernel/framework/vllm_ascend:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$INC_SHMEM_BUILD/lib:$INC_TORCH_BRIDGE:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export HCCL_DETERMINISTIC=true
```

不要把 `/workspace/shmem/examples/inc/fusion_kernel/framework` 加到 `PYTHONPATH`；其中的
目录名会遮蔽镜像内正式安装的 `vllm_ascend` 包。

当前 SHMEM cache 的等价配置是：

```bash
cmake -S /workspace/shmem -B "$INC_SHMEM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DUSE_EXAMPLES=ON -DBUILD_PYTHON=ON \
  -DSOC_TYPE=Ascend910B -DENABLE_CANN_BUILD=OFF

cmake --build "$INC_SHMEM_BUILD" --target \
  inc_fusion_plan_tests inc_fusion_benchmark_tests inc_fusion_api_tests \
  inc_fusion_compute_tests inc_fusion_route_pack_tests inc_fusion_e2e \
  _pyshmem -j8
```

当前 Torch bridge cache 的等价配置是：

```bash
cmake -S /workspace/shmem/examples/inc/fusion_kernel/framework/vllm_ascend/native \
  -B "$INC_TORCH_BRIDGE" -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/local/python3.11.14/bin/python \
  -DTorch_DIR=/usr/local/python3.11.14/lib/python3.11/site-packages/torch/share/cmake/Torch \
  -DTORCH_NPU_PATH=/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu \
  -DROUTE_PACK_LIBRARY="$INC_SHMEM_BUILD/lib/libinc_fusion_route_pack_kernel.so" \
  -DFUSION_API_LIBRARY="$INC_SHMEM_BUILD/lib/libinc_fusion_api.so" \
  -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-8.5.1/include
cmake --build "$INC_TORCH_BRIDGE" -j8
```

不要无条件构建仓库的全部 target；当前树中存在与本功能无关、依赖缺失 golden header 的
target。上面的定向 target 是已验证路径。`stage/python/shmem` 是隔离的可导入 Python
package；若从零重建 stage，需要从 `src/python/shmem` 复制 Python 文件，并把
`$INC_SHMEM_BUILD/lib` 中 `_pyshmem*.so`、`libshmem.so`、`libshmem_utils.so`、
`aclshmem_bootstrap_config_store.so` 放在同一 package 目录。当前 stage 已保留，无需重做。

## 4. 快速验证

纯 host/plan 测试不占 NPU：

```bash
"$INC_SHMEM_BUILD/bin/inc_fusion_plan_tests"
"$INC_SHMEM_BUILD/bin/inc_fusion_api_tests"
"$INC_SHMEM_BUILD/bin/inc_fusion_compute_tests"
"$INC_SHMEM_BUILD/bin/inc_fusion_route_pack_tests"

python -m unittest discover -s \
  /workspace/shmem/examples/inc/fusion_kernel/framework/vllm_ascend \
  -p 'test_*.py'
```

NPU E2E 前再次确认全机空闲。UID 必须由参与 SHMEM world 的 worker rank 0 生成并写入本次
运行的唯一 control 目录，不能在父进程预生成。runner 已处理 `set_conf_store_tls(False, "")`、
UID、W+1 bootstrap、sidecar READY/STOPPED 和 teardown，不应另写临时 bootstrap 脚本。

## 5. 发布候选跑数

发布主路径固定为 `fused_inc`。下面命令一次加载模型后测 16/64/256/1024 token；W 取 2
或 4，W4 使用 `--worker-devices 0,1,2,3 --inc-device 4`，W2 改为 `0,1` 和 `2`。

```bash
RUN_ROOT=/workspace/inc-runtime/logs/fusion_release_YYYYMMDD
W=4
OUT="$RUN_ROOT/fused_inc_w${W}_extended"
mkdir -p "$OUT"
HCCL_IF_BASE_PORT=62471 python \
  /workspace/shmem/examples/inc/fusion_kernel/framework/vllm_ascend/run_vllm_ascend_inc_baseline.py \
  --model /models/Qwen3-30B-A3B --mode fused_inc --workers "$W" \
  --worker-devices 0,1,2,3 --inc-device 4 \
  --max-model-len 1152 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.55 --warmup 2 --measure 5 \
  --tokens-per-wave 64 --activation-waves 1 \
  --scenarios-json '[{"name":"prefill_16_b1","input_len":16,"output_len":1,"batch_size":1},{"name":"prefill_64_b1","input_len":64,"output_len":1,"batch_size":1},{"name":"prefill_256_b1","input_len":256,"output_len":1,"batch_size":1},{"name":"prefill_1024_b1","input_len":1024,"output_len":1,"batch_size":1}]' \
  --output-json "$OUT/result.json" 2>&1 | tee "$OUT/launcher.log"
```

两个平面都确认空闲时可以并行各跑一个 case，但容器、NPU 平面、`HCCL_IF_BASE_PORT` 和
output/control 目录必须互不重叠。一个平面内同一时间只跑一个正式 case。

原生 vLLM graph 路径使用相同 1024-token scheduler cap；`W=2/4` 分别重放 W2/W4：

```bash
W=4
OUT="$RUN_ROOT/native_vllm_graph_w${W}_extended"
mkdir -p "$OUT"
HCCL_IF_BASE_PORT=62141 python \
  /workspace/shmem/examples/inc/fusion_kernel/framework/vllm_ascend/run_vllm_ascend_native_baseline.py \
  --model /models/Qwen3-30B-A3B --tensor-parallel-size "$W" \
  --max-model-len 1152 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.55 --warmup 2 --measure 5 \
  --scenarios-json '[{"name":"prefill_16_b1","input_len":16,"output_len":1,"batch_size":1},{"name":"prefill_32_b1","input_len":32,"output_len":1,"batch_size":1},{"name":"prefill_64_b1","input_len":64,"output_len":1,"batch_size":1},{"name":"prefill_128_b1","input_len":128,"output_len":1,"batch_size":1},{"name":"prefill_256_b1","input_len":256,"output_len":1,"batch_size":1},{"name":"prefill_512_b1","input_len":512,"output_len":1,"batch_size":1},{"name":"prefill_1024_b1","input_len":1024,"output_len":1,"batch_size":1}]' \
  --output-json "$OUT/result.json" 2>&1 | tee "$OUT/launcher.log"
```

追加 `--enforce-eager` 并将输出目录改为 `native_vllm_eager_w${W}_extended` 可得到执行模式
参考，但生产 baseline 必须保留默认 graph 路径。发布报告中的扩展对比正是按上述命令在两个
独立平面并行完成的；两个进程使用了不同 `HCCL_IF_BASE_PORT`。
实验模式仍可用 `MODE=serial_shmem|fused_shmem|serial_inc|fused_inc` 重放；当前 direct-SHMEM
在 vLLM 0.19.1 启动生命周期中存在已知失联，失败不能冒充为性能样本。

## 6. 成功判据与故障处理

- JSON 存在、measure 样本数正确、所有 worker setup 信息一致。
- INC case 必须看到 sidecar `PREPARED → READY → STOPPING → STOPPED`，所有进程正常退出。
- 结果必须按所有 worker 的请求完成时间比较；INC 模式额外消耗一张不加载模型的 NPU，
  需同时报告固定 worker 数和总 NPU 归一化口径。
- 不能把 native vLLM 与自定义路径的巨大差值归因于 D/C 融合；只有同通信后端的
  `serial_*` 对 `fused_*` 才隔离了交叠收益。
- `output_stable=true` 只表示同一路径重复输出一致；还必须对相同 prompt 比较 native 与
  自定义路径的 token/hidden golden。当前发布数据的最终 token 尚不一致，不能跳过此 gate。
- 若失败，先确认没有遗留进程和端口冲突，再查看本 case 的 `launcher.log`。不要复用失败
  case 的 control 目录；成功或失败退出后都可删除 `control/vllm-inc-*`。
- OOM 应先降低 `gpu_memory_utilization` 或缩小容量；不要通过改变 hidden/intermediate、
  top-k、权重布局或计算 kernel 来伪造同一 baseline。

当前结果索引见 `docs/inc/report/nb-borrow/FUSION_KERNEL_RESULTS.md`。
