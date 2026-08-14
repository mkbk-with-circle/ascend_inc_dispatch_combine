# 单 INC 最简 API 示例

[`inc_dc_inference_api_example.cpp`](inc_dc_inference_api_example.cpp) 是一份可运行的完整程序，覆盖
operator 初始化、Dispatch fan-out、模拟 expert 计算、Combine weighted reduction、
结果校验和资源释放。业务调用面只有：

```text
create → dispatch → expert compute → combine → route_release → destroy
```

`session / plan / workspace / request` 都由 `inc_dc_single_inc_api` 管理；需要
调度交叠时再使用同一头文件里的 `_async`、`request_wait/release`，无需改 API 层。

示例使用 CPU mock，因此不访问 NPU；构建仍需本仓库的 CANN/BiSheng toolchain。
mock 只替代 backend、device allocator 和 stream，正式 API 生命周期与真机一致。
真机把三者分别换成 `native_composite_backend`、ACL allocator 和 `aclrtStream`，
并在 worker 初始化时通过 `BindNativeSingleIncWorkerControl` 绑定常驻 INC
service。`inc_dc_native_full_example` 是包含故障注入的底层资格化 harness，可用于
核对 bootstrap 资源，不应照抄其手工 Easy API 热路径。

> mock 在 `enqueue` 内完成计算，仅验证 API、数值和生命周期，不是 INC 性能或交叠证据。

### 使用 CMake 构建运行

确保已加载 CANN 环境后，从源码目录执行：

```bash
cmake -S . -B build -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target inc_dc_single_inc_api_example -j
build/bin/inc_dc_single_inc_api_example
```

成功时末尾应看到：

```text
[6/8] 正确性校验 PASS
[8/8] route/plan/session 全部释放，workspace alloc/free=2/2
示例执行成功。
```

同一个 operator 有独立的 Dispatch/Combine slot，可在两个 stream 上保留一组
D/C 并发；同一 operation 的多并发请求应使用小型 plan pool。

## 文件

| 文件 | 用途 |
|---|---|
| `inc_dc_inference_api_example.cpp` | 完整、可运行的 CPU mock 生命周期与数值示例 |
| `inference_loop.c` | scheduler adapter 的轻量 C11 示例 |
