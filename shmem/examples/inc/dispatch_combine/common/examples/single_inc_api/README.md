# 单 INC 最简 API 示例

[`inc_dc_single_inc_api_example.cpp`](inc_dc_single_inc_api_example.cpp) 是一份可运行的完整程序，覆盖
operator 初始化、Dispatch fan-out、模拟 expert 计算、Combine weighted reduction、
结果校验和资源释放。业务调用面只有：

```text
create → dispatch → expert compute → combine → destroy
```

示例使用 `inc_dc_single_inc.hpp`：`batch` 自动持有 generation 和 Dispatch route，
Combine 成功后自动释放。内部 session、workspace 和 request 不属于公开 API。

示例使用 CPU mock，因此不访问 NPU；构建仍需本仓库的 CANN/BiSheng toolchain。
mock 只替代 backend、device allocator 和 stream，正式 API 生命周期与真机一致。
真机把三者分别换成 `native_composite_backend`、ACL allocator 和 `aclrtStream`，
并在 worker 初始化时通过 `BindNativeSingleIncWorkerControl` 绑定常驻 INC
service。`inc_dc_native_full_example` 是包含故障注入的底层资格化 harness，可用于
核对 bootstrap 资源，不应照抄其内部调用步骤。

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
[5/6] 正确性校验 PASS
[6/6] destroy 完成，workspace alloc/free=2/2
示例执行成功。
```

## 文件

| 文件 | 用途 |
|---|---|
| `inc_dc_single_inc_api_example.cpp` | 最短 C++ API 的完整、可运行 CPU mock 生命周期与数值示例 |
