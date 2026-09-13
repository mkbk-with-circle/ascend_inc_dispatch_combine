# Pull V2 acceptance plan

本文件只定义当前保留实现的验证方式。

## 必须构建

```text
inc_dc_pull_dispatch_v2_protocol
inc_dc_pull_dispatch_v2_device_kernel
inc_dc_pull_combine_v2_protocol
inc_dc_pull_combine_v2_device_kernel
inc_dc_pull_v2_api
inc_dc_pull_dispatch_v2_tests
inc_dc_pull_combine_v2_tests
inc_dc_pull_v2_api_tests
inc_dc_pull_combine_v2_contract_tests
inc_dc_pull_dispatch_v2_device_e2e
inc_dc_pull_combine_v2_npu_e2e
inc_dc_pull_v2_same_session_probe
inc_dc_pull_v2_api_example
```

构建图中出现旧 Framework/Easy/Inference、native stream、inline-route 或 Fusion
target 即视为清理回归。

## Host 与 contract gate

以下命令必须退出 0：

```bash
inc_dc_pull_dispatch_v2_tests
inc_dc_pull_combine_v2_tests
inc_dc_pull_v2_api_tests
inc_dc_pull_combine_v2_contract_tests
inc_dc_pull_v2_api_example
python3 -m unittest tests/test_pull_v2_overlap_qualification.py
```

覆盖要求：

- Dispatch：空 token、任意 top-k、重复目的、CSR、digest、容量与溢出；
- Combine：零行、W2/W4、非对称、hidden 尾部、cookie 与状态机；
- API：create/destroy、Dispatch→Combine handle、capacity、stale handle 和 dense
  top-k 乘法溢出；
- example：Dispatch、expert compute、Combine、数值检查和资源释放。

## Device correctness

每个设备 case 必须满足：

- 所有 rank 退出 0，status/cookie/generation/wave/ring-slot 一致；
- output、Journal、ACK、Completion 和 guard 全部正确；
- timeout 有限，错误不发布成功 completion；
- Source ACK 前不能复用 source/partial slot；
- Dispatch 的 Header/metadata PUT 远端完成后才发布 READY；
- Combine 的 128B READY descriptor PUT 远端完成后才发布 64B Notice。

至少运行 W2/W4 balanced、empty、ragged、非对齐 hidden 和连续 ring 复用。

## 性能口径

```text
Dispatch bandwidth =
    INC -> Worker fan-out hidden bytes / full Dispatch time

Combine bandwidth =
    Worker -> INC FP32 partial bytes / full Combine time
```

完整时间包含 READY/Notice 前的 descriptor PUT、通知、解析、payload GET、归约或
fan-out、远端完成、ACK 和 Completion。输入生成、expert compute 和结果验证在计时外。

GET+PUT 字节之和只可输出为 `aggregate_traffic_gb_s`，不得作为主带宽或与单向
link peak 比较。正式性能需 warmup >= 3、measure >= 10、全部正确、CV <= 5%。

任何来自其他 commit、不同 device library hash、旧协议或不同 workload 的数字，
必须标成历史；当前结论只写入
`docs/inc/report/nb-borrow/pull_v2_current_20260913/README.md`。
