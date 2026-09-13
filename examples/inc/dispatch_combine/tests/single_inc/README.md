# Pull V2 host tests

当前只保留三项直接覆盖维护路径的测试：

| 文件 | 覆盖内容 |
|---|---|
| `test_inc_dc_pull_dispatch_v2.cpp` | Source slot、路由、Journal 与边界校验 |
| `test_inc_dc_pull_combine_v2.cpp` | Notice、partial pull、归约计划与状态机 |
| `test_inc_dc_pull_v2_api.cpp` | 公共 API、handle 生命周期与错误传播 |

设备路径由 `single_inc/pull_combine` 目录中的 E2E 与 runner 验证。
