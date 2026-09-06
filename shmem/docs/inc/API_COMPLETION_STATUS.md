# Pull V2 API 补充状态（2026-09-06）

本轮不修改 Dispatch/Combine kernel。已完成前端生命周期补强和参考示例语义修正；
不能将本轮描述为完整 NPU API 接入完成。

## 已完成

- Dispatch 接收视图：DestinationRow、ExpertAssignment、expert counts。
- Session 实例身份、Batch lease 和 in-flight request 检查；旧 handle、跨 Session
  Completion、重复 Combine、未完成时 slot 复用均拒绝。
- Combine enqueue 后继续保护 slot，直到成功观察完成；timeout 保留资源。
- 修正非对称容量检查：本 rank 的发送 assignment 数不等于本 rank 接收容量，
  后端必须根据实际接收量验证；Combine 也不能用本 rank 的发送上限推断接收量。
- 参考示例：每 token/unique GPU 一份 hidden，所有 expert assignments 保留，
  B 端先加权归约，Combine 只做 partial 求和；3 个 compact rows、4 assignments、
  2 个输出 token 的 golden 检查通过。

## 仍需完成

1. 随库 NPU BackendOps：注册/分配内存、device pack、kernel launch 与完成事件。
2. 从真实 Dispatch 的设备 Journal 生成 Combine index，禁止在 Host 预先生成路由计划
   来冒充该链路。
3. Canonical partial 布局与任意 token-id 顺序的适配；重复 ID 需 owner/row 消歧。
4. BF16 partial 输入输出（FP32 累加）及设备端 send_count 支持。
5. 真实 API D→本地计算→C 的数值、随机、非对称和多代真机验证。

当前 Host 调用需串行化；Dispatch Completion 必须先成功观察，再提交其 Combine。
不同 ring slot 仍允许同时保有设备请求。按 owner 提前完成属于 kernel 调度变更，
不在本轮无数据面改动的补充范围内。

## PPT 大纲（按当前事实）

1. 整体角色：A/INC/B 与 token 往返路径。
2. Dispatch 输入准备：hidden + metadata + 每 wave 一次 READY。
3. INC Dispatch：GET、解析、目标去重、重整/fan-out、保存 Journal。
4. B 端计算与 Combine 准备：expert assignment、本地加权归约、FP32 partial、Notice。
5. INC Combine：读取 READY、GET partial、tile 累加、PUT owner。
6. 完成条件与边界：ACK、owner completion、buffer/Journal 生命周期；真实 NPU API
   后端及 D→C Journal 衔接尚未完成。
7. nb 实验配置与结果：双 HCCS 平面、W2/W4、56/112 GB/s raw、92% 指定门槛；
   设备算子的测量区间与分子单独注明，不标成公开 API 端到端性能。

尚未制作 PPT 文件；需在真实 API 接通后更新第 6 页的状态。
