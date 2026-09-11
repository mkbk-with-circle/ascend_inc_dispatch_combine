# 源分区测试状态（2026-09-10）

2026-09-11控制路径更新：128B描述先PUT到INC，随后64B Notice；12个设备case通过，三配置同卡组平均变化均小于0.1%。见[就绪发布回归](../../../../docs/inc/report/nb-borrow/ready_push_20260911/README.md)。

环境为nb-borrow/910B2C，仅同HCCS平面W2+1INC/W4+1INC。
[当前正式结果、计量口径与CSV](../../../../docs/inc/report/nb-borrow/random_pipeline_20260910/CURRENT_RESULTS.md) 是数据入口。

- 平面A随机路由19个case全部PASS；每个方向配置3种子、各3 warmup+10 measure。
- 并行Journal校验、错误owner拒绝后恢复、延迟READY独立完成已有设备验证。
- 空源、非对齐hidden、K8与多种H及ring复用已有通过记录。
- 平面B规则路由数值通过，但W4/K4尤其Dispatch的带宽有明显波动。
- 进一步+10%与全workload无回退尚未全部达成，不把正确性PASS写成性能PASS。

D使用fan-out字节，C使用归约输入字节，除以完整方向时间。
历史多打一均值42.734/85.478 GB/s是链路参照，不是严格理论上限。
