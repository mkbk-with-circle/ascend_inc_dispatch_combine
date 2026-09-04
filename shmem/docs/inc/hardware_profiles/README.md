# INC 硬件 Profile

当前仓库只保留经过当前 Pull V2 代码验证的 **910b2c-nb** profile：

| Profile | 用途 |
|---|---|
| [`910b2c-nb/`](910b2c-nb/) | nb-borrow，16×910B2C、双 8-card HCCS 平面；正式验收同平面 W2/W4 |

```bash
source docs/inc/configs/910b2c-nb.env
```

新集群不得复用本目录中的绝对带宽 gate。应复制环境模板，记录拓扑、CANN、AIV、
本机 roofline 和新 sweep，并在验证完成前标为未资格化。
