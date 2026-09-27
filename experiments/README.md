# 实验仓库说明

选题：《面向前缀复用的加密 KV Cache 恢复：关键路径剖析与认证组织优化》→
有效性关口后收敛为**不可信外部存储下 KV 缓存完整性的端到端评测与 AAD 身份绑定修复**（见 `../去留报告_一页_20260927.md`、`../方向搜寻_新选题_20260927.md`）。

软件固定提交：LMCache dev@`9a7b605`、vLLM main@`3b4566c`；环境与复现入口见 `exp_002_env_stack/`。

## 实验索引

| 目录 | 内容 |
| --- | --- |
| exp_001_a3_prototype | a3 六项组件行为复核（AAD=None、同 salt 替换被接受等） |
| exp_002_env_stack | 环境快照（pip freeze / 驱动 / GPU） |
| exp_003_upstream_example | 上游加密 L2 闭环：lmcache_driven 被 WSL2 CUDA IPC 阻塞；engine_driven 功能闭环通过 |
| exp_004_firstweek_measure | 第一周最小原型：认证组织等待的时间线与端到端对比 |
| exp_005_psweep | P 扫描（恢复 vs 重算）+ 窗口扫描（v1，含已作废口径，保留） |
| exp_006_engine_ttft | 引擎 TTFT v1（重算组被缓存命中污染，结论作废，保留） |
| exp_008_engine_ttft_fixed | 引擎 TTFT 修正版（salt 隔离重算 + 逐请求断言，双栈） |
| exp_009_psweep_fixed | 修正版原型扫描（单次执行轮转 + 边界细化 + 批量 I/O 基线） |
| exp_010_swap | 方向 S 关口：同 salt 同长度密文互换的**可重复静默错用**（端到端） |
| exp_011_aad_fix | AAD 修复（单变量）验收 + v2 证据链 + 端到端旧帧失效；src_patch/ 为对上游的唯一增改 |
| exp_012_overhead | 原版 vs AAD 版 AEAD 开销（预注册阈值判定 + 组件基准 + 吞吐相） |

## 纪律

- 每实验：claim.md（Claim 先行）→ params.yaml（参数全标注来源）→ raw/（原始数据）→ conclusion.md；
- 逐请求证据：response_id + 服务端日志窗口 + 磁盘哈希链；lookup 命中与 load 成功分开表述；
- 已知测量陷阱与修正记录在各 conclusion.md（测量有效性关口方法）。

## 不入库的内容（见根 .gitignore）

`models/`（公开 HF 模型）、`upstream/`（第三方克隆）、`*/runtime/`（引擎日志与 L2 盘对象）、
`*.blob`（KV 数据）、`master.key`（每次运行随机生成）、`.git-local-archive/`（本仓库 2026-09-27
前的旧 git 历史，含实验密钥与大文件，仅本地保存，禁止推送）。
