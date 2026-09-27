# related-work.md — 近邻工作对照与新颖性边界

> 状态：骨架（2026-09-27）。全部条目来自 a3/总览 §4.2（2026-09-26 在线核验）；定稿前需第 13–14 周复查 + 补核 SPADA 全文。

## 1. 六维对照（安全约束=使用前认证，不算"系统有某种安全机制"）

| 工作 | 认证块大小 | I/O 批量 | 按层消费 | 预取深度 | 部分前缀命中 | 使用前认证 |
| --- | --- | --- | --- | --- | --- | --- |
| LMCache 加密 MP 路径（@9a7b605） | 每对象独立帧；无系统粒度实验 | 有 | 钩子未实现（MP 路径） | 异步预取≠层数窗口 | 有前缀复用 | 对象认证成功才写入；批次后发布 |
| ObjectCache（arXiv 2605.22850） | 未见载荷 AEAD | 核心设计 | 核心设计 | 逐层预取 | 明确研究 | 未见相同约束 |
| py-kvcache（arXiv 2609.11744） | 未见载荷 AEAD | 异步直接 I/O+暂存 | 非核心卖点 | 请求等待期预加载 | 有 | 未见相同约束 |
| TZ-LLM（arXiv 2511.13717） | 解密任务拆分≠AEAD 记录粒度 | 参数恢复 I/O | 按参数/算子 | 流水调度 | 非前缀问题 | 不据"保护参数"推 KV 协议 |
| Fastrack（arXiv 2410.15240） | 明确认证分链 | 批次传输 | 非前缀 KV 逐层恢复 | 跨批次流水 | 不适用 | eager evaluation 与本题约束相反 |
| Tink Streaming AEAD | 成熟分段方案 | 交给应用 | 不理解模型层 | 交给应用 | 不理解前缀 | 提供认证后分段解密语义 |

## 2. 攻击/隐私侧（动机引用，不声称首创）

| 工作 | 出处 | 与本题关系 |
| --- | --- | --- |
| HijackKV | arXiv 2607.19957（USENIX Sec'26） | 位置无关 KV 复用劫持/投毒（单次 94%）——攻击侧 |
| "I Know What You Asked" | NDSS 2025 | KV 共享→提示泄露侧信道 |
| SafeKV | arXiv 2508.08438 | 选择性 KV 共享防时序侧信道 |
| Shadow in the Cache / KV-Cloak | arXiv 2508.09442 | KV 反演/碰撞/注入 + 可逆矩阵混淆 |

## 3. 原语与平台（防"首创"错觉）

- 地址无关加密原语：MICRO 2009（Chhabra 等，Address Independent Seed Encryption + Bonsai Merkle Trees）。
- 业界 CVM 迁移走重加密路线：Intel TDX TD Migration 规范（AES-GCM 迁移密钥、目的端重配 HKID）。
- CXL 安全分层内存切口已被占：IEEE CAL 2026-07（Seok 等，pp.291–294）。
- d-inference：已有加密 SSD 前缀缓存工程设计；kvpack（GitHub，无同行评审）：crash-safe 逐位一致 KV 重放。

## 4. 新颖性边界（a3 结论，措辞纪律）

检索支持"这是值得验证的交叉问题"，**不支持**"这是首次提出的机制"。未发现同时覆盖"外部 AEAD 加密 KV + 认证组织/粒度 + 逐层 verified-before-use + 强基线"的工作；但检索证明"没查到"，不证明"不存在"（兜底：第 13–14 周复查 + Kill 条件 4）。

## 5. 不可声称清单（每周对照）

- ❌ "首次加密 KV"（LMCache 2026-08 已上线）
- ❌ "首次逐层加载"（ObjectCache / LMCache 已有）
- ❌ "首次重叠认证与传输"（Fastrack 已有）
- ❌ "首次发现 KV 隐私风险"（NDSS'25 等已有）
- ❌ 单独成文的主张："换 AES-GCM"、"把 AAD 加上身份字段"、"首次安全 KV 恢复"
- ⚠️ SPADA（OpenReview 1FcjLaDk33）：全文未核验（浏览器验证页阻挡），a3 唯一悬置项，定稿前必须取得全文

## 6. 增补（2026-09-27 晚，方向搜寻轮核验）

- HijackKV（arXiv 2607.19957v2，USENIX Sec'26，代码公开）：论文自评重算类防御不足；其攻击者注入"合法计算的 KV"，AEAD 身份绑定不能直接防御——与本题目（不可信外部存储）威胁模型不同。
- KV-Cloak（NDSS 2026）：可逆矩阵混淆保护 KV 隐私（非完整性）。
- NVIDIA 官方指引（Structuring applications to secure the KV cache）：按用户隔离缓存（用户专属前缀标识/salt）作为共享缓存缓解。
- CVE-2026-7141（聚合站报告：vLLM KV Block Handler 弱哈希→碰撞→错误命中）：**NVD/官方源未能核实、版本信息矛盾**，仅作"缓存键身份完整性"动机注脚，不作选题支柱。
- 本地源码核验（LMCache@9a7b605）：L2 对象文件名明文编码身份（model@rank@group@chunk_hash[@salt].data）；上游无 AAD/认证清单代码或 TODO；fault_inject_l2_adapter 只建模可用性故障（丢 key），不建模完整性故障（篡改/撕裂/截断/改名/回放）。
- 待办：SPADA 全文核验（安全方向近邻义务，a3 遗留）。
