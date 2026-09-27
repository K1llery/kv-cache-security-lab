# exp_011 结论（2026-09-27 晚）：单变量 AAD 修复——真实引擎中阻止静默错用

## 判定

**修复有效**：把 serde 从 `aesgcm`（AAD=None）换成 `aesgcm_aad`（唯一变量：每个 AEAD 调用绑定预期对象身份的 AAD），exp_010 的全部静默错用在真实引擎中被阻止，且诚实缓存行为不变。验收四项全过（raw/verdict.json）：

| 验收项 | 结果 |
| --- | --- |
| A1 诚实缓存完整命中 | ✅ O_B_hit == O_B_true，(0 L1, 4 L2) 断言成立 |
| A2 全部互换被拒 | ✅ 4 目标 × 2 reps：输出全部回到无缓存基线（Tokyo），无任何 C 内容泄漏；服务端拒绝日志 3–9 行/次 |
| A3 翻字节/截断安全回退 | ✅ 拒绝 + 重算 + 输出 == 基线 |
| A4 旧帧失效（组件级） | ✅ AAD=None 封的帧在新 serde 下 InvalidTag |

## 修复设计（评审意见第 2 项的落实）

- **唯一变量**：新增模块 `lmcache/v1/distributed/serde/aesgcm_aad.py`（本仓库对固定提交 9a7b605 的唯一源码增改；帧格式 `[1B ver][12B IV][ct‖16B tag]`、HKDF 密钥派生、异步批处理与上游完全一致），serde 工厂名 `aesgcm_aad`。
- **AAD 规范编码**：`"LMCACHE-AAD-v1" ∥ len(model)∥model ∥ len(rank i32)∥rank ∥ len(group i64)∥group ∥ len(chunk_hash)∥chunk_hash ∥ len(salt)∥salt ∥ len(pt_len u64)∥pt_len`。
- **解密端身份来源**（关键）：请求的 ObjectKey（模型/布局身份、块身份）+ 自身目标缓冲区长度（预期长度）——**不从密文或盘上文件取得**。互换的密文带的是别人身份的 AAD，在本对象的身份下解密必然 InvalidTag。
- **旧缓存处理**：选择"失效重算"（无迁移）。组件级证据 A4；引擎级行为=miss→重算（LMCache 文档口径：标签失败即加载失败/miss）。

## 证据记录的修正（评审意见第 1 项，已完成）

- exp_010 的块序号归因：用 LMCache TokenHasher（blake3 滚动前缀）离线复算并对照盘上文件名，哈希配方标定成立；对齐集 = **{chunk2, chunk3}**（chunk2 与 C 共享文件哈希，互换它同时污染 C 的 chunk2），chunk3 对齐与状态等价论证一致；非对齐 = {chunk0, chunk1}（chunk1 两次重复输出不同——错位污染不保证确定）。完整文件名/全 sha256/块序号写入 `raw/evidence_strong.json`；强断言取代 verdict 的弱口径（全部试验改变诚实输出 + 全部 4/4 命中断言 + 对齐集含 chunk3 + 对齐试验确定）。
- "KV 状态严格相同"的表述已修正：状态等价由**装置构造**（共享前缀+同尾 token+同 salt）与**输出等价**联合保证，未直接转储 paged KV。
- params.yaml 重复数口径已按实际执行修正（每目标 2 次 × 4 目标）。
- master.key 卫生：12 个实验密钥文件已移出 git 跟踪（.gitignore `**/master.key`），后续运行使用每次随机的新密钥；历史提交中出现过这些本地一次性测试密钥，对外共享材料前需清洗历史或声明其为一次性实验材料（不推翻互换结果）。

## 回放与认证清单：暂缓（评审意见第 3 项，记录设计决定）

- AAD 绑定的是"这个密文属于哪个对象"，**不判断"是否最新版本"**——NIST SP 800-38D 明确认证成功不防回放。
- 引入清单前必须：① 构造同一对象身份下旧状态造成错误的**实际场景**（当前攻击面里"回放"仅是假设，尚未端到端演示危害）；② 说明清单的可信存放、重启后的新鲜性与原子更新。在此之前不引入清单、不声称防回放。
- exp_008 的 8/32 现象定性修正：等待落盘不足导致的**部分可见**，不是撕裂写样本；撕裂写场景另行构造。

## 下一步（按评审顺序）

1. **开销测量**（成文前的最后一项）：同一模型/提示词/命中状态/硬件下，原版 AEAD vs AAD 版 AEAD 的 TTFT/吞吐/离散度（n≥10，含存储写入侧）；明文 L2 仅作背景参照。
2. 扩展一个存储后端或版本组合；核验 SPADA 全文。
3. 与导师安排上游披露（PR 形态：aesgcm_aad 或直接替换 + 文档）；成文范围判定：若贡献仅为"一处 AAD 缺失及标准修复"→ 本科论文 + 上游 PR；主张更广研究论文需先证明跨层失效规律/适用范围/安全回退语义的可推广性（当前证据不支持，须如实说明）。

## 数据文件

raw/swap_results.jsonl（逐请求）、raw/verdict.json（验收断言）、raw/run.log；
runtime/<ts>/{lmcache.log, vllm.log, disk/}（master.key 不入库）；
源码增改：upstream/LMCache/lmcache/v1/distributed/serde/aesgcm_aad.py（新增）+ serde/__init__.py（+2 行导入）。

---

# v2 证据链修复（2026-09-27 深夜，评审意见第 1 项落实后重跑；v1 原始数据保留不覆盖）

- **增量日志读取**：每次请求前记录日志光标，请求后只解析新增行——消除"请求时间减 1 秒"窗口的相邻请求错误累加（v1 的 reject_lines 计数随窗口增长即此伪影）。
- **逐请求完整证据**（raw/swap_results_v2.jsonl）：response_id、目标文件完整名+全 sha256、lookup 断言行、**load 成功证据（Retrieved 行）**、具体 InvalidTag/加载失败行、输出、回退状态。术语修正：`retained keys 4/4` 是 **lookup** 命中；load 是否成功以 Retrieved 行与拒绝行区分——失败加载不再被表述为"成功取回"。
- **A4 端到端旧帧失效（真实 L2 文件）**：旧 serde 栈把 P_B 写入 L2（同一磁盘与主密钥）→ 新 serde 栈请求 P_B → lookup 命中、`Serde task (DESERIALIZE) failed` ×4、Retrieved=0、输出回到无缓存重算基线 ✅（组件级测试仍保留为补充）。
- **A3 修正**：截断的安全回退成立（输出==基线 ×2），但拒绝发生在读取层且无错误日志——如实记录为**静默 miss** 而非 InvalidTag 拒绝；翻字节为 InvalidTag 拒绝（×2）。
- **工厂参数校验对齐**：aes_bits ∈ (128,256)、不支持 provider、空路径三类 ValueError 与原版逐项一致（消除 AAD 以外的行为差异）。
- 验收（raw/verdict_v2.json）：A1 ✅、A2 ✅（4/4×2 全拒回基线、无 C 泄漏）、A3 ✅（翻字节=拒绝；截断=静默 miss，均安全）、A4 ✅（端到端+组件）、可逆 ✅。
