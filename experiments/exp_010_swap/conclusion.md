# exp_010 结论（2026-09-27 晚）：方向 S 第 1–3 天关口——**通过**

## 判定

**同 salt、同长度、不同对象的密文互换，在真实引擎（vLLM@3b4566c + LMCache@9a7b605，engine_driven，AES-128-GCM，fs L2）中造成可重复的静默错用**：请求按其自身身份从 L2 取回全部对象（逐请求断言），认证全部通过（无 InvalidTag），但服务的 KV 内容是另一对象的——输出随之改变。恢复文件字节后行为复原。翻 1 字节与截断对照全部按预期被拒绝并重算。→ **S 进入第 7–10 天的修复设计（AAD 绑定+认证清单），不触发转 F。**

## 证据链（对应用户指定的首要判据）

### (a) 确认本请求确实从 L2 取回该对象

- 每次 swap 请求都有**本请求 id** 的服务端行 `Prefetch request completed (L1+L2): 4/4 retained keys (0 L1, 4 L2)`（raw jsonl `prefetch` 字段）；
- 磁盘证据链：fC3 由 store_C 的唯一磁盘增量识别（chunk 0-2 同哈希去重共享）；互换前后逐文件 sha256 记录；试验后全部恢复（`files_rewritten_after_load=0`）。

### (b) 与无缓存重算比较 KV 状态及确定性输出

装置设计使"互换 chunk3 后的状态"与"诚实请求 P_C"在构造上等价（共享前 768 token 去重 + 最后 4 token 相同），**该等价由装置构造与输出等价联合保证，未直接转储 paged KV**；实测输出逐字符串相等：

| 条件 | 输出（temperature=0，确定性） |
| --- | --- |
| O_B_true（B，全新 salt 重算） | " Tokyo is known for its skyscrapers and bustling streets…" |
| O_B_hit（B，诚实 L2 命中） | 同 O_B_true（==true: True） |
| O_C_true / O_C_hit（C） | " Paris was bustling with activity, and the delegates…" |
| **O_B_swap（B，fC3 内容互换进其对象，×2 reps）** | **" Paris was bustling with activity…" == O_C_hit，!= O_B_true** |
| 恢复文件后 recheck ×2 | 回到 " Tokyo…"（REVERSIBLE: True） |

即：**请求 B 拿着 C 的 KV 继续推理，输出变成 C 的延续，全程无任何报错**。每条件 2 reps；对齐试验（==O_C_hit）逐字确定。

**块序号归因（2026-09-27 晚补充，annotate_evidence.py）**：用 LMCache TokenHasher（blake3 滚动前缀）离线复算并对照盘上文件名，哈希配方标定成立（c_ids 的 4 个块哈希全部命中文件，同块多 salt 变体正常）。4 个试验目标文件的块序号：对齐集 = **{chunk2, chunk3}**，chunk3 对齐与状态等价论证一致；chunk2 也对齐的原因是该文件哈希与 C 共享（chunk0-2 去重）——互换它同时污染了 C 的 chunk2，续写仍收敛到 Paris。非对齐 = {chunk0, chunk1}（输出改变但≠O_C_hit；chunk1 的两次重复输出不同——错位污染不保证确定）。结论句"两处输出等于 C 不足以反推具体块位置"已被本标定取代；"KV 状态严格相同"限定为 chunk3 的构造等价 + 输出等价。原始判定 primary_silent_misuse 已由 evidence_strong.json 的强断言取代（全部试验改变诚实输出、全部 4/4 命中断言、对齐集含 chunk3、对齐试验确定）。

### (c) 对照（应被拒绝）

| 对照 | 服务端拒绝日志行 | 输出 |
| --- | --- | --- |
| 翻 1 字节（ciphertext 区）×3 | 3–9 行（InvalidTag/fail 路径） | 全部 == O_B_true（重算，无错误输出、无崩溃） |
| 截断 64 字节 ×3 | 3–7 行 | 全部 == O_B_true |

→ 完整性失效的**检测**是好的（GCM 起效）；缺的是**身份绑定**：同 salt 下"别人的对象"能通过全部检查。

## 方法说明

- 文件识别不依赖哈希配方逆向：利用 chunk 去重（共享前缀 → 同文件）+ store 增量（store_C 恰新增 1 文件）无歧义定位 fC3；攻击对 4 个 P_B 对象逐一试验，对齐/错位两类结果都有判据。
- 预热请求的对象计入基线快照，不参与试验；每次试验后按字节恢复并核对全盘哈希。
- 命中状态一致性：每次正式请求前清 L1（HTTP /cache/clear），断言 (0 L1, 4 L2)。

## 局限

- 单点工作（1024 token / 4 chunk / 单机 fs L2 / 单 salt 对）；截断与部分落盘的系统化（含 exp_008 观察到的 8/32 现象）在下一步；跨 salt 与跨模型替换、回放（旧版本改名）尚未跑——属于第 4–6 天清单。
- 引擎内未做逐层插桩；"互换后与诚实 C 的 KV 状态逐位相同"由装置设计（共享前缀+同尾 token+同 salt）与输出等价联合保证，未直接转储 paged KV。

## 下一步（S 方向第 4–14 天）

1. 第 4–6 天：截断/部分落盘/回放（旧版本改名）变体系统化；把攻击面分类补全（哪些被检测、哪些静默）。
2. 第 7–10 天：**最小修复**——AAD = 哈希(模型, 布局版本, chunk 身份, 长度) + 本地认证清单（对象集合与版本）；在 serde 层实现后重放全部攻击，断言全部变为拒绝。
3. 第 11–14 天：修复开销（TTFT/吞吐，明文 L2 参照，n≥10，报告离散度）+ 依据总 Kill 条件做去留与成文判定。
4. 义务项：SPADA 全文核验；若走披露/PR 流程，先与导师确认。

## 数据文件

raw/swap_results.jsonl（逐请求：kind/salt/ttft/output/response_id/prefetch 断言/target_file）；
raw/verdict.json（primary_silent_misuse: true；reversible_recheck: true；bitflip/truncate_all_rejected: true）；
raw_run.log；runtime/<ts>/{lmcache.log, vllm.log, disk/, master.key(不入库)}。
