# exp_005 Claim 与参数（去留关口第 1 项）

**C5：存在前缀长度区间使"加密恢复+继续推理"快于"全量重算"；crossover 位置对冷/热读不敏感（本机 NVMe）。**

## 参数（params.yaml 有完整来源标注）

- P ∈ {2048, 8192, 32000}（热读另含一次 18001 旧文本档）；S=69（真实分词）；上下文上限 32768 实测
- 装置继承 exp_004（AES-128-GCM 帧 [1B ver][12B iv][ct‖16B tag]、AAD=None、HKDF(cache_salt)、
  bounded Queue 预取 W=2、逐层 verified-before-use、SC1–SC4）
- 方法学修正（按复核文档 2026-09-27）：变体轮转执行、n=5、报 min/max 离散度、
  记录每变体写入成本（store_fsync_ms）、GPU 峰值显存、受控冷读（fadvise DONTNEED）
- exp_007（W 扫描）复用本装置：P=8192、W∈{1,2,4,8}、n=7、变体轮转

## 数据文件

- raw/psweep_results.json（热读 2K/8K/18K）+ raw/psweep_results_p32000.json（热读 32K）
- raw/psweep_results_cold.json（冷读 2K/8K/32K）
- raw/psweep_P8192_W{1,4,8}.json（exp_007 W 扫描）
- raw/analysis_summary.json、figures/crossover.png、run_warm.log、run_cold.log、run_wsweep.log
