# exp_012 Claim（评审意见第 4 项：AAD 开销测量）

**C-S3：在真实引擎、同模型/提示词/token 长度/硬件/engine_driven/ext4/密钥规格/请求配置下，AES-GCM 加入对象身份 AAD（aesgcm_aad）相对原版 aesgcm 的端到端开销可忽略。**

主对照：仅 aesgcm vs aesgcm_aad（明文 L2 只作背景参照，引用 exp_008，不入分母）。

## 预注册判定阈值（实验前写定，运行后不得回调）

- **组件层**：identity_aad 规范编码单次 ≤ 10 µs；AAD(约 100 B) 使 GCM 加/解密耗时增加 ≤ 5%（12–96 KB 载荷，AES-NI）。
  理由：AAD 只进 GHASH 一次，长度 ~100 B；编码是 O(1) 字节拼接与定长整数打包。
- **端到端**：每个 P 点，median 差值 (aad − orig) 的 bootstrap 95% CI（1 万次重采样）：
  - 若 |median 差| ≤ 较慢方中位的 5%，或 CI 包含 0 且 |median 差| < 1 ms → 判"可忽略"；
  - 若 CI 宽度 > 较慢方中位的 10%（噪声淹没）→ 判"**尚不能判定**"（不写"零开销"）；
  - 否则如实报告幅度与方向。
  理由：exp_008 同装置 run 间漂移约 ±10–20%，5% 阈值低于噪声下限时只能判定"不可忽略"的方向性差异。
- **吞吐相**（仅 P=5000）：并发 4 × 3 波 × max_tokens=64；判据同上（波完成时间中位差）。

## 固定条件

模型 Qwen2.5-0.5B-Instruct（本地）；P ∈ {1024, 5000, 8192}（token-id 精确下发）；
engine_driven；fs L2（ext4，WSL 原生）；AES-128-GCM；hkdf(salt) 派生；每次运行随机主密钥；
L1=2GB；gpu_mem_util=0.35；l2hit 前清 L1；逐请求断言完整命中（4/19/32 块）+ Retrieved 行；
异步存储完成时间单独记录（不把请求结束当落盘完成）。

## 平衡顺序

每点两个 serde 块（每块 10 次 l2hit + 3 次重算基线）；块顺序按点轮转：
P1024: orig→aad；P5000: aad→orig；P8192: orig→aad。每块 2 次预热请求（不入统计）。
吞吐相在 P5000 的两个块内各执行（并发 4 × 3 波），与 TTFT 分开报告。
