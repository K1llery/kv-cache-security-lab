# exp_006 Claim（去留关口第 2 项）

**C6：在上游真实引擎路径（engine_driven 传输）内，可测得 L2 加密命中请求的真实 TTFT，并与重算/内存命中及服务端恢复时点对齐。**

- 装置：exp_003 已验证组合（LMCache@9a7b605 MP server + fs L2 + aesgcm serde + vLLM@3b4566c + engine_driven）；
  编排脚本自管服务生命周期（不走上游示例脚本——其退出 cleanup 会杀服务，见 conclusion.md 排障记录）。
- 方法：流式 /v1/completions（temperature=0，max_tokens=8），TTFT=首个非空 content chunk；
  四类请求：store_cold / L1 warm / L2 hit（页缓存热）/ L2 hit（fadvise DONTNEED 受控冷读）/ recompute（全新等长 prompt）；
  服务端阶段时点从 lmcache.log 墙钟抽取。
- 结果：conclusion.md；数据 raw/engine_ttft_results.json + runtime 日志。
