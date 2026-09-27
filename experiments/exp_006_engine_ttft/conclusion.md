# exp_006 结论（2026-09-27，去留关口第 2 项：engine_driven 真实 TTFT）

## Claim

**C6：在上游真实引擎路径（LMCache@9a7b605 MP server + fs L2 + aesgcm + vLLM@3b4566c，engine_driven 传输）内，L2 加密命中请求的真实 TTFT 与重算/内存命中可比，且服务端阶段时点可从日志固定。**

## 结果：成立（P=1024 无损益；P=4096 恢复慢 10–17%，n=3 起步数据）

真实流式 TTFT（首 content chunk，ms；n=3，temperature=0，max_tokens=8）：

| 工作点 | store_cold | L1 warm | L2 hit（页缓存热） | L2 hit（受控冷读） | recompute（全新等长 prompt） |
| --- | --- | --- | --- | --- | --- |
| P≈1024 | 555* | 138–222 | **145–152** | 142–185 | **132–156** |
| P≈4096 | 337 | 650–793 | **634–756** | 712–781 | **611–715** |

\* store_cold 含引擎首次请求预热，不作比较对象。

服务端日志锚点（同机墙钟对齐，来自 lmcache.log）：每次 L2 命中
`Prefetch request completed (L1+L2): 4/4 retained keys (0 L1, 4 L2) in 8.8–33.8 ms`
→ `Retrieved 1024 tokens in 0.080–0.593 s`；全程无 InvalidTag。

## 这告诉了我们什么

1. **P=1024：加密 L2 命中 ≈ 重算 ≈ L1 命中（都在 ~140–150ms）**——上游引擎在本规模下
   把认证成本完全藏在了请求处理噪声里；LMCache 官方断言在引擎级本工作点成立。
2. **P=4096：L2 命中（634–781）比重算（611–715）慢 ~10–17%（n=3，有重叠）**——
   引擎侧的 KV 安装路径（scatter_cpu_to_paged_kv，日志显示"unpinned CPU tensors
   dynamically pinning"）在恢复路径上引入了原型没有的额外拷贝/同步；这与原型 crossover
   （≈10K 才反超）方向一致：**引擎的恢复路径比原型更早失去优势**。
   若此观察在更多重复下稳定，"恢复/重算准入 + 引擎安装开销"就是 RQ1 的一个真实发现点。
3. L1 warm 在 P=4096 反而最慢（650–793）——疑似异步 store 尚未结束时的干扰
   （请求紧跟 store_cold 之后）。n=3 不足以定论；列入下轮修复（store 完成后再测 warm）。

## 排障记录（对后续实验重要的环境事实）

- 上游示例脚本自带演示流程，且**脚本退出时其 EXIT trap 会杀掉 lmcache server 与 vLLM**——
  第一、二次测量（本目录 runtime/ttft_20260927_13*）正是死于此"存活期竞态"：
  服务端日志显示 prefetch 全部成功，而客户端流在引擎被 SIGTERM 后无 content。
- 修复：编排脚本直接以 subprocess 拉起两个服务并自管生命周期（不走示例脚本），
  配置与 exp_003 已验证组合逐项一致（engine_driven、L1=2GB、util=0.35、AES-128-GCM、
  master key 16B、salt=tenant-a）。第三次运行全程稳定。

## 局限与下一步

- n=3 起步（复核文档允许的起步量）；P 只到 4096（4 chunk 对象）；P=1024 的 L2 命中
  未做受控多副本冷读交叉验证；读取/AEAD/H2D 的细粒度分解仍需 patch 插桩（列后续项，
  本轮先固定"日志可得时点"口径：prefetch 8.8–33.8ms + Retrieved 0.08–0.59s）。
- 下轮：n≥5、P∈{1K,4K,8K}、warm 测量与 store 完成解耦、把 Retrieved 窗口与 TTFT 的
  相关性做成散点（回答"服务端恢复时间有多少落在 TTFT 关键路径上"）。

## 数据文件

- raw/engine_ttft_results.json（逐请求 TTFT/total/chunks/raw SSE 头）
- raw/run_full.log；runtime/ttft_*/{lmcache.log, vllm.log, run.log, master.key(不入库), disk/}
