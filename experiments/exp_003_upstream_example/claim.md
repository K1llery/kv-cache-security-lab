# exp_003 Claim（2026-09-27 复核）

**C3：固定版本的 LMCache server + 文件 L2 + AES-GCM serde + vLLM MP 连接器能完成真实写入、清空 L1、L2 命中、认证恢复和推理。**

- **结论：成立，但需显式启用上游支持的 `engine_driven` 传输模式。** `lmcache_driven` 默认模式在本机跨进程 CUDA IPC 导入时失败；这只限定本机当前软硬件组合，不等于 WSL2 普遍不支持 CUDA IPC。
- 复现：在 `experiments/` 目录运行 `bash exp_003_upstream_example/run_engine_driven.sh`。脚本只在忽略的 `runtime/` 中生成上游示例的临时副本，加入 server `--supported-transfer-mode engine_driven` 和 connector `lmcache.mp.mp_transfer_mode=engine_driven`，不修改固定提交源码。
- 验证记录：`raw/engine_driven_verification.json`。两次 HTTP 推理成功；第一次存储 1024 token，产生 4 个首字节 `0x01` 的 L2 对象；清空 L1 后第二次预取显示 **4/4（0 L1，4 L2）**，取回 1024 token，无 `InvalidTag`。
- 默认 CUDA IPC 路径失败的独立复核：`raw/ipc_public_api_test.py` 用 PyTorch 公开的跨进程张量接口，在生产者仍存活时复现 `cudaErrorInvalidResourceHandle`；它与 LMCache 无关。
- 本实验是**功能性路径校验**。`engine_driven` 增加 worker 侧拷贝，不能把其时间直接当作默认 CUDA IPC 路径的性能基线，也不能据此回答认证组织优化的收益。
