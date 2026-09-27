# exp_003 结论（2026-09-27）：上游加密 L2 闭环已在本机完成

## Claim 与结果

固定 LMCache `9a7b605`、vLLM `3b4566c`，Qwen2.5-0.5B-Instruct，AES-128-GCM，文件 L2。使用固定版本已经提供的 `engine_driven` 传输模式后，LMCache MP 连接器完成 **写入 → 加密落盘 → 清空 L1 → L2 读取、认证解密 → vLLM 推理**。

| 关口 | 本机观测 |
| --- | --- |
| vLLM 与 LMCache server 启动 | 两者健康检查通过；两次 `/v1/completions` 返回 200 |
| 首次存储 | server 记录 `Stored 1024 tokens`；L2 有 4 个对象，每个首字节 `0x01` |
| L1 清空 | server 记录清空 4 个对象，HTTP 返回成功 |
| 第二次读取 | `Prefetch request completed (L1+L2): 4/4 retained keys (0 L1, 4 L2)`；记录 `Retrieved 1024 tokens` |
| 安全/输出 | 未出现 `InvalidTag`；两次示例打印的生成文本片段相同 |

机器可读断言在 `raw/engine_driven_verification.json`，完整日志在该文件指向的 `runtime/` 目录（运行密钥与日志不纳入 Git）。复现入口为 `run_engine_driven.sh`；脚本验证关口，任一关口失败返回非零值。

## WSL 限制的准确范围

原上游示例默认走 `lmcache_driven`：vLLM 导出 CUDA IPC 句柄，LMCache server 导入。本机在导入时出现 `cudaErrorInvalidResourceHandle`。早先第一轮另有显存不足；把 GPU 利用率下调至 0.35 后导出成功，导入依然失败。除原私有存储接口最小复现，新增 PyTorch 公开张量队列复现，仍在消费者进程出现相同错误。

因此可断言的是**本机当前 Windows 驱动 / WSL / PyTorch 组合的这条 CUDA IPC 路径失败**。[NVIDIA 当前 CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)列出 legacy CUDA IPC 支持，不能把本机结果写成“WSL2 一概不支持 CUDA IPC”。`engine_driven` 模式从 worker 侧拷贝 KV，绕过该导入关口，并未绕过加密 L2 或真实 vLLM 推理。

## 对论文的意义与下一步

03 现在提供了真实上游路径的**正确性锚点**，排除了“仅在自制原型里加密”的风险。它尚未给出可靠 TTFT、认证等待分解或逐层收益：传输模式改变，且本次只有 1024 token、两次请求。后续性能实验必须固定一种传输模式，在同模式内比较强基线与候选设计，并计入 worker 拷贝。

现阶段**不需要租算力服务器**。若在本地长前缀与强基线验证后，论文确实需要默认 CUDA IPC 路径的端到端性能对照，再使用可验证 CUDA IPC 的原生 Linux GPU 主机；先运行 `raw/ipc_public_api_test.py`，确认跨进程共享成功，再部署固定版本，并重新测所有对照。只为完成 03 的功能验证而租机没有必要。
