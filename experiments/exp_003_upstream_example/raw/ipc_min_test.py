# 最小 CUDA IPC 跨进程复现（判定 WSL2 环境是否支持，与 LMCache 无关）
# 进程 A: 分配 GPU 张量 -> _share_cuda_() 把句柄写入文件
# 进程 B: 读句柄 -> _new_shared_cuda() 重建张量 -> 校验值
import subprocess, sys, os, torch

HFILE = "/home/mr/projects/科研训练2.0/experiments/exp_003_upstream_example/raw/ipc_handle.bin"

if len(sys.argv) > 1 and sys.argv[1] == "child":
    torch.cuda.init()
    rec = torch.load(HFILE, weights_only=False)
    handle, size = rec["handle"], rec["size"]
    try:
        storage = torch.UntypedStorage._new_shared_cuda(0, *handle[1:])
        t = torch.empty((), device="cuda:0", dtype=torch.float32)
        t.set_(storage, 0, rec["shape"], rec["stride"])
        ok = bool((t == 3.14159).all().item())
        print(f"CHILD_IMPORT_OK value_match={ok}")
    except Exception as e:
        print(f"CHILD_IMPORT_FAIL {type(e).__name__}: {e}")
    sys.exit(0)

torch.cuda.init()
t = torch.full((1024, 1024,), 3.14159, device="cuda:0")
s = t.untyped_storage()
try:
    handle = s._share_cuda_()
except Exception as e:
    print(f"PARENT_EXPORT_FAIL {type(e).__name__}: {e}"); sys.exit(1)
torch.save({"handle": handle, "size": s.nbytes(), "shape": t.shape, "stride": t.stride()}, HFILE)
print("PARENT_EXPORT_OK")
r = subprocess.run([sys.executable, __file__, "child"], capture_output=True, text=True)
print(r.stdout.strip())
print("VERDICT:", "IPC_SUPPORTED" if "CHILD_IMPORT_OK value_match=True" in r.stdout else "IPC_BROKEN_IN_WSL")
