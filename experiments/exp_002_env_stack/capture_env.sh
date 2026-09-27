#!/usr/bin/env bash
# exp_002: 环境快照（a3 明确要求"源码固定≠已验证可运行组合"——本脚本记录实际可运行组合）
set -u
EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$EXPDIR/raw"
mkdir -p "$RAW"
PY=/home/mr/projects/科研训练2.0/experiments/.venv/bin/python

{
  echo "=== date ==="; date -u
  echo "=== uname ==="; uname -a
  echo "=== python ==="; "$PY" --version
  echo "=== nvidia-smi ==="; nvidia-smi
  echo "=== torch ==="; "$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'sm_'+''.join(map(str,torch.cuda.get_device_capability(0))))"
  echo "=== lmcache ==="; git -C /home/mr/projects/科研训练2.0/experiments/upstream/LMCache log -1 --format='%H %cI %s'; "$PY" -c "import lmcache; print('lmcache import OK:', lmcache.__file__)" 2>&1
  echo "=== lmcache version ==="; "$PY" -m pip show lmcache 2>/dev/null | head -3
  echo "=== vllm ==="; git -C /home/mr/projects/科研训练2.0/experiments/upstream/vllm log -1 --format='%H %cI %s'; "$PY" -c "import vllm; print('vllm import OK:', vllm.__version__)" 2>&1
  echo "=== cryptography/OpenSSL ==="; "$PY" -c "import cryptography; from cryptography.hazmat.backends.openssl import backend; print(cryptography.__version__, backend.openssl_version_text())"
  echo "=== AES-NI ==="; grep -o -m1 ' aes '[^ ]* /proc/cpuinfo | head -1
  echo "=== disk of L2 target ==="; df -h /home/mr/projects/科研训练2.0/experiments | tail -1; findmnt -no FSTYPE /home/mr 2>/dev/null || stat -f -c %T /home/mr
} > "$RAW/env_snapshot.txt" 2>&1

"$PY" -m pip freeze > "$RAW/pip_freeze.txt" 2>&1
echo "env snapshot written to $RAW"
