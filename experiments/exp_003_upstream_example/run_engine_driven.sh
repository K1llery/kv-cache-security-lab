#!/usr/bin/env bash
# Run the pinned upstream AES-GCM L2 example through its supported
# engine-driven transport (worker-side GPU/CPU copy, no CUDA IPC handle).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EXP_DIR}/.." && pwd)"
SOURCE="${ROOT}/upstream/LMCache/examples/serde/aesgcm/run_serde_aesgcm_example.sh"
RUNTIME="${EXP_DIR}/runtime/engine_driven_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUNTIME}"

# Make a runtime copy so the fixed upstream checkout remains untouched.
"${ROOT}/.venv/bin/python" - "${SOURCE}" "${RUNTIME}/upstream_patched.sh" <<'PY'
from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:])
script = source.read_text()
changes = (
    ('    --l1-size-gb "$L1_SIZE_GB" \\\n',
     '    --l1-size-gb "$L1_SIZE_GB" \\\n    --supported-transfer-mode engine_driven \\\n'),
    ('    "lmcache.mp.port": ${LMCACHE_PORT},\n',
     '    "lmcache.mp.port": ${LMCACHE_PORT},\n    "lmcache.mp.mp_transfer_mode": "engine_driven",\n'),
    ('    2>&1 | tee "$TMP_DIR/lmcache.log" &',
     '    > "$TMP_DIR/lmcache.log" 2>&1 &'),
    ('    2>&1 | tee "$TMP_DIR/vllm.log" &',
     '    > "$TMP_DIR/vllm.log" 2>&1 &'),
)
for old, new in changes:
    if script.count(old) != 1:
        raise SystemExit(f"Expected one occurrence of {old!r}, got {script.count(old)}")
    script = script.replace(old, new, 1)
target.write_text(script)
target.chmod(0o700)
PY

export PATH="${ROOT}/.venv/bin:${PATH}"
export MODEL="${ROOT}/models/Qwen2.5-0.5B-Instruct"
export L1_SIZE_GB=2
export GPU_MEM_UTIL=0.35
export VLLM_PORT=8321
export LMCACHE_PORT=6555
export LMCACHE_HTTP_PORT=8080
export TMP_DIR="${RUNTIME}"

bash "${RUNTIME}/upstream_patched.sh" 2>&1 | tee "${RUNTIME}/run.log"

"${ROOT}/.venv/bin/python" - "${RUNTIME}" "${EXP_DIR}/raw/engine_driven_verification.json" <<'PY'
from pathlib import Path
import json
import re
import sys

runtime, output = map(Path, sys.argv[1:])
log = (runtime / "run.log").read_text(errors="replace")
server_log = (runtime / "lmcache.log").read_text(errors="replace")
disk_files = sorted((runtime / "disk").glob("*.data"))
prefetch = re.search(r"Prefetch request completed \(L1\+L2\): (\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\)", server_log)
responses = re.findall(r"^Response: (.*)$", log, re.MULTILINE)
summary = {
    "experiment": "exp_003_upstream_example",
    "transport": "engine_driven (worker-side copy; no CUDA IPC)",
    "model": "Qwen2.5-0.5B-Instruct",
    "runtime_directory": str(runtime),
    "server_log": str(runtime / "lmcache.log"),
    "vllm_log": str(runtime / "vllm.log"),
    "run_log": str(runtime / "run.log"),
    "encrypted_disk_objects": len(disk_files),
    "all_frame_version_01": bool(disk_files) and all(p.read_bytes()[:1] == b"\x01" for p in disk_files),
    "l1_clear_confirmed": "L1Manager: cleared 4 objects" in server_log,
    "prefetch": None if prefetch is None else {
        "retained": int(prefetch[1]), "requested": int(prefetch[2]),
        "l1": int(prefetch[3]), "l2": int(prefetch[4]),
    },
    "two_successful_responses": len(responses) == 2,
    "response_excerpts_equal": len(responses) == 2 and responses[0] == responses[1],
    "store_1024_tokens_confirmed": "Stored 1024 tokens" in server_log,
    "retrieve_1024_tokens_confirmed": "Retrieved 1024 tokens" in server_log,
    "invalid_tag_seen": "InvalidTag" in server_log,
}
checks = [summary["encrypted_disk_objects"] == 4, summary["all_frame_version_01"],
          summary["l1_clear_confirmed"], summary["prefetch"] == {
              "retained": 4, "requested": 4, "l1": 0, "l2": 4},
          summary["two_successful_responses"], summary["store_1024_tokens_confirmed"],
          summary["retrieve_1024_tokens_confirmed"], not summary["invalid_tag_seen"]]
summary["verified"] = all(checks)
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print("EXP003_ENGINE_DRIVEN_VERIFIED", summary["verified"], "->", output)
if not summary["verified"]:
    raise SystemExit(1)
PY
