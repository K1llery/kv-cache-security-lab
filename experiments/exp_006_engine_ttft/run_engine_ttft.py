#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_006: engine_driven 模式内的真实 TTFT 与服务端阶段时点。

在 exp_003 已验证的组合上（LMCache@9a7b605 MP server + fs L2 + aesgcm serde +
vLLM@3b4566c + engine_driven 传输），用流式请求测：

  - recompute：从未缓存过的等长新 prompt → 引擎全量重算 TTFT
  - L1 warm：同 prompt 不清 L1 → 内存缓存命中 TTFT
  - L2 hit（页缓存热）：清 L1 → L2 命中 + AES-GCM 解密恢复 TTFT
  - L2 hit（受控冷读）：同上 + 对 L2 对象文件 fadvise(DONTNEED) 逐出页缓存

服务端阶段时点：从 lmcache.log 抽取 Prefetch/Retrieved 行墙钟时点，与请求
发送/首包时间对齐（同机墙钟）。读取/AEAD/H2D 细粒度分解需 patch 插桩，列为后续项。

用法：.venv/bin/python exp_006_engine_ttft/run_engine_ttft.py [--reps 3] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import httpx

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
VENV_BIN = os.path.join(ROOT, ".venv", "bin")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct")
SOURCE = os.path.join(ROOT, "upstream", "LMCache", "examples", "serde", "aesgcm",
                      "run_serde_aesgcm_example.sh")
RUNTIME = os.path.join(EXP, "runtime")
RAW = os.path.join(EXP, "raw")
FADV_DONTNEED = 4

VLLM_PORT = 8321
LMC_HTTP = 8080
SALT = "tenant-a"


def patched_script(target: str) -> None:
    script = open(SOURCE).read()
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
        assert script.count(old) == 1, old
        script = script.replace(old, new, 1)
    open(target, "w").write(script)
    os.chmod(target, 0o700)


def wait_url(url: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            time.sleep(2)
    return False


def fadvise_dir(path: str) -> int:
    n = 0
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".data"):
            fd = os.open(os.path.join(path, fn), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, FADV_DONTNEED)
            os.close(fd)
            n += 1
    return n


def stream_ttft(client: httpx.Client, prompt: str, max_tokens: int = 8):
    """返回 (ttft_ms, total_ms, first_chunk_text, n_chunks, raw_head)。

    TTFT=首个含非空 content 的 chunk；raw_head 保留前几条原始 SSE 行与任何
    含 error 的行，便于在无 content 时诊断（不改变测量口径）。
    """
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    first_text = None
    raw_head: list[str] = []
    with client.stream("POST", f"http://localhost:{VLLM_PORT}/v1/completions",
                       json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                             "temperature": 0.0, "cache_salt": SALT, "stream": True},
                       timeout=120.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if len(raw_head) < 8 or "error" in line.lower():
                raw_head.append(line[:300])
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "error" in d:
                raw_head.append("STREAM_ERROR: " + json.dumps(d)[:300])
                break
            chs = d.get("choices") or [{}]
            txt = chs[0].get("text", "") if isinstance(chs[0], dict) else ""
            if txt:
                chunks += 1
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                    first_text = txt
    total = (time.perf_counter() - t0) * 1000
    return ttft, total, first_text, chunks, raw_head


def make_prompt(n_reps: int) -> str:
    p = ""
    for _ in range(n_reps):
        p += ("The history and significance of the Roman empire spans more than a thousand "
              "years and profoundly shaped Western civilization. ")
        p += ("Its legal, architectural, linguistic, and political legacies persist to this "
              "day, influencing modern governments, languages, art, engineering, and law. ")
        p += ("The empire's trajectory from the founding of Rome through the Republic, the "
              "transition to the Principate under Augustus, the Pax Romana, the crisis of the "
              "third century, ")
        p += ("the Dominate under Diocletian, the adoption of Christianity under Constantine, "
              "the splitting into Western and Eastern halves, and the eventual collapse of the West "
              "is one of history's great narratives. Key figures include Julius Caesar, Augustus, "
              "Marcus Aurelius, Diocletian, Constantine, Justinian, and many others. ")
    return p + "Summarize the key transitions in one sentence."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    reps = 1 if args.quick else args.reps
    os.makedirs(RAW, exist_ok=True)
    run_dir = os.path.join(RUNTIME, f"ttft_{time.strftime('%Y%m%d_%H%M%S')}")
    disk_dir = os.path.join(run_dir, "disk")
    os.makedirs(disk_dir, exist_ok=True)

    # 直接拉起两个服务（不走上游示例脚本：该脚本自带演示流程且退出时
    # cleanup 会杀服务，见 exp_006 结论"存活期竞态"）。配置与 exp_003 已验证组合一致。
    master_key = os.path.join(run_dir, "master.key")
    with open(master_key, "wb") as f:
        f.write(os.urandom(16))
    os.chmod(master_key, 0o600)
    l2_adapter = json.dumps({
        "type": "fs", "base_path": disk_dir,
        "serde": {"type": "aesgcm", "key_provider": "hkdf",
                  "master_key_path": master_key, "aes_bits": 128}})
    env = dict(os.environ)
    env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
    srv_log = open(os.path.join(run_dir, "lmcache.log"), "w")
    srv = subprocess.Popen(
        ["lmcache", "server", "--l1-size-gb", "2", "--eviction-policy", "LRU",
         "--l2-store-policy", "default", "--l2-prefetch-policy", "default",
         "--supported-transfer-mode", "engine_driven",
         "--l2-adapter", l2_adapter, "--port", "6555", "--http-port", str(LMC_HTTP)],
        env=env, stdout=srv_log, stderr=subprocess.STDOUT)
    kv_cfg = json.dumps({
        "kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lmcache.mp.port": 6555,
            "lmcache.mp.mp_transfer_mode": "engine_driven",
            "lmcache.mp.mq_timeout": 10}})
    vllm_env = dict(env)
    vllm_env.pop("VLLM_PORT", None)
    vllm_env.update({"CUDA_VISIBLE_DEVICES": "0",
                     "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "PYTHONHASHSEED": "0"})
    vllm_log = open(os.path.join(run_dir, "vllm.log"), "w")
    vllm = subprocess.Popen(
        ["vllm", "serve", MODEL, "--port", str(VLLM_PORT), "--no-enable-prefix-caching",
         "--enforce-eager", "--gpu-memory-utilization", "0.35",
         "--kv-transfer-config", kv_cfg],
        env=vllm_env, stdout=vllm_log, stderr=subprocess.STDOUT)
    procs = (srv, vllm)
    try:
        if not wait_url(f"http://localhost:{LMC_HTTP}/healthcheck", 90):
            print("LMCACHE SERVER FAILED TO START"); return 1
        if not wait_url(f"http://localhost:{VLLM_PORT}/v1/models", 900):
            print("VLLM FAILED TO START"); return 1
        print("stack ready; measuring...")

        results = {"meta": {"model": MODEL, "transfer_mode": "engine_driven",
                            "reps": reps, "salt": SALT, "max_tokens": 8},
                   "requests": []}
        client = httpx.Client()

        def record(kind, P_rep, ttft, total, first_text, chunks, note=None, raw_head=None):
            results["requests"].append(dict(
                kind=kind, prompt_reps=P_rep, ttft_ms=ttft, total_ms=total,
                first_chunk=first_text, n_content_chunks=chunks,
                raw_head=raw_head or [], t_send_epoch=time.time(), note=note))
            ts = "None" if ttft is None else f"{ttft:8.1f}"
            print(f"  {kind:14s} P~{P_rep} ttft={ts}ms total={total:8.1f}ms chunks={chunks}")
            if ttft is None:
                print("    RAW_HEAD:", json.dumps(raw_head or [])[:500])

        for P_rep in (8, 32):  # ~1024 / ~4096 token prompt
            prompt = make_prompt(P_rep)
            fresh_prompts = [make_prompt(P_rep).replace("Summarize", f"Outline {i}: summarize")
                             for i in range(reps)]
            # 1) 首次请求：建立缓存（store 到 L1→L2）
            ttft, total, ft, ch, rh = stream_ttft(client, prompt)
            record("store_cold", P_rep, ttft, total, ft, ch, raw_head=rh)
            time.sleep(5)  # L2 异步落盘
            # 2) L1 warm
            for _ in range(2 if not args.quick else 1):
                ttft, total, ft, ch, rh = stream_ttft(client, prompt)
                record("l1_warm", P_rep, ttft, total, ft, ch, raw_head=rh)
            # 3) L2 hit（页缓存热）
            for i in range(reps):
                r = client.post(f"http://localhost:{LMC_HTTP}/cache/clear", timeout=10)
                ttft, total, ft, ch, rh = stream_ttft(client, prompt)
                record("l2hit_warm", P_rep, ttft, total, ft, ch, note=f"clear={r.status_code}", raw_head=rh)
                time.sleep(1)
            # 4) L2 hit（受控冷读）
            for i in range(reps):
                client.post(f"http://localhost:{LMC_HTTP}/cache/clear", timeout=10)
                n = fadvise_dir(disk_dir)
                ttft, total, ft, ch, rh = stream_ttft(client, prompt)
                record("l2hit_cold", P_rep, ttft, total, ft, ch, note=f"fadvise_files={n}", raw_head=rh)
                time.sleep(1)
            # 5) recompute 基线（全新 prompt，从未缓存）
            for i in range(reps):
                ttft, total, ft, ch, rh = stream_ttft(client, fresh_prompts[i])
                record("recompute", P_rep, ttft, total, ft, ch, raw_head=rh)

        with open(os.path.join(RAW, "engine_ttft_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("EXP006_OK ->", os.path.join(RAW, "engine_ttft_results.json"))
        return 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=25)
            except Exception:
                p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
