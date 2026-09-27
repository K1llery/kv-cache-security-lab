#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_008: 修正版引擎级 TTFT 对照（评审意见第 1、2 项）。

相对 exp_006 的测量有效性修正：
  1. 重算组 = 全新 cache_salt（LMCache 键含 salt → L1+L2 必然未命中），逐次记录 salt；
  2. 提示词以 token-id 数组下发（长度=实测分词长度），并记录引擎 usage 交叉核对；
  3. 每次正式请求前清 L1（命中状态一致）；
  4. t_send 在发送前记录；TTFT=首个非空 content chunk；
  5. 异步写完成 = 轮询服务端 "Stored N tokens" 行计数增加（不固定 sleep）；
  6. 明文 L2 / AES-GCM L2 两个栈，同引擎/传输/模型/提示词/预算；
  7. 预热请求单独标记，不入正式统计；
  8. 逐请求断言（服务端日志 + 请求 id），失败计入 failed_assertions。

用法：.venv/bin/python exp_008_engine_ttft_fixed/run_engine_fixed.py [--stacks aead_l2 plain_l2]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

import httpx
from transformers import AutoTokenizer

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
VENV_BIN = os.path.join(ROOT, ".venv", "bin")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct")
RAW = os.path.join(EXP, "raw")
FADV_DONTNEED = 4

VLLM_PORT = 8321
LMC_HTTP = 8080
POINTS = [1024, 5000, 8192]
REPS_WARM, REPS_COLD, REPS_RECOMP = 10, 5, 10
MAX_TOKENS = 8
LOG_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")


def now() -> float:
    return time.perf_counter()


def epoch_of_log_line(line: str) -> float | None:
    m = LOG_TS.search(line)
    if not m:
        return None
    t = time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return time.mktime(t) + int(m.group(2)) / 1000.0


class Stack:
    """一个 L2 栈（aead_l2 或 plain_l2）：自管服务生命周期。"""

    def __init__(self, stack_id: str, serde: str) -> None:
        self.id = stack_id
        self.serde = serde
        self.run_dir = os.path.join(EXP, "runtime", f"{stack_id}_{time.strftime('%Y%m%d_%H%M%S')}")
        self.disk_dir = os.path.join(self.run_dir, "disk")
        os.makedirs(self.disk_dir, exist_ok=True)
        self.server_log_path = os.path.join(self.run_dir, "lmcache.log")
        self.procs: list[subprocess.Popen] = []
        self._stored_seen = 0

    def start(self) -> None:
        master_key = os.path.join(self.run_dir, "master.key")
        with open(master_key, "wb") as f:
            f.write(os.urandom(16))
        os.chmod(master_key, 0o600)
        serde_cfg = ({"serde": {"type": "aesgcm", "key_provider": "hkdf",
                                "master_key_path": master_key, "aes_bits": 128}}
                     if self.serde == "aesgcm" else {})
        l2_adapter = json.dumps({"type": "fs", "base_path": self.disk_dir, **serde_cfg})
        env = dict(os.environ)
        env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
        srv = subprocess.Popen(
            ["lmcache", "server", "--l1-size-gb", "2", "--eviction-policy", "LRU",
             "--l2-store-policy", "default", "--l2-prefetch-policy", "default",
             "--supported-transfer-mode", "engine_driven",
             "--l2-adapter", l2_adapter, "--port", "6555", "--http-port", str(LMC_HTTP)],
            env=env, stdout=open(self.server_log_path, "w"), stderr=subprocess.STDOUT)
        kv_cfg = json.dumps({
            "kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.port": 6555,
                "lmcache.mp.mp_transfer_mode": "engine_driven",
                "lmcache.mp.mq_timeout": 10}})
        venv = dict(env)
        venv.pop("VLLM_PORT", None)
        venv.update({"CUDA_VISIBLE_DEVICES": "0",
                     "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "PYTHONHASHSEED": "0"})
        vllm = subprocess.Popen(
            ["vllm", "serve", MODEL, "--port", str(VLLM_PORT), "--no-enable-prefix-caching",
             "--enforce-eager", "--gpu-memory-utilization", "0.35",
             "--kv-transfer-config", kv_cfg],
            env=venv, stdout=open(os.path.join(self.run_dir, "vllm.log"), "w"),
            stderr=subprocess.STDOUT)
        self.procs = [srv, vllm]
        self._wait_url(f"http://localhost:{LMC_HTTP}/healthcheck", 90)
        self._wait_url(f"http://localhost:{VLLM_PORT}/v1/models", 900)

    def _wait_url(self, url: str, timeout_s: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if httpx.get(url, timeout=2.0).status_code == 200:
                    return
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"timeout waiting {url} ({self.id})")

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=25)
            except Exception:
                p.kill()

    # ---- 服务端日志工具 ----
    def _log_lines(self) -> list[str]:
        try:
            with open(self.server_log_path, errors="replace") as f:
                return f.readlines()
        except FileNotFoundError:
            return []

    def count_stored(self) -> int:
        return sum(1 for ln in self._log_lines() if "Stored " in ln and "tokens" in ln)

    def wait_store_done(self, prev_count: int, timeout_s: float = 120.0,
                        expect_objects: int | None = None) -> bool:
        # 评审修正：等待"Stored 日志行 + 落盘对象数达到期望值且连续两次采样稳定"，
        # 不再是固定 sleep；部分落盘时提前测量会导致部分命中（exp_008 run1 实测 8/32）。
        t0 = time.time()
        stable = 0
        last = -1
        while time.time() - t0 < timeout_s:
            if self.count_stored() > prev_count:
                n = self.disk_objects()
                if expect_objects is not None and n < expect_objects:
                    time.sleep(0.3)
                    continue
                if n == last:
                    stable += 1
                    if stable >= 2:
                        return True
                else:
                    stable = 0
                last = n
            time.sleep(0.3)
        return False

    def disk_objects(self) -> int:
        return len([f for f in os.listdir(self.disk_dir) if f.endswith(".data")]) \
            if os.path.isdir(self.disk_dir) else 0

    def fadvise_disk(self) -> int:
        n = 0
        for fn in sorted(os.listdir(self.disk_dir)):
            if fn.endswith(".data"):
                fd = os.open(os.path.join(self.disk_dir, fn), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, FADV_DONTNEED)
                os.close(fd)
                n += 1
        return n

    def clear_l1(self) -> bool:
        try:
            return httpx.post(f"http://localhost:{LMC_HTTP}/cache/clear", timeout=30).status_code == 200
        except Exception:
            return False


def request_once(client: httpx.Client, token_ids: list[int], salt: str):
    """发一次流式请求。返回 dict（含 t_send、ttft、usage、原始 chunk 计数、response id）。"""
    t_send_wall = time.time()
    t0 = now()
    ttft = None
    t_first_byte = None
    n_chunks = 0
    resp_id = None
    usage = None
    err = None
    try:
        with client.stream(
                "POST", f"http://localhost:{VLLM_PORT}/v1/completions",
                json={"model": MODEL, "prompt": token_ids, "max_tokens": MAX_TOKENS,
                      "temperature": 0.0, "cache_salt": salt, "stream": True,
                      "stream_options": {"include_usage": True}},
                timeout=180.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if t_first_byte is None and line.strip():
                    t_first_byte = (now() - t0) * 1000
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
                    err = json.dumps(d)[:300]
                    break
                if d.get("id"):
                    resp_id = d["id"]
                if d.get("usage"):
                    usage = d["usage"]
                chs = d.get("choices") or []
                if chs and isinstance(chs[0], dict) and chs[0].get("text"):
                    n_chunks += 1
                    if ttft is None:
                        ttft = (now() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    return dict(t_send_epoch=t_send_wall, ttft_ms=ttft, first_byte_ms=t_first_byte,
                total_ms=(now() - t0) * 1000, n_content_chunks=n_chunks,
                response_id=resp_id, usage=usage, salt=salt, error=err)


def assert_hits(stack: Stack, req: dict, kind: str, expect_chunks: int | None,
                prompt_len: int) -> dict:
    """逐请求服务端日志断言（评审要求：用请求 id + 服务端日志逐次断言命中状态）。"""
    rid = req.get("response_id") or ""
    lines = stack._log_lines()
    prefetch, retrieved, stored = [], [], []
    for ln in lines:
        ts = epoch_of_log_line(ln)
        if ts is not None and ts < req["t_send_epoch"] - 1.0:
            continue
        if "Prefetch request completed" in ln and rid and rid in ln:
            m = re.search(r"(\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\)", ln)
            if m:
                prefetch.append(dict(retained=int(m.group(1)), requested=int(m.group(2)),
                                     l1=int(m.group(3)), l2=int(m.group(4))))
        if "Retrieved" in ln and "tokens" in ln and ts is not None \
                and req["t_send_epoch"] - 1.0 <= ts <= time.time():
            retrieved.append(ln.strip()[:160])
        if "Stored " in ln and "tokens" in ln and ts is not None \
                and req["t_send_epoch"] - 1.0 <= ts <= time.time():
            stored.append(ln.strip()[:160])
    ok, detail = False, ""
    if kind in ("l2hit_warm", "l2hit_cold"):
        hit = [p for p in prefetch if p["l2"] > 0]
        ok = bool(hit) and all(p["l2"] == expect_chunks for p in hit) \
            and all(p["l1"] == 0 for p in hit)
        detail = f"prefetch={prefetch} retrieved_n={len(retrieved)}"
    else:  # recompute
        bad = [p for p in prefetch if p["retained"] > 0]
        ok = (not bad) and bool(stored)
        detail = f"prefetch={prefetch} stored_n={len(stored)}"
        if bad:
            detail += f" UNEXPECTED_HITS={bad}"
    return dict(kind=kind, response_id=rid, asserted_ok=ok, detail=detail)


def build_prompt_ids(tok, n_tokens: int) -> list[int]:
    base = ("The history and significance of the Roman empire spans more than a thousand "
            "years and profoundly shaped Western civilization. Its legal, architectural, "
            "linguistic, and political legacies persist to this day. ") * 400
    ids = tok(base, add_special_tokens=False).input_ids
    return ids[:n_tokens]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", nargs="+", default=["aead_l2", "plain_l2"])
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    out_path = os.path.join(RAW, "engine_ttft_fixed.jsonl")
    outf = open(out_path, "w")
    summary = {}

    for stack_id in args.stacks:
        serde = "aesgcm" if stack_id == "aead_l2" else "none"
        stack = Stack(stack_id, serde)
        print(f"=== stack {stack_id} starting ===")
        stack.start()
        client = httpx.Client()
        rows: list[dict] = []
        try:
            # 预热（单独标记，不入正式统计）
            warm_ids = build_prompt_ids(tok, 512)
            for i in range(2):
                r = request_once(client, warm_ids, salt=f"warmup-{i}")
                r.update(kind="warmup", prompt_len=len(warm_ids), stack=stack_id)
                rows.append(r)
                print(f"  warmup{i}: ttft={r['ttft_ms']}")
            time.sleep(1)

            for n_tok in POINTS:
                ids = build_prompt_ids(tok, n_tok)
                # LMCache 只存整 256-token 块；尾部 n%256 token 由引擎重算（真实行为，记录）
                expect_chunks = n_tok // 256
                tail_tokens = n_tok % 256
                group_salt = f"{stack_id}-{n_tok}"
                # store：建立 L2 内容
                prev_stored = stack.count_stored()
                r = request_once(client, ids, salt=group_salt)
                r.update(kind="store", prompt_len=len(ids), stack=stack_id, salt=group_salt)
                rows.append(r)
                ok_store = stack.wait_store_done(prev_stored, expect_objects=expect_chunks)
                print(f"  P={len(ids)} stored: {ok_store} disk_objs={stack.disk_objects()}")

                # L2 hit（页缓存热）×10
                for i in range(REPS_WARM):
                    assert stack.clear_l1()
                    r = request_once(client, ids, salt=group_salt)
                    a = assert_hits(stack, r, "l2hit_warm", expect_chunks, len(ids))
                    r.update(kind="l2hit_warm", prompt_len=len(ids), stack=stack_id,
                             expect_chunks=expect_chunks, assertion=a)
                    rows.append(r)
                # L2 hit（受控冷读）×5
                for i in range(REPS_COLD):
                    assert stack.clear_l1()
                    n_adv = stack.fadvise_disk()
                    r = request_once(client, ids, salt=group_salt)
                    a = assert_hits(stack, r, "l2hit_cold", expect_chunks, len(ids))
                    r.update(kind="l2hit_cold", prompt_len=len(ids), stack=stack_id,
                             expect_chunks=expect_chunks, fadvise_files=n_adv, assertion=a)
                    rows.append(r)
                # 真重算（全新 salt）×10
                for i in range(REPS_RECOMP):
                    assert stack.clear_l1()
                    salt = f"{stack_id}-{n_tok}-recompute-{i}"
                    r = request_once(client, ids, salt=salt)
                    a = assert_hits(stack, r, "recompute", None, len(ids))
                    r.update(kind="recompute", prompt_len=len(ids), stack=stack_id,
                             salt_used=salt, assertion=a)
                    rows.append(r)
                # 写一行进度
                for r in rows[-REPS_WARM - REPS_COLD - REPS_RECOMP - 1:]:
                    outf.write(json.dumps(r, ensure_ascii=False) + "\n")
                outf.flush()
        finally:
            stack.stop()
            client.close()

        # 汇总（仅断言通过的正式请求）
        valid = [r for r in rows if r.get("assertion", {}).get("asserted_ok")
                 and r["kind"] in ("l2hit_warm", "l2hit_cold", "recompute") and r["ttft_ms"]]
        stat = {}
        for kind in ("l2hit_warm", "l2hit_cold", "recompute"):
            for n_tok in POINTS:
                ts = sorted(r["ttft_ms"] for r in valid
                            if r["kind"] == kind and r["prompt_len"] == n_tok)
                if ts:
                    stat[f"{stack_id}/{kind}/P{n_tok}"] = {
                        "n": len(ts), "median_ms": ts[len(ts) // 2] if len(ts) % 2 else (ts[len(ts)//2 - 1] + ts[len(ts)//2]) / 2,
                        "min_ms": ts[0], "max_ms": ts[-1],
                        "all": [round(x, 1) for x in ts]}
        failed = [r for r in rows if r.get("assertion") and not r["assertion"]["asserted_ok"]]
        summary[stack_id] = {"stats": stat, "n_failed_assertions": len(failed),
                             "failed_detail": [r["assertion"]["detail"][:200] for r in failed[:5]]}
        print(json.dumps(summary[stack_id], indent=1, ensure_ascii=False)[:1500])

    outf.close()
    with open(os.path.join(RAW, "engine_ttft_fixed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("EXP008_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
