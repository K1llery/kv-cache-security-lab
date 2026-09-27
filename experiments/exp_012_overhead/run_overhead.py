#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_012: 原版 aesgcm vs aesgcm_aad 开销测量（主对照仅此二者）。

3 点（1024/5000/8192）× 2 serde 块，平衡顺序；每块：预热×2 → store（异步完成时间
单独记录）→ l2hit×10（清 L1 + 流式 TTFT + 逐请求 lookup=(0,N) 断言 + Retrieved 证据）
→ 重算基线×3（全新 salt）；P=5000 块内吞吐相（并发 4 × 3 波 × 64 token，与 TTFT 分开）。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import httpx
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "exp_011_aad_fix"))
from run_aad_fix import Stack  # noqa: E402

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
MODEL = os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct")
RAW = os.path.join(EXP, "raw")
VLLM_PORT = 8321
POINTS = [1024, 5000, 8192]
BLOCK_ORDER = {1024: ["aesgcm", "aesgcm_aad"],
               5000: ["aesgcm_aad", "aesgcm"],
               8192: ["aesgcm", "aesgcm_aad"]}
EXPECTED_CHUNKS = {1024: 4, 5000: 19, 8192: 32}
REPS_L2HIT = 10
REPS_RECOMP = 3
MAX_TOKENS_TTFT = 8
THR = dict(point=5000, concurrency=4, waves=3, max_tokens=64)


def stream_request(token_ids: list[int], salt: str, kind: str, block: str,
                   max_tokens: int = MAX_TOKENS_TTFT) -> dict:
    """无日志窗口的流式请求（吞吐/并发用；命中证据由波级窗口统一解析）。"""
    t_send = time.time()
    t0 = time.perf_counter()
    rid, text, err, ttft, usage = None, "", None, None, None
    c = httpx.Client()
    try:
        with c.stream("POST", f"http://localhost:{VLLM_PORT}/v1/completions",
                      json={"model": MODEL, "prompt": token_ids, "max_tokens": max_tokens,
                            "temperature": 0.0, "cache_salt": salt, "stream": True,
                            "stream_options": {"include_usage": True}},
                      timeout=180.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
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
                    err = json.dumps(d)[:200]
                    break
                if d.get("id"):
                    rid = d["id"]
                if d.get("usage"):
                    usage = d["usage"]
                chs = d.get("choices") or []
                if chs and isinstance(chs[0], dict) and chs[0].get("text"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    text += chs[0]["text"]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:200]
    finally:
        c.close()
    return dict(kind=kind, block=block, salt=salt, prompt_len=len(token_ids),
                t_send_epoch=t_send, ttft_ms=round(ttft, 1) if ttft else None,
                total_ms=round((time.perf_counter() - t0) * 1000, 1),
                output_text=text, response_id=rid, usage=usage, error=err)


class WindowedRequester:
    """单请求 + 日志增量窗口（TTFT 相用；串行调用）。"""

    def __init__(self, stack: Stack, block: str) -> None:
        self.stack = stack
        self.block = block

    def __call__(self, token_ids: list[int], salt: str, kind: str,
                 max_tokens: int = MAX_TOKENS_TTFT, expect_chunks: int | None = None) -> dict:
        pos = self.stack.mark()
        r = stream_request(token_ids, salt, kind, self.block, max_tokens)
        new_lines = self.stack.lines_since(pos)
        lookup, retrieved, rejects = [], [], []
        for ln in new_lines:
            if "Prefetch request completed" in ln and r["response_id"] and r["response_id"] in ln:
                import re
                m = re.search(r"(\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\)", ln)
                if m:
                    lookup.append(dict(retained=int(m.group(1)), requested=int(m.group(2)),
                                       l1=int(m.group(3)), l2=int(m.group(4))))
            if "Retrieved" in ln and "tokens" in ln:
                retrieved.append(ln.strip()[:160])
            low = ln.lower()
            if "invalidtag" in low or ("fail" in low and "serde" in low) or "ERROR" in ln:
                rejects.append(ln.strip()[:200])
        r["lookup"] = lookup
        r["retrieved_n"] = len(retrieved)
        r["reject_lines"] = rejects
        r["hit_asserted"] = (bool(lookup)
                             and any(p["l1"] == 0 and p["l2"] == expect_chunks
                                     for p in lookup)) if expect_chunks else None
        return r


def wave_throughput(ids: list[int], salt: str, block: str, st: Stack,
                    concurrency: int, max_tokens: int) -> list[dict]:
    """一波并发请求 + 波级日志窗口的命中证据（避免并发下逐请求光标错位）。"""
    pos = st.mark()
    t0 = time.perf_counter()
    outs: list[dict] = []
    lock = threading.Lock()

    def worker(k: int) -> None:
        r = stream_request(ids, salt, f"thr", block, max_tokens)
        r["worker"] = k
        with lock:
            outs.append(r)

    ths = [threading.Thread(target=worker, args=(k,)) for k in range(concurrency)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = round((time.perf_counter() - t0) * 1000, 1)
    # 波级命中证据：窗口内每个 response_id 都应有 (0, N) lookup 与 Retrieved 行
    new_lines = st.lines_since(pos)
    for r in outs:
        rid = r.get("response_id") or ""
        lk = 0
        for ln in new_lines:
            if rid and rid in ln and "Prefetch request completed" in ln:
                lk += 1
        r["lookup_count_in_wave"] = lk
        r["wave_wall_ms"] = wall
    print(f"  thr wave: wall={wall}ms ttfts={[r['ttft_ms'] for r in outs]} lookups={[r['lookup_count_in_wave'] for r in outs]}")
    return outs




def wait_store_files(st, expected_total: int, timeout_s: float = 120.0) -> tuple[float, bool]:
    """等待盘上对象数达到期望且稳定。返回 (耗时, "Stored" 汇总行是否出现)。"""
    t0 = time.time()
    last, stable = -1, 0
    while time.time() - t0 < timeout_s:
        n = len(st.disk_files())
        if n >= expected_total and n == last:
            stable += 1
            if stable >= 3:
                summary = any("Stored " in ln and "tokens" in ln for ln in st.log_lines())
                return time.time() - t0, summary
        else:
            stable = 0
        last = n
        time.sleep(0.3)
    summary = any("Stored " in ln and "tokens" in ln for ln in st.log_lines())
    return time.time() - t0, summary


def wait_files_stable(st, rounds: int = 3, timeout_s: float = 120.0) -> float:
    """等待盘上对象数连续多轮不变（预热异步存储完成；不依赖 Stored 行计数）。"""
    t0 = time.time()
    last, stable = -1, 0
    while time.time() - t0 < timeout_s:
        n = len(st.disk_files())
        if n == last:
            stable += 1
            if stable >= rounds:
                return time.time() - t0
        else:
            stable = 0
        last = n
        time.sleep(0.3)
    raise RuntimeError("wait_files_stable timeout")


def main() -> int:
    os.makedirs(RAW, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = {}
    for n in POINTS:
        base = ("The history and significance of the Roman empire spans more than a "
                "thousand years and profoundly shaped Western civilization. Its legal, "
                "architectural, linguistic, and political legacies persist to this day. ")
        ids = (tok(base, add_special_tokens=False).input_ids * 800)[:n]
        assert len(ids) == n
        prompts[n] = ids

    all_rows: list[dict] = []
    block_meta: list[dict] = []

    for point in POINTS:
        for serde in BLOCK_ORDER[point]:
            tag = f"{serde}_P{point}"
            run_dir = os.path.join(EXP, "runtime", f"ovh_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
            disk_dir = os.path.join(run_dir, "disk")
            os.makedirs(disk_dir, exist_ok=True)
            master_key = os.path.join(run_dir, "master.key")
            open(master_key, "wb").write(os.urandom(16))
            os.chmod(master_key, 0o600)
            st = Stack(serde, run_dir, disk_dir, master_key)
            st.start()
            rq = WindowedRequester(st, tag)
            ids = prompts[point]
            nch = EXPECTED_CHUNKS[point]
            salt = f"ovh-{point}"
            try:
                for i in range(2):
                    all_rows.append(rq(ids[:512], f"warmup-{tag}-{i}", "warmup"))
                wait_files_stable(st)

                files_before = len(st.disk_files())
                all_rows.append(rq(ids, salt, "store", expect_chunks=nch))
                t_store, summary_seen = wait_store_files(st, files_before + nch)
                block_meta.append(dict(block=tag, point=point, serde=serde,
                                       store_completion_s=round(t_store, 2),
                                       expected_chunks=nch,
                                       stored_summary_line_seen=summary_seen))
                print(f"[{tag}] store files stable {t_store:.1f}s "
                      f"(summary line seen: {summary_seen})")

                for i in range(REPS_L2HIT):
                    st.clear_l1()
                    r = rq(ids, salt, "l2hit", expect_chunks=nch)
                    all_rows.append(r)
                n_ok = sum(1 for r in all_rows
                           if r["block"] == tag and r["kind"] == "l2hit" and r["hit_asserted"])
                print(f"[{tag}] l2hit asserted {n_ok}/{REPS_L2HIT}")

                for i in range(REPS_RECOMP):
                    st.clear_l1()
                    all_rows.append(rq(ids, f"{salt}-recomp-{i}", "recompute"))

                if point == THR["point"]:
                    for wave in range(THR["waves"]):
                        st.clear_l1()
                        all_rows.extend(wave_throughput(ids, salt, tag, st,
                                                        THR["concurrency"],
                                                        THR["max_tokens"]))
            finally:
                st.stop()

    with open(os.path.join(RAW, "overhead_results.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(RAW, "block_meta.json"), "w") as f:
        json.dump(block_meta, f, indent=2)
    print("EXP012_RUN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
