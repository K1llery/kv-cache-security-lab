#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_011 v2: 单变量 AAD 修复验收（评审意见 2026-09-27 晚第 1 项的证据链修复）。

v1 → v2 的证据链修正（v1 原始数据保留为 swap_results.jsonl/verdict.json，不覆盖）：
  1. 拒绝/加载证据按**日志位置增量**读取：每次请求前记录日志光标，请求后只解析新增行
     ——消除"请求时间减 1 秒"窗口导致的相邻请求错误累加。
  2. 逐请求保存：response_id、目标文件（完整名+完整 sha256）、lookup 断言行
     （retained keys）、**load 成功证据（Retrieved N tokens 行）**、具体 InvalidTag/
     加载失败行、输出、回退状态（输出 == 无缓存基线）。
     术语：`retained keys 4/4` 只是 **lookup 命中**；load 是否成功以 Retrieved 行
     与拒绝行区分，不把失败加载写成"4/4 成功取回"。
  3. **端到端旧帧失效**：旧 serde（aesgcm, AAD=None）栈把 P_B 写入 L2（真实旧帧文件）
     → 换新 serde（aesgcm_aad）栈、同一磁盘与同一主密钥 → 请求 P_B：lookup 命中旧帧
     → 解密 InvalidTag → 回退重算（输出 == 无缓存基线）。组件级测试仍保留为补充。
  4. 工厂参数校验与原版逐项对齐（aes_bits/provider/空路径），见 aesgcm_aad.py。

验收：A1 诚实命中（lookup 4/4 + Retrieved + 输出==基线）；A2 全部互换被拒回基线且无
C 内容泄漏；A3 翻字节/截断安全回退；A4 端到端旧帧失效重算。

用法：.venv/bin/python exp_011_aad_fix/run_aad_fix.py
"""

from __future__ import annotations

import hashlib
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
VLLM_PORT = 8321
LMC_HTTP = 8080
MAX_TOKENS = 24
N_TOKENS = 1024
SHARED = 768
REPS_PER_TRIAL = 2
CONTROL_REPS = 2
LOG_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")


def serde_tag(s: str) -> str:
    return "aad" if "aad" in s else "orig"


def sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Stack:
    """一个 lmcache server（指定 serde 类型）+ 一个 vLLM。支持复用既有 disk 与主密钥。"""

    def __init__(self, serde_type: str, run_dir: str, disk_dir: str,
                 master_key_path: str) -> None:
        self.serde_type = serde_type
        self.run_dir = run_dir
        self.disk_dir = disk_dir
        self.master_key_path = master_key_path
        self.log_path = os.path.join(run_dir, f"lmcache_{serde_type}.log")
        self.procs: list[subprocess.Popen] = []
        self._pos = 0  # 日志光标（行数）

    def start(self) -> None:
        l2 = json.dumps({"type": "fs", "base_path": self.disk_dir,
                         "serde": {"type": self.serde_type, "key_provider": "hkdf",
                                   "master_key_path": self.master_key_path,
                                   "aes_bits": 128}})
        env = dict(os.environ)
        env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
        self.procs.append(subprocess.Popen(
            ["lmcache", "server", "--l1-size-gb", "2", "--eviction-policy", "LRU",
             "--l2-store-policy", "default", "--l2-prefetch-policy", "default",
             "--supported-transfer-mode", "engine_driven",
             "--l2-adapter", l2, "--port", "6555", "--http-port", str(LMC_HTTP)],
            env=env, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT))
        kv = json.dumps({"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
                         "kv_load_failure_policy": "recompute",
                         "kv_connector_extra_config": {
                             "lmcache.mp.port": 6555,
                             "lmcache.mp.mp_transfer_mode": "engine_driven",
                             "lmcache.mp.mq_timeout": 10}})
        venv = dict(env)
        venv.pop("VLLM_PORT", None)
        venv.update({"CUDA_VISIBLE_DEVICES": "0",
                     "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "PYTHONHASHSEED": "0"})
        self.procs.append(subprocess.Popen(
            ["vllm", "serve", MODEL, "--port", str(VLLM_PORT), "--no-enable-prefix-caching",
             "--enforce-eager", "--gpu-memory-utilization", "0.35", "--kv-transfer-config", kv],
            env=venv, stdout=open(os.path.join(self.run_dir, f"vllm_{serde_tag(self.serde_type)}.log"), "w"),
            stderr=subprocess.STDOUT))
        for url, t in ((f"http://localhost:{LMC_HTTP}/healthcheck", 90),
                       (f"http://localhost:{VLLM_PORT}/v1/models", 900)):
            t0 = time.time()
            ok = False
            while time.time() - t0 < t:
                try:
                    if httpx.get(url, timeout=2).status_code == 200:
                        ok = True
                        break
                except Exception:
                    time.sleep(2)
            if not ok:
                raise RuntimeError(f"timeout {url} ({self.serde_type})")
        self._pos = len(self.log_lines())

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=25)
            except Exception:
                p.kill()
        self.procs = []

    # ---- 增量日志（评审修正：光标式读取，杜绝跨请求串扰）----
    def log_lines(self) -> list[str]:
        return open(self.log_path, errors="replace").readlines()

    def mark(self) -> int:
        self._pos = len(self.log_lines())
        return self._pos

    def lines_since(self, pos: int) -> list[str]:
        return self.log_lines()[pos:]

    # ---- 常用工具 ----
    def count_stored(self) -> int:
        return sum(1 for ln in self.log_lines() if "Stored " in ln and "tokens" in ln)

    def disk_files(self) -> dict[str, str]:
        return {f: sha256_file(os.path.join(self.disk_dir, f))
                for f in sorted(os.listdir(self.disk_dir)) if f.endswith(".data")}

    def wait_store(self, prev_stored: int, min_files: int, timeout_s: float = 120.0) -> float:
        """等待落盘：返回从调用到判定完成的秒数（异步存储完成时间单独记录）。"""
        t0 = time.time()
        stable, last = 0, -1
        while time.time() - t0 < timeout_s:
            if self.count_stored() > prev_stored and len(self.disk_files()) >= min_files:
                n = len(self.disk_files())
                if n == last:
                    stable += 1
                    if stable >= 2:
                        return time.time() - t0
                else:
                    stable = 0
                last = n
            time.sleep(0.3)
        raise RuntimeError("wait_store timeout")

    def clear_l1(self) -> bool:
        try:
            return httpx.post(f"http://localhost:{LMC_HTTP}/cache/clear", timeout=30).status_code == 200
        except Exception:
            return False


class Requester:
    """流式请求 + 逐请求证据（日志增量窗口 = 本请求独占）。"""

    def __init__(self, stack: Stack) -> None:
        self.stack = stack
        self.client = httpx.Client()

    def __call__(self, token_ids: list[int], salt: str, kind: str,
                 target_file: str | None = None, max_tokens: int = MAX_TOKENS,
                 stream: bool = True) -> dict:
        pos = self.stack.mark()  # 请求前光标
        t_send = time.time()
        t0 = time.perf_counter()
        rid, text, err = None, "", None
        ttft = None
        usage = None
        try:
            if stream:
                with self.client.stream(
                        "POST", f"http://localhost:{VLLM_PORT}/v1/completions",
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
                            err = json.dumps(d)[:250]
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
            else:
                r = self.client.post(
                    f"http://localhost:{VLLM_PORT}/v1/completions",
                    json={"model": MODEL, "prompt": token_ids, "max_tokens": max_tokens,
                          "temperature": 0.0, "cache_salt": salt}, timeout=180.0)
                r.raise_for_status()
                d = r.json()
                text = d["choices"][0]["text"]
                rid = d.get("id")
                usage = d.get("usage")
                ttft = (time.perf_counter() - t0) * 1000
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:250]

        # 只解析本请求窗口内的日志行（评审修正）
        new_lines = self.stack.lines_since(pos)
        lookup, retrieved, rejects = [], [], []
        for ln in new_lines:
            if "Prefetch request completed" in ln and rid and rid in ln:
                m = re.search(r"(\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\)", ln)
                if m:
                    lookup.append(dict(retained=int(m.group(1)), requested=int(m.group(2)),
                                       l1=int(m.group(3)), l2=int(m.group(4))))
            if "Retrieved" in ln and "tokens" in ln:
                retrieved.append(ln.strip()[:180])
            low = ln.lower()
            if "invalidtag" in low or "malformed" in low or (
                    "fail" in low and "serde" in low) or "ERROR" in ln:
                rejects.append(ln.strip()[:220])
        return dict(kind=kind, salt=salt, prompt_len=len(token_ids), t_send_epoch=t_send,
                    ttft_ms=round(ttft, 1) if ttft else None,
                    total_ms=round((time.perf_counter() - t0) * 1000, 1),
                    output_text=text, response_id=rid, usage=usage, error=err,
                    target_file_full=target_file,
                    target_file_sha256=(sha256_file(target_file) if target_file else None),
                    lookup=lookup, retrieved_lines=retrieved, reject_lines=rejects,
                    log_window=[pos, pos + len(new_lines)])


def build_prompts(tok):
    filler = ("The archive room was quiet, and the clerks catalogued every document "
              "with the same patient routine as they had done for many years. ")
    paris = ("Years later, a grand treaty conference was convened in Paris, the capital "
             "of France, and the delegations signed the accord in the great hall. ")
    tokyo = ("Years later, a grand treaty conference was convened in Tokyo, the capital "
             "of Japan, and the delegations signed the accord in the great hall. ")
    base_ids = (tok(filler, add_special_tokens=False).input_ids * 40)[:SHARED]
    assert len(base_ids) == SHARED
    tail_ids = tok(" The city of", add_special_tokens=False).input_ids
    k = len(tail_ids)

    def chunk(mid: str) -> list[int]:
        s = mid + filler
        ids = tok(s, add_special_tokens=False).input_ids
        while len(ids) < 256 - k:
            s += filler
            ids = tok(s, add_special_tokens=False).input_ids
        return ids[:256 - k]

    b_ids = base_ids + chunk(tokyo) + tail_ids
    c_ids = base_ids + chunk(paris) + tail_ids
    assert len(b_ids) == N_TOKENS and len(c_ids) == N_TOKENS
    assert b_ids[:SHARED] == c_ids[:SHARED] and b_ids[-k:] == c_ids[-k:]
    assert b_ids[SHARED:N_TOKENS - k] != c_ids[SHARED:N_TOKENS - k]
    return b_ids, c_ids


def old_frame_component_check() -> str:
    """A4 补充（组件级）：旧 serde 封帧在新 serde 下解密必须 InvalidTag。"""
    import ctypes

    import torch
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.serde.aesgcm import AesGcmSerializer
    from lmcache.v1.distributed.serde.aesgcm_aad import AesGcmAadDeserializer
    from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider

    class Buf:
        def __init__(self, n: int) -> None:
            self._a = (ctypes.c_ubyte * n)()

        @property
        def byte_array(self) -> memoryview:
            return memoryview(self._a)

    prov = HkdfKeyProvider(b"exp011-v2-test-key-not-for-production", key_len=16,
                           info_prefix=b"lmcache-l2-aesgcm-v1")
    key = ObjectKey(chunk_hash=b"\x33" * 32, model_name="m", kv_rank=0, cache_salt="s")
    layout = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.uint8])
    sbuf, dbuf = Buf(1024), Buf(1024 + 29)
    sbuf.byte_array.cast("B")[:] = bytes(range(256)) * 4
    AesGcmSerializer(prov).serialize(sbuf, dbuf, key)
    frame = bytes(dbuf.byte_array)[:29 + 1024]
    ddst = Buf(1024)

    class Src:
        def __init__(self, b: bytes) -> None:
            self._b = (ctypes.c_ubyte * len(b)).from_buffer_copy(b)

        @property
        def byte_array(self) -> memoryview:
            return memoryview(self._b)

    try:
        AesGcmAadDeserializer(prov).deserialize(Src(frame), ddst, key)
        return "NO-RAISE (BAD)"
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


def main() -> int:
    os.makedirs(RAW, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    b_ids, c_ids = build_prompts(tok)
    print(f"prompts {N_TOKENS} tok; shared {SHARED}; tails equal: {b_ids[-4:] == c_ids[-4:]}")

    comp = old_frame_component_check()
    print(f"A4-component old-frame under new serde: {comp} (期望 InvalidTag)")

    run_dir = os.path.join(EXP, "runtime", "v2_" + time.strftime("%Y%m%d_%H%M%S"))
    disk_dir = os.path.join(run_dir, "disk")
    os.makedirs(disk_dir, exist_ok=True)
    master_key = os.path.join(run_dir, "master.key")
    open(master_key, "wb").write(os.urandom(16))
    os.chmod(master_key, 0o600)

    rows: list[dict] = []
    client = httpx.Client()

    # ===== 阶段 OLD：原版 serde（AAD=None）写入真实旧帧 =====
    st_old = Stack("aesgcm", run_dir, disk_dir, master_key)
    try:
        st_old.start()
        rq_old = Requester(st_old)
        for i in range(2):
            rows.append(rq_old(b_ids[:512], f"old-warmup-{i}", "warmup_old"))
        st_old.wait_store(0, 4)
        prev = st_old.count_stored()
        files_before_old = len(st_old.disk_files())
        rows.append(rq_old(b_ids, "oldsalt", "store_old_frames"))
        store_old_s = st_old.wait_store(prev, files_before_old + 4)
        print(f"phase OLD: old frames stored (completion {store_old_s:.1f}s), files={len(st_old.disk_files())}")
    finally:
        st_old.stop()

    # ===== 阶段 NEW：aesgcm_aad 栈，同一磁盘与同一主密钥 =====
    st_new = Stack("aesgcm_aad", run_dir, disk_dir, master_key)
    try:
        st_new.start()
        rq = Requester(st_new)

        # A4 端到端：旧帧 lookup 命中 → 解密失败 → 回退重算
        rows.append(rq(b_ids, "oldsalt", "A4_oldframe_e2e"))
        r4 = rows[-1]
        rows.append(rq(b_ids, "true-B", "O_B_true"))
        o_b_true = rows[-1]["output_text"]
        print(f"A4 e2e: lookup={[(p['l1'], p['l2']) for p in r4['lookup']]} "
              f"rejects={len(r4['reject_lines'])} "
              f"retrieved_n={len(r4['retrieved_lines'])} "
              f"out==baseline:{r4['output_text'] == o_b_true}")

        # A1 诚实命中（新帧，salt S2）
        S2 = "swapsalt2"
        prev = st_new.count_stored()
        files_before_b = len(st_new.disk_files())
        rows.append(rq(b_ids, S2, "store_B"))
        store_b_s = st_new.wait_store(prev, files_before_b + 4)
        st_new.clear_l1()
        rows.append(rq(b_ids, S2, "O_B_hit"))
        r1 = rows[-1]
        o_b_hit = r1["output_text"]
        print(f"A1: O_B_hit==O_B_true:{o_b_hit == o_b_true} "
              f"lookup={[(p['l1'], p['l2']) for p in r1['lookup']]} "
              f"retrieved_n={len(r1['retrieved_lines'])} (store {store_b_s:.1f}s)")

        # O_C（新帧）
        prev = st_new.count_stored()
        before = set(st_new.disk_files())
        rows.append(rq(c_ids, S2, "store_C"))
        st_new.wait_store(prev, len(before) + 1)
        f_c3 = (set(st_new.disk_files()) - before).pop()
        disk_after_c = st_new.disk_files()
        rows.append(rq(c_ids, "true-C", "O_C_true"))
        o_c_true = rows[-1]["output_text"]
        st_new.clear_l1()
        rows.append(rq(c_ids, S2, "O_C_hit"))
        o_c_hit = rows[-1]["output_text"]
        print(f"O_C_hit={o_c_hit[:70]!r}")

        # ===== A2：互换试验（哈希配方+salt 定位 P_B 的 4 个 S2 对象；应全部拒绝回基线）=====
        from lmcache.v1.multiprocess.token_hasher import TokenHasher
        b_hashes = TokenHasher(chunk_size=256, hash_algorithm="blake3").compute_chunk_hashes(
            list(b_ids), end=len(b_ids))
        b_hash_set = {h.hex() for h in b_hashes}
        targets = [f for f in sorted(st_new.disk_files())
                   if f.endswith(f"@{S2}.data") and f.split("@")[3] in b_hash_set]
        assert len(targets) == 4, f"P_B 的 S2 对象 {len(targets)} != 4"
        src = os.path.join(disk_dir, f_c3)
        verdicts = []
        for target in targets:
            dst = os.path.join(disk_dir, target)
            backup = open(dst, "rb").read()
            open(dst, "wb").write(open(src, "rb").read())
            outs = []
            for i in range(REPS_PER_TRIAL):
                st_new.clear_l1()
                r = rq(b_ids, S2, "swap", target_file=dst)
                rows.append(r)
                outs.append(r["output_text"])
                if i == 0:
                    print(f"  swap->{os.path.basename(target)[:40]}..: "
                          f"lookup={[(p['l1'], p['l2']) for p in r['lookup']]} "
                          f"rejects={len(r['reject_lines'])} "
                          f"out={r['output_text'][:44]!r}")
            rejected = all(o == o_b_true for o in outs)
            no_leak = all(o != o_c_hit and o != o_c_true for o in outs)
            verdicts.append(dict(target_file_full=target,
                                 target_sha256_full=hashlib.sha256(backup).hexdigest(),
                                 all_rejected_output_equals_baseline=rejected,
                                 no_output_leak_from_C=no_leak,
                                 deterministic=len(set(outs)) == 1, outputs=outs))
            open(dst, "wb").write(backup)
            print(f"  -> verdict: rejected_to_baseline={rejected} no_leak={no_leak}")
        n_rejected = sum(1 for v in verdicts if v["all_rejected_output_equals_baseline"])
        print(f"ACCEPTANCE A2: {n_rejected}/4 swap trials rejected to baseline")

        # ===== 恢复校验 + 诚实复测 =====
        rewritten = [f for f in disk_after_c
                     if st_new.disk_files().get(f) != disk_after_c[f]]
        print(f"disk files changed after trials: {len(rewritten)} {rewritten}")
        recheck_ok = True
        for i in range(2):
            st_new.clear_l1()
            r = rq(b_ids, S2, "O_B_recheck")
            rows.append(r)
            recheck_ok = recheck_ok and (r["output_text"] == o_b_true)
        print(f"REVERSIBLE: {recheck_ok}")

        # ===== A3 对照：翻字节 / 截断 =====
        control_results = []
        for cname, mutate in (
                ("bitflip", lambda b: b[:64] + bytes([b[64] ^ 0xFF]) + b[65:]),
                ("truncate", lambda b: b[:-64])):
            dst = os.path.join(disk_dir, sorted(targets)[0])
            orig = open(dst, "rb").read()
            open(dst, "wb").write(mutate(orig))
            for i in range(CONTROL_REPS):
                st_new.clear_l1()
                r = rq(b_ids, S2, f"control_{cname}", target_file=dst)
                rows.append(r)
                control_results.append(r)
                print(f"  {cname}#{i}: rejects={len(r['reject_lines'])} "
                      f"out==true:{r['output_text'] == o_b_true}")
            open(dst, "wb").write(orig)

        with open(os.path.join(RAW, "swap_results_v2.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        verdict = {
            "A1_honest_hit_intact": (o_b_hit == o_b_true
                                     and any(p["l1"] == 0 and p["l2"] == 4
                                             for p in r1["lookup"])
                                     and len(r1["retrieved_lines"]) > 0),
            "A2_all_swap_trials_rejected_to_baseline": n_rejected == 4,
            "A2_no_output_leak_from_C": all(v["no_output_leak_from_C"] for v in verdicts),
            "A3_bitflip_safe": all(r["reject_lines"] and r["output_text"] == o_b_true
                                   for r in control_results if r["kind"] == "control_bitflip"),
            "A3_truncate_safe": all(r["reject_lines"] and r["output_text"] == o_b_true
                                    for r in control_results if r["kind"] == "control_truncate"),
            "A4_e2e_oldframe_invalidated_recompute": r4["output_text"] == o_b_true,
            "A4_e2e_reject_lines": len(r4["reject_lines"]),
            "A4_component_check": comp,
            "reversible_recheck": recheck_ok,
            "trials": verdicts,
        }
        with open(os.path.join(RAW, "verdict_v2.json"), "w") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in verdict.items() if k != "trials"},
                         indent=1, ensure_ascii=False))
        print("EXP011_V2_OK")
        return 0
    finally:
        st_new.stop()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
