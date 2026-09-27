#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_010: 方向 S 端到端密文互换实验（首要判据见 params.yaml）。

装置设计（用去重增量识别文件，避免离线哈希配方逆向）：
  P_B = base(768) + Tokyo 块内容 + 共享尾 " The city of"；P_C = base(768) + Paris 块
  内容 + 同一共享尾。同 salt 存储：chunk 0-2 哈希相同 → 去重共享文件；
  store_C 的磁盘增量恰为 fC3（唯一新文件，无歧义）；store_B 的增量为 P_B 的 4 个对象。
  攻击：把 fC3 的密文内容逐一复制到 P_B 的 4 个对象文件（每次试验后恢复字节）。
  - 落在 chunk3（位置对齐）→ 引擎状态 == 诚实请求 P_C → 判据 O == O_C_hit
    且 != O_B_true/O_B_hit（静默错用；续写应给出 Paris 而非 Tokyo）。
  - 落在 chunk0-2（错位）→ 静默错位污染（输出既非诚实 B 也非诚实 C）。
  对照：翻 1 字节 / 截断 64 字节 → 应被拒绝（InvalidTag/malformed → 重算 → O == O_B_true）。

用法：.venv/bin/python exp_010_swap/run_swap.py
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
SALT = "swapsalt"
MAX_TOKENS = 24
N_TOKENS = 1024
SHARED = 768
REPS_PER_TRIAL = 2
CONTROL_REPS = 3
LOG_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")


def sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


class Stack:
    def __init__(self) -> None:
        self.run_dir = os.path.join(EXP, "runtime", time.strftime("%Y%m%d_%H%M%S"))
        self.disk_dir = os.path.join(self.run_dir, "disk")
        os.makedirs(self.disk_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "lmcache.log")
        self.procs: list[subprocess.Popen] = []

    def start(self) -> None:
        master_key = os.path.join(self.run_dir, "master.key")
        open(master_key, "wb").write(os.urandom(16))
        os.chmod(master_key, 0o600)
        l2 = json.dumps({"type": "fs", "base_path": self.disk_dir,
                         "serde": {"type": "aesgcm", "key_provider": "hkdf",
                                   "master_key_path": master_key, "aes_bits": 128}})
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
            env=venv, stdout=open(os.path.join(self.run_dir, "vllm.log"), "w"),
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
                raise RuntimeError(f"timeout {url}")

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=25)
            except Exception:
                p.kill()

    def log_lines(self) -> list[str]:
        return open(self.log_path, errors="replace").readlines()

    def count_stored(self) -> int:
        return sum(1 for ln in self.log_lines() if "Stored " in ln and "tokens" in ln)

    def disk_files(self) -> dict[str, str]:
        return {f: sha256_file(os.path.join(self.disk_dir, f))
                for f in sorted(os.listdir(self.disk_dir)) if f.endswith(".data")}

    def wait_store(self, prev_stored: int, min_files: int, timeout_s: float = 120.0) -> bool:
        t0 = time.time()
        stable, last = 0, -1
        while time.time() - t0 < timeout_s:
            if self.count_stored() > prev_stored and len(self.disk_files()) >= min_files:
                n = len(self.disk_files())
                if n == last:
                    stable += 1
                    if stable >= 2:
                        return True
                else:
                    stable = 0
                last = n
            time.sleep(0.3)
        return False

    def clear_l1(self) -> bool:
        try:
            return httpx.post(f"http://localhost:{LMC_HTTP}/cache/clear", timeout=30).status_code == 200
        except Exception:
            return False

    def prefetch_for(self, rid: str) -> list[dict]:
        out = []
        for ln in self.log_lines():
            if "Prefetch request completed" in ln and rid and rid in ln:
                m = re.search(r"(\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\)", ln)
                if m:
                    out.append(dict(retained=int(m.group(1)), requested=int(m.group(2)),
                                    l1=int(m.group(3)), l2=int(m.group(4))))
        return out

    def reject_lines_since(self, since_epoch: float) -> list[str]:
        out = []
        for ln in self.log_lines():
            m = LOG_TS.search(ln)
            if not m:
                continue
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1000
            if ts >= since_epoch - 1 and ("InvalidTag" in ln or "malformed" in ln.lower()
                                          or "fail" in ln.lower() or "ERROR" in ln
                                          or "WARNING" in ln):
                out.append(ln.strip()[:220])
        return out


def request_once(client: httpx.Client, token_ids: list[int], salt: str, kind: str,
                 target_file: str | None = None) -> dict:
    t_send = time.time()
    t0 = time.perf_counter()
    rid, text, err = None, "", None
    try:
        r = client.post(f"http://localhost:{VLLM_PORT}/v1/completions",
                        json={"model": MODEL, "prompt": token_ids, "max_tokens": MAX_TOKENS,
                              "temperature": 0.0, "cache_salt": salt}, timeout=180.0)
        r.raise_for_status()
        d = r.json()
        text = d["choices"][0]["text"]
        rid = d.get("id")
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:200]
    return dict(kind=kind, salt=salt, prompt_len=len(token_ids), t_send_epoch=t_send,
                total_ms=round((time.perf_counter() - t0) * 1000, 1), output_text=text,
                response_id=rid, error=err, target_file=target_file)


def build_prompts(tok):
    # base：中性填充（避免与块内容竞争续写）；块尾共享 " The city of"，
    # 续写直接吐出城市名（P_B→Tokyo，P_C→Paris），最大化判别力。
    filler = ("The archive room was quiet, and the clerks catalogued every document "
              "with the same patient routine as they had done for many years. ")
    paris = ("Years later, a grand treaty conference was convened in Paris, the capital "
             "of France, and the delegations signed the accord in the great hall. ")
    tokyo = ("Years later, a grand treaty conference was convened in Tokyo, the capital "
             "of Japan, and the delegations signed the accord in the great hall. ")

    base_ids = (tok(filler, add_special_tokens=False).input_ids * 40)[:SHARED]
    assert len(base_ids) == SHARED, f"base {len(base_ids)} != {SHARED}"

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
    assert b_ids[:SHARED] == c_ids[:SHARED]
    assert b_ids[-k:] == c_ids[-k:]
    assert b_ids[SHARED:N_TOKENS - k] != c_ids[SHARED:N_TOKENS - k]
    return b_ids, c_ids


def main() -> int:
    os.makedirs(RAW, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    b_ids, c_ids = build_prompts(tok)
    print(f"prompts {N_TOKENS} tok; shared {SHARED}; tails equal: {b_ids[-4:] == c_ids[-4:]}")
    stack = Stack()
    stack.start()
    client = httpx.Client()
    rows: list[dict] = []
    try:
        # 预热（其文件计入基线快照 W0，不参与试验）
        for i in range(2):
            rows.append(request_once(client, b_ids[:512], f"warmup-{i}", "warmup"))
        stack.wait_store(stack.count_stored(), len(stack.disk_files()))
        w0 = set(stack.disk_files())
        print(f"warmup ok (W0={len(w0)} files)")

        # store P_B：增量恰 4 个对象
        prev = stack.count_stored()
        rows.append(request_once(client, b_ids, SALT, "store_B"))
        assert stack.wait_store(prev, len(w0) + 4), "store_P_B 未完成"
        delta_b = set(stack.disk_files()) - w0
        assert len(delta_b) == 4, f"store_B 增量 {len(delta_b)} != 4"
        print(f"store_B ok: +4 objects")

        # 基线输出
        rows.append(request_once(client, b_ids, "true-B", "O_B_true"))
        stack.clear_l1()
        rows.append(request_once(client, b_ids, SALT, "O_B_hit"))
        o_b_true = rows[-2]["output_text"]
        o_b_hit = rows[-1]["output_text"]
        print(f"O_B_true={o_b_true[:70]!r}")
        print(f"O_B_hit ={o_b_hit[:70]!r} ==true:{o_b_hit == o_b_true}")

        # store P_C：增量恰 1（= fC3；chunk0-2 去重共享）
        prev = stack.count_stored()
        before = set(stack.disk_files())
        rows.append(request_once(client, c_ids, SALT, "store_C"))
        assert stack.wait_store(prev, len(before) + 1), "store_P_C 未完成（期望增量 1）"
        delta = set(stack.disk_files()) - before
        assert len(delta) == 1, f"P_C 增量 {len(delta)} != 1"
        f_c3 = delta.pop()
        disk_after_c = stack.disk_files()
        print(f"fC3 identified: {f_c3[:50]}..")

        # O_C_true / O_C_hit
        rows.append(request_once(client, c_ids, "true-C", "O_C_true"))
        stack.clear_l1()
        rows.append(request_once(client, c_ids, SALT, "O_C_hit"))
        o_c_true = rows[-2]["output_text"]
        o_c_hit = rows[-1]["output_text"]
        print(f"O_C_true={o_c_true[:70]!r}")
        print(f"O_C_hit ={o_c_hit[:70]!r} ==true:{o_c_hit == o_c_true}")

        # ===== 攻击：fC3 内容逐一互换到 P_B 的 4 个对象文件 =====
        src = os.path.join(stack.disk_dir, f_c3)
        verdicts = []
        for target in sorted(delta_b):
            dst = os.path.join(stack.disk_dir, target)
            backup = open(dst, "rb").read()
            open(dst, "wb").write(open(src, "rb").read())
            outs = []
            for i in range(REPS_PER_TRIAL):
                stack.clear_l1()
                r = request_once(client, b_ids, SALT, "swap", target_file=target)
                r["prefetch"] = stack.prefetch_for(r.get("response_id") or "")
                rows.append(r)
                outs.append(r["output_text"])
                if i == 0:
                    hits = [(p["l1"], p["l2"]) for p in r["prefetch"]]
                    print(f"  swap->{target[:30]}..: hit={hits} out={r['output_text'][:56]!r}")
            aligned = all(o == o_c_hit for o in outs)
            misaligned = all(o != o_c_hit and o != o_b_true and o != o_b_hit for o in outs)
            verdicts.append(dict(target_file=target,
                                 hash_orig=hashlib.sha256(backup).hexdigest()[:16],
                                 aligned=aligned, misaligned=misaligned,
                                 deterministic=len(set(outs)) == 1, outputs=outs))
            open(dst, "wb").write(backup)
        n_aligned = sum(1 for v in verdicts if v["aligned"])
        n_misaligned = sum(1 for v in verdicts if v["misaligned"])
        print(f"PRIMARY: aligned trials={n_aligned}/1; misaligned silent-corruption trials={n_misaligned}/3")

        now_files = stack.disk_files()
        rewritten = [f for f in disk_after_c if now_files.get(f) != disk_after_c[f]]
        print(f"disk files changed after swap trials: {len(rewritten)} "
              f"(引擎 kv_both 读回后重写属正常，记录 IV 变化): {rewritten}")
        # 恢复后诚实复测：输出必须回到 Tokyo
        recheck_ok = True
        for i in range(2):
            stack.clear_l1()
            r = request_once(client, b_ids, SALT, "O_B_recheck")
            rows.append(r)
            ok = r["output_text"] == o_b_true
            recheck_ok = recheck_ok and ok
            print(f"  recheck#{i}: out==O_B_true:{ok} out={r['output_text'][:50]!r}")
        print(f"REVERSIBLE: {recheck_ok}")

        # ===== 对照：翻 1 字节 / 截断（应被拒绝 → 重算 → O == O_B_true）=====
        control_results = []
        for cname, mutate in (
                ("bitflip", lambda b: b[:64] + bytes([b[64] ^ 0xFF]) + b[65:]),
                ("truncate", lambda b: b[:-64])):
            dst = os.path.join(stack.disk_dir, sorted(delta_b)[0])
            orig = open(dst, "rb").read()
            open(dst, "wb").write(mutate(orig))
            for i in range(CONTROL_REPS):
                stack.clear_l1()
                r = request_once(client, b_ids, SALT, f"control_{cname}")
                r["reject_lines"] = stack.reject_lines_since(r["t_send_epoch"])
                r["output_matches_true"] = r["output_text"] == o_b_true
                rows.append(r)
                control_results.append(r)
                print(f"  {cname}#{i}: reject_logs={len(r['reject_lines'])} "
                      f"out==true:{r['output_matches_true']} out={r['output_text'][:46]!r}")
            open(dst, "wb").write(orig)

        with open(os.path.join(RAW, "swap_results.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        verdict = {
            "n_aligned_trials": n_aligned,
            "n_misaligned_trials": n_misaligned,
            "swap_all_deterministic": all(v["deterministic"] for v in verdicts),
            "primary_silent_misuse": n_aligned >= 1,
            "reversible_recheck": recheck_ok,
            "files_rewritten_after_load": len(rewritten),
            "bitflip_all_rejected": all(r["reject_lines"] and r["output_matches_true"]
                                        for r in control_results if r["kind"] == "control_bitflip"),
            "truncate_all_rejected": all(r["reject_lines"] and r["output_matches_true"]
                                         for r in control_results if r["kind"] == "control_truncate"),
            "trials": verdicts,
        }
        with open(os.path.join(RAW, "verdict.json"), "w") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in verdict.items() if k != "trials"},
                         indent=1, ensure_ascii=False))
        print("EXP010_OK")
        return 0
    finally:
        stack.stop()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
