#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_005: 恢复 vs 重算的前缀长度扫描（去留关口第 1 项）。

问题（复核文档 2026-09-27）：存在前缀长度区间使"加密恢复+继续推理"快于"全量重算"吗？
若在合理前缀范围内恢复始终慢于重算 → 停止"加速恢复"叙事（Kill 条件 ①/② 判定依据）。

相对 exp_004 的方法学修正（按复核文档要求）：
  1. P ∈ {2048, 8192, 32000}（32000+69 ≤ 模型上下文上限 32768，逐点记录真实 S 与上限）；
  2. 变体轮转执行（每 rep 轮转一次顺序），消除 exp_004 固定顺序的 warmup 偏差；
  3. 记录每阶段 GPU 峰值显存（torch.cuda.max_memory_allocated）；
  4. --cold 模式：worker 读前对该 blob 执行 posix_fadvise(DONTNEED) 逐出页缓存
     （文件属主可做，无需 root），与热读分开报告；
  5. 记录每变体写入成本（store ms 与字节数）。

变体与 sanity checks 与 exp_004 完全一致（SC1-SC4 见其 claim.md；本实验运行时重跑 SC2/SC3）。

用法：
  .venv/bin/python exp_005_psweep/run_psweep.py [--reps 5] [--cold] [--quick]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

EXP = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(EXP, "data")
RAW = os.path.join(EXP, "raw")
MODEL_PATH = "/home/mr/projects/科研训练2.0/experiments/models/Qwen2.5-0.5B-Instruct"

PREFIX_LIST = [2048, 8192, 32000]
SUFFIX_TOKENS = 69
BLOCK_TOKENS = 256
AES_BITS = 128
HKDF_PREFIX = b"exp004-aesgcm-verify"
SALT = "tenant-a"
WINDOW_W = 2
MASTER_KEY = b"exp004-local-test-master-key-32b!!"
IV_LEN, TAG_LEN, VERSION, HDR = 12, 16, 1, 13
FADV_DONTNEED = 4  # POSIX_FADV_DONTNEED on Linux


def now() -> int:
    return time.perf_counter_ns()


def derive_key(salt: str) -> bytes:
    return HKDF(algorithm=SHA256(), length=AES_BITS // 8, salt=None,
                info=HKDF_PREFIX + salt.encode()).derive(MASTER_KEY)


def seal(key: bytes, pt: bytes) -> bytes:
    iv = os.urandom(IV_LEN)
    return bytes([VERSION]) + iv + AESGCM(key).encrypt(iv, pt, None)


def open_seal(key: bytes, frame: bytes, pt_len: int) -> bytes:
    if len(frame) < HDR + pt_len + TAG_LEN or frame[0] != VERSION:
        raise ValueError("malformed frame")
    return AESGCM(key).decrypt(frame[1:HDR], frame[HDR:HDR + pt_len + TAG_LEN], None)


def worker_run(index_path: str, blob_path: str, out_q, use_aead: bool,
               tamper_rec: int | None, t0: int, cold: bool) -> None:
    key = derive_key(SALT) if use_aead else None
    with open(index_path) as f:
        index = json.load(f)["records"]
    fd = os.open(blob_path, os.O_RDONLY)
    try:
        if cold:
            os.posix_fadvise(fd, 0, 0, FADV_DONTNEED)  # 受控冷读：读前逐出页缓存
        for rec in index:
            tr = {"rec_id": rec["rec_id"]}
            tr["read0"] = now() - t0
            raw = os.pread(fd, rec["seal_len"], rec["offset"])
            if tamper_rec is not None and rec["rec_id"] == tamper_rec:
                b = bytearray(raw)
                b[HDR + 4] ^= 0xFF
                raw = bytes(b)
            tr["read1"] = now() - t0
            if use_aead:
                tr["dec0"] = now() - t0
                try:
                    pt = open_seal(key, raw, rec["pt_len"])
                except Exception as e:
                    out_q.put(("ERROR", rec["rec_id"], type(e).__name__))
                    return
                tr["dec1"] = now() - t0
            else:
                pt = raw
                tr["dec0"] = tr["dec1"] = tr["read1"]
            out_q.put(("OK", rec, pt, tr))
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="受控冷读（fadvise 逐出）")
    ap.add_argument("--quick", action="store_true", help="reps=2, P=[2048, 8192]")
    ap.add_argument("--ps", type=int, nargs="+", default=None, help="覆盖 P 列表")
    ap.add_argument("--W", type=int, default=WINDOW_W, help="预取窗口（exp_007 扫描用）")
    ap.add_argument("--variants", type=str, nargs="+", default=None,
                    help="只跑指定恢复变体（recompute 恒跑）")
    ap.add_argument("--out", default="psweep_results.json")
    args = ap.parse_args()
    reps = 2 if args.quick else args.reps
    ps = [2048, 8192] if args.quick else (args.ps or PREFIX_LIST)
    window = args.W
    regime = "cold_fadvise" if args.cold else "warm_pagecache"
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    m = model.model
    cfg = m.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    ctx_limit = getattr(cfg, "max_position_embeddings", None)
    key = derive_key(SALT)
    results = {"meta": {"n_layers": n_layers, "n_kv": n_kv, "hd": hd, "reps": reps,
                        "window": window, "aes_bits": AES_BITS, "aad": None,
                        "ctx_limit": ctx_limit, "regime": regime,
                        "block_tokens": BLOCK_TOKENS},
               "points": []}
    print(f"ctx_limit={ctx_limit} regime={regime} reps={reps}")

    suffix_text = ("Question: who was the first emperor of Rome and when did he rule? "
                   "Answer briefly.") * 4
    suffix_ids_all = tok(suffix_text, return_tensors="pt").input_ids[0]
    S = min(SUFFIX_TOKENS, suffix_ids_all.shape[0])
    suffix_ids = suffix_ids_all[:S].unsqueeze(0).cuda()

    base_text = ("The history and significance of the Roman empire spans more than a "
                 "thousand years and profoundly shaped Western civilization. " * 2000)

    for P in ps:
        assert P + S <= ctx_limit, f"P={P}+S={S} exceeds ctx {ctx_limit}"
        point = {"P_requested": P, "variants": {}, "store": {}, "peak_gpu_mem": {}}
        prefix_ids = tok(base_text, return_tensors="pt").input_ids[0][:P].unsqueeze(0).cuda()
        P_real = prefix_ids.shape[1]
        point["P_real"] = P_real
        point["S_real"] = suffix_ids.shape[1]

        # ---- 冷前向：真实前缀 KV ----
        torch.cuda.synchronize()
        t0 = now()
        out = model(prefix_ids, use_cache=True)
        torch.cuda.synchronize()
        pk = out.past_key_values
        gold_kv = [(pk.layers[i].keys.contiguous(), pk.layers[i].values.contiguous())
                   for i in range(n_layers)]
        cold_fwd_ms = (now() - t0) / 1e6
        point["cold_prefill_ms"] = cold_fwd_ms
        point["peak_gpu_mem"]["prefix_cold_forward_MiB"] = \
            torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        del out
        torch.cuda.empty_cache()

        kv_layer_bytes = 2 * n_kv * P_real * hd * 2
        point["kv_bytes_total"] = kv_layer_bytes * n_layers
        point["kv_bytes_per_token"] = kv_layer_bytes * n_layers / P_real
        print(f"[P={P_real}] cold prefill {cold_fwd_ms:.0f}ms, "
              f"KV={kv_layer_bytes * n_layers >> 20}MiB "
              f"({point['kv_bytes_per_token']:.0f} B/token), peak {point['peak_gpu_mem']['prefix_cold_forward_MiB']:.0f}MiB")

        # ---- STORE ----
        def to_bytes(layer_ids, t0_, t1_):
            parts = []
            for li in layer_ids:
                k, v = gold_kv[li]
                parts.append(k[:, :, t0_:t1_].contiguous().view(torch.uint8).cpu().numpy().tobytes())
                parts.append(v[:, :, t0_:t1_].contiguous().view(torch.uint8).cpu().numpy().tobytes())
            return b"".join(parts)

        granularities = {
            "whole": [dict(layers=list(range(n_layers)), t0=0, t1=P_real)],
            "layer": [dict(layers=[i], t0=0, t1=P_real) for i in range(n_layers)],
            "layer_block": [dict(layers=[i], t0=t, t1=min(t + BLOCK_TOKENS, P_real))
                            for i in range(n_layers)
                            for t in range(0, P_real, BLOCK_TOKENS)],
        }
        for gname, defs in granularities.items():
            for use_aead in (True, False):
                tag = ("aead_" if use_aead else "plain_") + gname
                if tag not in ("aead_whole", "aead_layer", "aead_layer_block", "plain_layer"):
                    continue
                records, offset = [], 0
                blob = os.path.join(DATA, f"P{P_real}_{tag}.blob")
                t0s = now()
                with open(blob + ".tmp", "wb") as f:
                    for rec_id, d in enumerate(defs):
                        pt = to_bytes(d["layers"], d["t0"], d["t1"])
                        frame = seal(key, pt) if use_aead else pt
                        f.write(frame)
                        records.append(dict(rec_id=rec_id, layers=d["layers"], t0=d["t0"],
                                            t1=d["t1"], pt_len=len(pt), seal_len=len(frame),
                                            offset=offset))
                        offset += len(frame)
                    f.flush()
                    os.fsync(f.fileno())
                store_ms = (now() - t0s) / 1e6
                os.replace(blob + ".tmp", blob)
                if args.cold:
                    bfd = os.open(blob, os.O_RDONLY)
                    os.posix_fadvise(bfd, 0, 0, FADV_DONTNEED)
                    os.close(bfd)
                with open(os.path.join(DATA, f"P{P_real}_{tag}.index.json"), "w") as f:
                    json.dump({"records": records}, f)
                point["store"][tag] = {"records": len(records), "bytes": offset,
                                       "store_fsync_ms": store_ms}
        torch.cuda.reset_peak_memory_stats()

        # ---- 恢复 + 续算（与 exp_004 相同语义）----
        def load_index(tag):
            with open(os.path.join(DATA, f"P{P_real}_{tag}.index.json")) as f:
                idx = json.load(f)["records"]
            need = {i: [] for i in range(n_layers)}
            for r in idx:
                for li in r["layers"]:
                    need[li].append(r["rec_id"])
            return idx, need

        def install_segments(cache, li, items):
            for rec, pt in sorted(items, key=lambda x: x[0]["t0"]):
                T = rec["t1"] - rec["t0"]
                per = n_kv * T * hd * 2
                pos = 0
                for lix in rec["layers"]:
                    if lix == li:
                        k = torch.frombuffer(bytearray(pt[pos:pos + per]),
                                             dtype=torch.bfloat16).view(1, n_kv, T, hd).cuda()
                        v = torch.frombuffer(bytearray(pt[pos + per:pos + 2 * per]),
                                             dtype=torch.bfloat16).view(1, n_kv, T, hd).cuda()
                        cache.update(k, v, li)
                    pos += 2 * per

        def run_variant(tag, use_aead, tamper_rec=None):
            idx_path = os.path.join(DATA, f"P{P_real}_{tag}.index.json")
            blob_path = os.path.join(DATA, f"P{P_real}_{tag}.blob")
            index, need = load_index(tag)
            rows = []
            for rep in range(reps):
                cache = DynamicCache(config=cfg)
                ctx = mp.get_context("fork")
                q = ctx.Queue(maxsize=window)
                torch.cuda.synchronize()
                t0 = now()
                w = ctx.Process(target=worker_run,
                                args=(idx_path, blob_path, q, use_aead, tamper_rec, t0, args.cold))
                w.start()
                pending = {i: [] for i in range(n_layers)}
                error = None
                layer_recv = [None] * n_layers
                layer_start = [None] * n_layers
                layer_end = [None] * n_layers
                got_layer0 = False

                def absorb(item):
                    nonlocal error
                    if item[0] == "ERROR":
                        error = (item[1], item[2])
                        return False
                    _, rec, pt, tr = item
                    for lx in rec["layers"]:
                        pending[lx].append((rec, pt))
                        if len(pending[lx]) == len(need[lx]):
                            layer_recv[lx] = now() - t0
                    return True

                while not got_layer0 and error is None:
                    if absorb(q.get()) and len(pending[0]) == len(need[0]):
                        install_segments(cache, 0, pending[0])
                        got_layer0 = True
                if error is None:
                    emb = m.embed_tokens(suffix_ids)
                    position_ids = torch.arange(P_real, P_real + S, device=emb.device).unsqueeze(0)
                    cache_position = torch.arange(P_real, P_real + S, device=emb.device)
                    mask_kwargs = {"config": cfg, "inputs_embeds": emb, "attention_mask": None,
                                   "past_key_values": cache, "position_ids": position_ids}
                    mask_map = {"full_attention": create_causal_mask(**mask_kwargs)}
                    position_embeddings = m.rotary_emb(emb, position_ids)
                    h = emb
                    for li in range(n_layers):
                        if li > 0:
                            while len(pending[li]) < len(need[li]) and error is None:
                                absorb(q.get())
                            if error is not None:
                                break
                            install_segments(cache, li, pending[li])
                        torch.cuda.synchronize()
                        layer_start[li] = now() - t0
                        lout = m.layers[li](h, attention_mask=mask_map[cfg.layer_types[li]],
                                            position_ids=position_ids, past_key_values=cache,
                                            use_cache=True, cache_position=cache_position,
                                            position_embeddings=position_embeddings)
                        h = lout[0] if isinstance(lout, tuple) else lout
                        torch.cuda.synchronize()
                        layer_end[li] = now() - t0
                if error is None:
                    logits = model.lm_head(m.norm(h)[:, -1:])
                    torch.cuda.synchronize()
                    t_end = now() - t0
                    w.join(timeout=10)
                    row = dict(rep=rep, ttft_proxy_ms=t_end / 1e6,
                               first_tok=int(logits[0, 0].argmax().item()), error=None,
                               timeline=dict(layer_recv_us=layer_recv,
                                             layer_start_us=layer_start,
                                             layer_end_us=layer_end))
                    if rep == 0 and use_aead:
                        bad = [li for li in range(n_layers)
                               if not (torch.equal(cache.layers[li].keys[:, :, :P_real].contiguous()
                                                   .view(torch.uint8),
                                                   gold_kv[li][0].view(torch.uint8))
                                       and torch.equal(cache.layers[li].values[:, :, :P_real].contiguous()
                                                       .view(torch.uint8),
                                                       gold_kv[li][1].view(torch.uint8)))]
                        row["kv_byte_identity"] = len(bad) == 0
                        row["kv_identity_bad_layers"] = bad
                    rows.append(row)
                else:
                    w.terminate()
                    w.join(timeout=10)
                    rows.append(dict(rep=rep, ttft_proxy_ms=None,
                                     error=f"record {error[0]}: {error[1]}",
                                     tamper_rejected=True, output_committed=False))
            return rows

        variant_defs = [("recompute", None), ("plain_layer", False), ("aead_whole", True),
                        ("aead_layer", True), ("aead_layer_block", True)]
        if args.variants:
            variant_defs = [v for v in variant_defs
                            if v[0] == "recompute" or v[0] in args.variants]

        # recompute（无恢复，单独测）
        rec_rows = []
        for rep in range(reps):
            torch.cuda.synchronize()
            t0 = now()
            out = model(torch.cat([prefix_ids, suffix_ids], dim=1), use_cache=True)
            torch.cuda.synchronize()
            rec_rows.append(dict(rep=rep, ttft_proxy_ms=(now() - t0) / 1e6,
                                 first_tok=int(out.logits[0, -1].argmax().item()), error=None))
            del out
            torch.cuda.empty_cache()
        point["variants"]["recompute"] = rec_rows
        point["peak_gpu_mem"]["recompute_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        med = statistics.median(r["ttft_proxy_ms"] for r in rec_rows)
        print(f"[P={P_real}] recompute median {med:.0f}ms")

        # 恢复变体：轮转执行（rep r 从第 r 个变体开始）
        for rep in range(reps):
            order = variant_defs[1:] + variant_defs[1:]
            start = rep % len(variant_defs[1:])
            for tag, use_aead in order[start:start + len(variant_defs) - 1]:
                rows = point["variants"].setdefault(tag, [])
                rows.extend(r for r in run_variant(tag, use_aead) if r["rep"] == rep)

        for tag, _ in variant_defs[1:]:
            ok = [r for r in point["variants"][tag] if r["error"] is None]
            if ok:
                ts = [r["ttft_proxy_ms"] for r in ok]
                print(f"[P={P_real}] {tag:18s} median {statistics.median(ts):7.0f}ms "
                      f"min {min(ts):7.0f} max {max(ts):7.0f} SC2={ok[0].get('kv_byte_identity', 'n/a')}")
        point["peak_gpu_mem"]["recovery_peak_MiB"] = torch.cuda.max_memory_allocated() / 2**20

        # SC3 在每个 P 的 aead_layer_block 上做一次
        tam = run_variant("aead_layer_block", True, tamper_rec=3)
        point["sc3_tamper_rejected"] = bool(tam[0].get("tamper_rejected", False))
        point["sc3_error"] = tam[0].get("error")
        print(f"[P={P_real}] SC3 tamper rejected: {point['sc3_tamper_rejected']} ({point['sc3_error']})")

        results["points"].append(point)
        # 释放当前 P 的 gold KV，进入下一个 P
        del gold_kv[:]
        torch.cuda.empty_cache()

    with open(os.path.join(RAW, args.out), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("EXP005_OK ->", os.path.join(RAW, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
