#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_009: 修正版原型扫描（评审意见第 3 项）。

修正点（相对 exp_005/007）：
  1. 每个变体每轮恰好执行一次：run_single() 只跑一个 rep；外层按轮转顺序调用，
     真实执行顺序记录在结果里（exp_005 的 run_variant 内层循环了全部 reps、
     外层又只保留一条——隐藏执行影响预热与负载，已废除）。
  2. 边界细化：P ∈ {8192, 12288, 16384, 20480}。
  3. aead_layer_batch：批量 I/O 基线（顺序 16MB 缓冲读 + 逐记录认证 + 固定预取窗口），
     取代"逐记录 pread"的读取路径。
  4. W 扫描只称"窗口扫描"。

用法：
  .venv/bin/python exp_009_psweep_fixed/run_fixed.py boundary --reps 7
  .venv/bin/python exp_009_psweep_fixed/run_fixed.py windowsweep --reps 7
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

SUFFIX_TOKENS = 69
BLOCK_TOKENS = 256
AES_BITS = 128
HKDF_PREFIX = b"exp004-aesgcm-verify"
SALT = "tenant-a"
MASTER_KEY = b"exp004-local-test-master-key-32b!!"
IV_LEN, TAG_LEN, VERSION, HDR = 12, 16, 1, 13
BATCH_BUF = 16 << 20
FADV_DONTNEED = 4


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
               tamper_rec: int | None, t0: int, cold: bool, batch_io: bool) -> None:
    """预取进程。batch_io=False：逐记录 os.pread（exp_005 路径）；
    batch_io=True：顺序 16MB 缓冲读 + 逐记录认证（批量 I/O 基线）。"""
    key = derive_key(SALT) if use_aead else None
    with open(index_path) as f:
        index = json.load(f)["records"]
    if batch_io:
        fobj = open(blob_path, "rb", buffering=BATCH_BUF)
    else:
        fd = os.open(blob_path, os.O_RDONLY)
        if cold:
            os.posix_fadvise(fd, 0, 0, FADV_DONTNEED)

    def read_at(off: int, n: int) -> bytes:
        if batch_io:
            return fobj.read(n)  # 顺序读：调用方保证按 offset 顺序取
        return os.pread(fd, n, off)

    try:
        cursor = 0
        for rec in index:
            assert rec["offset"] == cursor, "batch_io 需要顺序记录布局"
            tr = {"rec_id": rec["rec_id"]}
            tr["read0"] = now() - t0
            raw = read_at(rec["offset"], rec["seal_len"])
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
            cursor += rec["seal_len"]
    finally:
        if batch_io:
            fobj.close()
        else:
            os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["boundary", "windowsweep"])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--W", type=int, default=2, dest="window",
                    help="预取窗口（windowsweep 每次调用测一个 W）")
    ap.add_argument("--cold", action="store_true")
    args = ap.parse_args()
    reps = args.reps
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    torch.set_grad_enabled(False)
    WINDOW = args.window

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

    suffix_ids_all = tok(("Question: who was the first emperor of Rome and when did he "
                          "rule? Answer briefly.") * 4,
                         return_tensors="pt", add_special_tokens=False).input_ids[0]
    S = min(SUFFIX_TOKENS, suffix_ids_all.shape[0])
    suffix_ids = suffix_ids_all[:S].unsqueeze(0).cuda()
    base_text = ("The history and significance of the Roman empire spans more than a "
                 "thousand years and profoundly shaped Western civilization. " * 2000)

    if args.mode == "boundary":
        ps = [8192, 12288, 16384, 20480]
        variants = [("recompute", None), ("plain_layer", False), ("aead_layer", True),
                    ("aead_layer_batch", True), ("aead_layer_block", True),
                    ("aead_whole", True)]
        out_name = "boundary_results.json"
    else:
        ps = [8192]
        variants = [("aead_layer", True), ("aead_layer_batch", True)]
        out_name = f"windowsweep_W{args.window}_results.json"
    results = {"meta": {"mode": args.mode, "reps": reps, "n_layers": n_layers,
                        "n_kv": n_kv, "hd": hd, "ctx_limit": ctx_limit,
                        "window": WINDOW, "cold": args.cold,
                        "note": "windowsweep≠完整强基线：批量I/O已实现(顺序16MB缓冲读)，"
                                "但线程/批次仍未扫描"},
               "points": []}

    for P in ps:
        point = {"P_requested": P, "variants": {}, "store": {}, "peak_gpu_mem": {},
                 "execution_order": []}
        prefix_ids = tok(base_text, return_tensors="pt",
                         add_special_tokens=False).input_ids[0][:P].unsqueeze(0).cuda()
        P_real = prefix_ids.shape[1]
        point["P_real"] = P_real
        point["S_real"] = S
        assert P_real + S <= ctx_limit

        out = model(prefix_ids, use_cache=True)
        pk = out.past_key_values
        gold_kv = [(pk.layers[i].keys.contiguous(), pk.layers[i].values.contiguous())
                   for i in range(n_layers)]
        del out
        torch.cuda.empty_cache()
        kv_layer_bytes = 2 * n_kv * P_real * hd * 2
        point["kv_bytes_total"] = kv_layer_bytes * n_layers
        point["kv_bytes_per_token"] = kv_layer_bytes * n_layers / P_real
        torch.cuda.reset_peak_memory_stats()

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
        want = {"plain_layer", "aead_whole", "aead_layer", "aead_layer_batch", "aead_layer_block"}
        for gname, defs in granularities.items():
            for use_aead in (True, False):
                tag = ("aead_" if use_aead else "plain_") + gname
                if tag not in want:
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
                os.replace(blob + ".tmp", blob)
                if args.cold:
                    bfd = os.open(blob, os.O_RDONLY)
                    os.posix_fadvise(bfd, 0, 0, FADV_DONTNEED)
                    os.close(bfd)
                with open(os.path.join(DATA, f"P{P_real}_{tag}.index.json"), "w") as f:
                    json.dump({"records": records}, f)
                point["store"][tag] = {"records": len(records), "bytes": offset,
                                       "store_fsync_ms": (now() - t0s) / 1e6}

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

        def run_single(tag: str, use_aead: bool, batch_io: bool, rep: int,
                       tamper_rec: int | None = None) -> dict:
            """恰好执行一次恢复+续算（修复 exp_005 的嵌套重复）。"""
            storage_tag = "aead_layer" if tag == "aead_layer_batch" else tag
            idx_path = os.path.join(DATA, f"P{P_real}_{storage_tag}.index.json")
            blob_path = os.path.join(DATA, f"P{P_real}_{storage_tag}.blob")
            index, need = load_index(storage_tag)
            cache = DynamicCache(config=cfg)
            ctx = mp.get_context("fork")
            q = ctx.Queue(maxsize=WINDOW)
            torch.cuda.synchronize()
            t0 = now()
            w = ctx.Process(target=worker_run,
                            args=(idx_path, blob_path, q, use_aead, tamper_rec, t0,
                                  args.cold, batch_io))
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
                mk = {"config": cfg, "inputs_embeds": emb, "attention_mask": None,
                      "past_key_values": cache, "position_ids": position_ids}
                mask_map = {"full_attention": create_causal_mask(**mk)}
                pe = m.rotary_emb(emb, position_ids)
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
                                        position_embeddings=pe)
                    h = lout[0] if isinstance(lout, tuple) else lout
                    torch.cuda.synchronize()
                    layer_end[li] = now() - t0
            if error is None:
                logits = model.lm_head(m.norm(h)[:, -1:])
                torch.cuda.synchronize()
                t_end = now() - t0
                w.join(timeout=15)
                row = dict(rep=rep, ttft_proxy_ms=t_end / 1e6,
                           first_tok=int(logits[0, 0].argmax().item()), error=None,
                           timeline=dict(layer_recv_us=layer_recv, layer_start_us=layer_start,
                                         layer_end_us=layer_end))
                if use_aead and rep == 0:
                    bad = [li for li in range(n_layers)
                           if not (torch.equal(cache.layers[li].keys[:, :, :P_real].contiguous()
                                               .view(torch.uint8), gold_kv[li][0].view(torch.uint8))
                                   and torch.equal(cache.layers[li].values[:, :, :P_real].contiguous()
                                                   .view(torch.uint8), gold_kv[li][1].view(torch.uint8)))]
                    row["kv_byte_identity"] = len(bad) == 0
                    row["kv_identity_bad_layers"] = bad
                return row
            w.terminate()
            w.join(timeout=15)
            return dict(rep=rep, ttft_proxy_ms=None,
                        error=f"record {error[0]}: {error[1]}",
                        tamper_rejected=True, output_committed=False)

        # recompute 基线（每轮一次）
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

        # 修复后的轮转：每轮每个变体恰好一次，记录真实执行顺序
        recovery = [v for v in variants if v[0] != "recompute"]
        for rep in range(reps):
            order = recovery[rep % len(recovery):] + recovery[:rep % len(recovery)]
            point["execution_order"].append({"rep": rep, "order": [t for t, *_ in order]})
            for tag, use_aead in order:
                batch_io = tag.endswith("_batch")
                row = run_single(tag, use_aead, batch_io, rep)
                row["exec_index"] = len(point["execution_order"]) - 1
                point["variants"].setdefault(tag, []).append(row)

        for tag, _ in recovery:
            ok = [r for r in point["variants"][tag] if r["error"] is None]
            if ok:
                ts = [r["ttft_proxy_ms"] for r in ok]
                print(f"[P={P_real}] {tag:18s} median {statistics.median(ts):7.0f}ms "
                      f"min {min(ts):7.0f} max {max(ts):7.0f} SC2={ok[0].get('kv_byte_identity', 'n/a')}")
        med = statistics.median(r["ttft_proxy_ms"] for r in rec_rows)
        print(f"[P={P_real}] recompute          median {med:7.0f}ms")
        point["peak_gpu_mem"]["recovery_peak_MiB"] = torch.cuda.max_memory_allocated() / 2**20

        tam = run_single("aead_layer_block", True, False, 0, tamper_rec=3)
        point["sc3_tamper_rejected"] = bool(tam.get("tamper_rejected", False))
        point["sc3_error"] = tam.get("error")
        print(f"[P={P_real}] SC3 rejected: {point['sc3_tamper_rejected']} ({point['sc3_error']})")

        results["points"].append(point)
        gold_kv.clear()
        torch.cuda.empty_cache()

    with open(os.path.join(RAW, out_name), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"EXP009_{args.mode.upper()}_OK -> {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
