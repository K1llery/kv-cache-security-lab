#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_004: 第一周最小原型——真实 KV → AES-GCM 保存 → 释放 → 认证恢复 → 继续推理。

背景：exp_003 证明 LMCache MP 加密路径被 WSL2 CUDA IPC 阻塞（环境问题）。
按 10 号文件预案，本原型以单进程方式建立"实验者实际拥有的加密缓存→恢复→推理闭环"，
回答第一周问题：**是否存在"数据已可读、但因认证组织方式不能及时计算"的可观测等待？**

变体（恢复侧全部 verified-before-use；recompute 无缓存概念）：
  recompute        重算前缀（不读缓存）
  plain_layer      明文逐层记录 + 预取窗口 W（无认证，性能归因）
  aead_whole       整份 KV = 1 条 AEAD 记录（收齐并认证完才能用任何层）
  aead_layer       每层 1 条记录 + 预取 W
  aead_layer_block 每(层, 256-token 块) 1 条记录 + 预取 W
  aead_layer_tamper 故障注入：翻 1 字节 → InvalidTag → 终止请求、不提交输出

Sanity checks（08 纪律 Step 3）：
  SC1 zero-cost：明文记录路径端到端可跑
  SC2 字节一致：AEAD 恢复的前缀 KV 与保存前逐位相同
  SC3 故障终止：篡改记录导致请求终止且无输出提交
  SC4 首 token：resume 与 recompute 的下一个 token id（报告，不强求一致——
      sdpa 形状差异可引入浮点非结合性，同上游示例 README 的说明）

计时：wall-clock perf_counter_ns（CLOCK_MONOTONIC，跨进程可比）；
GPU 段前后 torch.cuda.synchronize()；本次各变体固定顺序执行，存在顺序敏感性；
页缓存为热态（无 root，冷读留作后续受控实验——a3 警告已遵守）。

用法：.venv/bin/python exp_004_firstweek_measure/run_firstweek.py [--quick]
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

# 60 秒无进展则自动向 stderr 打印全部线程栈（排障用；不影响正常路径）
faulthandler.dump_traceback_later(90, repeat=True)

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

PREFIX_TOKENS = 1024
SUFFIX_TOKENS = 128
BLOCK_TOKENS = 256
AES_BITS = 128
HKDF_PREFIX = b"exp004-aesgcm-verify"
SALT = "tenant-a"
WINDOW_W = 2
REPS = 5
MASTER_KEY = b"exp004-local-test-master-key-32b!!"
IV_LEN, TAG_LEN, VERSION, HDR = 12, 16, 1, 13


def now() -> int:
    return time.perf_counter_ns()


def derive_key(salt: str) -> bytes:
    return HKDF(algorithm=SHA256(), length=AES_BITS // 8, salt=None,
                info=HKDF_PREFIX + salt.encode()).derive(MASTER_KEY)


def seal(key: bytes, pt: bytes) -> bytes:
    """帧格式镜像 LMCache aesgcm：[1B version][12B iv][ct||16B tag]，AAD=None。"""
    iv = os.urandom(IV_LEN)
    return bytes([VERSION]) + iv + AESGCM(key).encrypt(iv, pt, None)


def open_seal(key: bytes, frame: bytes, pt_len: int) -> bytes:
    if len(frame) < HDR + pt_len + TAG_LEN or frame[0] != VERSION:
        raise ValueError("malformed frame")
    return AESGCM(key).decrypt(frame[1:HDR], frame[HDR:HDR + pt_len + TAG_LEN], None)


# ---------------------------------------------------------------------------
# worker：预取进程（读 + 解密认证）。绝不触碰 CUDA（fork 安全）。
# bounded Queue(maxsize=W) = 固定预取窗口 W。
# ---------------------------------------------------------------------------

def worker_run(index_path: str, blob_path: str, out_q, use_aead: bool,
               tamper_rec: int | None, t0: int) -> None:
    key = derive_key(SALT) if use_aead else None
    with open(index_path) as f:
        index = json.load(f)["records"]
    fd = os.open(blob_path, os.O_RDONLY)
    try:
        for rec in index:
            tr = {"rec_id": rec["rec_id"]}
            tr["read0"] = now() - t0
            raw = os.pread(fd, rec["seal_len"], rec["offset"])
            if tamper_rec is not None and rec["rec_id"] == tamper_rec:
                b = bytearray(raw)
                b[HDR + 4] ^= 0xFF  # 翻密文第 4 字节（IV/版本不动）
                raw = bytes(b)
            tr["read1"] = now() - t0
            if use_aead:
                tr["dec0"] = now() - t0
                try:
                    pt = open_seal(key, raw, rec["pt_len"])
                except Exception as e:  # InvalidTag/ValueError → 协议：终止整个请求
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
    ap.add_argument("--quick", action="store_true", help="reps=2 冒烟")
    args = ap.parse_args()
    reps = 2 if args.quick else REPS
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
    key = derive_key(SALT)
    results = {"meta": {"n_layers": n_layers, "n_kv": n_kv, "hd": hd, "reps": reps,
                        "window": WINDOW_W, "aes_bits": AES_BITS, "aad": None},
               "store": {}, "variants": {}}

    # ---- 负载：prefix（被缓存前缀）+ suffix（未命中后缀）----
    base_text = ("The history and significance of the Roman empire spans more than a "
                 "thousand years and profoundly shaped Western civilization. " * 200)
    prefix_ids = tok(base_text, return_tensors="pt").input_ids[0][:PREFIX_TOKENS].unsqueeze(0).cuda()
    suffix_text = ("Question: who was the first emperor of Rome and when did he rule? "
                   "Answer briefly.") * 4
    suffix_ids = tok(suffix_text, return_tensors="pt").input_ids[0][:SUFFIX_TOKENS].unsqueeze(0).cuda()
    P, S = prefix_ids.shape[1], suffix_ids.shape[1]
    results["meta"]["P"], results["meta"]["S"] = P, S

    # ---- 冷前向：生成真实前缀 KV（被缓存对象）----
    out = model(prefix_ids, use_cache=True)
    pk = out.past_key_values
    gold_kv = [(pk.layers[i].keys.contiguous(), pk.layers[i].values.contiguous())
               for i in range(n_layers)]
    gold_next = int(out.logits[0, -1].argmax().item())
    del out
    torch.cuda.empty_cache()

    kv_layer_bytes = 2 * n_kv * P * hd * 2
    print(f"P={P} S={S} L={n_layers} kv/layer={kv_layer_bytes >> 10}KiB "
          f"total={(kv_layer_bytes * n_layers) >> 20}MiB")

    # ---- STORE：三种记录组织（+明文对照），各写一份 blob+index ----
    def to_bytes(layer_ids, t0, t1):
        parts = []
        for li in layer_ids:
            k, v = gold_kv[li]
            parts.append(k[:, :, t0:t1].contiguous().view(torch.uint8).cpu().numpy().tobytes())
            parts.append(v[:, :, t0:t1].contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return b"".join(parts)

    granularities = {
        "whole": [dict(layers=list(range(n_layers)), t0=0, t1=P)],
        "layer": [dict(layers=[i], t0=0, t1=P) for i in range(n_layers)],
        "layer_block": [dict(layers=[i], t0=t, t1=min(t + BLOCK_TOKENS, P))
                        for i in range(n_layers)
                        for t in range(0, P, BLOCK_TOKENS)],
    }
    want_tags = {"aead_whole", "aead_layer", "aead_layer_block", "plain_layer"}
    for gname, defs in granularities.items():
        for use_aead in (True, False):
            tag = ("aead_" if use_aead else "plain_") + gname
            if tag not in want_tags:
                continue
            records, offset = [], 0
            blob = os.path.join(DATA, tag + ".blob")
            t0 = now()
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
            store_ms = (now() - t0) / 1e6
            os.replace(blob + ".tmp", blob)
            with open(os.path.join(DATA, tag + ".index.json"), "w") as f:
                json.dump({"records": records}, f)
            results["store"][tag] = {"records": len(records), "bytes": offset,
                                     "store_fsync_ms": store_ms}
            print(f"store {tag}: {len(records)} rec, {offset >> 10}KiB, {store_ms:.1f}ms(含fsync)")

    # ---- 恢复 + 续算 ----
    def load_index(tag):
        with open(os.path.join(DATA, tag + ".index.json")) as f:
            idx = json.load(f)["records"]
        need = {i: [] for i in range(n_layers)}
        for r in idx:
            for li in r["layers"]:
                need[li].append(r["rec_id"])
        return idx, need

    def install_segments(cache, li, items):
        """把层 li 的全部已认证记录按 token 顺序装入 cache（DynamicCache.update 追加）。"""
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

    def run_variant(tag: str, use_aead: bool, tamper_rec: int | None = None):
        idx_path = os.path.join(DATA, tag + ".index.json")
        blob_path = os.path.join(DATA, tag + ".blob")
        index, need = load_index(tag)
        rows = []
        for rep in range(reps):
            cache = DynamicCache(config=cfg)
            ctx = mp.get_context("fork")
            q = ctx.Queue(maxsize=WINDOW_W)
            torch.cuda.synchronize()
            t0 = now()
            w = ctx.Process(target=worker_run,
                            args=(idx_path, blob_path, q, use_aead, tamper_rec, t0))
            w.start()
            pending: dict[int, list] = {i: [] for i in range(n_layers)}
            worker_tr: list[dict] = []
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
                worker_tr.append(tr)
                for lx in rec["layers"]:
                    pending[lx].append((rec, pt))
                    if len(pending[lx]) == len(need[lx]):
                        layer_recv[lx] = now() - t0
                return True

            while not got_layer0 and error is None:
                if absorb(q.get()):
                    if len(pending[0]) == len(need[0]):
                        install_segments(cache, 0, pending[0])
                        got_layer0 = True
            if error is None:
                emb = m.embed_tokens(suffix_ids)
                position_ids = torch.arange(P, P + S, device=emb.device).unsqueeze(0)
                cache_position = torch.arange(P, P + S, device=emb.device)
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
                    # transformers 5.17 的 DecoderLayer 直接返回张量（旧版本返回元组）
                    h = lout[0] if isinstance(lout, tuple) else lout
                    torch.cuda.synchronize()
                    layer_end[li] = now() - t0

            if error is None:
                logits = model.lm_head(m.norm(h)[:, -1:])
                torch.cuda.synchronize()
                t_end = now() - t0
                w.join(timeout=5)
                row = dict(rep=rep, ttft_proxy_ms=t_end / 1e6,
                           first_tok=int(logits[0, 0].argmax().item()), error=None,
                           timeline=dict(layer_recv_us=layer_recv,
                                         layer_start_us=layer_start,
                                         layer_end_us=layer_end,
                                         worker_records=worker_tr))
                if rep == 0 and use_aead:
                    bad = [li for li in range(n_layers)
                           if not (torch.equal(cache.layers[li].keys[:, :, :P].contiguous()
                                               .view(torch.uint8),
                                               gold_kv[li][0].view(torch.uint8))
                                   and torch.equal(cache.layers[li].values[:, :, :P].contiguous()
                                                   .view(torch.uint8),
                                                   gold_kv[li][1].view(torch.uint8)))]
                    row["kv_byte_identity"] = len(bad) == 0
                    row["kv_identity_bad_layers"] = bad
                rows.append(row)
            else:
                w.terminate()
                w.join(timeout=5)
                rows.append(dict(rep=rep, ttft_proxy_ms=None,
                                 error=f"record {error[0]}: {error[1]}",
                                 tamper_rejected=True, output_committed=False))
        return rows

    # ---- recompute 基线 ----
    rec_rows = []
    for rep in range(reps):
        torch.cuda.synchronize()
        t0 = now()
        out = model(torch.cat([prefix_ids, suffix_ids], dim=1), use_cache=True)
        torch.cuda.synchronize()
        rec_rows.append(dict(rep=rep, ttft_proxy_ms=(now() - t0) / 1e6,
                             first_tok=int(out.logits[0, -1].argmax().item()), error=None))
        del out
    results["variants"]["recompute"] = rec_rows
    print(f"recompute median: {statistics.median(r['ttft_proxy_ms'] for r in rec_rows):.1f}ms "
          f"first_tok={rec_rows[0]['first_tok']}")

    # ---- 恢复变体（固定顺序；后续 sweep 应随机化/轮转）----
    variants = [("plain_layer", False), ("aead_whole", True),
                ("aead_layer", True), ("aead_layer_block", True)]
    for tag, use_aead in variants:
        results["variants"][tag] = run_variant(tag, use_aead)
        ok = [r for r in results["variants"][tag] if r["error"] is None]
        if ok:
            med = statistics.median(r["ttft_proxy_ms"] for r in ok)
            ident = ok[0].get("kv_byte_identity", "n/a")
            print(f"{tag} median: {med:.1f}ms first_tok={ok[0]['first_tok']} SC2字节一致={ident}")
        else:
            print(f"{tag}: ALL REPS FAILED: {results['variants'][tag][0].get('error')}")

    # ---- SC3 故障注入 ----
    results["variants"]["aead_layer_tamper"] = run_variant(
        "aead_layer_block", True, tamper_rec=3)
    trow = results["variants"]["aead_layer_tamper"][0]
    print(f"SC3 tamper: rejected={trow.get('tamper_rejected', False)} error={trow.get('error')}")

    results["meta"]["gold_next_token_prefix_only"] = gold_next
    with open(os.path.join(RAW, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("EXP004_OK ->", os.path.join(RAW, "results.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
