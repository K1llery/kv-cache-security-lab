#!/usr/bin/env python
"""exp_012 组件耗时基准：AAD 编码成本 + GCM 有/无 AAD 的加解密成本（解释端到端结果）。"""

from __future__ import annotations

import json
import os
import statistics
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.aesgcm_aad import identity_aad

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def bench(fn, reps: int = 2000) -> float:
    fn()  # 预热
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts) / 1000  # µs


def main() -> int:
    os.makedirs(RAW, exist_ok=True)
    key = HKDF(algorithm=SHA256(), length=16, salt=None,
               info=b"lmcache-l2-aesgcm-v1tenant-a").derive(b"exp012-bench-key")
    okey = ObjectKey(chunk_hash=bytes(range(32)), model_name="Qwen2.5-0.5B-Instruct",
                     kv_rank=0, object_group_id=0, cache_salt="tenant-a")
    dek = key

    # 1) identity_aad 编码成本
    enc_us = bench(lambda: identity_aad(okey, 12288))
    print(f"identity_aad encode: {enc_us:.2f} µs/call")

    # 2) GCM 加/解密：AAD vs 无 AAD（载荷 12KB/48KB/96KB = 1K/4K/8K token 的 KV 量级）
    gcm = AESGCM(dek)
    results = {"aad_encode_us": enc_us, "gcm": {}}
    for name, size in (("12KB", 12 * 1024), ("48KB", 48 * 1024), ("96KB", 96 * 1024)):
        pt = os.urandom(size)
        iv = os.urandom(12)
        aad = identity_aad(okey, size)
        e_no = bench(lambda: gcm.encrypt(iv, pt, None), reps=300)
        e_ad = bench(lambda: gcm.encrypt(iv, pt, aad), reps=300)
        ct = gcm.encrypt(iv, pt, None)
        ct_ad = gcm.encrypt(iv, pt, aad)
        d_no = bench(lambda: gcm.decrypt(iv, ct, None), reps=300)
        d_ad = bench(lambda: gcm.decrypt(iv, ct_ad, aad), reps=300)
        results["gcm"][name] = {
            "encrypt_us_no_aad": round(e_no, 2), "encrypt_us_with_aad": round(e_ad, 2),
            "decrypt_us_no_aad": round(d_no, 2), "decrypt_us_with_aad": round(d_ad, 2),
            "enc_overhead_pct": round((e_ad - e_no) / e_no * 100, 2),
            "dec_overhead_pct": round((d_ad - d_no) / d_no * 100, 2),
        }
        print(f"{name}: enc {e_no:.1f}→{e_ad:.1f} µs (+{results['gcm'][name]['enc_overhead_pct']}%), "
              f"dec {d_no:.1f}→{d_ad:.1f} µs (+{results['gcm'][name]['dec_overhead_pct']}%)")

    thr = {"aad_encode_under_10us": enc_us <= 10,
           "gcm_overhead_under_5pct": all(abs(v["enc_overhead_pct"]) <= 5 and abs(v["dec_overhead_pct"]) <= 5
                                          for v in results["gcm"].values())}
    results["component_thresholds"] = thr
    print(json.dumps(thr))
    json.dump(results, open(os.path.join(RAW, "component_bench.json"), "w"), indent=2)
    print("BENCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
