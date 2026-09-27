#!/usr/bin/env python
"""exp_012 analysis: 中位/尾部/离散度/差值 bootstrap CI + 预注册阈值判定。"""

import json
import os
import random
import statistics

EXP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(EXP, "raw")
POINTS = [1024, 5000, 8192]
NEG_PCT = 5      # e2e_negligible_pct
NEG_ABS = 1.0    # e2e_negligible_abs_ms
UND_CID_PCT = 10  # undecidable_ci_width_pct
B = 10000


def med(xs):
    return statistics.median(xs)


def boot_ci_diff(a: list[float], b: list[float], iters: int = B):
    """a - b 的中位差 bootstrap 95% CI（a=aad, b=orig）。"""
    rng = random.Random(20260927)
    diffs = []
    for _ in range(iters):
        ma = med(rng.choices(a, k=len(a)))
        mb = med(rng.choices(b, k=len(b)))
        diffs.append(ma - mb)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main() -> int:
    rows = [json.loads(l) for l in open(os.path.join(RAW, "overhead_results.jsonl"))]
    meta = json.load(open(os.path.join(RAW, "block_meta.json")))

    # 命中断言完整性
    l2hits = [r for r in rows if r["kind"] == "l2hit"]
    n_assert = sum(1 for r in l2hits if r.get("hit_asserted"))
    print(f"l2hit rows: {len(l2hits)}, hit_asserted: {n_assert} (必须等于行数)")

    stats = {}
    for point in POINTS:
        per = {}
        for serde in ("aesgcm", "aesgcm_aad"):
            ts = [r["ttft_ms"] for r in l2hits
                  if r["block"] == f"{serde}_P{point}" and r["ttft_ms"]]
            per[serde] = dict(median=med(ts), min=min(ts), max=max(ts), n=len(ts),
                              values=sorted(round(x, 1) for x in ts))
        a = [r["ttft_ms"] for r in l2hits if r["block"] == f"aesgcm_aad_P{point}" and r["ttft_ms"]]
        b = [r["ttft_ms"] for r in l2hits if r["block"] == f"aesgcm_P{point}" and r["ttft_ms"]]
        lo, hi = boot_ci_diff(a, b)
        diff = med(a) - med(b)
        slower = max(med(a), med(b))
        ci_width_pct = (hi - lo) / slower * 100
        if abs(diff) <= slower * NEG_PCT / 100 or (lo <= 0 <= hi and abs(diff) < NEG_ABS):
            verdict = "可忽略"
        elif ci_width_pct > UND_CID_PCT:
            verdict = "尚不能判定"
        else:
            verdict = "显著"
        per["diff"] = dict(median_diff_ms=round(diff, 1), ci95=[round(lo, 1), round(hi, 1)],
                           ci_width_pct=round(ci_width_pct, 1), verdict=verdict)
        stats[f"P{point}"] = per
        print(f"P{point}: orig med={per['aesgcm']['median']:.1f} "
              f"aad med={per['aesgcm_aad']['median']:.1f} "
              f"diff={diff:+.1f}ms CI95=[{lo:+.1f},{hi:+.1f}] -> {verdict}")
        print(f"      orig values: {per['aesgcm']['values']}")
        print(f"      aad  values: {per['aesgcm_aad']['values']}")

    # 重算基线（块内）
    for point in POINTS:
        for serde in ("aesgcm", "aesgcm_aad"):
            ts = [r["ttft_ms"] for r in rows
                  if r["block"] == f"{serde}_P{point}" and r["kind"] == "recompute" and r["ttft_ms"]]
            if ts:
                print(f"recompute P{point} ({serde} run): median {med(ts):.1f} n={len(ts)}")

    # 吞吐相（P5000，各 serde 3 波）
    thr = {}
    for serde in ("aesgcm", "aesgcm_aad"):
        walls = [r["wave_wall_ms"] for r in rows
                 if r["kind"] == "thr" and r["block"] == f"{serde}_P5000"]
        ttfts = [r["ttft_ms"] for r in rows
                 if r["kind"] == "thr" and r["block"] == f"{serde}_P5000" and r["ttft_ms"]]
        lookups_ok = all(r.get("lookup_count_in_wave", 0) >= 1 for r in rows
                         if r["kind"] == "thr" and r["block"] == f"{serde}_P5000")
        thr[serde] = dict(wave_wall_ms=[walls], median_wave_wall=med(walls),
                          ttft_under_concurrency_median=med(ttfts),
                          all_lookups_present=lookups_ok)
        print(f"throughput {serde}: waves={walls} median_wall={med(walls):.0f}ms "
              f"ttft@conc4 median={med(ttfts):.0f}ms lookups_ok={lookups_ok}")
    a_w = thr["aesgcm_aad"]["median_wave_wall"]
    o_w = thr["aesgcm"]["median_wave_wall"]
    print(f"throughput median wall diff (aad-orig): {a_w - o_w:+.0f}ms")

    # store 完成时间
    for m in meta:
        print(f"store {m['block']}: {m['store_completion_s']}s "
              f"(summary line: {m.get('stored_summary_line_seen')})")

    json.dump({"ttft": stats, "throughput": thr, "store": meta},
              open(os.path.join(RAW, "analysis_summary.json"), "w"), indent=2)
    print("ANALYSIS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
