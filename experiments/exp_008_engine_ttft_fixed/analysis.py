#!/usr/bin/env python
"""exp_008 analysis: 离线重断言（Stored 宽窗口）+ 汇总表 + 原型/引擎对照。

在线断言中 recompute 组的 "Stored 行" 检查窗口只到响应结束；异步写完成可能晚于
响应（本 run1/run2 均有 4-1 例伪影）。此处以最终日志在 [t_send, t_send+120s]
内重查 Stored 行；命中断言（prefetch 行）维持原口径不变。
"""

import glob
import json
import os
import re
import statistics
import time

EXP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(EXP, "raw")
LOG_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")


def epoch_of(line: str) -> float | None:
    m = LOG_TS.search(line)
    if not m:
        return None
    return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1000


def stack_log_map() -> dict:
    """stack_id -> 最新 runtime 日志路径（目录名 = <stack_id>_<date>_<time>）。"""
    out = {}
    for d in sorted(glob.glob(os.path.join(EXP, "runtime", "*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d):
            continue
        sid = base.rsplit("_", 2)[0]  # 去掉 _YYYYMMDD_HHMMSS
        lp = os.path.join(d, "lmcache.log")
        if os.path.exists(lp):
            out[sid] = lp  # 排序后覆盖 → 最新
    return out


def main() -> int:
    rows = [json.loads(l) for l in open(os.path.join(RAW, "engine_ttft_fixed.jsonl"))]
    logs = stack_log_map()
    fixed = 0
    for r in rows:
        a = r.get("assertion")
        if not a or a.get("asserted_ok") or not r.get("response_id"):
            continue
        lp = logs.get(r["stack"])
        if not lp:
            continue
        stored_ok = False
        for line in open(lp, errors="replace"):
            ts = epoch_of(line)
            if ts is None:
                continue
            if r["t_send_epoch"] - 1 <= ts <= r["t_send_epoch"] + 120 \
                    and "Stored " in line and "tokens" in line:
                stored_ok = True
                break
        if r["kind"] == "recompute" and stored_ok:
            a["asserted_ok"] = True
            a["detail"] += " | offline: Stored line found in widened window"
            fixed += 1
    print(f"offline re-assertion fixed {fixed} rows")

    valid = [r for r in rows if r.get("assertion", {}).get("asserted_ok")
             and r["kind"] in ("l2hit_warm", "l2hit_cold", "recompute") and r["ttft_ms"]]
    failed = [r for r in rows if r.get("assertion") and not r["assertion"]["asserted_ok"]]

    summary = {}
    for stack in sorted({r["stack"] for r in valid}):
        for kind in ("l2hit_warm", "l2hit_cold", "recompute"):
            for pl in sorted({r["prompt_len"] for r in valid if r["stack"] == stack}):
                ts = sorted(r["ttft_ms"] for r in valid
                            if r["stack"] == stack and r["kind"] == kind
                            and r["prompt_len"] == pl)
                if not ts:
                    continue
                med = statistics.median(ts)
                summary[f"{stack}/{kind}/P{pl}"] = {
                    "n": len(ts), "median_ms": round(med, 1),
                    "min_ms": round(ts[0], 1), "max_ms": round(ts[-1], 1)}
    for k, v in summary.items():
        print(f"{k:34s} n={v['n']:2d} median={v['median_ms']:8.1f} "
              f"[{v['min_ms']:8.1f}..{v['max_ms']:8.1f}]")
    print(f"still-failed rows: {len(failed)}")
    for r in failed:
        print("  FAIL:", r["stack"], r["kind"], "P"+str(r["prompt_len"]),
              r["assertion"]["detail"][:100])

    json.dump({"summary": summary, "n_fixed_offline": fixed,
               "still_failed": len(failed)},
              open(os.path.join(RAW, "engine_ttft_fixed_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print("EXP008_ANALYSIS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
