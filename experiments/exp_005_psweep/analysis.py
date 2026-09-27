#!/usr/bin/env python
"""exp_005 analysis: 恢复 vs 重算 crossover 曲线（热/冷两个 regime 各一张）+ 汇总表。"""

import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(EXP, "raw")
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"recompute": "#888888", "plain_layer": "#4C72B0", "aead_whole": "#C44E52",
          "aead_layer": "#55A868", "aead_layer_block": "#8172B2"}
LABELS = {"recompute": "recompute", "plain_layer": "plaintext per-layer+pf",
          "aead_whole": "AEAD whole", "aead_layer": "AEAD per-layer+pf",
          "aead_layer_block": "AEAD per-(layer,block)+pf"}


def summarize(paths):
    out = None
    for path in paths:
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        if out is None:
            out = {"regime": r["meta"]["regime"], "points": []}
        for pt in r["points"]:
            row = {"P": pt["P_real"], "S": pt["S_real"], "kv_MiB": pt["kv_bytes_total"] / 2**20,
                   "store": pt["store"], "peak": pt["peak_gpu_mem"],
                   "sc3": pt.get("sc3_tamper_rejected"), "variants": {}}
            for tag, rows in pt["variants"].items():
                ts = [x["ttft_proxy_ms"] for x in rows if x["error"] is None]
                if ts:
                    row["variants"][tag] = {"median": statistics.median(ts), "min": min(ts),
                                            "max": max(ts), "n": len(ts)}
            out["points"].append(row)
    # 同一 P 多文件出现时合并样本
    if out:
        merged = {}
        for row in out["points"]:
            key = row["P"]
            if key in merged:
                for tag, v in row["variants"].items():
                    merged[key]["variants"].setdefault(tag, {"n": 0})
                    # 以样本量加权合并中位数的近似：直接保留样本多的那份并累计 n
                    if v["n"] >= merged[key]["variants"][tag].get("n", 0):
                        merged[key]["variants"][tag] = v
                for k in ("kv_MiB", "store", "peak", "sc3", "S"):
                    merged[key][k] = row[k]
            else:
                merged[key] = row
        out["points"] = [merged[k] for k in sorted(merged)]
    return out


warm = summarize([os.path.join(RAW, "psweep_results.json"),
                  os.path.join(RAW, "psweep_results_p32000.json")])
cold_path = os.path.join(RAW, "psweep_results_cold.json")
cold = summarize([cold_path]) if os.path.exists(cold_path) else None

for name, s in (("warm", warm), ("cold", cold)):
    if s is None:
        continue
    print(f"=== regime={s['regime']} ===")
    for row in s["points"]:
        cells = " | ".join(f"{LABELS[t]}: {v['median']:.0f}ms" for t, v in row["variants"].items())
        print(f"P={row['P']:6d} S={row['S']} KV={row['kv_MiB']:.0f}MiB :: {cells}")

# crossover 图
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, (name, s) in zip(axes, [("warm page cache", warm), ("controlled cold read (fadvise)", cold)]):
    if s is None:
        ax.set_visible(False)
        continue
    for tag in COLORS:
        xs, ys, lo, hi = [], [], [], []
        for row in s["points"]:
            v = row["variants"].get(tag)
            if v:
                xs.append(row["P"]); ys.append(v["median"]); lo.append(v["min"]); hi.append(v["max"])
        ax.plot(xs, ys, marker="o", color=COLORS[tag], label=LABELS[tag])
        ax.fill_between(xs, lo, hi, color=COLORS[tag], alpha=0.12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted({row["P"] for row in s["points"]}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("prefix tokens P (S=69, log)")
    ax.set_ylabel("TTFT proxy (ms, log)")
    ax.set_title(f"exp_005: recovery vs recompute — {name}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "crossover.png"), dpi=150)
print("saved crossover.png")

json.dump({"warm": warm, "cold": cold}, open(os.path.join(RAW, "analysis_summary.json"), "w"), indent=2)
print("ANALYSIS_OK")
