#!/usr/bin/env python
"""exp_004 analysis: 从 raw/results.json 出图与等待分解（原始数据→图，不引入新假设）。

时间线原始值单位为 ns（键名 _us 是历史命名，见 run_firstweek.py）。
等待分解口径（v2，修正 v1 的两个错误：单位换算与"暴露"定义）：
  到达时间 arrival[0] ≈ worker 首记录 read0（主循环到层0的时点）；
  arrival[li] = layer_end[li-1]（上一层算完，主循环来取第 li 层数据的时点）。
  exposed_auth_wait[li] = max(0, recv[li] - arrival[li])   # 数据晚到，主循环阻塞=关键路径
  hidden_margin[li]     = max(0, arrival[li] - recv[li])   # 数据早到，等待已被计算隐藏
  注：layer_start[li] 记录于阻塞之后，不能用于判断是否等过（v1 的错误来源）。

输出：
  figures/e2e_compare.png   五变体端到端（中位 + min/max 胡须）
  figures/timeline.png      aead_whole 与 aead_layer 各 rep0 的逐层时间线
  raw/analysis_summary.json 分解指标
"""

import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(EXP, "raw")
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)

R = json.load(open(os.path.join(RAW, "results.json")))
V = R["variants"]
meta = R["meta"]
NS2MS = 1e6

# ---- 1. 端到端对比 ----
variants = ["recompute", "plain_layer", "aead_whole", "aead_layer", "aead_layer_block"]
labels = {"recompute": "recompute\n(no cache)", "plain_layer": "plaintext\nper-layer+pf",
          "aead_whole": "AEAD whole\n(1 record)", "aead_layer": "AEAD per-layer\n+pf W=2",
          "aead_layer_block": "AEAD per-(layer,block)\n+pf W=2"}
meds, los, his = [], [], []
for v in variants:
    ts = [r["ttft_proxy_ms"] for r in V[v] if r["error"] is None]
    meds.append(statistics.median(ts))
    los.append(min(ts))
    his.append(max(ts))
    print(f"{v:20s} median={statistics.median(ts):7.1f}ms  min={min(ts):7.1f}  max={max(ts):7.1f}  n={len(ts)}")

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.bar(range(len(variants)), meds, color=["#888888", "#4C72B0", "#C44E52", "#55A868", "#8172B2"])
ax.errorbar(range(len(variants)), meds,
            yerr=[[m - l for m, l in zip(meds, los)], [h - m for h, m in zip(his, meds)]],
            fmt="none", ecolor="black", capsize=4)
ax.set_xticks(range(len(variants)))
ax.set_xticklabels([labels[v] for v in variants], fontsize=8.5)
ax.set_ylabel("TTFT proxy (ms)\n[P=%d tokens prefix, S=%d suffix, warm page cache]" % (meta["P"], meta["S"]))
ax.set_title("exp_004: end-to-end first-token latency by recovery organization (n=%d reps, min-max whiskers)" % meta["reps"], fontsize=10)
for i, m in enumerate(meds):
    ax.text(i, m + 3, f"{m:.0f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "e2e_compare.png"), dpi=150)
print("saved e2e_compare.png")

# ---- 2. 时间线（rep0）：认证可见时点 vs 层消费 ----
fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4))
for ax, tag in zip(axes, ["aead_whole", "aead_layer"]):
    tl = V[tag][0]["timeline"]
    n_layers = meta["n_layers"]
    recv = [x / NS2MS for x in tl["layer_recv_us"]]
    st = [x / NS2MS for x in tl["layer_start_us"]]
    en = [x / NS2MS for x in tl["layer_end_us"]]
    for li in range(n_layers):
        ax.barh(li, max(en[li] - st[li], 0.02), left=st[li], height=0.6,
                color="#55A868", label="layer compute" if li == 0 else None)
        ax.plot(recv[li], li, marker="|", color="#C44E52", markersize=10, mew=2,
                label="auth complete (visible)" if li == 0 else None)
    ax.set_yticks(range(0, n_layers, 4))
    ax.set_ylabel("layer")
    ax.set_title(f"{tag} rep0 (green=compute span; red tick=auth-visible time)", fontsize=9)
    ax.invert_yaxis()
axes[1].set_xlabel("ms since request start")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("exp_004 timeline: authentication visibility vs layer consumption", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "timeline.png"), dpi=150)
print("saved timeline.png")

# ---- 3. 等待分解（v2 口径）----
summary = {"end_to_end": {}, "auth_wait_decomposition": {},
           "note": "arrival[0]=first record read0; arrival[li]=layer_end[li-1]; ns->ms"}
for v in variants:
    ts = [r["ttft_proxy_ms"] for r in V[v] if r["error"] is None]
    summary["end_to_end"][v] = {"median_ms": statistics.median(ts), "min_ms": min(ts),
                                "max_ms": max(ts), "n": len(ts)}

for tag in ["aead_whole", "aead_layer", "aead_layer_block"]:
    per_rep = []
    for r in V[tag]:
        if r["error"] is not None:
            continue
        tl = r["timeline"]
        recv = [x / NS2MS for x in tl["layer_recv_us"]]
        end = [x / NS2MS for x in tl["layer_end_us"]]
        first_read0 = min(w["read0"] for w in tl["worker_records"]) / NS2MS
        arrival = [first_read0] + end[:-1]
        exposed = [max(0.0, rv - av) for rv, av in zip(recv, arrival)]
        hidden = [max(0.0, av - rv) for av, rv in zip(arrival, recv)]
        per_rep.append({
            "ttft_ms": r["ttft_proxy_ms"],
            "exposed_auth_wait_ms_total": sum(exposed),
            "hidden_margin_ms_total": sum(hidden),
            "exposed_by_layer_ms": exposed,
            "hidden_by_layer_ms": hidden,
            "first_auth_visible_ms": min(recv),
            "last_auth_visible_ms": max(recv),
        })
    summary["auth_wait_decomposition"][tag] = per_rep
    e = statistics.median(x["exposed_auth_wait_ms_total"] for x in per_rep)
    h = statistics.median(x["hidden_margin_ms_total"] for x in per_rep)
    f = statistics.median(x["first_auth_visible_ms"] for x in per_rep)
    l = statistics.median(x["last_auth_visible_ms"] for x in per_rep)
    print(f"{tag}: exposed auth wait median={e:.1f}ms | hidden margin={h:.1f}ms | "
          f"auth visible span {f:.1f}..{l:.1f}ms")

json.dump(summary, open(os.path.join(RAW, "analysis_summary.json"), "w"), indent=2)
print("ANALYSIS_OK")
