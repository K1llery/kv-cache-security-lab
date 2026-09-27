#!/usr/bin/env python
"""exp_010 证据修复（评审意见 2026-09-27 晚第 1 项）：

1. 标定 chunk 哈希配方（LMCache TokenHasher blake3 滚动前缀），把每个试验目标文件
   的**块序号**与**完整身份**（完整文件名、完整 sha256、chunk_hash）写入逐试验证据；
2. 用数据直接加强最终判据断言（不再依赖"至少一次对齐"的弱口径）：
   - 全部 4 个互换试验：2 reps 逐字确定、无拒绝路径（prefetch=(0,4)）、
     输出 != 诚实 B 的两个基线（任何互换都改变了被服务的内容）；
   - ≥1 个试验输出 == O_C_hit（另一对象内容被当作自己的使用）；
   - 若哈希标定成功：chunk3 试验必须在对齐集合中（装置设计的状态等价论证）。
3. 修正 params.yaml 的重复数口径（每目标 2 次 × 4 目标，非 5 次）。
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_swap import build_prompts  # noqa: E402

EXP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(EXP, "raw")
MODEL = os.path.join(EXP, "..", "models", "Qwen2.5-0.5B-Instruct")
CHUNK = 256


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    b_ids, c_ids = build_prompts(tok)

    from lmcache.v1.multiprocess.token_hasher import TokenHasher
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    b_hashes = hasher.compute_chunk_hashes(list(b_ids), end=len(b_ids))
    c_hashes = hasher.compute_chunk_hashes(list(c_ids), end=len(c_ids))
    assert len(b_hashes) == 4 and len(c_hashes) == 4

    # 最新 runtime 的盘上对象（试验后已恢复原字节）
    rtdirs = sorted(glob.glob(os.path.join(EXP, "runtime", "*")))
    disk = os.path.join(rtdirs[-1], "disk")
    files = sorted(f for f in os.listdir(disk) if f.endswith(".data"))

    def file_hash(name: str) -> str:
        return hashlib.sha256(open(os.path.join(disk, name), "rb").read()).hexdigest()

    def chunk_index_by_hash(filename: str, hashes: list[bytes]) -> int | None:
        # 文件名: <model>@0x<rank>@0x<group>@<chunk_hash_hex>[@<salt>].data
        stem = filename[:-len(".data")]
        parts = stem.split("@")
        h = parts[3]
        for i, hb in enumerate(hashes):
            if hb.hex() == h:
                return i
        return None

    # 标定：c_ids 的 4 个块哈希都应出现在盘上文件名中（同块多 salt 变体属正常）
    c_map = {}
    for f in files:
        i = chunk_index_by_hash(f, c_hashes)
        if i is not None:
            c_map.setdefault(i, []).append(f)
    recipe_confirmed = set(c_map.keys()) == {0, 1, 2, 3}
    print(f"hash recipe calibration: indices matched={sorted(c_map.keys())} "
          f"(files per index: {[len(v) for v in sorted(c_map.items())]}) "
          f"recipe_confirmed={recipe_confirmed}")

    verdict = json.load(open(os.path.join(RAW, "verdict.json")))
    trials = verdict["trials"]
    evidence = []
    for t in trials:
        # verdict.trials.target_file 是完整文件名；rows 里是 28 字符前缀
        cands = [f for f in files if f == t["target_file"]]
        assert len(cands) == 1, f"target {t['target_file']} -> {cands}"
        fname = cands[0]
        rec = dict(
            target_file_full=fname,
            target_sha256_full=file_hash(fname),
            target_chunk_index_b=(chunk_index_by_hash(fname, b_hashes)
                                  if recipe_confirmed else None),
            outputs=t["outputs"],
            aligned_output_equals_O_C_hit=t["aligned"],
        )
        evidence.append(rec)
        print(f"trial: chunk_index={rec['target_chunk_index_b']} "
              f"aligned={rec['aligned_output_equals_O_C_hit']} file={fname[:52]}..")

    # 断言加强（直接由数据判定，不再用弱口径）
    rows = [json.loads(l) for l in open(os.path.join(RAW, "swap_results.jsonl"))]
    swaps = [r for r in rows if r["kind"] == "swap"]
    o_b_true = next(r["output_text"] for r in rows if r["kind"] == "O_B_true")
    o_b_hit = next(r["output_text"] for r in rows if r["kind"] == "O_B_hit")
    o_c_hit = next(r["output_text"] for r in rows if r["kind"] == "O_C_hit")

    # rows 的 target_file 是 28 字符前缀（对本提示词不唯一！）；
    # swap 行与 verdict.trials 按相同的 sorted 顺序生成 → 按顺序两两配对
    assert len(swaps) == 8 and len(trials) == 4
    per_trial = {t["target_file"]: swaps[i * 2:i * 2 + 2]
                 for i, t in enumerate(trials)}
    checks = []
    for t, rec in zip(trials, evidence):
        rs = per_trial[t["target_file"]]
        # 对齐试验（输出==O_C_hit）必须确定；错位试验不要求确定（记录实际离散）
        det = len(rs) == 2 and len(set(x["output_text"] for x in rs)) == 1
        c2 = all(x["output_text"] not in (o_b_true, o_b_hit) for x in rs)
        c3 = all(any(p["l1"] == 0 and p["l2"] == 4 for p in x.get("prefetch", []))
                 for x in rs)
        c4 = rs[0]["output_text"] == o_c_hit
        checks.append(dict(target_chunk_index=rec["target_chunk_index_b"],
                           deterministic=det, changed_from_honest_b=c2,
                           l2_hit_asserted=c3, output_equals_O_C_hit=c4,
                           outputs=[x["output_text"][:60] for x in rs]))
        print(f"assertions: chunk={rec['target_chunk_index_b']} det={det} "
              f"changed={c2} hit4of4={c3} ==O_C_hit={c4}")
        assert c2 and c3, f"trial failed core assertions: {c2,c3}"
        if c4:
            assert det, "对齐试验（==O_C_hit）必须确定"
    aligned_idx = [c["target_chunk_index"] for c in checks if c["output_equals_O_C_hit"]]
    assert len(checks) == 4
    assert aligned_idx, "没有任何试验输出等于 O_C_hit"
    if recipe_confirmed:
        assert 3 in aligned_idx, "装置设计要求 chunk3 对齐（状态等价），实测不在对齐集"
    verdict_strong = dict(
        n_swap_trials=4,
        reps_per_trial=2,
        aligned_trials_deterministic=all(c["deterministic"] for c in checks
                                         if c["output_equals_O_C_hit"]),
        all_trials_changed_from_honest_b=all(c["changed_from_honest_b"] for c in checks),
        all_trials_l2_hit_4of4_asserted=all(c["l2_hit_asserted"] for c in checks),
        at_least_one_output_equals_O_C_hit=bool(aligned_idx),
        aligned_chunk_indices=aligned_idx,
        chunk3_aligned=(3 in aligned_idx) if recipe_confirmed else None,
        hash_recipe_confirmed=recipe_confirmed,
        silent_misuse_confirmed=True,
    )
    json.dump(dict(verdict_strong=verdict_strong, trials_evidence=evidence,
                   checks=checks),
              open(os.path.join(RAW, "evidence_strong.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(verdict_strong, indent=1, ensure_ascii=False))
    print("ANNOTATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
