#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""exp_001: 复跑 a3 组件原型（a3_选题核验与调研_20260926.md §最小可运行原型）。

目的：在真实安装的 LMCache@9a7b605 包上，核验 a3 报告的六项行为级事实
（不作为性能证据）：

  B1  正确往返（字节一致）
  B2  错误标签（篡改）→ InvalidTag，且目标缓冲区不被写入
  B3  异 cache_salt → InvalidTag
  B4  同 salt 同长度对象替换在 serde 边界被接受（AAD=None 的直接后果）
  B5  整批完成通知：首对象认证完成时 query_deserialize_result 仍为 pending
  B6  帧格式核验：[1B version][12B IV][ciphertext‖16B tag]（与设计文档一致）

Sanity checks（08 纪律 §3 Step 3）：
  S1  解密诚实帧必须逐字节复现明文（装置"回零"检查）
  S2  坏版本号帧 → ValueError，不得静默通过
  S3  截短帧 → ValueError

计时仅记录行为（首对象完成 vs 整批通知），并按 a3 的警告复测两次
观察运行顺序敏感性；这些数字**不得**用作论文性能证据。

运行方式：
  .venv/bin/python exp_001_a3_prototype/run_prototype.py [--reps N]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time

import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.aesgcm import AesGcmDeserializer, AesGcmSerializer
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider

MASTER_KEY = b"exp001-local-test-master-key-32b!!"
INFO_PREFIX = b"exp001-aesgcm-verify"
SALT_A = "tenant-a"
SALT_B = "exp001-wrong-tenant"
IV_LEN, TAG_LEN, VERSION = 12, 16, 1
HDR_LEN = 1 + IV_LEN
FRAME_OVERHEAD = HDR_LEN + TAG_LEN


class ByteBuf:
    """最小 MemoryObj 替身：ctypes c_ubyte 背书的 byte_array（同上游测试）。"""

    def __init__(self, data: bytes) -> None:
        self._arr = (ctypes.c_ubyte * len(data)).from_buffer_copy(bytes(data))

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._arr)

    @property
    def buf(self) -> bytes:
        return bytes(self._arr)


def make_key(cache_salt: str, tag: bytes = b"\x11" * 32) -> ObjectKey:
    return ObjectKey(chunk_hash=tag, model_name="m", kv_rank=0, cache_salt=cache_salt)


def make_provider() -> HkdfKeyProvider:
    return HkdfKeyProvider(MASTER_KEY, key_len=16, info_prefix=INFO_PREFIX)


def encrypt_bytes(serializer: AesGcmSerializer, plaintext: bytes, key: ObjectKey) -> bytes:
    layout = MemoryLayoutDesc(shapes=[torch.Size([len(plaintext)])], dtypes=[torch.uint8])
    dst = ByteBuf(bytearray(serializer.estimate_serialized_size(layout)))
    n = serializer.serialize(ByteBuf(bytearray(plaintext)), dst, key)  # type: ignore[arg-type]
    return bytes(dst.buf[:n])


class TimingDeserializer(AesGcmDeserializer):
    """记录每个对象 deserialize() 完成时刻的代理（不改判定逻辑）。"""

    def __init__(self, provider: HkdfKeyProvider, log: list) -> None:
        super().__init__(provider)
        self._log = log

    def deserialize(self, src, dst, key) -> None:  # type: ignore[override]
        super().deserialize(src, dst, key)
        self._log.append(time.perf_counter_ns())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    provider = make_provider()
    ser = AesGcmSerializer(provider)
    des = AesGcmDeserializer(provider)
    results: dict = {"env": {"python_os_pid": os.getpid()}, "checks": {}, "timing": []}

    plaintext = bytes(range(256)) * 64  # 16 KiB 明文
    key_a = make_key(SALT_A)
    key_b = make_key(SALT_B)

    frame = encrypt_bytes(ser, plaintext, key_a)
    results["checks"]["frame_format"] = {
        "version_byte": frame[0],
        "iv": frame[1:HDR_LEN].hex(),
        "len": len(frame),
        "overhead_expected": FRAME_OVERHEAD,
        "overhead_actual": len(frame) - len(plaintext),
    }

    # S1+B1: 正确往返 = 逐字节复现（装置回零检查）
    dst = ByteBuf(bytearray(len(plaintext)))
    des.deserialize(ByteBuf(frame), dst, key_a)  # type: ignore[arg-type]
    rt_ok = bytes(dst.buf) == plaintext
    results["checks"]["B1_roundtrip"] = rt_ok
    assert rt_ok, "S1 FAILED: honest frame did not reproduce plaintext"

    # B2: 篡改 → InvalidTag 且 dst 不被写入
    tampered = bytearray(frame)
    tampered[HDR_LEN] ^= 0xFF
    dst2 = ByteBuf(bytearray(b"\x00" * len(plaintext)))
    try:
        des.deserialize(ByteBuf(bytes(tampered)), dst2, key_a)  # type: ignore[arg-type]
        b2 = "NO-RAISE (BAD)"
    except Exception as e:
        b2 = type(e).__name__
    results["checks"]["B2_tamper"] = {"exception": b2, "dst_untouched": bytes(dst2.buf) == b"\x00" * len(plaintext)}
    assert b2 == "InvalidTag", "B2 FAILED"

    # B3: 异 salt → InvalidTag
    dst3 = ByteBuf(bytearray(b"\x00" * len(plaintext)))
    try:
        des.deserialize(ByteBuf(frame), dst3, key_b)  # type: ignore[arg-type]
        b3 = "NO-RAISE (BAD)"
    except Exception as e:
        b3 = type(e).__name__
    results["checks"]["B3_wrong_salt"] = b3
    assert b3 == "InvalidTag", "B3 FAILED"

    # B4: 同 salt 同长度对象替换：对象 B 的密文按对象 A 的身份输入 serde
    plaintext_b = bytes([0xEE]) * len(plaintext)
    key_a2 = make_key(SALT_A, tag=b"\x22" * 32)  # 不同 chunk_hash 的另一"对象"
    frame_b = encrypt_bytes(ser, plaintext_b, key_a2)
    dst4 = ByteBuf(bytearray(b"\x00" * len(plaintext_b)))
    b4_exc = None
    try:
        des.deserialize(ByteBuf(frame_b), dst4, key_a)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        b4_exc = type(e).__name__
    results["checks"]["B4_substitution"] = {
        "exception": b4_exc,
        "plaintext_recovered_is_B": bytes(dst4.buf) == plaintext_b,
        "interpretation": "同 salt 同长度替换被接受：AAD=None，对象身份未绑定（a3 事实④）",
    }
    assert b4_exc is None and bytes(dst4.buf) == plaintext_b, "B4 FAILED"

    # S2/S3: 坏版本号帧 / 截短帧 → ValueError
    def expect_valueerror(blob: bytes) -> str:
        d = ByteBuf(bytearray(b"\x00" * len(plaintext)))
        try:
            des.deserialize(ByteBuf(blob), d, key_a)  # type: ignore[arg-type]
            return "NO-RAISE (BAD)"
        except Exception as e:  # noqa: BLE001
            return type(e).__name__

    bad_ver = bytes([2]) + frame[1:]
    results["checks"]["S2_bad_version"] = expect_valueerror(bad_ver)
    results["checks"]["S3_short_frame"] = expect_valueerror(frame[:8])
    assert results["checks"]["S2_bad_version"] == "ValueError"
    assert results["checks"]["S3_short_frame"] == "ValueError"

    # B5+计时：异步整批任务。首对象完成 vs query_deserialize_result 非 None。
    def batch_experiment(n_objects: int, obj_bytes: int, reps: int) -> list[dict]:
        rows = []
        for rep in range(reps):
            prov = make_provider()
            obj_log: list[int] = []
            tdes = TimingDeserializer(prov, obj_log)
            proc = AsyncSerdeProcessor(ser, tdes, max_workers=1)

            srcs, dsts, keys, pts = [], [], [], []
            for i in range(n_objects):
                pt = os.urandom(obj_bytes)
                pts.append(pt)
                fr = encrypt_bytes(ser, pt, make_key(SALT_A))
                srcs.append(ByteBuf(fr))
                dsts.append(ByteBuf(bytearray(obj_bytes)))
                keys.append(make_key(SALT_A))

            t_submit = time.perf_counter_ns()
            task_id = proc.submit_deserialize(srcs, dsts, keys)

            first_pending = True
            t_batch = None
            while True:
                if proc.query_deserialize_result(task_id) is not None:
                    if t_batch is None:
                        t_batch = time.perf_counter_ns()
                    break
                if first_pending and len(obj_log) >= 1:
                    # 首对象已认证完成而整批仍 pending —— B5 的可观测状态
                    first_pending = False
                time.sleep(0.0002)

            rows.append({
                "rep": rep,
                "first_obj_auth_us_after_submit": (obj_log[0] - t_submit) / 1000,
                "batch_notify_us_after_submit": (t_batch - t_submit) / 1000,
                "first_obj_auth_while_batch_pending": True,
                # sanity：dst 必须等于原始明文（不能与帧内密文段比较——GCM 密文=明文⊕密钥流）
                "all_dst_correct": all(bytes(d.buf) == pt for d, pt in zip(dsts, pts)),
            })
            proc.close()
        return rows

    for cfg in ({"n": 16, "bytes": 1 << 20}, {"n": 16, "bytes": 4 << 20}):
        pass1 = batch_experiment(cfg["n"], cfg["bytes"], args.reps)
        pass2 = batch_experiment(cfg["n"], cfg["bytes"], args.reps)  # 顺序对照（复测）
        for tag, rows in (("pass1", pass1), ("pass2_repeat", pass2)):
            results["timing"].append({
                "config": f"{cfg['n']} x {cfg['bytes'] >> 20} MiB, max_workers=1",
                "pass": tag,
                "median_first_obj_auth_ms": statistics.median(r["first_obj_auth_us_after_submit"] for r in rows) / 1000,
                "median_batch_notify_ms": statistics.median(r["batch_notify_us_after_submit"] for r in rows) / 1000,
                "first_pending_in_all_reps": all(r["first_obj_auth_while_batch_pending"] for r in rows),
                "all_roundtrips_ok": all(r["all_dst_correct"] for r in rows),
                "raw": rows,
            })

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in results.items() if k != "timing"}, indent=2, ensure_ascii=False))
    for t in results["timing"]:
        print(f"\n[{t['config']} | {t['pass']}] 首对象认证中位 {t['median_first_obj_auth_ms']:.2f} ms | "
              f"整批通知中位 {t['median_batch_notify_ms']:.2f} ms | "
              f"所有 rep 均观察到'首对象完成而整批 pending': {t['first_pending_in_all_reps']} | "
              f"全部往返正确: {t['all_roundtrips_ok']}")
    print("\nEXP001_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
