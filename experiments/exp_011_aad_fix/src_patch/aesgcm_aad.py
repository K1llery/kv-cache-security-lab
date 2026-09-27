# SPDX-License-Identifier: Apache-2.0
"""AES-GCM serde with object-identity AAD binding (research addition).

Single-variable variant of :mod:`lmcache.v1.distributed.serde.aesgcm`:
identical frame format ``[1B version][12B IV][ciphertext || 16B tag]`` and
key derivation, but every encrypt/decrypt call binds an associated-data
(AAD) encoding of the *expected* object identity:

    ``b"LMCACHE-AAD-v1"`` || model_name || kv_rank || object_group_id ||
    chunk_hash || cache_salt || plaintext_length

The deserializer takes the identity from the *request's* ObjectKey and the
expected plaintext length from its own destination buffer -- never from the
ciphertext or the on-disk file. A ciphertext moved to a different object
(same salt, same length) therefore fails the tag check instead of being
silently accepted.

Old caches: frames written without this AAD fail verification under the new
serde and are treated as load misses (invalidation + recompute). No
migration path is implemented.
"""

# Standard
import os
import struct

# Third Party
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.aesgcm import (
    _IV_LEN,
    _VERSION,
    KeyProvider,
    _plaintext_bytes,
)
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider

_AAD_DOMAIN = b"LMCACHE-AAD-v1"
_TAG_LEN = 16
_HDR_LEN = 1 + _IV_LEN


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def identity_aad(key: ObjectKey, pt_len: int) -> bytes:
    """Canonical AAD encoding of the expected object identity.

    Both sides derive this from their own trusted view of the identity:
    the serializer from the store request's ObjectKey and source length,
    the deserializer from the lookup request's ObjectKey and destination
    buffer length. The ciphertext and the on-disk file contribute nothing.
    """
    return b"".join([
        _AAD_DOMAIN,
        _field(key.model_name.encode("utf-8")),
        _field(struct.pack(">i", key.kv_rank)),
        _field(struct.pack(">q", key.object_group_id)),
        _field(bytes(key.chunk_hash)),
        _field(key.cache_salt.encode("utf-8")),
        _field(struct.pack(">Q", pt_len)),
    ])


class AesGcmAadSerializer(Serializer):
    def __init__(self, key_provider: KeyProvider) -> None:
        self._keys = key_provider

    def serialize(self, src, dst, key: ObjectKey) -> int:
        dek = self._keys.get_key(key.cache_salt)
        iv = os.urandom(_IV_LEN)
        pt = bytes(src.byte_array)
        aad = identity_aad(key, len(pt))
        ciphertext = AESGCM(dek).encrypt(iv, pt, aad)
        out = memoryview(dst.byte_array).cast("B")
        out[0:1] = bytes((_VERSION,))
        out[1:_HDR_LEN] = iv
        out[_HDR_LEN:_HDR_LEN + len(ciphertext)] = ciphertext
        return _HDR_LEN + len(ciphertext)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        return _plaintext_bytes(layout_desc) + _HDR_LEN + _TAG_LEN


class AesGcmAadDeserializer(Deserializer):
    def __init__(self, key_provider: KeyProvider) -> None:
        self._keys = key_provider

    def deserialize(self, src, dst, key: ObjectKey) -> None:
        out = memoryview(dst.byte_array).cast("B")
        pt_len = len(out)  # 预期长度来自调用方缓冲区，不来自密文
        blob = bytes(src.byte_array)
        frame_end = _HDR_LEN + pt_len + _TAG_LEN
        if len(blob) < frame_end or blob[0] != _VERSION:
            raise ValueError("aesgcm_aad serde: malformed frame or unknown version")
        dek = self._keys.get_key(key.cache_salt)
        aad = identity_aad(key, pt_len)
        plaintext = AESGCM(dek).decrypt(blob[1:_HDR_LEN], blob[_HDR_LEN:frame_end], aad)
        out[: len(plaintext)] = plaintext


_SUPPORTED_AES_BITS = (128, 256)


def _create_aesgcm_aad_serde(kwargs: dict[str, object]):
    """与 aesgcm._create_aesgcm_serde 相同的构造与参数校验，仅换 serde 类（单变量）。

    校验顺序与错误类型逐项对齐原版：aes_bits 必须在 (128, 256)；
    key_provider 仅支持 hkdf；master_key_path 非空（空路径 ValueError，
    与原版一致——可读性错误的异常类型不因本扩展而改变）。
    """
    provider_name = str(kwargs.get("key_provider", "hkdf"))
    aes_bits = int(kwargs.get("aes_bits", 128))  # type: ignore[call-overload]
    if aes_bits not in _SUPPORTED_AES_BITS:
        raise ValueError(
            f"aes_bits must be one of {_SUPPORTED_AES_BITS}, got {aes_bits}"
        )
    if provider_name != "hkdf":
        raise ValueError(
            f"unsupported key_provider {provider_name!r} (only 'hkdf' is implemented)"
        )
    master_key_path = str(kwargs.get("master_key_path", ""))
    if not master_key_path:
        raise ValueError("aesgcm_aad serde 'hkdf' requires 'master_key_path'")
    with open(master_key_path, "rb") as f:
        master_key = f.read()
    provider: KeyProvider = HkdfKeyProvider(
        master_key, key_len=aes_bits // 8, info_prefix=b"lmcache-l2-aesgcm-v1"
    )
    max_workers = int(kwargs.get("max_workers", 1))
    return AsyncSerdeProcessor(
        AesGcmAadSerializer(provider),
        AesGcmAadDeserializer(provider),
        max_workers=max_workers,
    )


register_serde_factory("aesgcm_aad", _create_aesgcm_aad_serde)
