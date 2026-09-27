# 源码补丁存档（对固定提交 9a7b605 的唯一增改）

- `aesgcm_aad.py`：新增 serde（AAD 绑定对象身份），从 `upstream/LMCache/lmcache/v1/distributed/serde/aesgcm_aad.py` 复制。
- `serde_init.py`：`serde/__init__.py` 的修改副本（仅 +2 行导入注册）。
- `upstream/` 本体（第三方代码克隆）不入库；复现方式 = 检出 9a7b605 后放置/套用本目录两个文件。
