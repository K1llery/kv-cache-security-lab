# 论文工作区说明

题目（中文）：**面向前缀复用的加密 KV Cache 恢复：关键路径剖析与认证组织优化**
英文工作题：*Authentication-Aware Recovery of Encrypted Offloaded KV Caches: Critical-Path Analysis and Authentication-Record Organization*

> 题目依据：`集大成_项目总览与迁移指南_20260926.md` §3（10 号文件原题的"协同优化"已被 a3/a4 修正为测量先行）。

## 文件组织（07 模板 §9 的 Paper 更新对象）

| 文件 | 内容 | 当前状态 |
| --- | --- | --- |
| `problem.md` | 问题、三 RQ、威胁模型、安全铁律、非目标 | 骨架完成（依据 a3/a4/10，无新增未核验断言） |
| `design.md` | 机制候选与状态机定义、强基线定义 | 骨架完成；**全部为设计假设，无实验支持，不得写成果** |
| `evaluation.md` | 六组实验、基线表、指标、公平性与数据纪律 | 骨架完成；结果栏一律 TODO |
| `related-work.md` | 近邻工作六维对照、新颖性边界、不可声称清单 | 骨架完成（全部条目来自 a3/总览 §4.2 已核验表格） |

## 写作纪律（来自 07/08，每周执行）

1. 任何"实验表明"前面必须有 raw data（`experiments/exp_*/raw/`）；没有数据就写 TODO。
2. 不可声称清单（见 related-work.md 末节）每周对照一次。
3. Claim 由学生确认后才写入正文；AI 不得自动生成"提升明显"类结论。
4. 每周五更新本目录，并在周报 §9 记录改了什么。
