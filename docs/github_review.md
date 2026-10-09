# GitHub 对照与改进记录

核查日期：2026-10-09。使用 GitHub 连接读取上游源码，独立实现本项目改动；没有复制上游实验数字。

## 对照来源

- [FuxiCTR 的特征 embedding 实现](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/layers/embeddings/feature_embedding.py)：按特征规范设置 padding_idx，并区分保留位置与已知类别的初始化。本项目将未知 ID 0 明确设为 padding_idx，新增实际含未知类别的优化器更新测试，防止其表示意外漂移。
- [FuxiCTR 的训练基类](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/models/rank_model.py)：监控指标、早停及加载最佳 checkpoint。本项目已实现验证主任务 LogLoss 早停，继续保留；新增训练前后 prepared parquet 指纹检查，以及评估和校准前的校验，防止只比较 manifest 而漏掉数据文件变化。
- [DeepCTR-Torch 的特征输入实现](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/inputs.py)：明确类别/数值/变长特征规范。本项目保留显式特征白名单与训练词表，增加历史特征消融来验证现有特征设计的价值。
- [DeepCTR-Torch 的 PLE 实现](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/ple.py)：按层组织任务专家与共享专家。作为后续架构参考；当前并未新增 PLE，也没有宣称其收益。先补现有基线的可解释对照。

## 本轮实际实现

1. `integrity.py` 统一校验模型、预测和数据文件；加载 pickle/checkpoint 前要求对应模型指纹齐全，拒绝越过产物目录的文件名。
2. 新训练运行记录六个 prepared split 的 SHA256，训练前后检查数据未变；冻结测试恢复和随机流量校准也检查实际数据。旧运行缺少 prepared 文件指纹，报告明确标为覆盖不足，原始指标不改写。
3. `verify_artifacts.py` 可独立检查全部原始运行，避免先反序列化模型或读取测试标签。已核验 14 个运行、70 个逐行预测文件及所有模型产物。
4. 未知类别 embedding/线性字段设 padding_idx=0，优化器更新测试覆盖三种神经架构；旧 checkpoint 的权重保持原样。
5. `run_history_ablation.py` 在相同训练事件和数据上，比较完整 LR 与移除 12 个用户/视频/作者历史数值特征的 LR，只评估验证流量，不重新使用原测试集选配置。

本轮测试 28 项通过。旧实验结果仍是原版本的结果；新消融单独存档，不能与原来的六模型组混为同一实验。

## 历史特征消融实测

两组 LR 使用相同 seed、同一批 250,000 条训练事件、相同类别特征和其他数值特征。移除 12 个前日用户/视频/作者历史数值特征，仅重拟合训练词表及数值归一化。

| 验证流量 | 特征 | AUC | LogLoss | 用户 GAUC |
| --- | --- | ---: | ---: | ---: |
| 普通完整验证 | 有历史 | 0.763998 | 0.521614 | 0.661040 |
| 普通完整验证 | 无历史 | 0.735883 | 0.541141 | 0.666266 |
| 随机验证 | 有历史 | 0.698236 | 0.490160 | 0.572072 |
| 随机验证 | 无历史 | 0.601541 | 0.481355 | 0.548759 |

有历史的普通验证 AUC 高 **0.028114（2.8114 个百分点）**；LogLoss 差（有历史减无历史）为 **-0.019527**，用户级 2,000 次 paired bootstrap 的 95% 区间为 **[-0.020699, -0.018369]**。但 GAUC 反而略低，说明整体概率分辨改善不等于每个用户内排序都改善。随机验证中历史模型 AUC 高而原始 LogLoss 较差，也说明排序分辨能力和概率校准需分别评估。

这些结果来自验证流量，不能称作新的测试集提升或因果效应。原测试不再次用于调参。新消融运行的六个数据文件、模型及三组预测均通过字节校验；最初 14 个运行的模型/预测/manifest 校验通过，但不存在历史 prepared 文件指纹，不事后补造。

## 复现命令

```powershell
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe scripts/run_history_ablation.py
```

第二条需有本机原 LR 模型/报告，或将 `--baseline` 指向自行复现的完整 LR 运行；已存在的消融运行不会被覆盖。克隆仓库不含数据或模型，需要先按 README 复现。结果见 `results/history_ablation.json` 和 `results/artifact_verification.json`。
