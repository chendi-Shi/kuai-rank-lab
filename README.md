# KuaiRank Lab：真实短视频数据上的召回与精排链路

从用户 ID 出发，完成热门 / ItemCF / 双塔召回、FAISS 向量检索、已看过滤、候选融合、前日历史特征、长播精排和作者多样性。代码使用真实 KuaiRand-Pure 曝光日志，实验结果由训练脚本生成。

先读 [从一条请求讲清楚项目](docs/walkthrough.md)。这个完整链路针对训练期已知的 **7,538 视频池**，不是快手全平台内容；服务使用历史日期快照，供离线复现和本机演示。

完整链路实测见 [端到端报告](docs/pipeline_report.md)：测试 ItemCF Recall@300 **51.45%**，热门基线 **43.50%**。验证集选择 ItemCF 且模型排序权重为 **0**，原曝光分类模型没有改善当前候选排序指标；默认列表保留召回顺序，模型分数作为诊断返回。

## 已完成的实测

已处理 **262 万真实曝光**，用同一批 25 万训练样本完成 **6 种精排模型配置、14 个运行**；神经精排模型各 3 个 seed，所有测试均在验证选择冻结后评估。完整链路加入 1 个 ID 双塔初始化及 ItemCF、热门召回，**37 项本地测试通过**。

- 验证集选择 LR，普通测试 AUC **0.7514**；MMoE 在不同 seed 下的收益不稳定，容量匹配对照亦未证明稳定优势。
- LightGBM 的随机曝光测试 LogLoss 经独立 Platt 校准从 **0.4833 降至 0.2787**，优于平均正例率常数基线 **0.2960**；排序与 AUC 不变。
- 旧版评分接口的 24 候选、4 并发本机 CPU 压测：HTTP p95 **123 ms**，候选编码诊断及评分 p95 **30 ms**；该口径只包含评分，完整推荐链路性能另见端到端报告。

查看 [实验报告](docs/experiment_report.md)、[自动指标表](results/comparison.html) 和 [简历条目与面试准备](docs/resume.md)。数据、训练模型和逐行预测已保留本机，Git 不包含这些大文件。

2026-10-09 的 [GitHub 对照与改进记录](docs/github_review.md) 补充了历史特征消融、独立产物核验和未知类别训练保护。旧实验不重写；新训练增加六个数据切分的文件指纹。

历史特征消融已实测：保留前日历史时，普通验证 AUC **0.7640**，移除后 **0.7359**；但用户 GAUC 略低，结论按指标分别解释。该对照只访问验证标签，详情及用户级 bootstrap 见改进记录。

## 要回答的问题

1. 加入点击/有效播放和点赞任务，能否改善长播预测，还是产生负迁移？
2. 模型的收益来自历史特征、模型容量，还是多任务共享？
3. 在普通推荐日志上较好的模型，是否仍然适用于随机曝光日志？
4. 主任务质量、稀疏反馈、概率校准和候选评分延迟如何权衡？

## 实现范围

- 官方下载、分段续传、完整 MD5 校验和安全解压。
- UTC+8 全局时间切分；普通曝光与随机曝光分别保存。
- 用户、视频、作者的前日累计曝光及平滑行为率；编码器仅在训练样本拟合。
- LR、LightGBM、DeepFM、DeepFM Shared-Bottom、DeepFM MMoE，以及与 MMoE 总参数量匹配的单任务 DeepFM。
- 长播为主任务，辅助任务为 `is_click` 和 `is_like`。
- AUC、按有效用户曝光量加权的 GAUC、LogLoss、Brier、ECE、已曝光 user-day 集合的 NDCG@10。
- 冷启动/曝光频次分桶、MMoE gate 分布、配置与预测指纹、自动对照报告。
- 独立随机曝光 calibration 切分上的 Platt 概率校准，验证与测试分别报告。
- 加载真实训练产物的 FastAPI 候选评分接口。

当前已实现**候选池内的完整推荐链路和演示服务**：输入用户 ID，系统自行召回、生成特征、评分、约束作者后返回推荐。实时业务数据管道、全平台视频库与线上 A/B 需要真实业务环境；本项目不宣称工业上线。PLE 不是本项目完整性的必需条件，现有多任务对照采用 Shared-Bottom 和 MMoE。

## 完整链路运行

本机已有原精排模型，可直接运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval]"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m uvicorn kuai_rank.recommend_api:create_app --factory --host 127.0.0.1 --port 8766
```

打开 `http://127.0.0.1:8766/` 演示页或 `/docs`。`POST /recommend` 接受 `{"user_id":"某用户ID","k":10}`，不需要手工提供特征。未知用户使用热门兜底；跨越快照当天的时间戳被拒绝。默认使用 2022-05-01 快照，中午请求场景。

`run_end_to_end.py` 建立 2022-04-22 和 2022-05-01 两个历史快照，以 10 万去重长播用户视频对训练 ID 双塔，保存 ItemCF 邻居与 FAISS Flat/HNSW 索引；在验证集选择召回路由和排序融合权重，保存指纹后再报告测试。每个日期最多固定抽取 1,000 个具有池内新长播目标的用户。只知道已观测正例，不能把池内未曝光视频当作已知不相关。

扩展任务沿用原精排已报告过的测试日期，不称作一份新的盲测数据。已看过滤只使用普通历史日志。

产物位于 `artifacts/end_to_end`，汇总报告位于 `results/end_to_end.json`。所有指标依脚本生成；完成报告不覆盖，未完成索引阶段可恢复已训练权重。

本机已保留冻结实验产物，可直接启动服务。原离线评估代码对应提交 `93830e6`；后续优化只减少服务中的无用计算，已有冻结清单仍指向原版本。当前代码重新做实验时，请使用下方独立配置及报告路径；不要覆盖仓库的已报告指标，或绕过旧冻结清单的源代码检查。

从 GitHub 克隆后，先按下文复现精排到 `artifacts/reproduction`，再生成独立配置指向本机模型：

```powershell
$pipelineConfig = Get-Content configs/end_to_end.json -Raw | ConvertFrom-Json
$pipelineConfig.ranker = "artifacts/reproduction/lr_s2026"
$pipelineConfig | ConvertTo-Json -Depth 10 | Set-Content artifacts/pipeline-local.json -Encoding utf8
.\.venv\Scripts\python.exe scripts/run_end_to_end.py --config artifacts/pipeline-local.json --out artifacts/pipeline-local --report artifacts/pipeline-local-report/report.json
```

使用独立输出目录后，部署配置也在该目录，启动时设置对应路径：

```powershell
$env:KUAI_DEPLOYMENT = "artifacts/pipeline-local/deployment.json"
.\.venv\Scripts\python.exe -m uvicorn kuai_rank.recommend_api:create_app --factory --host 127.0.0.1 --port 8766
```

对应的独立复现压测：`python scripts/benchmark_recommendation.py --example artifacts/pipeline-local-report/end_to_end_example.json --snapshot artifacts/pipeline-local/test_snapshot --out artifacts/pipeline-local-report/recommendation_benchmark.json`。

模型、向量、索引和特征快照都不提交 Git，复现需自行生成。`results/end_to_end.json` 是本机原实验记录，并不代表克隆后已有可加载模型。

全链路 HTTP 测试：

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_recommendation.py
```

该压测覆盖本机内存中的召回、特征、精排与多样性，混合多个实际用户和未知用户，不含外部特征数据库网络开销。索引邻居一致率与推荐 Recall 分开报告。

## 数据与防泄漏协议

数据：[KuaiRand 官方仓库](https://github.com/chongminggao/KuaiRand)，[Zenodo 公开数据](https://zenodo.org/records/10439422)。数据遵循其 CC BY-SA 4.0 许可；代码许可见 LICENSE。原始数据不提交到 Git。

| 流量 | 切分 | UTC+8 日期 |
| --- | --- | --- |
| 普通曝光 | train | 2022-04-08 至 2022-04-21 |
| 普通曝光 | valid | 2022-04-22 至 2022-04-30 |
| 普通曝光 | test | 2022-05-01 至 2022-05-08 |
| 随机曝光 | calibration | 2022-04-22 至 2022-04-25 |
| 随机曝光 | valid | 2022-04-26 至 2022-04-30 |
| 随机曝光 | test | 2022-05-01 至 2022-05-08 |

以 `time_ms` 为准，边界为左闭右开。基础模型训练不使用随机曝光标签。calibration 仅用于独立的后处理概率校准，不进入基础模型训练或模型早停。

额外报告 `standard_aligned_valid`：普通验证流量中 4 月 26–30 日的子集，与 `random_valid` 对齐日期。两者仍可能有用户、视频和场景支持集差异；日期对齐不等于因果识别。

历史特征仅使用**请求日期之前的普通曝光**。所有同一天样本使用同一前日快照，同一时间戳不会交换标签。验证和测试采用观察策略下的逐日回放，允许使用前一天已经发生的普通曝光及反馈；假设前日反馈已结算。这不是模型自身推荐策略下的闭环仿真。

不使用 `video_features_statistic_pure.csv`，因为该文件是整月统计，包含未来信息。不使用无明确快照时间的用户画像，也不使用 `visible_status` 等动态视频状态。当前曝光的行为和播放时间仅作为标签来源，不作为特征。

KuaiRand-Pure 仅保留候选池内日志，历史是不完整的候选池历史。`is_click` 在不同 UI 中可以代表点击或有效播放，不能统一解释成广告 CTR；主任务 `long_view` 使用官方长播标签。用户日内 NDCG 是**日志曝光集合内排序**，不等于全库推荐 NDCG，也不等于在线收益。

Pure 中许多 user-day 只有少量候选，一候选正样本组的 NDCG 恒为 1。`scripts/ranking_diagnostics.py` 补充集合大小分位数、单候选比例，并仅在至少 5 条曝光且同时含正负反馈的组上报告额外的 NDCG。简历主要使用可解释的分类、用户 GAUC 和校准结果，不将高日志集合 NDCG 包装成全库推荐效果。

## Windows 快速运行

推荐 Python 3.10–3.13。项目当前本机环境位于 `.venv`；它继承本机现有 CPU PyTorch 环境。迁移时应创建独立环境，并按 CPU/GPU 选择安装 PyTorch。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m kuai_rank.cli prepare
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_experiments.py --out artifacts/reproduction --seeds 2026 --models lr lightgbm deepfm sharedbottom mmoe deepfm_matched
```

默认 `configs/pure.json` 使用固定抽样的 25 万训练曝光，验证集不抽样。所有模型及 seed 使用同一批训练样本；ID 词表和数值归一化也由该批样本拟合。设置 `max_train_rows` 为 0 可使用完整训练集。相同 seed 的结果目录不会被覆盖，不完整的运行需要检查后选择新的目录重跑。

仓库内 `results/initial` 和 `results/repeated` 是已完成实验的指标归档，克隆时不包含模型、编码器或逐行预测。复现请使用上面的新目录 `artifacts/reproduction`，避免已归档 `metrics.json` 被调度器视为已完成而跳过训练。省略 `--seeds 2026` 可按配置运行三个 seed。

LR 和 LightGBM 各运行一个 seed，四种神经模型配置各运行三个 seed，共 14 个运行。当前本机软件版本保存在 `configs/local_environment.json`；每次训练也独立记录版本与源码哈希。

本机在首轮训练后使用 `configs/cpu.json` 的单线程配置继续剩余实验，减少 CPU 争用；每个运行记录真实线程数。训练调度在单个进程内复用已加载依赖，不为每个模型重新启动 Python。

完成验证集上的配置选择后，再冻结配置并开启最终测试：

```powershell
.\.venv\Scripts\python.exe scripts/finalize_evaluation.py --runs artifacts/reproduction
```

测试结果应一次性用于报告，不应继续据此调参。不同 seed 的标准差衡量初始化波动，不能代替用户级 bootstrap 的置信区间。报告同时列出模型参数量；`deepfm_matched` 在相同训练样本和编码器条件下提供容量匹配对照。

`finalize_evaluation.py` 先保存验证集选择结果及运行指纹，再加载既有 checkpoint 评估测试集，不重新训练，也不根据测试结果挑选模型。已有 test freeze 不会被覆盖；中断后只能在模型、数据和验证产物未变化的情况下继续尚未完成的运行。

可追加 `--models deepfm_matched`，程序将根据同一训练编码器的真实词表大小，自动扩大单任务 DeepFM 隐层，使总参数量尽量接近默认 MMoE。结果同时记录目标和实际参数量，用于检验多任务收益是否只是容量增加造成的。

## 产物与代码

```text
configs/                 训练、任务权重和时间边界
scripts/download_data.py 官方文件校验与下载
scripts/run_experiments.py 可续跑的实验调度
src/kuai_rank/data.py     数据审计、时间切分、前日特征
src/kuai_rank/models.py   DeepFM / Shared-Bottom / MMoE
src/kuai_rank/train.py    编码、训练、早停、预测与模型保存
src/kuai_rank/metrics.py  分类、用户 GAUC、日志集合 NDCG 与分桶
src/kuai_rank/report.py   JSON + 独立 HTML 实验报告
src/kuai_rank/serving.py  候选评分 API
tests/                   时间边界、防泄漏和数值正确性测试
data/processed/          六个切分、audit.json、manifest.json
results/<run>/           模型、编码器、metrics.json 和逐行预测
```

模型早停使用验证集主任务 LogLoss。LR 和 LightGBM 只预测长播；多任务模型输出三个概率，排序服务默认按长播分数排序。辅助任务质量在结果 JSON 中单独报告，不预设复杂模型必然获胜。

## 候选评分服务

```powershell
$env:PYTHONPATH = "src"
$env:KUAI_RANK_RUN = "results/initial/lightgbm_s2026"
.\.venv\Scripts\python.exe -m uvicorn kuai_rank.serving:create_app --factory --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs`。`POST /rank` 接收 1–500 个候选，每个候选需提供训练配置中完整的 categorical/numeric 特征字典；特征值为编码前值。调用者负责确保特征在请求时刻可用。接口拒绝重复候选、缺失/多余特征和非有限数值，并返回长播排序及未知类别比例。

本机既有模型可直接用上述路径；从克隆复现则将 `KUAI_RANK_RUN` 和校准命令的 `--run` 改为 `artifacts/reproduction/lightgbm_s2026`。原实验冻结清单位于 `results/test_freeze.json`。

这是模型评分服务，未包含在线特征生成或召回服务。只应加载由本项目产生、可信的本地模型和 pickle 文件。

在另一个终端生成真实曝光特征请求并测试接口：

```powershell
.\.venv\Scripts\python.exe scripts/make_api_payload.py
.\.venv\Scripts\python.exe scripts/benchmark_api.py
```

该压测重复一个已曝光 user-day 的固定候选特征批次，报告本机 HTTP 往返及候选评分的 p50/p95/p99。它不包含召回或在线特征获取耗时，不能写成端到端推荐系统的生产性能。

固定模型对照的用户级不确定性：

```powershell
.\.venv\Scripts\python.exe scripts/paired_bootstrap.py --baseline results/initial/deepfm_s2026/predictions_valid.npz --candidate results/initial/mmoe_s2026/predictions_valid.npz --out results/initial/mmoe_vs_deepfm_bootstrap.json
```

可用 `scripts/ranking_diagnostics.py` 汇总所有运行的非平凡曝光集合排序指标；安装 `[plot]` 可运行 `scripts/plot_results.py` 从 JSON 产物生成静态科学图表。

随机曝光校准：

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_random.py --run results/initial/lightgbm_s2026 --out results/calibration_lightgbm.json
```

只使用 `random_calibration` 标签拟合 logit 分数的斜率与截距，评估 `random_valid` 上的 LogLoss/Brier/ECE。基础模型测试冻结后可添加 `--evaluate-test`。这是有监督的目标流量概率校准，不是 IPS 或因果去偏；斜率为正时排序不变，校准变好不能写成推荐排序收益。

## 项目扩展顺序

1. 任务权重与历史特征消融，验证当前复杂模型收益不稳定的原因。
2. 共同视频支持集及 UI/日期分层诊断，进一步解释曝光分布变化。
3. PLE 与梯度冲突分析，在真实负迁移问题上验证价值。
4. 迁移 KuaiRand-1K 扩大完整历史和视频覆盖；当前 Pure 已实现热门/ItemCF/双塔和 FAISS，但不等于全平台召回。

## 参考与简历表述

算法与实验组织参考：[FuxiCTR](https://github.com/reczoo/FuxiCTR)、[DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)、[KuairandRec](https://github.com/Under-the-dome/KuairandRec)、[debiased-video-recsys](https://github.com/shuaihuang028/debiased-video-recsys)。本项目实现独立编写，未复制上述项目的实验结果。

已完成的真实实验及适用边界见 `docs/experiment_report.md`；可修改使用的简历条目和面试准备见 `docs/resume.md`。不宣称工业上线，不把短视频行为概率称为广告 CVR。
