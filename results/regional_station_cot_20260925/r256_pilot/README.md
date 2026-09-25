# 区域 R256 单卡工程试跑

PBS `210632.tc6000` 已 `C/exit_status=0`。在湖南 256×256 上，用按月份抽样的 60 张 train 和 12 张 val 原始 AGRI/CPP 图，随机初始化旧 COTUNet 架构的新实例，试跑 3 epoch。只是区域数据链、批量和显存可行性检查。权重及 `val_pred_log1p.npy` 留在服务器隔离目录 `SolarCOTIFS_20260907/experiments/regional_station_cot_20260925/r256_smoke_seed42_v1/`，没有替换或更新原冻结的 16×16 R。

一张 A100 40GB；测试 batch 2/4/8/16，短样本试跑选择 batch16，约306图/s、PyTorch峰值分配1.98 GiB。batch2包含冷启动，吞吐异常低；完整数据训练须重新 profile 单卡批量和数据加载。保存预测在12张val的有效CPP像元上，`log1p(COT)` MAE 1.352179，与 [`complete.json`](complete.json) 一致。3轮训练和12张验证图不足以评价区域反演质量，也没有做GHI实验。

原始 CPP 在训练期 39,589 个唯一时刻中只覆盖 12,319 个；来源批次到脚本/权重的绑定和独立 QA 均未闭合。本试跑不提供论文物理 COT 或 GHI 增益证据。下一步先扩大 train/val 区域标签缓存，并在保留完整 GHI 样本的前提下单独处理 CPP 缺失。
