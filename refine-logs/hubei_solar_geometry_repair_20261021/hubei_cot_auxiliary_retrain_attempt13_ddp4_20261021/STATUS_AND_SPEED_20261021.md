# 四卡训练完成与实际加速

2026-10-21 22:48–22:54 CST实时核查。

## 完成状态

- PBS 215453于22:03:53正常退出，Exit_status=0。COT辅助组完成50轮、12800次优化器更新，global batch16；`PAIR_COMPLETE_attempt13.json`与`run_metadata.json`均已生成，最佳权重保留。
- 迁移前完成17轮/4352次更新；迁移后完成33轮/8448次更新。训练剩余时间为0。四卡已经释放。
- 当前结果仍需锁定checkpoint后的同样本独立推理评分；训练过程中最佳验证指标不替代该步骤，也不作为GHI收益。

## 实测加速

| 范围 | 单卡A800 | 四卡A800 |
|---|---:|---:|
| 完整轮次平均耗时（含验证） | 434.77秒 | 209.82秒 |
| 完整轮次中位耗时 | 433.29秒 | 209秒 |
| 轮次计时样本 | 15个相邻轮间隔 | 33个完整轮次 |

平均耗时比为2.072倍，单轮耗时减少约51.74%。四卡33轮总轮内耗时6924秒（1小时55分24秒）；若沿用原单卡平均速度，预计14347秒（3小时59分7秒），节省约7423秒（2小时3分43秒）。

口径：单卡采用原COT辅助训练结构化日志的相邻轮验证时间戳；四卡采用各轮282/282结束时最后elapsed，282=256训练batch+26验证batch。含每轮训练、验证和轮内回调，不包含一次性迁移试跑、等待与初始化；不是4倍理论加速或纯GPU kernel吞吐。原始逐轮值与复算脚本为`receipts/TRAINING_SPEED_AUDIT.json`和`audit_training_speed_20261021.py`。

## 后续评分失败与修复

- 215454于22:05:37失败，原因是评分脚本要求`stage_root/stage_manifest.csv`的kind/source/destination格式，但修正数据缓存只保留data/aux，权威清单保存在`prepared_attempt12/stage_manifest.csv`，采用data_path/aux_path等字段。215455因afterok依赖失败被PBS删除，未执行。
- 实时核验：原节点本地data/aux仍存在；不是训练数据丢失、不是训练失败，不需要重训。原失败日志保留为`infer_score_retry4.log`。
- 新建独立兼容视图`inference_stage_view_manifestfix1`，映射到相同本地payload；20946个文件存在且全部215664个train/val索引引用均能对应。无数据复制/改写，test/buffer读取0；不修改共享推理源码或旧清单。
- 评分重试215501已R，AGRI护栏215502为H并依赖215501。日志改为`infer_score_manifestfix1.log`与`agri_guard_manifestfix1.log`，原完成训练和失败证据均保留。最终评分尚未完成，剩余评估时间需以实际推理进度更新。
