# CPP 缺失原因与旧数据处理核查

2026-09-07 只读核查。数量单位均为站点×时刻，两个站点共享同一个源NC时刻。

## 当前清单的时间分布

对修复ledger按relative_path保留最后记录，合计130,940条：成功46,036，源文件未找到84,904；对应唯一源路径分别23,018和42,452。

| UTC/BJT分钟（两时区分钟相同） | 对齐成功 | 源缺失 |
|---|---:|---:|
| 00 | 31,298 | 1,314 |
| 15 | 4,912 | 27,312 |
| 30 | 4,910 | 28,150 |
| 45 | 4,916 | 28,128 |

84,904条缺失中，83,590条在非整点，占98.45%。因此主要解释是源目录多数时期只覆盖整点，而AGRI样本按15min构建。这里说明已观测到的文件覆盖规律，不宣称已经查明上游为何只生成/保存整点。

月份也明显不均：2024-04成功1,418/总5,578；2025-06至09在本次既有样本清单内无missing_source（不代表整个月所有理论时刻齐全）；2026-03仅成功84/总5,730。还存在额外日期覆盖缺口，不能把全部缺失归因于分辨率。

## 之前就存在缺失

旧数据源 `/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined/latest_summary.json` 记录构建完成时间2026-08-03 09:20:53 BJT，totals.missing_cpp=85,039，written=130,750，skipped=190。

旧构建代码 `/home/Data_Pool_3/chenyi/GHI/scripts/legacy_root/build_hunan_station16_samples.py`：crop_cpp在源文件不存在或读取异常时，均返回NaN CPP、全False掩膜、cpp_present=False；仍保存该站点时刻的AGRI/GHI，并累加missing_cpp。因此旧数混合缺源与读取异常，且构建时跳过190个已有输出，不能把85,039与84,904的差135直接解释成修复回收数。

旧COT清单程序 `/home/Data_Pool_3/chenyi/GHI/cloud_retrieval_hunan/make_split.py` 先扫描实际存在的CPP时刻，再筛选有效白天COT像元（默认至少16个）。旧训练dataset也拒绝完全没有有效COT像元的记录。这解释了为何之前能训练，但全量合并数据并不意味着CPP完整。

当前证据支持“缺失早已存在，本次按原样记录并明确区分原因”，不支持“本次对齐导致源文件消失”。空间一像元修正与源文件时间覆盖是两类问题。

旧缓存抽查已完成：跨站点/年份选取18条本次missing_source记录，旧文件全部cpp_present=False；另18条本次ok记录，旧文件全部cpp_present=True。此为有界抽查，不冒充全量旧缓存读取。

## 影响与后续

COT监督的时段/季节覆盖不均，需要按月份与整点/非整点记录训练分布和检索诊断；不能宣称覆盖充分。新R训练只能使用实际有效参考，预测阶段可对每15min外推AGRI生成COT。不能用插值伪造独立CPP标签，也不能因此按未来CPP可用性筛选GHI主表。

只读核查脚本：`scripts/audit_cpp_missing_history.py`；服务器结果：`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/audits/cpp_missing_history_20260907.json`（脚本完成后写出，额外包含旧缓存抽查）。
