# 湖南标签来源核查与湖北主分析方向

同日后续用户决定：湖南同期COT价值分析保留单帧主候选，三帧作为补充对照。下列三帧结果仍作为已完成证据保留；不据两轮不同合同的增益百分比判定帧数最优或删站。区域主次及标签质量规则不变，实验组织见`../refine-logs/cot_value_first_20261013/EXPERIMENT_PLAN.md`首节。

核查日期：2026-10-20，北京时间。用户提出第二篇论文以湖北牛首、王寨、米庄为主要研究站，湖南作为补充证据，并要求追溯湖南标签来源。本记录落实湖北主分析方向；湖南最终保留哪些站点须依据观测质量，不能依据 COT 增益的正负决定。当前没有删除整站、修改标签、重训或打开 test/buffer。

## 1. 标签究竟来自哪里

湖南地面标签来自复旦共享盘上的两份工作簿：

| 站点 | 当前源文件 |
|---|---|
| 四里 Sili | `/home/Data_Pool_3/chenyi/GHI/data/hunan/raw/stations/四里光伏.xlsx` |
| 竺家 Zhujia | `/home/Data_Pool_3/chenyi/GHI/data/hunan/raw/stations/竺家光伏.xlsx` |

旧登记目录为 `/home/Data_Pool_3/chenyi/辐照度预报/data/hunan_stations_raw/`。两目录的既有 SHA 登记相同：四里 `c13c6a066cc7ca3ee0d902bb8a9d6c8ea1cc1618659e86eafa4971bbe9a879f4`，竺家 `38e4b708a009a69d16bc49ff80ea330a88a6b32e1d69ff6ee53dc49ee13783a0`。本轮通过 FD-201 只读 stat 确认现路径文件存在，分别为 1,039,392 和 1,311,997 字节，mtime 均为 2026-07-23；本轮未重新计算原表哈希或打开原表数值。历史哈希不能当作本轮重新哈希。

两表各有一个 `结果集` sheet。四里身份列为 `名称· / OBJ_TYPE / DATA_DATE`，竺家为 `名称· / DATA_DATE`；之后是 288 个 `VAL0005` 至 `VAL2400` 数值列。工程将其解释为北京时间的 5 分钟区间结束记录。既有源库存日期为 2024-01-01 至 2026-06-30，末日 `VAL2400` 对应 2026-07-01 00:00；这不表示每天完整，更不表示整个库存均进入实验。

15 分钟目标时刻 T 的标签取 T−10、T−5、T 三个值的算术平均，对应 `(T−15 min,T]`。空值、非有限或负值令目标无效；有限的零值在原始处理里保留，不以插值补齐。

来源证据：

- `audits/spatial_reaudit_static_20260908/station_schema.json`，workbooks 段。
- `refine-logs/cot_value_first_20261013/results/trainval_data_20261013/workbook_label_extraction_audit.json`。
- `refine-logs/cot_value_first_20261013/build_trainval_labels_from_workbooks.py`，表头解析、请求时刻及三值均值。
- `docs/SOURCE_FLATLINE_QC_20260908.md`，原始列与时间含义。

## 2. 当前湖南实验仍使用这条来源链

```text
两份原始 Excel
  → unique_target_labels_train_val.csv
  → ghi_kt_train_val.csv
  → ghi_head_pack_new_r_v1
  → 三帧 data_attempt2
  → 湖南四组云概率/COT 同期 GHI 实验
```

本轮经 zjnu-hpc 只读核对下列 manifest：

- 工程根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/`。
- `monthly_R_station16_20261013_v1/data/ghi_head_pack_new_r_v1/pack_manifest.json`：COMPLETE，54,937 行；`label_sha256=a86c2425ac4e6e8259fcaa4b5b2e083ee427d16e12d4eb1b44aa87751b75988d`。
- `cot_interval_history_20261015_v1/data_attempt2/pack_manifest.json`：COMPLETE，51,929 行，train 47,666、validation 4,263；`source_pack_sha256=27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4`，`primary_labels_modified=false`。

上述标签哈希与 `refine-logs/cot_value_first_20261013/W0_AUDIT_20261013.md` 的工作簿提取/晴空合并登记吻合。当前湖南四组从三帧包读取 `rows.csv` 的 `ghi`，对已确认无效的四里 2024-09-05 整日应用共同排除规则后，验证为 4,226 行。

标签不是由 CPP、COT 或卫星反演模型生成。`clear_sky` 用于构造 kt 和回算 GHI，也不替代原始地面观测。

## 3. 已经证实的源数据问题，以及尚未知的内容

既有源 XML 审计对 train/validation 请求的白天零值逐一回查。白天定义为 5 分钟区间中点太阳高度至少 10°；以下数量来自已保存审计，不是本轮重读全部源表。

| 项目 | 四里 | 竺家 |
|---|---:|---:|
| 源表显式为零的 5 分钟白天记录 | 1,731 | 2,092 |
| 其中属于连续至少 60 分钟零段 | 1,128 | 1,541 |
| 连续零段数量 | 11 | 24 |

3,823 个零值全部在源 Excel 中显式存在，与缓存一致，不是均值、空值填补或后续网络生成的零。两站均有这类问题；不能据此建立“一个站全好、另一个站全坏”的结论。含有至少一个白天零值的日期也不等于整日无效。

已确认并落实的整日排除只有四里 2024-09-05：原单帧标签包含 39 个全零目标，当前三帧队列覆盖其中 37 个。该日源表已回查的 06:00–16:00 共 121 个五分钟单元格全为零。具体是仪器故障、通信故障还是上游占位零，尚未得到设备记录确认。依据用户对该日无效的确认执行整日排除，不能扩称为整站无效。

证据：

- `results/interval_alignment_20261015/ZERO_SOURCE_AUDIT_20261015.md`。
- `results/interval_alignment_20261015/workbook_zero_xml_audit.json`。
- `results/cot_interval_history_20261015/audit/workbook_audit.json`。
- `docs/GHI_STATION_DAY_EXCLUSIONS_20261019.json`。
- `audits/sili_invalid_day_20261019/REPORT.md`。

更上游的来源仍未查清：供数单位、采集系统、仪器型号、水平/倾斜安装、校准维护记录、原始单位说明、五分钟值的实际平均/瞬时定义，以及缺测是否导出为 0。旧 schema 的单位字段明确写作 `W m-2 (inferred from IRRADIANCE field and value range)`，属于工程推断，尚不是供数方字段字典的独立确认。四里有 `OBJ_TYPE=IRRADIANCE`，支持其为辐照度字段；竺家缺少此字段。现有资料不足以确认全部读数就是经过质量控制的水平面总辐照度，也不足以断言它们不是。

湖北也仍有仪器校准/字段字典未独立核验的缺口，不能因为湖北模型分数更好就认定其观测完全可靠；见 `docs/ACTUAL_DATA_INVENTORY_HUNAN_HUBEI_20261016.md`。

## 4. “退化站”不能替代标签质量判断

2026-10-20 最新湖南三帧四组同期验证，三种子平均 RMSE（W/m²），来自 `refine-logs/cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1/per_run_metrics_fp64.csv`：

| 站点 | AGRI+几何 | 加 COT | 加云概率 | 加 COT 和云概率 |
|---|---:|---:|---:|---:|
| 四里 | 87.250 | 84.788 | 86.184 | 84.863 |
| 竺家 | 91.720 | 91.739 | 90.188 | 90.727 |

四里 COT-only 点估计改善约 2.82%；竺家只增加约 0.019 W/m²（约 0.02%），基本持平。已有 COT 后再增加云概率时，四里略退、竺家改善，方向又不同。因此方法增量的正负不能用作仪器质量判据。此表是重复使用的区域内验证集；不同输入模式、旧单帧结果、未来预报结果不能混在一起选择“有收益站”。

## 5. 论文方向与下一步

1. **湖北牛首、王寨、米庄作为第二篇的主要分析区域**，围绕同期 GHI 的 COT 增量及云况差异组织主结果。第一篇范围和已运行实验不变。
2. **湖南降为补充区域证据**。用户提出保留一站，具体站点待对两站使用同一套、与模型收益无关的观测质量规则后确定。本轮不删除整站或覆盖旧结果。
3. 如果质量证据只支持排除某些日期，实施共同的站点日规则；如果得到整站系统性问题的证据，再说明整站不纳入主分析的理由。已完成的另一站结果和质量说明保留在补充材料/记录中，不隐去负结果。
4. 湖南既有验证集已反复选模和分析，只能称区域补充验证。独立测试必须使用此前未打开、预先保留的 test 日期，并在方法及质量规则固定后评价；本轮没有打开 test。两省各自训练，也不能称湖北模型到湖南的零样本迁移。
5. 优先向原始供数方确认：辐射变量是 GHI 还是倾斜面 POA；单位和五分钟平均/瞬时定义；零值编码；仪器安装、校准与故障日志。内部代码和缓存的一致性无法补齐这些信息。

本轮只修改方向记录和来源说明；没有修改双语论文正文、数据或模型，也没有增加 GPU 作业。
