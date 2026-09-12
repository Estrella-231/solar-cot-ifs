# R(real AGRI) / R(predicted AGRI) 配对来源清点

2026-09-08 完成的只读 metadata/header inventory。不是误差传播结果，未打开 train/val 图像、CPP 或掩膜的数值，未读取任何 test 行或 payload，未打开 GHI 标签/性能、未载入模型、未使用 GPU。权威机器结果为 `audits/cot_pair_inventory_20260908.json`，入口 `scripts/inventory_cot_pairs.py`。

已先读本篇 AGENTS、UPSTREAM_PROJECT_RULES、IRRADIANCE_ERROR_PREVENTION 和 HOURLY_NEXT_ACTIONS。下一步保留现有 S/R、负 pilot 和数值 warning；本清点不放行扩大训练或 test。实际配对推理由主任务另行核验后执行。

## 1. 权威资产与精确合同

学院 SSH 别名 `zjnu-hpc`；项目根：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
```

Python 为 `/public/home/slfu/miniconda3/envs/swc/bin/python`。下列相对路径均从上述服务器根解释，不能使用本地路径交接服务器操作。

R pack 完整路径：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/data/cot_repaired_pack_20260907_v1
```

当前 `pack_status.json` 为 COMPLETE：train 11,978、validation 6,536，共 18,514；`test_payloads_read=0`。本次重新核对 rows.csv、norm.json、pack_config.json 的 SHA 与原 registry 一致。大数组仅核对 NPY header/大小，原数组 SHA 从已完成验收的 registry 引用，本次没有为了重算 SHA 扫描数组内容。

| 文件 | 已读 header 的形状/dtype | 数值含义 |
|---|---|---|
| x_raw.npy | `[18514,16,16,16]`, float32 | 未归一化的物理 AGRI 13 通道 + 3 几何通道 |
| target.npy | `[18514,1,16,16]`, float32 | 修正 CPP COT / 100；恢复参考值时乘 100 |
| mask.npy | `[18514,1,16,16]`, bool | sidecar 原始 cot_reference_mask，必须同一像元集比较两端 |
| rows.csv | 18,514 条，仅 train/validation | 数组行、站点、UTC/BJT、原文件/sidecar 路径及 SHA 的关联 |

rows.csv 字段精确为：`index,station_id,timestamp_utc,timestamp_bjt,split,original_path,sidecar_path,original_sha256,sidecar_sha256,status,valid_pixels`。读取 `valid_pixels` 字段的 CSV 元数据不等于读取 mask 数组；本次没有按该数值选择、排序或调整样本。

R 输入通道依次为 C01–C06、C09–C15、cosSOZ、cosRAA、day_mask；打包代码从原 AGRI 的 `[0,1,2,3,4,5,8,9,10,11,12,13,14]` 取物理通道，其后三项为 `cos(deg2rad(agri[18]))`、`cos(deg2rad(agri[17]-agri[15]))`、`agri[19]>0.5`。R norm 仅由 train 拟合，输入缺失经归一化后填零。

参考 mask 来自修正版 sidecar：源 COT 有限、范围 `[0,100]`、day > 0.5、有效太阳几何且 SOZ < 80°。这是数值/研究几何规则，不是独立官方 QA 解码。mask 外的 target=0 不代表晴空，不能计入误差。后续两端使用这个完全相同的 mask，不按预测输出重算 mask。

冻结 R：`runs/cot_repaired_seed42_v1/best.pt`，SHA `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`；输出是非负 `log1p(COT)`、无上限裁剪。其 checkpoint 与本次 cache contract 中的 R 配置一致；本次未载入权重。

Forecast cache 完整路径：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/data/frozen_forecast_trainval_20260907_v1
```

根 complete.json 为 COMPLETE_FEATURE_CACHE。train 27,675 序列/1,154 分片；val 4,258 序列/178 分片。index.csv SHA 与各自 complete.json 一致。没有读取包含 test 的 S 全量 manifest，也没有读取 head labels.csv。

每个分片 `shard_NNNNN.npz` 为未压缩 ZIP_STORED；本次只读固定选择涉及的 59 个 val 分片的 ZIP 目录和 NPY header，不打开数组数值。成员为：

| 成员 | 形状，B 为该分片序列数 |
|---|---|
| indices | `[B]` |
| predicted_agri_physical | `[B,2,16,13,16,16]` |
| geometry | `[B,2,16,3,16,16]` |
| cot_log1p | `[B,2,16,16,16]` |
| cot_features | `[B,2,16,4]` |
| r_latent_mean | `[B,2,16,16]` |

无需回取 256×256 实测/预测 full grids。物理预测 AGRI 已逆变换 S norm；R 输入仍需使用冻结 R norm。

## 2. 时间、序列、站点和标签键

两站顺序固定为 `sili,zhujia`，即 station_index 0、1。每个 split 的 index.csv 字段为 `index,seq_id,split,init_time_utc`；index 是 split 内 cache_index，不能与另一个 split 混用。`seq_id=hunan_YYYYmmddHHMM` 的后缀是 UTC 序列起点；所有行都验证：

```text
init_time_utc = seq_id 的 UTC 序列起点 + 7*15 min
lead_index = lead_minutes/15 - 1
target_time_utc = init_time_utc + lead_minutes
R key = (相同 split, station_id, timestamp_utc == target_time_utc)
pair key = (split, station, init_time_utc, target_time_utc)
```

从 complete.json 的分片样本数累加定位 shard 和 shard_row。后续读取数值时还须核对 `indices[shard_row] == cache_index`；header 本身不能证明 indices 的值。本次确认 index 连续、init/seq 唯一、R 数组行连续、R station/target 无冲突、BJT/UTC 表示同一时刻、原文件名确为 UTC，以及 R split 符合 BJT 边界。任何同站同 target 多 R 行必须报错，不能任挑。

特别区分：原 Combined 站点文件名/嵌入 `timestamp_utc` 为 UTC（已逐行核验）；这不是上游 matched 文件名 BJT 的合同。R 的 AGRI/CPP 是文件 nominal target 时刻的图像参考，不是 15 min GHI 均值标签。这里不从 Combined 的 `ghi_5min` 取标签。其 schema 明示旧 GHI 字段对应文件后 +5/+10/+15 min，不能拿来冒充同刻末端标签。若将来关联 GHI，仅可另接已封存 `(station,init,target)` 和 canonical 区间 `(target-15 min,target]`。

## 3. 全部 train/val 来源交集（只按键）

“可配对”指 R target 对应至少一条该 split 的 forecast init；不是 GHI cohort，也未按 GHI/CPP误差筛选。

| split / 站点 | R unique target | 至少1 lead | 完整16 lead | 全无 lead | 总 pair |
|---|---:|---:|---:|---:|---:|
| train / sili | 6,011 | 5,764 | 4,966 | 247 | 86,191 |
| train / zhujia | 5,967 | 5,725 | 4,924 | 242 | 85,570 |
| validation / sili | 3,246 | 3,078 | 2,465 | 168 | 44,597 |
| validation / zhujia | 3,290 | 3,119 | 2,478 | 171 | 45,168 |

16 时效交集如下；每列缺失数可用上表 R target 数减该格，JSON 也逐格保存 `unmatched_R_targets`。

| lead/min | train sili | train zhujia | val sili | val zhujia |
|---:|---:|---:|---:|---:|
| 15 | 5347 | 5309 | 2748 | 2788 |
| 30 | 5344 | 5306 | 2751 | 2791 |
| 45 | 5363 | 5325 | 2756 | 2794 |
| 60 | 5373 | 5335 | 2761 | 2799 |
| 75 | 5369 | 5331 | 2767 | 2805 |
| 90 | 5367 | 5329 | 2774 | 2812 |
| 105 | 5389 | 5351 | 2781 | 2819 |
| 120 | 5398 | 5359 | 2789 | 2827 |
| 135 | 5396 | 5358 | 2796 | 2834 |
| 150 | 5399 | 5362 | 2804 | 2842 |
| 165 | 5417 | 5380 | 2814 | 2852 |
| 180 | 5430 | 5392 | 2822 | 2861 |
| 195 | 5426 | 5388 | 2826 | 2865 |
| 210 | 5416 | 5378 | 2825 | 2864 |
| 225 | 5389 | 5344 | 2817 | 2835 |
| 240 | 5368 | 5323 | 2766 | 2780 |

R train 主要为小时整点但包含 15/30/45 min，validation 覆盖四个分钟位置；不能推断只有小时 CPP。R validation target 从 BJT 2025-07-01 06:30 到 2025-09-22 14:15；forecast val init 从 BJT 2025-07-01 06:45 到 2025-09-10 14:15，此外还存在内部缺口。截止日期和边界解释部分缺配；不补造、插值或跨 split 取缓存。

## 4. 在误差前固定的下一步样本

规则在任何 tensor/CPP 值读取前固定：每站取全部 R validation 唯一 target，按 UTC 升序；第 k 个选择秩为 `floor(k*(n-1)/31), k=0,...,31`。每站 32 个，共 64 个 target。排序不使用误差、参考 COT、valid_pixels、forecast 覆盖率或 GHI。对每个 target 尝试全部 16 lead，只保留已有可匹配的 pair，同时完整记录缺失；不将零配对 target 替换成其他日期。

固定选择的规则+keys SHA：`8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318`。全部目标、秩、R row、原/sidecar 路径和 SHA、forecast seq/index/shard/row/lead 及缺配原因均在 JSON `selection` 中，须原样复用。

| 站点 | 固定 target | 有配对 target | 已有 pair | 缺 lead |
|---|---:|---:|---:|---:|
| sili | 32 | 28 | 431 | 81 |
| zhujia | 32 | 30 | 441 | 71 |
| 合计 | 64 | 58 | 872 | 152 |

下一步只对 58 个有配对 target 验证空间/几何并读用于诊断的数值；其余 6 个仍在固定清单中列为无配对。参考可用性只限定单列诊断的可观测范围，不改变既有 GHI 主 cohort、训练、选模或下游输入。

## 5. 空间与几何的最小门禁：无需重建数据

本次已通过现有密钥 SSH `FD-107` 只读返回原 `schema.json` 和修复 `run_config.json`；当前 schema 的两站 agri_bounds 与冻结 S 配置一致。完整 source 路径为：

```text
FD-107:
/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined/schema.json
/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined/{station}/{year}/{yyyymm}/{yyyymmdd}/{yyyymmddHHMM}.npz
/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/data/cpp_aligned_20260905_v2/run_config.json
/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/data/cpp_aligned_20260905_v2/samples/{station}/{year}/{yyyymm}/{yyyymmdd}/{yyyymmddHHMM}.npz
/home/Data_Pool_3/wangyc/irradiance_paper/hunan_shortterm_m0_20260903_r101_r103_v1/spatial_audit.json

zjnu-hpc:
/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/grid_static.npz
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/configs/s_frozen_hunan_seed42.json
```

模板内的所有选中 source 绝对路径由 JSON selection 给出，不需要枚举目录。例：首个有匹配的四里样本为 `/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined/sili/2025/202507/20250705/202507050230.npz`，sidecar 使用同一后缀。

| 站点 | S patch `[r0,r1,c0,c1]` | 最近点/patch中心 | 站点纬度、经度 |
|---|---|---|---|
| sili | `[155,171,140,156]` | `[163,148]` / `[8,8]` | 25.9478611111, 112.347916667 |
| zhujia | `[65,81,142,158]` | `[73,150]` / `[8,8]` | 29.566666667, 112.45 |

冻结 S grid SHA 为 `54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62`，与已保存 M0 spatial_audit 的 grid SHA 相同。本次未重新读取 grid 内容，旧 M0 数值不是本次逐个选中样本的坐标证明。

后续坐标门禁具体执行：

1. 仅在固定的 58 个有配对 target 上读取原 NPZ 的 `station_id,timestamp_utc,grid_lat,grid_lon`，以及 sidecar 的这些字段和 `original_sha256,contract_sha256,schema_version,cpp_rows,cpp_cols`。不读原 `agri/ghi_5min/cpp`，也不读 sidecar `cpp/cot_reference_mask`，先把这一阶段限制为身份与坐标。每个文件均是 JSON 中指定的 exact path；不补挑其他日期。
2. 对 `grid_static.npz` 核对冻结 grid SHA，仅取 `lat,lon,row_index,col_index` 的两个上述 16×16 patch。比较原样本与 sidecar 的 grid_lat/grid_lon，再与 S patch 的 lat/lon 逐像元比较，保存 shape、dtype、最大坐标差及规范化坐标 bytes 的新 SHA。主任务在读取58个样本前预定绝对差门限 `<=1e-5 deg`；所有差异均记录，任一超限就停止该空间合同的误差解释，不事后扩大容差。索引字段描述真实 HDF 行列，不能拿地理 crop 加常数冒充。
3. 原 sidecar schema `cpp_aligned_sidecar_v1` **没有独立 canonical_grid_sha256 字段**。`contract_sha256` 是 COT规则 JSON 的 SHA，不是坐标 hash；不得以它证明坐标相等。本次 run_config 返回 contract SHA `2a3f2f9bc93c6b82db6529fa3b8d442f12a1fe39c6d4291f32d76ab216c03d48`。已有每个 sidecar SHA、内嵌 original_sha256 与两个 grid 数组负责来源绑定；后续明确新增坐标 hash 到新诊断 JSON，无需改旧数据。修复代码用源 NC LAT/LON 映射产生 cpp_rows/cpp_cols，旧 schema 的 cpp_bounds 已过时，不得复用。
4. 空间通过后，仅读取 R `x_raw[r_pack_index,13:16]` 与相同 target 的 forecast `geometry[shard_row,station_index,lead_index]`。先检查 forecast 同 target、不同 init/lead 的 geometry 自身是否一致，再逐通道比较实测端与预测端，保存最大差、RMSE、有限性和 day_mask 不同像元数。不能静默替换三项几何。若存在差异，native 输入配对描述的是整套输入变化，不得声称 AGRI 单独造成；是否新增 common-geometry 诊断臂须先单独预定，不根据误差选臂。

当前 pack 本身不保存 grid_lat/grid_lon 或每行 geometry 来源；因此“shape/patch配置一致”不能替代上述逐样本坐标和几何检查。源 CPP NC 批次与生成 checkpoint 的绑定依然未核实，不把参考当独立真值。

## 6. 有界读取与之后的数值步骤

本次读取 metadata 约 10.54 MB，另加 3 个 NPY header、59 个分片 ZIP 目录/header；没有 tensor element 访问。下列是后续预算，不代表已读：

| 后续内容 | 预算 |
|---|---:|
| 58 个有配对 R 行的 x_raw + target + mask | 1,024,512 bytes |
| 若以固定64目标计的 R 行保守上限（JSON预算） | 1,130,496 bytes |
| 872 个 forecast 物理AGRI+geometry+已存cot patch | 15,179,776 bytes |
| 对59个selected shard全文件SHA流式核验 | 792,519,506 bytes；约755.81 MiB |
| 若重验R三数组完整SHA，额外顺序读取 | 327,031,680 bytes；无test、无raw fullgrid |
| 仅58个R目标三几何通道 | 178,176 bytes，包含于上述R预算 |
| 仅872个forecast三几何通道 | 2,678,784 bytes，包含于上述forecast预算 |

58 个 target 原/sidecar 两套 lat/lon 数组若均 float32，仅237,568 bytes；若 float64，上限475,136 bytes，另加很小的标量身份、CPP映射索引和静态 grid。NPZ 压缩成员按成员读取，绝不把整份样本解包后再“丢弃” GHI。source/static 文件实际压缩大小与 dtype 先读 header 再登记；遇到来源访问失败记录门禁缺口，不改样本。

Forecast 是 ZIP_STORED，可在核对选中分片 SHA、shape/dtype/indices 后，按 JSON `selected_shards[].members[].data_file_offset` 与 C-order 索引 seek 只取对应连续 station/lead patch。普通 `z['predicted_agri_physical'][i]` 会先载入整个成员，避免采用该方式批量扩大读取。R 的 3 个 npy 使用只读 memmap 后取固定行。原文件只读，诊断另名保存。

数值阶段只使用冻结 R 且 eval/no_grad，采用先前数值见证明确的同一 backend、dtype 和 batch 规则，对真实端和预测端都重新前向；已保存 cot_log1p 仅作缓存重放差值参考，不把不同 backend 的差混作输入域效应。207626 原设置重放旧 cache 仍有 max 0.001033 略超原0.001门限，warning 保留；benchmark/TF32关闭 GPU/CPU高度一致也不能单独证明所有旧cache通过或TF32是唯一原因。

两端输出用同一 `log1p(100*target)` 和同一 mask 计算逐 target/lead 的参考误差；可同时保存物理COT误差，必须从保存输出/参考/mask重算。按站点、16lead完整列样本数、有效像元数和缺配，不能因为某lead或天气不利而不报；共享 target 的多lead不是独立真值样本。此有限、validation-selected R上的描述性诊断不构成 test/generalization 证据，不进入可部署输入，不训练头，也不预言 domain harm。

## 7. 复核入口及实施边界

本次脚本已在学院 Python 实际运行完成，assert 检查通过；没有另写镜像实现的单元测试。首次运行中的“候选全为小时整点”假设在只读 metadata 阶段被真实15min记录拒绝，已改为实际分钟计数与合法15min检查；采样规则未变，任何数值都尚未读取。现有正式 JSON 不覆盖；需复核时另命名输出：

```bash
/public/home/slfu/miniconda3/envs/swc/bin/python \
  /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/scripts/inventory_cot_pairs.py \
  --root /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907 \
  --output /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/audits/cot_pair_inventory_20260908_check.json
```

没有来源/代码改变则无需反复运行。本轮仅新增本工程文档、inventory脚本和audit JSON，不修改 status 文档、旧训练数据、缓存、模型、论文或任何测试集。完成空间/几何门禁和独立脚本复核前，实际 R(real AGRI)/R(predicted AGRI) oracle 误差计算保持未执行。
