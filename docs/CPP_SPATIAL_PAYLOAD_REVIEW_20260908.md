# CPP 源像元至 R pack 的独立空间复核

2026-09-08，固定 12 行全部通过：源 CPP NetCDF 经独立 LAT/LON 坐标推导后，COT 与 corrected sidecar 逐像元相等；原参考 mask 和 R pack 的 `COT/100` float32 逐值相等。此范围内未发现转置、翻转、零/一基索引或 ±1 像元错误，不需要据此重训。

这是同模型家族独立代理的暂定审查。代码没有导入生产 `cpp_grid_mapping`、`cpp_quality_mask` 或 `pack_cot_training` 函数，没有模型前向、GPU、GHI/test 读取或旧资产修改。结论限于 12 行和它们绑定的 Combined 网格，不能据此保证全库、上游 CPP 反演质量或所有站点绝对地理定位；站点/静态网格/HDF 的对应属于并行空间审计。

## 固定样本与来源

样本在读取值前由主代理冻结：每站 × train/validation，按 UTC 排序的首行、中间 `floor((N−1)/2)` 行和末行，共 12 行。不按 COT、误差、覆盖率或有效像元数选样，不替换缺源。本次 12 行均有源，共 8 个唯一 NetCDF 文件。

- selection 文件 SHA：`d2557c7e37d6d55d60a295fb3c7c16a207783cd3928b6e9a8a37f0fb1054a49c`。
- selection 内容 SHA：`ae0cbeab4116ba3452f4f0e329b4ca545844c04fa8ad75457e0b6a9c4b50cb2b`。
- R pack 行号：四里 train `0/3005/6010`、validation `6011/7633/9256`；竺家 train `9257/12240/15223`、validation `15224/16868/18513`。
- HPC 根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。
- 原 NC 根：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP`；按选中 UTC 构造 `YYYY/YYYYMMDD/FY4B_AGRI_YYYYMMDDHHMMSS.nc`，并与 sidecar 的 `source_nc` 相等。

HPC 先核对 `rows.csv`、pack 哈希清单、`target.npy` 和 `mask.npy` 的全文件 SHA，重建固定排序身份，仅导出所选 12 行 target/mask。FD-107 检查导出报告/值 SHA 后读取选中原文件和 sidecar，核对两者全文件 SHA 与身份；8 个唯一 NC 均做全文件 SHA，并检查读取前后 size/mtime 未变。完整文件哈希包括文件所有字节，语义解码只限坐标、身份、原 day/SOZ、CPP COT 和固定 pack 切片，不解码 GHI 字段。

## 独立计算及结果

逐个目标像元在单调源轴上找最近 LAT/LON，要求与目标网格在原合同 `1e-4` 度内重合，验证行列不折叠、网格方向及 COT 维度含义。使用一个坐标决定的有界基本切片读取源 COT，再在 NumPy 中按配对二维索引取值，不依赖生产 NetCDF 双数组 fancy indexing。

所有源 COT 均为 float32，维度 `('LAT','LON')`、形状 `[4051,4051]`；LAT 递减，LON 递增，`Range='0 - 100'`、FillValue 为 NaN。所有样本源坐标最大误差为 **0 度**。

| 站点 | CPP 零基行索引（含端点） | CPP 零基列索引（含端点） | patch 纬度范围 | patch 经度范围 |
|---|---|---|---|---|
| 四里 | 1368–1383 | 2201–2216 | 25.68–26.28°N | 112.04–112.64°E |
| 竺家 | 1278–1293 | 2203–2218 | 29.28–29.88°N | 112.12–112.72°E |

这两组是 **CPP LAT/LON 网格索引**，不能当作 FY-4B 原始 HDF 行列。12 行的独立行列与 sidecar `cpp_rows/cpp_cols` 完全一致。转置、上下/左右翻转、双翻转、行列各 ±1，以及零/一基错读的预列坐标假设，全部不满足坐标容差；没有通过 COT 值寻找“更优位置”。

mask 独立按已有规则重建：源 COT 有限且在 `[0,100]`，day 有限且 `>0.5`，SOZ 有限且在 `[0,180]` 并 `<80`。它与 sidecar、pack 原 mask 完全一致；12 patch 共 3,072 像元、2,735 个有效参考像元。四里 train 末行保留 157 个，validation 首行保留 18 个；其余各 256 个，不删改任何 mask。

源物理 COT 与 sidecar COT（含 NaN 位置）逐值相等，有限位置最大差 **0**。按源 COT 和原 mask 重建 `where(mask, COT/float32(100), 0).astype(float32)`，与 pack target 逐值相等。单独记录的 float32 `÷100` 后 `×100` 最大有效像元舍入差为 `3.814697265625e-6` COT；它不是区域或索引错误。

从保存的小值又用独立 NumPy 片段复核：原生产公式的首列/首行 `argmin` 与新逐像元 searchsorted 坐标的完整二维索引完全相同；源/sidecar COT、mask、pack target 重建仍全部 exact。未回跑首次慢实现，也未重新访问全 NC 做第二次空间搜索。

## 中断记录与工程变更

首次只读审计在首个样本输出前被中断。中断栈停在坐标 `argmin`，不能据此认定它就是耗时瓶颈；此前对压缩块反复解压的怀疑未被证实，完整源文件哈希等步骤也可能耗时。原脚本和未完成 JSON（state 为 RUNNING）保留，另有明确的 interruption 记录说明该次已中断。

随后仅采用等价工程优化：单调轴 searchsorted 最近邻、一个有界基本切片；样本、坐标合同、mask、数据和模型均未改变。新输出独立命名为 `cpp_spatial_payload_20260908_v1_hyperslab`，正常完成，脚本内计时约 6.52 秒；不将其速度与冷缓存首次尝试作科学比较。

## 权威产物

FD-107 审计目录：`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/audits/cpp_spatial_payload_20260908_v1`。以下产物亦同步到 HPC 根的 `audits/` 和本地仓库。

| 产物 | SHA-256 |
|---|---|
| `cpp_spatial_payload_20260908_v1_pack.json` | `46de243207b7a98aba9b1ce39b8517319529454cee8e229476b876ad60d9913e` |
| `cpp_spatial_payload_20260908_v1_hyperslab.json` | `5c3d8d26d8ba4a3b16667a0562cb54a236cb5c5a50be5b59ddd943131a6296ce` |
| `cpp_spatial_payload_20260908_v1_hyperslab.npz`（317,536 bytes） | `7f2151daab0a06b5e2ecde73127f4d4fc1d51c5100da6954dade955ae34ddb77` |
| `cpp_spatial_payload_20260908_v1_mapping_equivalence.json` | `22ff26c0f470de7be459964ee852844c751d1a3bb03ff284dcf4d9e2db6201f3` |
| 当前 `scripts/check_cpp_spatial_payload_20260908.py` | `8c5f49da358676a3a13066c4895cfbffef026cb2655c599e3af9d1e58f53d2ab` |

完整报告状态为 `PASS_12_SOURCE_SIDECAR_PACK_PIXEL_CHAINS`；小值复核为 `PASS_12_SMALL_ARRAY_MAPPING_AND_PAYLOAD_RELOAD`。NPZ 保留每行源 LAT/LON 轴、目标网格、推导行列、源与 sidecar COT、day/SOZ、原/重建 mask、pack/重建 target，可直接独立复查。源 NC 到生成 checkpoint 的教师绑定仍未由本次空间一致性审计解决。
