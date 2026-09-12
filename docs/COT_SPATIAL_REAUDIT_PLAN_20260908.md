# 用户要求的 COT 空间与输入链复查

2026-09-08 用户提出“可能是 COT 的区域等等错误了”，要求修改每小时任务并立即检查。此授权重开空间/输入合同审计，覆盖此前仅做无变化巡检的优先级。此前负收益结果仍保留；不因怀疑直接重训，也不因旧判定 closed 而停止本项检查。

## 要检验的问题

先前 58 目标的 original/sidecar/S-grid 经纬度字段相同、872 配对几何相同，只排除了被比较字段的差异。它们不能独立排除坐标共同误标、实际 CPP/AGRI 取错片、源 NC 写入时转置/翻转、上游生成时刻错位或缓存装配错误。本轮沿源实际像元复查，不能用这些旧 PASS 替代。

## 事前固定样本

从当前 R pack 的 rows.csv 中，对每站 sili/zhujia × train/validation，按 UTC 排序取 rank 0、floor((N−1)/2)、N−1，共 12 行。只用索引与时间选择，未按 COT/GHI 误差、覆盖、天气或参考值选择；缺源保留不换。完整选择位于 audits/cot_spatial_reaudit_selection_20260908_v1.json。

- rows.csv SHA：7741fda8e5a232c23598b9824811bf67ca5ba48d50b392f1c99a21b298177c69。
- selection 内容 SHA：ae0cbeab4116ba3452f4f0e329b4ca545844c04fa8ad75457e0b6a9c4b50cb2b。
- selection 文件 SHA：d2557c7e37d6d55d60a295fb3c7c16a207783cd3928b6e9a8a37f0fb1054a49c。

## 并行检查与判据

1. **CPP 实际像元链。** 独立从源 NC 的 LAT/LON 与目标 AGRI 经纬度计算索引，不导入生产映射函数；检查维度名、轴单调性、方向、行列顺序和零基索引。坐标沿生产合同 ≤1e-4 度，不事后放宽。原 NC COT 与 sidecar 实际像元、finite 模式、记录行列须精确对应；独立重建既有 mask 和 float32 masked COT/100，应与 R pack target/mask 精确相等。原训练值反乘 100 的舍入不能当区域偏移。固定 ±1、转置/翻转对照只用于解释识别，不以哪个误差更小选择新位置。
2. **AGRI 实际像元链。** 对同 12 行验证 Combined 20×16×16 与 full 湖南 AGRI 对应片，验证 R pack 实际 13 通道与三几何、S 的输入通道与 aux 有效位掩膜。无变换的原样切片要求 finite/NaN 模式与有限值精确一致；若不同实现存在已知 float32 数学误差，应在读取差值前声明对应容差和理由，不能事后为通过放宽。核实文件 UTC、站点轴、crop bounds、通道/单位，不能只验证标签经纬度。
3. **源 CPP 生成地理绑定。** 在已有具体路径内查生成/推理/NetCDF writer 与实际 batch/checkpoint 关联，检查输出 COT 方向是否与写入 LAT/LON 一致。邻近训练代码不等于实际生产代码，不能推断未找到的上游绑定；只读有界代码清点，不全树扫描。
4. **独立站点及网格。** 核查当前湖南整图、两站官方工程坐标、最近像元与16×16片中心；必要时沿已有 geolocation/HDF row_index/col_index 来源验证，而非把裁剪常数当原始 HDF 索引。湖南云场上下文矩形大于省界本身不构成裁错证据。

本轮只读 train/validation 和必要静态网格/来源代码；不解码 GHI、不打开 test、不训练，不修改源数据、缓存、norm、mask 或 checkpoint。先用 CPU、小 patch 完成；无必要不占 GPU。

## 结果与后续动作

每条证据记录实际源路径、SHA或源小片SHA的明确范围、时间、站点、轴、索引、数值差、finite模式和失败原因，另存报告和可复核小数组。原失败输出保留；扩大样本仅为定位具体未决差异，并先固定规则。主代理独立复算关键数组/判据，不能把执行者自报 PASS 当唯一证据。

若查出明确数据/代码错误，先定位受影响资产和范围，另名修复并验收，再按既有 original/QC1、A/AC、两档LR、seed42预算公平复跑。只有确实影响冻结 S/R 的错误才重训相应上游资产，保留旧版本。若未发现错误，报告本轮实际排除的错误与剩余未核实范围，不能证明所有区域问题不存在，也不把负收益自动归因于域差异。

服务器根：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907；来源 FD-107，CPP 位于 /home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP。源 NC 是再网格化经纬度产品，其 LAT/LON 索引不能用原 FY4B HDF 像元索引替代。

## 原 HDF 小范围见证（执行前补充固定）

选择固定12行中的首个train UTC目标2024-04-02T00:00:00Z、两站、C02/C13两通道，共4个station/channel片。读取aux绑定的单个原HDF，按静态grid真实row/col读取DN并独立用原Calibration LUT和valid_range/FillValue恢复物理量，执行既有可见通道夜间置0规则，再与full AGRI同片逐值比较，既有aux valid bit也应exact。不全文件hash该102MB HDF，记录aux指向及源size/mtime稳定性、实际小数组SHA；不将此称为独立绝对地理校准。源attrs显示Begin Line/Pixel均0，LUT4096项，DN0..4095；源观察时刻比名义起始晚秒级，不据此移动数据时间。
