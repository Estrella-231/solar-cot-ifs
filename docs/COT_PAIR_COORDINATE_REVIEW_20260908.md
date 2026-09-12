# 固定 COT 配对坐标检查的独立复核

2026-09-08，复核结论为 `PASS_INDEPENDENT_COORDINATE_PROVENANCE_AND_REPORT_CONSISTENCY`。未发现会改变本次58目标坐标判断的实现错误。这里是另一执行者的代码、来源与报告一致性复核；不是重新读取 FD-107 的116个源文件，也不是 COT 误差、模型增益或独立真值验证。语义审阅属 same-family/provisional。

## 审阅与实际执行范围

已完整审阅 `scripts/check_cot_pair_coordinates.py`。读取当前 inventory、静态网格导出与坐标报告的 JSON；随后在学院 Python 实际执行只读一致性断言，并独立从冻结静态网格重建两站参考 patch。没有使用 GPU、载入模型、解码 AGRI/GHI/CPP 值或打开 test。

学院项目根为 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`；解释器为 `/public/home/slfu/miniconda3/envs/swc/bin/python`。核对的文件及 SHA 如下，local 与学院文件一致：

| 文件 | SHA-256 |
|---|---|
| scripts/check_cot_pair_coordinates.py | 2144890bc0795e6cb64207f47cf29e43a1157558d36b4975858a565dbb82febb |
| audits/cot_pair_inventory_20260908.json | 7e7f5f87cced284797ed46b47104c16ad62770c62eb6f611f88036637fa8cf2d |
| audits/cot_pair_grid_reference_20260908.json | 2bcc6000383a176ab00cbda102d5b21a8f0d30dcdce5ad855dd734bed63b94a1 |
| audits/cot_pair_coordinates_20260908.json | 4298da1f31e36dcba206bc5ebbd3e12298ad5a39748e9f9f6140e99b633a551d |

固定规则+选择内容 SHA 重新计算为 `8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318`。不能只核对 JSON 自报的 selection_sha256；本次独立断言也从 `selection_rule` 与 `selection` 重新序列化计算了一遍。

## 代码判断

脚本先验证固定选择内容 SHA 与64目标数量，只跳过清单中原本没有 pair 的6目标。它不根据坐标差、CPP值或任何误差重新挑样本，最终要求58完成/6缺配，异常只会生成 `ERROR_INCOMPLETE_COORDINATE_BINDING` 并重新抛出，不会成为 PASS。

每个 original/sidecar 先完整读取并比对 frozen selection 中的文件 SHA，再从同一份已核对 bytes 解码坐标与标量身份；不会出现“hash一个版本、load另一个版本”的读入间隙。原与sidecar的站点、目标 instant 一致性，以及 sidecar 的 original_sha256/schema_version 均有断言。`timestamp_utc` 中的无时区字符串按明确字段合同解释为 UTC，随后转 aware UTC instant 与冻结 target 比较；这一处理不会把字符串写法差异当作时间错配。

`compare()` 要求双方为16×16且全部有限，转 float64 后计算原值差。original 对 S、sidecar 对 S、original 对 sidecar 的纬度和经度全部使用读取样本前预定的 `1e-5°` 绝对门限，并另记 exact_equal。没有在比较前把源坐标量化成 float32；float32仅用于固定参考导出和规范化坐标 hash。坐标 hash 的字节规则为 little-endian float32、C-order、先 lat 后 lon，引用方与检查方一致。

主脚本的 `source_files_hashed_in_full=True` 与 `GHI_CPP_AGRI_values_decoded=False` 表述准确：full-file SHA会读取源文件全部字节，但只解码 grid_lat/grid_lon 和身份字段。不能把结果简写成“完全没有读取源文件字节”。原始数据未写回。

## 静态参考导出独立重建

只检查“导出 JSON 的 grid_sha256 字段等于冻结值”不足以证明 patch 真来自该 grid，因此本次独立读取学院实际文件：

```text
/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/grid_static.npz
```

文件2,098,626 bytes，重新计算 SHA 为 `54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62`，同时与冻结 S config、inventory 和导出 JSON 一致。导出内 S_config_sha256 也与学院实际 `configs/s_frozen_hunan_seed42.json` 文件 SHA 一致。

独立根据冻结 S 的 bounds 切取 `lat,lon,row_index,col_index`，四项均逐值等于导出 JSON 的对应16×16数组；重新生成坐标 hash 一致：

| 站点 | bounds | 重建结果 | coordinate SHA-256 |
|---|---|---|---|
| sili | [155,171,140,156] | 四项全部 exact equal | 562c09893b816024c0377ce88a99f3679608e51075436db2e082d6b9fe2229c2 |
| zhujia | [65,81,142,158] | 四项全部 exact equal | 8e6622519b7bb0f82952b4a961872913cbd7212d3836c6dfaed910854cafb8bd |

该执行没有重开116份 FD-107 original/sidecar；静态网格来源的独立重建弥补了主检查器依赖外部导出 JSON 的那一段来源核验。

## 报告与冻结清单的逐项一致性

独立检查确认报告的 inventory/reference/checker SHA 对应当前文件；58个坐标报告 key 无重复，恰好等于固定 selection 中有pair的58个 key，6个 unmatched key也恰好对应原清单。每条 R pack index、pair数量、S bounds、original/sidecar完整路径和登记SHA均与 selection 一致，合计872 pairs。

| 内容 | 独立复核结果 |
|---|---|
| 固定目标 | 64，未替换 |
| 有匹配目标 | 58（sili 28、zhujia 30） |
| 无匹配目标 | 6，原样保留 |
| 已有配对 | 872，原来的152个缺lead未补造 |
| 报告source记录 | 116，均按 original/sidecar 成对对应正确目标 |
| 原样本/sidecar/S 纬经度 | 所有报告比较 max_abs=0、exact_equal=True |
| 每个source坐标hash | 等于对应站点的静态参考hash |

这支持“固定58个有匹配目标的坐标绑定已通过，当前报告没有选中样本的网格错位证据”。不能将其扩展为未检查目标、CPP参考检索值、掩膜质量或 COT 输入域效应的验证。完整源文件哈希/身份执行来自已复核的主检查器，本独立复核没有重复全部源I/O。

下一步仍须使用单独的太阳几何检查结果解释 R 真实端与预测端的3个geometry通道是否相同，并保留此前cache数值warning。坐标通过本身不放行“AGRI单独导致损害/收益”的结论，也不改变固定选择、GHI cohort、现有模型或训练预算。
