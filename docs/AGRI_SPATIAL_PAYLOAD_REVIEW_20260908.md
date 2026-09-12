# AGRI 实际像元链复查：固定12个train/validation样本

2026-09-08，本次实际数值检查为 `PASS_FIXED_12_AGRI_PAYLOAD_CHAIN`。12个预先固定的源行全部可读，没有替换日期。原 Combined 的20通道16×16像元与FD-107、学院对应full AGRI片逐值相同；依照冻结R通道与几何公式重建的16通道，与R pack固定行逐值相同，所有这些比较最大绝对差为0、非有限模式相同。这补充了此前仅靠坐标字段无法证明的payload链。

本结果只覆盖12个检查点，不直接证明所有日期、历史节点缓存、全部HDF校准通道或COT检索参考都正确。它不评价GHI或COT误差，不解释负pilot原因。未训练、未载入模型、未读取test；原Combined文件为核对完整SHA而读取bytes，只有AGRI及身份字段被解码，GHI/CPP数值没有解码。

## 固定选择和事前判断规则

选择由主任务在数值读取前封存：每站×train/validation按UTC排序，选秩0、floor((N−1)/2)、N−1。selection文件为 `audits/cot_spatial_reaudit_selection_20260908_v1.json`，文件SHA `d2557c7e37d6d55d60a295fb3c7c16a207783cd3928b6e9a8a37f0fb1054a49c`，选择SHA `ae0cbeab4116ba3452f4f0e329b4ca545844c04fa8ad75457e0b6a9c4b50cb2b`。未按CPP值、有效像元数量、forecast覆盖或误差筛选。

| 站点 | split | 首 / 中 / 末 UTC |
|---|---|---|
| sili | train | 2024-04-02 00:00 / 2025-01-04 03:00 / 2025-06-30 10:30 |
| sili | validation | 2025-06-30 22:30 / 2025-08-04 05:00 / 2025-09-22 06:15 |
| zhujia | train | 2024-04-02 00:00 / 2025-01-04 04:00 / 2025-06-30 10:30 |
| zhujia | validation | 2025-06-30 22:30 / 2025-08-04 02:15 / 2025-09-22 06:15 |

9月22日不在现有forecast缓存覆盖中，但这次检查的是真实source链，仍按固定清单检查该日，不换成有forecast的日期。

复制链预定要求 `exact_equal_nan`，并分别保存finite、NaN、+Inf、−Inf模式。R物理13通道要求exact；重建的三项三角函数/掩膜事前绝对门限为2e-6，实际也全部exact/max0。没有随结果扩大容差。

空间诊断事前固定3×3偏移（dr,dc∈−1,0,1）、transpose、上下翻转、左右翻转、双翻转、full-grid行列起点互换。它们只作错误模式检查，不根据哪个候选降低预测误差来选择新crop；任何输出、配置与模型都不改写。

## 实际输入链与源代码

FD-107当前 Combined 源构建脚本快照：

```text
/home/Data_Pool_3/chenyi/GHI/scripts/legacy_root/build_hunan_station16_samples.py
```

快照SHA `8a38195d276199a1a9000ad5a9c33798221a865ba4231ff55afdd1e36bcd3be9`，保存于 `audits/agri_spatial_payload_20260908_v1_sources/build_hunan_station16_samples.py`。`process_timestamp`要求full为float32 `[20,256,256]`，直接执行 `agri_full[:,r0:r1,c0:c1]`，未转置/翻转/重排或归一化20通道。当前快照不是当年逐样本内嵌的构建代码hash，不能声称恢复了所有历史构建版本。

冻结站点patch分别为sili `[155,171,140,156]`、zhujia `[65,81,142,158]`。本次从原Combined的 `agri_crop_bounds` 验证这两个矩形，再直接比较像元。full文件按nominal UTC路径寻址：

```text
FD-107:
/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed/data/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHHMM.npy

zjnu-hpc:
/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/data/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHHMM.npy
```

20通道元数据顺序是C01–C15、SAA、SAZ、SOA、SOZ、day_mask。C01–C06为反射率；C07/C08为3.75μm不同增益的等效亮温，C09–C15为亮温；三个角度相关数组使用度，day_mask使用SOZ<85°。原数据在SOZ≥90°将可见光置零，并将SOA置零；aux几何在nominal时刻计算。这些单位/意义来自当前channel_info与dataset_meta，本子任务没有独立重做全部DN/LUT标定。

R输入依次抽取原通道 `[0,1,2,3,4,5,8,9,10,11,12,13,14]`，追加 `cos(deg2rad(SOZ))`、`cos(deg2rad(SOA-SAA))`、`day_mask>.5`。将这段合同作用于实际Combined数组后，与学院：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/data/cot_repaired_pack_20260907_v1/x_raw.npy
```

的固定12行逐通道比较。rows.csv和norm.json当前SHA与封存一致；x_raw使用只读memmap读取12行，未重新扫描整数组SHA。每个选中行的payload hash单独写入新报告。

S实际读取代码来自：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPDDP_20260907/scripts/hunan_data.py
/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPDDP_20260907/scripts/train_ddp.py
```

当前SHA分别为 `f4b4aab55101bde575f474d5daa4a184466c16399b401c2ac67bf666c5f28b91`、`3a47895d04f9b3f6ba9cfb530c8f373bcc9f3079b1a0db34de519c32b813880b`，均与frozen forecast contract绑定的source SHA一致；cache_frozen_forecast.py SHA也重新验证。S选择同一13通道，但还应用aux的 `agri_valid_bits`，无效值归一化后置0。R只处理归一化后的nonfinite，因此两个有效性规则并非天然同一规则；本次12行未发现有限值被S QA额外mask的像元。

没有调用会写入节点缓存的 `CachedData.frame`，没有生成或重建旧缓存。当前source代码一致不能替代全部历史node-local缓存的payload验收。

## 数值结果与方向检查

| 对照 | 检查规模 | 结果 |
|---|---:|---|
| Combined20通道 vs FD full canonical片 | 12×20×16×16 | 全部exact，max0 |
| Combined20通道 vs HPC full canonical片 | 12×20×16×16 | 全部exact，max0 |
| FD vs HPC full 18×18 halo | 12×20×18×18 | 全部exact，非有限模式一致 |
| Combined重建16通道 vs R x_raw行 | 12×16×16×16 | 全部exact，max0 |
| raw三几何公式 vs同target aux | 12×3×16×16 | 全部exact，max0 |
| FD vs HPC aux三几何及valid_bits | 12个固定target | 全部exact |
| 每样本13个非canonical空间候选 | 12×13 | 无候选与13通道原片完全一致 |

所有样本的canonical矩形都是预定候选中的唯一exact匹配。没有出现“一像元偏移、行列互换、transpose或flip后才一致”的现象。替代候选的聚合RMSE只保留为描述性调试值，各通道量纲不同，不作为科学性能比较或选偏移依据。

上述真实geometry一致不等于所有forecast cache中的同target三几何都已重新核验；该项由另一份forecast配对检查负责。特别是R反演数值误差、S预测输出域和CPP参考值均不在本子任务推理范围内。

## 读取量、产物和重现

主脚本仅CPU、numpy线程上限2。每个full NPY读取18×18 halo和行列起点互换的16×16小片；未materialize或传回full-grid数组。记录exact路径、size/mtime前后一致及小片SHA，没有重算整份full NPY SHA。aux NPZ按四个命名成员读取，NumPy会先解压完整该成员再切站点片，因此不能声称辅助文件也只发生patch字节I/O。

FD导出的NPZ为611,966 bytes，实际array payload 599,808 bytes，仅含12行Combined AGRI、FD halo、站点aux geometry和valid_bits，无CPP/GHI。这份小包在学院进行R/full交叉检查，原始/旧数据没有改写。

| 新产物 | SHA-256 |
|---|---|
| scripts/check_agri_spatial_payload_20260908.py | 69eaffe40dea7c655cd8bb48a29975a7ebda70f73eea723b42a2b2889b9131bc |
| audits/agri_spatial_payload_20260908_v1_fd.json | 9d3e6352e71f5ecfbe98b372aeffcab2e5337064dff661fc00ca3b38590978f6 |
| audits/agri_spatial_payload_20260908_v1_fd.npz | 8aa4d94ee55f020a845c717e7d25e41b4a738795d67ae498cbc94ebf50f4d30d |
| audits/agri_spatial_payload_20260908_v1_hpc.json | 3bb5dd7a4ef729394fd890ecb70bbb41d80d03a3009cacba3a30482427bad061 |
| audits/agri_spatial_payload_20260908_v1_source_metadata.json | 6c8ce1063dea42b56cee501a92914de5a1cafbf886ed11d8fc984a06b47b954b |

source_metadata补充保存12行exact aux路径、source_hdf、source_size_bytes、source_mtime_ns，以及两站static真实HDF row/col片，用于主任务另行预定的C02/C13原HDF校准见证。主payload检查阶段没有读取原HDF；后述独立补充仅重读两张小LUT和属性，不重读DN。

学院脚本已通过py_compile，两阶段实际执行均完成。若证据变化需要复查，使用新output-prefix，不能覆盖本次JSON/NPZ或失败尝试。FD阶段示例：

```bash
/home/Data_Pool/zjnu/.conda/envs/swc/bin/python -u \
  /tmp/check_agri_spatial_payload_20260908.py --stage fd \
  --selection /tmp/cot_spatial_reaudit_selection_20260908_v1.json \
  --output-prefix /tmp/agri_spatial_payload_20260908_v2_fd
```

学院阶段使用 `/public/home/slfu/miniconda3/envs/swc/bin/python` 和服务器项目根下脚本，提供 `--stage hpc --selection <同一冻结JSON> --fd-report <新FD JSON> --fd-arrays <新FD NPZ> --output-prefix <新的学院输出前缀>`。本次成功后无需重复全流程；若进一步调查，应针对尚未核对的具体上游环节或另行预定样本，而不是不断重新检查同一批exact配对。

## 追加：另一执行链的站点/HDF见证独立复核

本子任务随后只读复核主任务的 `check_station_grid_contract_20260908.py`、`check_raw_hdf_patch_20260908.py` 及其保存产物。结果另存 `audits/agri_spatial_payload_20260908_v1_peer_review.json`：`PASS_INDEPENDENT_STATIC_AND_SAVED_HDF_WITNESS_REVIEW`。没有重开原HDF或Combined，没有运行模型。语义代码审阅属same-family/provisional。

静态最近点采用与原检查不同的单位球Cartesian距离实现，独立得到sili `[163,148]`、zhujia `[73,150]`，均为16×16 patch内 `[8,8]`；与报告距离的差小于7e-13 km。站点schema/config/grid/source脚本SHA与原报告sources逐个匹配。

固定2024-04-02 00:00 UTC、两站各C02/C13的4份保存校准片，均与保存full AGRI片以及本子任务独立读取的Combined对应通道exact相等，valid/aux bits也exact。保存HDF行列片与静态grid中两个站点片exact相等。原Combined该时刻SOZ全<90°，本次可见光比较不靠night-zeroing掩盖DN/LUT差异。

原HDF校准脚本使用真实grid row/col索引，从最小science bounding block按每像元DN索引LUT，检查DN和LUT范围、填值、有限性；并核对source path/size/mtime与aux及读取前后一致。本次未发现改变该4组结果的逻辑错误。原HDF没有全文件SHA；这不等于独立地理定位标定，也不覆盖其他日期和其他13个通道。第一阶段peer_review只重算保存的校准值/full值/独立Combined值一致性，因为当时LUT未保存；该阶段报告保留原样。随后按明确授权补完下述两张LUT复算，没有扩样。

原执行脚本只检查report.json不存在，存在未来失败重试时覆盖遗留values.npz的风险；本次没有发生覆盖。主任务已将当前脚本改为拒绝非空目录，当前SHA为 `bfcf68233fd0d49b6ccf70b859967a87323cd1c5674cc4e83c846f7ba83b5e09`。本复核确认 `audits/cot_raw_hdf_witness_20260908_v1/run_script_snapshot.py` SHA仍为报告中的原执行SHA `adaf368fbdce8e38ca632ce0b8854179c31d6cec3d4bdf70311f863c076a1b2f`，没有以修改后的脚本冒充原执行版本，也没有为这项防护重跑HDF像元读取。

本地独立复核首次在读取UTF-8站点schema时触发Windows默认GBK解码错误，发生在数组复算前；失败说明保留为 `agri_spatial_payload_20260908_v1_peer_review_attempt1_failure.json`。随后显式UTF-8读取完成复核，未改数据、阈值或选择。

## 追加：同一目标的独立DN→LUT复算已完成

`audits/agri_spatial_payload_20260908_v1_peer_lut_review.json` 的状态为 `PASS_FOUR_SAVED_DN_INDEPENDENT_LUT_RECONSTRUCTION`。仍只检查2024-04-02 00:00 UTC两站各C02/C13；DN来自原见证已保存的values.npz，未重新读取原HDF DN、full AGRI或Combined源文件。辅助身份字段与本次HDF读取前后的size/mtime相符，aux SHA也与原报告相符。源HDF为：

```text
/home/Data_Pool/data/FY/FY4B/AGRI/4KM/2024/20240402/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240402000000_20240402001459_4000M_V0001.HDF
```

本次仅从 `Calibration/CALChannel02`、`Calibration/CALChannel13` 读取两张4096项的LUT及属性，共32,768 decoded array bytes；同时仅读取 `Data/NOMChannel02`、`Data/NOMChannel13` 的valid_range和FillValue属性，不读取这两个Data dataset的数组。数据DN有效范围均为[0,4095]、填值65535；C02 LUT范围[0,1.5]，C13为[100,500]，两张LUT的Slope=1、Intercept=0、FillValue=−65535、原dtype均为big-endian `>f4`。这些小数组及所有CAL属性被保存，单位属性原文为NUL，物理单位仍由主链channel_info合同解释。

独立代码先按新读取的DN属性做范围/填值/LUT长度检查，再按保存DN索引新LUT，检查LUT范围和有限性并输出float32物理值及valid。结果如下：

| 站点/通道 | 保存DN范围 | valid像元 | 重建值=原校准值=full=独立Combined | valid=保存valid=aux | 最大绝对差 |
|---|---|---|---|---|---|
| sili C02 | 256–1146 | 256 | exact | exact | 0 |
| zhujia C02 | 604–1411 | 256 | exact | exact | 0 |
| sili C13 | 1861–2461 | 256 | exact | exact | 0 |
| zhujia C13 | 2871–3517 | 256 | exact | exact | 0 |

四片无夜间置零触发。此步骤补齐了第一阶段缺少LUT时不能独立重算DN标定的缺口，但不增加日期/通道覆盖，也不把源HDF的path/size/mtime绑定称为整文件SHA验证。

| 新产物 | SHA-256 |
|---|---|
| audits/agri_spatial_payload_20260908_v1_peer_lut_snapshot.json | 6278bdad93468b12e78a7ddcb18fa8bb0eac4b51def45783c865c571b6934f33 |
| audits/agri_spatial_payload_20260908_v1_peer_lut_snapshot.npz | ef9e657d9d2bb419ef6da59dbd3a685a6cfa3a64926263af52746bf25208ab9f |
| audits/agri_spatial_payload_20260908_v1_peer_lut_review.json | b1352716a887e86364f9b8a47d1d72d89d6677b114fda471ddd9fffa99536eb5 |
| audits/agri_spatial_payload_20260908_v1_peer_lut_review.npz | c9c76bdec61e51408fbff03c7f81209b884c8e51f263e5aa37f39982f70760ff |
| audits/agri_spatial_payload_20260908_v1_sources/peer_lut_review.py | 68a45436e44a83bab53f9badd78e259316af2ba8bd7bc4b4c6ddaa0aae2385f3 |

新snapshot NPZ为24,973 bytes，复算输出NPZ为32,622 bytes。首个读取尝试使用native float32相等判断，拒绝源文件正常的big-endian `>f4`，在第一张4096项LUT读出后停止，没有写出snapshot，也没有读取DN。该失败与原脚本另存 `agri_spatial_payload_20260908_v1_peer_lut_attempt1_failure.json`、`agri_spatial_payload_20260908_v1_sources/peer_lut_review_attempt1.py`；随后仅把类型检查改为四字节浮点并保留原字节序，未放宽数值exact要求。包含失败尝试，此补充累计解码12,288个LUT元素、49,152 bytes；这是数组量，不是对HDF实际块I/O字节的估计。

本地和学院项目根下均保存上述新产物。FD只读阶段脚本为 `/tmp/agri_spatial_payload_20260908_v1_peer_lut_review_attempt2.py`，快照为 `/tmp/agri_spatial_payload_20260908_v1_peer_lut_snapshot.json/.npz`；FD解释器 `/home/Data_Pool/zjnu/.conda/envs/swc/bin/python`。独立复算在本地只读取保存的小数组，代码入口为 `peer_lut_review.py --stage local --project <项目根> --output <全新输出前缀>`。
