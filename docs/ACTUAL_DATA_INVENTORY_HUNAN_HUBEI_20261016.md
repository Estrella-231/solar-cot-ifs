# 湖南、湖北实际使用数据清单

核查日期：2026-10-16，北京时间。范围：第二篇 COT 项目当前同期 GHI 实验及其共享/独立卫星场外推资产。依据服务器实际 pack、rows、训练 run_metadata、索引和来源脚本；没有用历史计划替代实际训练记录。所有日期范围均为北京时间边界，并不意味着逐日、逐时无缺测。

## 1. 当前实际使用概览

| 项目 | 湖南 | 湖北 |
|---|---|---|
| 站点 | Sili 四里、Zhujia 竺家 | Niushou 牛首、Wangzhai 王寨、Mizhuang 米庄 |
| 当前数据范围 | 2024-04-02—2026-06-30 | 2025-08-01—2026-07-14；7月14日实测仅到上午约08:55 |
| 卫星数据 | FY-4B AGRI；256×256区域预处理场 | FY-4B AGRI；已有预处理场/站点patch，另补六月至七月14日冻结train/val的HDF |
| 实际模型使用的光谱通道 | C01—C06、C09—C15，共13个 | 相同13个通道 |
| 太阳几何 | cosSOZ、cosRAA、day_mask；R/S用3项，已完成的同期GHI头用前2项 | 同左；各自区域坐标、索引和归一化 |
| COT监督参考 | Global_CPP中的FY4B AGRI CPP NetCDF，含COT和CLP | `cpp_16_fixed_v2`，采用有效COT平面；旧`cpp_16`不作本轮监督 |
| 地面实测 | 四里、竺家权威原始工作簿的5分钟区间末记录 | 三站原始分钟记录中的“总辐射1瞬时值”，来自按采集点月表及六月/七月分段表 |
| GHI标签 | `(T−15min,T]`内三条5分钟记录，即T−10、T−5、T的均值 | `(T−15min,T]`内15个分钟箱；箱内重复记录先均值，再对15箱等权均值；缺箱不插值 |
| 晴空辐照度 | pvlib/Ineichen计算的15分钟区间晴空值，用于kt及回算GHI | pvlib计算的15分钟区间晴空值，用于kt及回算GHI |
| 当前A/AC/AP输入空间 | 站点中心3×3 AGRI与对齐COT；R先在16×16上反演 | 各站中心3×3；R在覆盖三站的16×16共享patch上反演 |
| IFS是否实际进入本轮A/AC/AP | 没有 | 没有 |
| test | 当前核查资产均为train/val；test载荷关闭 | 同左，buffer载荷也不进入本轮训练 |

当前实际训练同时包含晴空、有云等不同日间场景。“晴空辐照度大于0”是归一化分母有效性要求，不等于只选择晴空天气。GHI实测是标签/评价量，不是模型输入；COT来自AGRI经冻结R反演，CPP不直接成为部署输入。

## 2. 已完成同期GHI对照实际读取的样本

样本单位为唯一`(station,target_time)`。同站点A、AC、AP使用同样的键、标签和有效掩膜。

| 区域/站点 | train | validation | train目标时间范围BJT | validation目标时间范围BJT |
|---|---:|---:|---|---|
| 湖南四里 | 24,155 | 2,128 | 2024-04-02 06:30—2026-06-30 19:15 | 2024-04-19 06:15—2026-06-25 19:15 |
| 湖南竺家 | 26,303 | 2,351 | 同上 | 同上 |
| 湖南合计 | **50,458** | **4,479** | | |
| 湖北牛首 | 9,104 | 856 | 2025-08-01 06:00—2026-07-14 08:45 | 2025-08-22 06:15—2026-06-29 19:30 |
| 湖北王寨 | 8,959 | 784 | 2025-08-01 06:00—2026-07-14 08:30 | 同上 |
| 湖北米庄 | 8,832 | 816 | 2025-08-01 06:00—2026-07-14 08:45 | 同上 |
| 湖北合计 | **26,895** | **2,456** | | |

湖南pack共54,937行，湖北pack共29,351行。两省分别从随机初始化训练各自GHI头，均已存在seed42/43/44的A/AC/AP产物。不能把湖南两年多的样本量和湖北不足一年的样本量当成同一数据规模；两省不合并训练、不互相初始化。

## 3. COT反演器实际见过的CPP监督，比GHI数据期短

| 项目 | 湖南R | 湖北R |
|---|---|---|
| 独立训练模型 | `monthly_station16_COTUNet_seed42_v1` | `hubei_R_station16_fixedv2_v1` |
| R输入 | 13 AGRI + 3几何，站点中心16×16 | 同类16通道，覆盖三站的共享16×16 |
| R监督样本train/val | **15,414 / 1,398**个站点patch | 原包**3,178 / 301**帧；实际非空CPP监督**2,811 / 259**帧 |
| train监督时间边界 | 2024-04-02 08:00—2026-02-27 17:00 | 原R包2025-08-01 06:00—2026-02-27 18:00；不是每一包行都含有效CPP监督 |
| validation时间边界 | 2024-04-19 07:00—2026-03-02 17:00 | 原R包2025-08-22 06:15—2026-03-02 18:00；其中42行没有有效CPP像元 |
| 监督掩膜 | 有限COT∈[0,100]、CLP∈{0,1,2}、13通道均有效、day_mask>0.5且cosSOZ>cos80° | 实际pack为CPP COT有限且∈[0,100]；全无效CPP行不进R损失/选模。本轮不能称其使用了湖南完全相同的CLP/太阳高度掩膜 |
| 归一化支持 | 由train监督支持拟合 | norm实际记录使用3,178个train行的有限输入像元；无val/test参与，不能笼统称仅用2,811个非空CPP行 |
| CPP的身份 | 检索参考，含有效0；缺失与无效不当晴空 | 同左；367个train和42个val空监督行仍保留供后续AGRI推理 |

湖南R pack第一次构建记录13个train源帧读取错误（两站共26行），作为不可用来源留档；后续实际COT bank对55,733帧已全部生成，无源错误。不能把这两阶段的计数混用。

**两省的CPP监督支持都只到2026年3月初，而GHI样本延伸到6月或7月。** 之后的AGRI仍可经冻结R生成COT，但这属于R在未获该阶段CPP监督日期上的应用；不能称已有同月份CPP独立准确度验证。

湖南COT bank：51,236 train帧+4,497 val帧=55,733帧，每帧各有两站16×16输出。湖北扩展COT bank：22,660 train帧+2,045 val帧=24,705帧，共享16×16输出。bank帧数不同于GHI有效站点样本数。

湖北三站在共享16×16内的中心像元分别为牛首(8,6)、王寨(9,8)、米庄(7,10)，均按各站中心取3×3；湖南两站各自重定心patch中心为(8,8)。

## 4. 日期划分与外推数据

两省都按月留连续2天validation、另2天test，每个留块前后1天buffer，其他合格日期train；日期分配依据文件/时间支持，非GHI大小或效果。稀疏月份只train。

| 项目 | 湖南 | 湖北 |
|---|---|---|
| split ID | `hunan_monthly_block_2d_buffer1d_seed42_v1` | `hubei_monthly_block_2d_buffer1d_seed42_202508_202607_frozen_v1` |
| 日期数量train/val/test/buffer | 588 / 52 / 52 / 104 | 250 / 22 / 22 / 44 |
| 稀疏月份 | 2025-09仅train | 2026-07仅train；结束于7月14日 |
| 最新实际场外推输入 | 8历史帧×16通道×256×256 | 同形状 |
| 预测目标 | 未来16帧×13 AGRI，+15…+240分钟 | 同左 |
| 实际train/val序列数 | **35,320 / 3,140**；当前matched-control/retrieval续训run_metadata再次确认 | **4,086 / 407**；正式strict24-same-day-v3使用的严格索引 |

湖北正式S进一步要求24帧同一BJT日、同一split；跨日/跨split窗口已剔除。旧5,153/509索引及旧retry bank不能替代正式4,086/407。湖北S的监督为真实未来AGRI，不使用GHI标签；它与同期GHI pack是两个数据任务。

湖北六月—七月补充中：站点patch持久化取得2,644帧；全场新增清单为2,512帧（部分时次原已有全场）。这两个数不是矛盾。按实际产品日历每完整UTC日95次、常规缺UTC17:00；在28个冻结train/val BJT日，2,660个合法发布槽取得2,644个，仍缺16个，不插值。不能继续把按96次/日算出的44个差集全称为真实缺档。

本轮未打开test数据；表内test日期数仅来自元数据。湖南新的同月重训合同与旧论文/旧split结果须区分。

## 5. 最新三帧/区间对齐分支

湖南另有已完成的三帧真实AGRI同期诊断：T−30/T−15/T，各帧局地3×3，实际pack为train47,666、val4,263，共51,929行；GHI几何修正但冻结R/COT来源保留原版。该分支不能与上述54,937行单帧同期主包混算。

本聊天推进的`interval_alignment_20261015/tensor_pack_v1`是另一份按实际扫描时间权重构建的候选。2026-10-16现场PBS作业214221仍Q，尚无tensor pack产物，不能将其54,461条manifest候选写成已用于训练。独立的corrected-geometry R目录也不能因计划存在就称已替换同期主包R。

## 6. 原始来源和当前实际训练位置

### 湖南

- AGRI迁移副本：`zjnu-hpc:/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/`，数据/辅助为`data/`、`aux_256/`、`grid_static.npz`。历史R rows仍保存复旦原来源绝对路径，不代表训练时始终从该旧路径读取。
- AGRI原来源：`/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed/`。
- CPP实际行示例：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/2024/20240401/FY4B_AGRI_20240401160000.nc`；该文件名UTC对应2024-04-02 00:00 BJT，非有监督白天样本的例子。监督范围/数量以上文SUPERVISED状态计。
- 原始工作簿：`/home/Data_Pool_3/chenyi/GHI/data/hunan/raw/stations/四里光伏.xlsx`与`竺家光伏.xlsx`。源工作簿库存期2024-01-01至2026-07-01；本轮实际候选范围更窄，不能把整个工作簿期写成实际训练期。
- 已完成GHI实际pack：`zjnu-hpc:/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/monthly_R_station16_20261013_v1/data/ghi_head_pack_new_r_v1/`。
- 当前卫星场matched续训：同工程根下`experiments/hunan_simvp_history_retrieval_v2_20261016/`；仍用35,320/3,140序列，无COT输入。它不自动替换第二篇固定外推资产。

### 湖北

- 原数据根：`/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/`。`data16_label/`只作为AGRI/几何来源，文件名中的旧辐照度与旧标签不作为本轮15分钟标签。
- CPP：上述根下`cpp_16_fixed_v2/`；R实际训练包在`zjnu-hpc:/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_r_runtime_gate_20261015_v1/r_pack_trainval/`。
- 地面月表源：`/home/Data_Pool_3/chenyi/GHI/data/hubei/raw/station_aggregates/按采集点聚合时间/`，从`天气数据_202508.xlsx`至`天气数据_202605.xlsx`，另两份`天气数据_2026-06-01-2026-07-06.xlsx`、`天气数据_2026-07-07-2026-07-14.xlsx`。本轮依据冻结train/val键重建分钟箱标签，不沿用legacy_root两个月表的限制。
- 补充全场HDF清单中的实际源示例：`/home/Data_Pool/data/FY/FY4B/real_time_data/AGRI/4KM/2026/20260610/`；保留`source_hdf`、UTC开始时刻、BJT目标、输出与逐文件SHA。
- FD-201当前工程根：`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/`。
- GHI实际pack：该根下`pack/`；标签表`labels/ghi15_train_val_daylight_clear.csv`；AGRI总清单`manifests/agri_paths_trainval.csv`；冻结日期`input/day_assignment.csv`。
- 正式S：该根下`runs/b3_simvp_seed42_20261016_strict24_same_day_v3/`，stage输入在本机`/tmp/...`；实际模型参数副本已归档。

湖北W/m²沿用原工程“总辐射1瞬时值”的操作量纲；1300是旧归一化常数而非单位转换。仪器校准/导出字段字典仍未独立核验，这一限制不应省略。湖南源显式日间0值仍保留；可疑连续零仅标记/敏感性分析，不按误差删标签。

## 7. 本次现场证据

归档目录：`results/actual_data_inventory_20261016/`。

- `hunan_live_metadata.json`：HPC实际R/GHI pack、bank、训练metadata、标签join审计及CSV逐行计数/时间边界/源路径示例。
- `hubei_live_metadata.json`：FD-201实际GHI pack、run、扩展AGRI/COT bank、R来源、strict索引与train-only norm。
- `hubei_r_live_metadata.json`：HPC实际湖北R pack、行清单、日期与QA。内部原pack状态含逐行source metadata，故文件比其他快照大。
- `hunan_current_simvp_run_metadata.json`：当前HPC matched-retrieval训练直接读取的run_metadata。
- `hubei_current_simvp_model_param.json`：FD-201正式strict-v3场外推训练参数。
- 读取脚本：`scripts/audit_actual_regional_data_20261016.py`，只读train/val清单及既有JSON，没有打开test数组或原工作簿test值。

本清单用于工程溯源；没有修改模型、数据、PBS任务或论文结果。
