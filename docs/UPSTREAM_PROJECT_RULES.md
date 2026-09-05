# Irradiance Forecasting Project Rules

本文件是本工作区辐照度预报任务的最高优先级项目记忆。进行数据构建、训练、评估、绘图或报告前，必须同时阅读：

`IRRADIANCE_ERROR_PREVENTION.md`

## 1. 规则优先级

本项目规则分为两类，不能混淆：

1. **不可违反的数据与评估规则**：时间系统、空间索引、标签定义、因果可用性、样本公平性和结果溯源。
2. **可选择的模型合同**：通道数、网络结构和 checkpoint。模型合同由当前已批准方案决定，不得把一次候选实验永久写成全项目强制要求。

如果二者冲突，以数据与评估完整性为先。改变模型合同必须单独命名、保存新输出并记录原因，但不需要为了符合旧候选合同而重训已经可用的正式模型。

## 2. 当前批准的论文路线（2026-09-03）

### 2.1 湖南主训练、短临为主

- 论文研究区域固定为**湖南**。论文主模型和所有 learned baseline 均使用湖南数据完成训练、验证、选模和独立时间测试，不采用湖北训练权重迁移到湖南，也不把湖北数据并入论文训练集。湖北资产只允许用于论文之外的历史复现，不进入正文、附录或主结果。
- 正文以 `0–4 h`、15 min 分辨率的卫星短临辐照度预报为核心；IFS `D+1～D+3` 逐小时预报作为短临结束后的中期衔接模块，独立成节、独立评价，不与短临指标直接混排。
- 当前确定性 SimVP、Direct-GHI 和 IFS-only ResNet 均视为可复现基线资产，不是不可替换的最终创新。后续允许在湖南数据上训练更先进的时序模型、运动引导 diffusion 和站点预测头，但必须使用独立配置、checkpoint、结果目录和 provenance。
- 当前首选方法研究方向为“确定性时序骨干 + 固定因果运动/输运 + transport-frame innovation diffusion + 云物理/太阳几何/NWP 条件 + 站点 GHI 输出”。该方向必须通过严格对照和消融后才能写成论文贡献，不能预先宣称有效。

### 2.2 既有 16 通道 SimVP 作为正式基线资产

- 既有契约：过去 8 帧预测未来 16 帧，15 min 时间分辨率；输入 `[8,16,256,256]`。
- 16 个输入通道：13 个 AGRI（C01–C06、C09–C15）+ `cosSOZ` + `cosRAA` + `day_mask`；预测目标为 13 个 AGRI 通道。
- C07/C08 不进入该 checkpoint。该事实仅说明旧资产合同，不代表已证明删除 C07/C08 普遍更优。
- 真实覆盖历史末帧后的 `+15,+30,...,+240 min`。旧产物把第一张未来帧称为 `issue_time`，因此旧 `lead_minutes` 写成 `0,15,...,225`；这是坐标原点命名差异，不是缺少 +240 min 帧。

既有正式 checkpoint：

```text
/home/Data_Pool_3/chenyi/辐照度预报/SimVPv2/work_dirs/hbagri_2h4h_15min_gSTA/checkpoints/best.ckpt
SHA-256: db620b622f986598068775bf90423414929a5357e4ca8f4bb36c5b737b9471d0
```

### 2.3 新模型与通道升级

`20 个原始 AGRI + solar_altitude + azimuth_abs_diff` 不是强制合同，也不因旧 checkpoint 使用 16 通道而被禁止。任何新的通道组合、时序骨干或 diffusion 实验都必须独立命名，并保存自己的数据合同、配置、checkpoint、结果目录和消融依据；不得与旧 16 通道 checkpoint 混用或冒充同一模型。

### 2.4 当前论文中删除或降级的路线

- 旧 area-to-point v2 不进入当前论文方法和实验矩阵；这是用户于 2026-08-14 明确作出的范围决定。
- 不因为旧文档曾把 v2 写成 fallback，就重新加入或重训它。
- “湖北训练、湖南零样本外推”从当前论文实验矩阵中删除。不得以湖北 checkpoint 初始化湖南论文主模型并将其描述为湖南独立训练；若未来研究迁移学习，必须另建独立项目和结果目录。
- 禁止使用已确认存在时间/行索引/scan 标签问题的旧 scan-time v3 checkpoint 作为最终模型或汇报模型。
- 注意：正式 `direct_ghi_v3_cached_<Station>/best_model.pth` 是交接文档核验过的 Direct-GHI 模型；它与上述被禁的旧 scan-time v3 不是同一资产，不能因目录名含 `v3` 而误禁。

## 3. 不可违反的数据规则

### 3.1 时间系统

- 已有 matched `.npz` 文件名时间为北京时间（BJT）。
- 匹配 FY-4B HDF 时使用：`HDF UTC = matched filename BJT - 8 h`。
- 必须区分初始化时间 `init_time` 与目标时间 `target_time`。
- 统一标准：`init_time = history_end_time = 最后一张输入帧时间`，`target_time = init_time + lead_minutes`。
- 正式主时效为 `lead_minutes={15,30,...,240}`。读取旧产物时使用：`canonical_lead = legacy_lead + 15 min`，并把旧 `issue_time` 重命名理解为 `forecast_start_time/first_target_time`。

### 3.2 空间范围和 HDF 索引

- 湖南是当前主区域，必须使用湖南自己的 `grid_static`、裁剪范围、站点经纬度和真实 HDF `row_index/col_index`；不得复用湖北像元索引。
- 下列湖北常量仅在复现湖北历史资产或外部验证时使用，不是湖南论文主实验的空间合同：
- 湖北 256 crop：`R1=1112, R2=1368, C1=2077, C2=2333`。
- 湖北三站 16×16 patch：`CROP_ROW=103, CROP_COL=118, CROP_SIZE=16`。
- 读取 `NOMObsTime` 时必须使用 `grid_static.npz["row_index"]` 和 `grid_static.npz["col_index"]`；禁止用 `R1 + CROP_ROW` 或 `C1 + CROP_COL` 代替真实 HDF 像元索引。

### 3.3 标签与输入边界

- `target_time = init_time + k×15 min` 统一定义为第 `k` 个未来 15 min 交付区间的结束时刻；对应标签窗口为 `(init_time+(k-1)×15 min, init_time+k×15 min]`。因此 16 个标签完整覆盖初始化后的 0～240 min，不得把 `target_time` 当窗口起点再向后取 15 min。
- 权威湖南工作簿和正式 `HuNan_Station16_Combined/schema.json` 均按 5 min 区间结束时刻记时；一个完整 15 min 标签取 `target_time-10 min`、`target_time-5 min` 和 `target_time` 三个记录。旧缓存若以某个文件时间作为区间起点并取 `+5/+10/+15 min`，只有将其标签坐标显式重映射到“文件时间+15 min”的 canonical target 后才可使用。
- 论文主标签为同一 15 min 窗口内 `mean(GHI)/mean(clear_sky_GHI)` 定义的区间 `kt`；GHI 用于标签和评价。这样 `pred_GHI = pred_kt × mean(clear_sky_GHI)` 可精确回算。
- 站点实测辐照度不得作为模型输入。
- 真实未来 AGRI、CPP/COT 真值和目标时刻之后发布的 NWP 不得进入推理输入。
- 不使用精确 2–3 min scan GHI 作为权重为 0.3 的强辅助回归目标；该标签比 15 min 均值噪声更大。

### 3.4 Excel 与站点映射

- 必须读取所有相关 sheet，不能只读第一个 sheet。
- 湖南主站至少覆盖：`Sili/四里`、`Zhujia/竺家`，以权威工作簿和站点坐标表为准。
- 若进行湖北外部验证，站点映射至少覆盖：`Niushou/牛首供电所`、`Wangzhai/王寨供电所`、`Mizhuang/米庄供电所`。

## 4. NWP 因果性与单位

- IFS/GFS 必须记录 `init_time`、`release_time`、`valid_time`、step/forecast hour 和变量单位。
- 业务因果评价只能使用 `release_time <= project init_time` 的产品；离线有效时刻匹配必须明确标为非业务回放。
- `ssrd` 等累计辐射量必须按相邻 step 差分并除以时间间隔；`DSWRF` 等平均通量不得重复差分或除以 3600。每个文件以元数据为准。
- 当前正式端到端回放的 `ecmwf_mode=legacy` 是按目标有效时刻匹配，不是严格 issue-time 业务回放；论文不得把它描述成已完成严格实时验证。

## 5. 评估与绘图合同

- 所有方法必须使用完全相同的 `(station, init_time, target_time)`、实测值和白天掩膜后再比较。
- 标准坐标下不存在同刻 lead-0，主结果为 +15～+240 min。旧文件中的 `legacy lead=0` 是第一张未来帧，即标准 `+15 min`，必须保留并转换，不能误删。
- 观测默认按以 `target_time` 为结束时刻的目标 15 min 窗口平均；不得把原生 1/5 min 折线与 15 min 预测指标混算。
- 报告必须明确使用真实历史 AGRI 还是 SimVP 外推 AGRI。
- `forecast AGRI` 是预测/外推云图，不是真实历史 AGRI；两种模式的指标不得混为同一实验。
- 绘图横轴必须说明是初始化时间还是目标时间；不得将不同意义的时间点连成一条未注明的曲线。
- IFS/GFS 缺少的有效时刻应断线或标注，不得用跨时段直线伪装成完整逐小时产品。

## 6. 模型资产与结果治理

- 旧 16 通道 SimVP、Direct-GHI 和 IFS-only ResNet 作为基线复用时，必须先核验 checkpoint、配置、归一化文件和 SHA-256。
- 新湖南主模型可以重新训练时序骨干、diffusion 和站点算子，但不得覆盖或改写旧基线资产；每种模型使用独立配置、结果目录和数据 manifest。
- 若某个实验声明冻结 SimVP/COT，则不得对其反向传播；若做端到端联合训练，必须作为单独实验明确命名，不能沿用“冻结基线”名义。
- 修改模型、输入合同或归一化文件时，必须同步更新交接文档、checkpoint 哈希和结果 provenance。
- 新实验不得覆盖已完成的正式回放目录。
- 指标必须由保存的逐样本预测和真值重算；不得为了突出项目优势而调整样本、天气或时间窗口。

## 7. 当前开始新工作的最小检查表

论文正文仅保留支撑科学问题、方法复现和结论所必需的信息，服务器路径、模型哈希、历史错误、修复过程和内部限制统一放入工程文档或补充材料。此后修改论文时，英文 LaTeX 正文与中文对照 Markdown 必须同步更新，并保持章节结构、公式、数据口径和结论完全一致。

```text
[ ] 已读取 AGENTS.md 和 IRRADIANCE_ERROR_PREVENTION.md
[ ] 当前任务是否属于湖南主训练/验证/测试；若使用湖北，是否明确标为外部验证或历史复现
[ ] 使用旧 16 通道 SimVP 还是新模型；二者的数据、配置、checkpoint 和结果目录是否完全分开
[ ] matched 文件名按 BJT 处理，HDF 查找使用 BJT-8h
[ ] NOMObsTime 使用 grid_static 的真实 row_index/col_index
[ ] 湖南 grid_static/crop/站点坐标使用正确；未复用湖北索引
[ ] 站点实测只作标签/评价，不进入模型输入
[ ] 15 min 标签窗口、init/target/lead 定义一致
[ ] NWP 的发布时间、有效时间、单位和累计/平均口径已审计
[ ] 所有比较使用同一站点、时刻、真值和白天掩膜
[ ] 已区分真实 AGRI 与 forecast AGRI
[ ] 标准 init 使用历史末帧；旧 lead 0–225 已转换为标准 lead 15–240
[ ] 未使用被禁的旧 scan-time v3，也未误禁正式 direct_ghi_v3_cached 模型
```
