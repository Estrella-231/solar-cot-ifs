> **2026-10-16 撤回CPP评价：已确认参考CPP的全表索引/val局部索引错配。下文历史指标不可作为模型性能结论。详见本目录 INVALID_CPP_JOIN.md 与 audits/cpp_reference_join_20261016/AUDIT.json。**

# 湖南 S→R COT 预报验证结果（仅验证集）

日期：2026-10-14（BJT）  
状态：FP64独立复算完成；探索性验证结果，不是独立测试结论。

## 数据与计算合同

- 冻结湖南月度 split 的 validation；271 个至少有一个 CPP 有效目标像元的 8→16 窗口，覆盖 8 个 BJT 目标日期。
- 每个窗口两站、16 个目标时效（+15 至 +240 min）。CPP 参考掩膜内共 550,400 个像元；两站、16 时效均有支持的完整窗口只有 8 个。
- 主比较为冻结 SimVP 预测 AGRI→冻结新月度 R（S→R）与最后历史帧 R 持续性（Persistence-R）。`Real-target AGRI→R` 单列为离线反演参考线，使用未来实测 AGRI，不属于可部署预报方法、不参与排名。
- 所有方法在相同 `(window, station, lead)`、CPP参考值及有效掩膜上评分。对 `log1p(COT)` 仅做一次 FP64 `expm1`；物理预测不截断。
- 主尺度为站点周围 3×3 原像元：先逐站、逐时效池化共同有效像元计算误差，再两站等权、16 时效等权汇总。站点中心像元和16×16 patch另列。
- 厚云固定阈值为 COT≥30。日期块 bootstrap 重采样单位为 BJT 目标日期，共 2,000 次；因只有8个目标日期，区间仅描述不确定性，不作稳定性/显著性结论。
- 测试载荷读取数为0。CPP是检索参考而非独立地面COT真值。

## 点估计

| 尺度 | 方法 | 平均逐时效站点宏平均 RMSE | MAE | Bias | 对持续性的平均逐时效 FS |
|---|---|---:|---:|---:|---:|
| 中心像元 | S→R | 25.422 | 21.293 | -13.711 | 6.51% |
| 中心像元 | Persistence-R | 28.051 | 22.565 | -4.507 | 0 |
| 3×3（主） | S→R | 25.538 | 21.393 | -13.791 | 6.68% |
| 3×3（主） | Persistence-R | 28.252 | 22.650 | -4.488 | 0 |
| 16×16 | S→R | 26.113 | 21.702 | -13.804 | 4.09% |
| 16×16 | Persistence-R | 27.731 | 22.260 | -5.064 | 0 |

COT为无量纲；负 Bias 表示低估。离线 `Real-target AGRI→R` 的 3×3 RMSE 为25.868、MAE为21.874、Bias为-12.201。它反映实际目标AGRI经R反演与CPP参考的一致性，不是可部署预报，也不是独立真值。

3×3按站点、16个 lead 等权平均的 RMSE：Sili 为29.17（S→R）/30.88（持续性），Zhujia 为21.91/25.62。逐 lead 数据见 `cot_forecast_by_lead.csv`。

补充口径：将同一站点、尺度的全部有效像元跨16个 lead 一并池化后，再两站等权，RMSE 为1×1 `25.40/28.32/25.77`、3×3 `25.52/28.54/25.84`、16×16 `26.09/27.84/26.01`（依次为S→R / Persistence-R / 离线真实目标AGRI→R）。这是跨lead池化补充量，不替代上述预注册的“逐 lead 计算、lead等权”主终点。对应的连续2日日期块bootstrap敏感性区间也全部跨0；例如16×16 S→R减Persistence-R为约`[-5.70,+0.87]`，S→R减离线参考为约`[-2.90,+1.99]`。完整表见 bank 目录内 `FORECAST_COT_VAL_BANK_RESULTS_20261014.md` 和 `forecast_cot_metrics_fp64.json`。

## 厚云诊断与不确定性

3×3全部共同像元上，固定 COT≥30 的 S→R 厚云 CSI 为0.082（POD 0.090，FAR 0.522）；Persistence-R CSI为0.189（POD 0.254，FAR 0.579）。因此，较低的总体 COT RMSE 不代表厚云识别更好；需同时报告该退化。

主尺度 S→R 相对 Persistence-R 的平均逐 lead 宏 RMSE 差为 -2.714 COT；8日期配对 bootstrap 的描述性95%区间为[-6.476, 0.493]，包含0。日期数未达到20，不能据此声称稳定改善或统计显著。另有大量重叠预报窗口，但它们不增加独立天气日期数。

### 补充结构指标：log1p-COT SSIM

按预注册口径，仅用新月度R训练包中 CPP 有效的训练参考值确定尺度：15,414个监督训练patch、3,921,824个有效像元，物理COT q99.9=74.0397，`L=log1p(q99.9)=4.3180`。验证时对 `log1p(max(COT,0))/L` 截断到[0,1]，使用11×11 Gaussian窗口（σ=1.5、K1=0.01、K2=0.03、population covariance），只计算整个窗口CPP mask均有效的中心；不填补缺测，也不改变未截断物理COT误差。

在有效窗口均值后按站点和lead等权的描述性汇总中，S→R SSIM=0.2356，Persistence-R=0.2468；S→R未优于持续性。两者各有77,400个有效11×11窗口中心，占全部理论窗口中心的24.8%。验证参考有626个共同有效像元高于训练q99.9；SSIM映射因此会截顶。预测值在SSIM映射中截顶的像元数为S→R 0、持续性13,377；主物理误差仍不截断。该指标只评价小站点patch中的结构相似性，不替代COT物理误差，也不构成独立真值或部署优势证据。

逐窗逐站逐lead记录与完整尺度收据见 `../cot_forecast_eval_20261014_v5/cot_ssim_by_window.csv` 和 `cot_ssim_metrics.json`；训练尺度源COT/mask SHA-256分别为 `43dab24e9b6c5cf9b9db09f6bce16b3dede60750b312c6e808b192ec265fb63c`、`2614ca0b589ff3e3367f08f0e8f8ed8067226f80446f4e27f3478d5c675c340f`。

逐时效曲线显示，持续性在短 lead 与 S→R 接近，之后误差快速增大；S→R在约+150 min以后点估计较低。该形态仅是这8个CPP支持日期上的描述性现象。S→R仍有约-13.8的系统性低估，需与 R 对CPP的同期反演偏差及厚云漏报一并解释。

## 结论边界与下一步

当前结果支持“冻结 SimVP→R 在有限 CPP 支持验证样本上可产生有限的 COT 预报，并可与持续性完成同支持评价”；不支持“COT预报已稳定优于持续性”“厚云预报更好”“GHI预报收益”或独立测试泛化。严格16 lead完整样本和独立目标日期很少，扩大有效CPP验证日期覆盖是首要数据缺口。

本轮没有发现可用且与当前新月度双站共同键对齐的既有输运预测bank；依计划不另开发运动模型，也不将早期Sili-only旧bank并入比较。train-only尺度和验证SSIM已按预注册口径完成，但结构指标未显示优于持续性。M03六案例/Cloud-FNO协议表已完成；此外，Yi/Sun匹配原则表和固定规则单例曲线已补充完成，见`../COT_REFERENCE_COMPARISON_20261014.md`与`../cot_forecast_case_20261014_v1/`。案例只作形态诊断，不代表总体。T17只批准进入B3并明确GHI预报头保持门控；当前B3证据不满足稳定优势主张，因此不自动启动B4训练。不得通过筛选有利案例改写总体验证结论。

## 产物与来源

- `cot_forecast_metrics.json`：点估计、逐 lead 宏汇总、厚云计数和配对日期bootstrap。
- `cot_forecast_by_lead.csv`、`cot_thick_cloud_by_lead.csv`：站点×lead×尺度的复算明细。
- 输入 bank：`../forecast_cot_val_bank_20261014_v1/forecast_cot_val_bank.npz`，SHA-256 `fd7a431148ebf301140614bdafa8f05e09e99bc5c83a34bb0c456d6cf88b27ac`。
- profile收据：`../forecast_cot_val_bank_20261014_v1/profile.json`，SHA-256 `412a26c2a2779b7ad8bcf746baf3d92f2fbcb8441312f1a7a3013ee25f6a6be9`。
- 推理：PBS `213858.tc6000`，node22单A800，exit_status=0；271窗端到端35.47秒，batch=8，6 workers，GPU峰值预留8,496 MiB；测试载荷读取数0。
- GPU `dmon` 对整段作业的1秒采样均值SM利用率约9%、峰值100%；该作业含启动与1窗写盘自检，测量时段很短且呈突发，不代表持续吞吐利用率。
