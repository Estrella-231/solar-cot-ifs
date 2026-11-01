# COT外推到达时间与峰值案例诊断

案例：hunan_202508210145，目标日 2025-08-21 BJT。按预先规则选择该日：earliest BJT target date among windows with CPP support at each of 16 leads for both stations; lexical sequence_id tie-break。

在两站固定3×3邻域、每个lead的CPP共同有效像元上，计算物理COT空间均值。到达定义为首个均值达到固定厚云阈值COT≥30的15分钟lead，不做时效插值。

| 站点 | 序列 | CPP首达时效 | 本方法首达时效 | 本方法峰值及lead | 相对CPP峰值差 | 支持像元/lead |
|---|---|---:|---:|---:|---:|---:|
| Sili | SimVP→R | +135 min | 240 min内未达到 | 29.61 @ +15 min | -24.11 | 9 |
| Sili | 持续性→R | +135 min | 240 min内未达到 | 28.34 @ +15 min | -25.38 | 9 |
| Sili | 真实未来AGRI→R（离线参考） | +135 min | +15 min | 32.37 @ +15 min | -21.34 | 9 |
| Zhujia | SimVP→R | +30 min | 240 min内未达到 | 0.47 @ +15 min | -43.55 | 9 |
| Zhujia | 持续性→R | +30 min | 240 min内未达到 | 1.84 @ +15 min | -42.19 | 9 |
| Zhujia | 真实未来AGRI→R（离线参考） | +30 min | 240 min内未达到 | 6.11 @ +240 min | -37.92 | 9 |

## 结果解释

四里CPP参考在+135 min首次达到厚云阈值，并在+240 min达到3×3均值峰值53.72；SimVP→R和持续性在预报窗内均未达到阈值，峰值分别只有29.61和28.34。竺家CPP参考在+30 min已达到阈值、+195 min峰值44.03；SimVP→R与持续性在整段时效都没有达到阈值，3×3均值峰值分别为0.47和1.84。该固定案例显示预测链没有复现参考厚云信号及其到达过程。

离线真实未来AGRI→R在四里+15 min越过阈值但随后没有追上CPP参考峰值；竺家整段未到阈值。它只能帮助区分外推输入误差与反演链差异，不能作为可部署预测成绩。该图/表是单一、机械选出的验证案例，不能估计总体到达误差；CPP也不是独立物理真值。

来源为既有`cot_forecast_case_values.csv`与`case_selection_and_metrics.json`；脚本逐站逐lead校验方法和CPP有效像元数相同。完整机器结果见`arrival_diagnostic.json`。test载荷读取为0。
