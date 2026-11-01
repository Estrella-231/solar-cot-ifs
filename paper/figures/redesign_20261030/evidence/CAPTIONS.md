# Forecast GHI figure: bilingual caption and evidence scope

## English caption

**Historical and forecast COT contributions to GHI forecasting in Hubei.** (a) GHI RMSE at 16 forecast leads for four paired input variants. “No COT” uses historical observed AGRI and SimVP-predicted future AGRI; “Historical COT” adds eight retrieved historical COT fields. The two correction heads share the historical-COT base and have equal trainable capacity, but only the forecast-COT head receives COT retrieved from predicted future AGRI. (b) The additional effect of forecast COT, relative to the correction head without future COT, at each lead. Shading indicates pointwise 95% paired target-date bootstrap intervals. (c) Historical-COT-minus-no-COT differences by station and across the three stations. (d) Forecast-COT-head-minus-control-head differences. Diamonds are means across three random seeds; open circles show individual-seed differences; whiskers are 95% paired target-date bootstrap intervals. Negative differences indicate improvement. Panels use different numerical scales. Metrics are first computed separately for each station and seed, then averaged equally. All variants use the same 15,275 station–initialization–target records across 22 Beijing-time target dates (22 August 2025–29 June 2026); overlapping forecast windows are resampled together by target date (5,000 draws). Labels are 15-minute mean GHI, and lead is relative to the last observed input frame. No future observed AGRI or CPP is an inference input. These are development-validation results, not an independent test.

## 中文图注

**历史及未来预测 COT 对湖北 GHI 预报的贡献。**（a）四组配对输入在 16 个时效上的 GHI RMSE。“无 COT”使用历史真实 AGRI 与 SimVP 外推的未来 AGRI；“历史 COT”额外加入 8 帧历史反演 COT 场。两个修正头共享历史 COT 基底，且可训练参数量相同，只有未来 COT 修正头读取从预测未来 AGRI 反演得到的 COT。（b）各时效上，未来 COT 修正头相对不读取未来 COT 的修正头所产生的增量；阴影表示按目标日期配对重采样得到的逐时效 95% 区间。（c）各站及三站平均的“历史 COT−无 COT”差值。（d）各站及三站平均的“未来 COT 修正头−同容量对照头”差值。菱形表示三个随机种子的平均差值，空心圆表示单种子差值，横线表示目标日期配对 bootstrap 的 95% 区间；负值为改善。不同面板的数值尺度不同。RMSE 先逐站、逐种子计算，再对站点和种子等权平均。四组使用相同的 15,275 条站点—初始化—目标时刻记录，覆盖 22 个北京时间目标日期（2025年8月22日至2026年6月29日）；重叠预报窗口按目标日期整组重采样，共 5,000 次。标签为 15 分钟平均 GHI，时效相对最后一张历史输入帧。推理输入不包含真实未来 AGRI 或 CPP。本图属于开发验证结果，不是独立测试。

## Reading the figure / 图中可支持的判断

- Historical COT: equal-station/equal-seed RMSE 105.595 → 103.670 W m⁻². Difference −1.925 W m⁻²; 95% interval [−4.734, +0.670]. The point estimate improves, but the interval includes no gain.
- Forecast COT compared with its capacity control: 103.783 → 103.808 W m⁻². Difference +0.026 W m⁻²; 95% interval [−0.003, +0.059]. No additional forecast-COT benefit is demonstrated.
- The lower panels distinguish between seed variability and weather-date uncertainty. Three seeds do not create three independent sets of weather.
- Per-lead and per-station intervals are descriptive pointwise intervals, with no multiplicity correction. They must not be cherry-picked as new independent findings.
- This figure does not compare algorithms on new independent dates, does not measure COT retrieval accuracy, and does not prove that forecast COT is universally useless.

## Figure role

Recommended role: replace the *content* of the forecast-application figure when the surrounding Methods, Results and bilingual descriptions are updated to this exact four-arm experiment. Do not attach it to the old 77.150/78.155 W m⁻² paragraph or label it as same-time estimation.

The all-lead error panel and paired-difference panels deliberately use position, not zero-truncated bars. No smoothing, axis break, omitted lead or error-based filtering is used. The correction curves almost coincide because the saved outputs are almost equivalent; the difference panels resolve their actual numerical scale instead of artificially separating the curves.

## Source and reproduction

- Source experiment: `refine-logs/hubei_forecast_cot_increment_20261022`.
- Source predictions: 36 immutable `*_validation_predictions.npz` files in `evidence_attempt1/training_attempt1/pairs`.
- Primary scoring receipt: `evidence_attempt1/scored_attempt1/score.json`.
- Independent previous receipt: `evidence_attempt1/INDEPENDENT_AUDIT.json`.
- New figure script: `plot_forecast_ghi.py`; consumes the saved predictions directly.
- `forecast_ghi_manifest.json` binds every source NPZ and completion receipt by SHA-256.
- Exports: `forecast_ghi.pdf` (vector), `forecast_ghi.svg` (editable text), `forecast_ghi.png` (350 dpi). Canvas is exactly 177 × 132 mm; this is a provisional publication design, not a claim of specific-journal compliance.
- `station_seed_arm_metrics.csv`: 36 station × seed × arm metrics.
- `paired_comparisons.csv`: station and overall paired changes and intervals.
- `lead_metrics.csv`: 16 leads and forecast increment intervals.
- `sample_support.csv`: unique rows and dates by station and lead.

## 同期主收益图的来源缺口

本轮没有把湖北旧 Fig.4 的 4.49% 换成 10.54%。前者使用已确认 CPP 错时监督的旧反演器；后者来自 2026-10-20 的正确 UTC 四组同期实验，但后续 2026-10-21 又修复了均时差公式、太阳几何及反射率昼夜掩膜，不能仅依据 10月20日较早的几何检查即认定其为修复后最终来源。本轮有界检索未找到将该同期四组结果绑定到修复后同一 R、几何和共同样本的重新训练/评分记录。因此，本图只使用来源清晰的后续四臂预报结果；同期主收益和反演精度仍需单独核实，不以预报数据替代。

湖南固定权重、无效日剔除后的单帧 3.870% 是另一组已保存的重评分证据，但不是湖北同期来源修复，也不是重新选模后的独立评价。它不与本图合并计算区域总分。
