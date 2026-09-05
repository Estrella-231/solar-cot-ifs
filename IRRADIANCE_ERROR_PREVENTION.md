# Irradiance Forecasting Error Prevention

## 0. Short Chinese Memory

- v3 旧 scan-time 结果不能用于最终模型或汇报：主要问题是 `row 错`、`BJT/UTC 错`、`精确瞬时标签噪声大`；正式 `direct_ghi_v3_cached` 不属于这批禁用资产。
- 当前论文全部使用湖南数据完成训练、验证、选模和独立时间测试，不做湖北到湖南的数据或 checkpoint 迁移；`0–4 h` 短临是核心，IFS `D+1～D+3` 属于论文之外的独立中期工程。
- 已有 16 通道 SimVP（13 AGRI + `cosSOZ` + `cosRAA` + `day_mask`）是正式基线资产，不再是必须冻结复用的唯一论文骨干；新模型必须独立命名和保存，不能与旧 checkpoint 混用。
- 8→16 的真实时效是相对历史末帧 `+15...+240 min`；旧结果的 `lead=0...225` 是从第一张未来帧计时，转换规则为 `canonical_lead=legacy_lead+15`。
- 新实验必须先确认：时间体系、真实 HDF 行列、湖南自己的 grid/crop/站点坐标、所选 checkpoint 的准确通道合同、主标签定义和评估口径。

This document records the major mistakes found during v1/v2/v3 irradiance forecasting experiments and the checks required to avoid repeating them.

## 1. Time System Error: BJT vs UTC

### What Went Wrong

Existing matched `.npz` filenames are in BJT, but v3 scan-time matching initially treated them as UTC.

Example:

- `20250801_0400.npz` has `day_mask=0`, solar altitude about `-19 deg`, and GHI labels near zero.
- Therefore `20250801_0400` is 04:00 BJT, not 04:00 UTC.

The wrong v3 build used `20250801_0400` as 04:00 UTC, so it paired:

- main label: 04:00 BJT, night, GHI near 0;
- scan label: 04:02 UTC = 12:02 BJT, midday, high GHI.

This created contradictory multitask supervision.

### Correct Rule

For existing matched files:

```text
matched filename time = BJT
FY4B HDF lookup time = BJT - 8h = UTC
NOMObsTime = UTC
Excel lookup time = scan UTC + 8h = BJT
```

### Required Check

After building scan-time data, inspect at least:

- `*_0000.npz`: scan BJT should be about 00:02-00:03 and scan GHI should be night/near zero.
- `*_0400.npz`: scan BJT should be about 04:02-04:03 and scan GHI should be night/near zero.
- `*_1200.npz`: scan BJT should be about 12:02-12:03 and scan GHI may be high.

Never accept data where 04:00 BJT main labels are zero but scan labels are midday high values.

## 2. Scan Row Error: NOMObsTime Needs Real HDF Row Indices

### What Went Wrong

The Hubei crop constants are geolocation crop coordinates, not direct FY4B science-array indices.

The wrong v3 scan-time build used:

```text
R1 + CROP_ROW = 1112 + 103 = 1215
```

to read `NOMObsTime`.

But the model patch is sampled from FY4B arrays using geolocation lookup tables. The actual row indices are stored in:

```text
grid_static.npz["row_index"]
grid_static.npz["col_index"]
```

For the three-station patch, actual HDF row indices are approximately:

```text
row_index min / median / max = 550 / 556.5 / 563
col_index min / median / max = 1529 / 1536 / 1543
```

### Correct Rule

When reading FY4B `NOMObsTime` for the 16x16 patch:

```text
rows = grid_static["row_index"][103:119, 118:134]
read NOMObsTime[np.unique(rows), :]
take median scan time
```

Do not use `R1 + CROP_ROW` or `C1 + CROP_COL` directly for `NOMObsTime`.

### Required Check

For Hubei patch scan time:

```text
correct scan time should be around T + 2-3 minutes
wrong raw-row scan time was around T + 5-6 minutes
```

## 3. Region and Patch Consistency

### Current Main Region: Hunan

湖南是当前论文的主训练和评价区域。任何湖南数据构建、训练、推理和绘图必须使用湖南自己的 `grid_static`、裁剪范围、站点坐标以及真实 HDF `row_index/col_index`。不得将下列湖北 crop 或 patch 常量用于湖南。

### Historical Hubei Region

Hubei 256 crop:

```text
R1=1112, R2=1368
C1=2077, C2=2333
```

Three-station 16x16 patch inside the 256 crop:

```text
CROP_ROW=103
CROP_COL=118
CROP_SIZE=16
```

The patch geolocation is approximately:

```text
lon: 111.8-112.4E
lat: 31.8-32.4N
```

Nearest patch pixel distance to Niushou, Wangzhai, and Mizhuang is about 1.8-2.0 km.

### Required Check

When reproducing the historical Hubei experiments, training, actual-history evaluation, 0526 inference, and plotting must all use the same Hubei patch:

```text
full[:, 103:119, 118:134]
```

Never silently switch to another crop.

## 4. Model Channel Contract

### Existing formal SimVP baseline

The existing formal SimVP checkpoint uses:

```text
input shape: [8,16,256,256]
13 AGRI: C01-C06, C09-C15
+ cosSOZ + cosRAA + day_mask
```

The checkpoint does not contain C07/C08. This is a legacy baseline contract, not evidence that removing C07/C08 is universally better and not a restriction on separately named Hunan models.

### New Hunan model experiments

No single new channel contract is mandatory. `20 AGRI + solar_altitude + azimuth_abs_diff`, alternative AGRI subsets, newer temporal backbones and diffusion models may be tested on Hunan only when each has a separately named dataset/config/checkpoint/result directory and a clear ablation. They may not be mixed with the existing 16-channel checkpoint.

### Rules

- Never change channel count or order between the training contract and inference contract of the same checkpoint.
- Do not use SimVP outputs as a replacement for real AGRI input unless the experiment is explicitly named as a forecast-AGRI experiment.
- Record the exact channel list in every run manifest.

## 5. Label Meaning and Data Leakage

### Correct Meaning

The model input uses historical AGRI patch frames only.

The labels are future observations:

```text
main target = future 15min mean kt
pred_ghi = pred_kt * clear_sky_ghi
```

Canonical short-term labels use interval-ending target coordinates. For
`target_time = init_time + k*15min`, the label window is
`(init_time+(k-1)*15min, init_time+k*15min]`. The authoritative Hunan
workbooks and `HuNan_Station16_Combined/schema.json` timestamp the 5-minute
records by interval end, so the label averages `target-10min`, `target-5min`,
and `target`. A cache keyed by interval start and containing `+5/+10/+15min`
records is usable only after its coordinate is explicitly remapped to a
canonical target 15 minutes later.

Observed irradiance from Excel/matched files is only used for labels and evaluation. It must not be used as a model input.

### Future Labels Are Not Leakage

Using future observed irradiance as a training label is normal supervised learning. It becomes leakage only if future observation values are included in model inputs or feature construction available at inference time.

## 6. Exact 2-3 Minute Scan Window Can Interfere

### What We Learned

The exact FY4B scan-time observation is not simply a better version of the 15min mean label. It is a more instantaneous value and can differ strongly under cloud variability.

Measured diagnostic:

```text
v2 0-5min auxiliary vs 15min mean:
MAE about 19 W m^-2
RMSE about 34 W m^-2
correlation about 0.993

true 2-3min scan auxiliary vs 15min mean:
MAE about 88 W m^-2
RMSE about 149 W m^-2
p90 absolute difference about 305 W m^-2
correlation about 0.863
```

### Rule

Do not train exact scan GHI as a strong auxiliary regression target with `scan_aux_weight=0.3`.

Safer options:

- reduce scan weight to `0.03` or `0.05`;
- gate scan loss when `|scan_ghi - mean_ghi|` is too large;
- use exact scan as a quality/cloud-variability feature, not a strong shared-encoder loss;
- keep v2 0-5min mean auxiliary for 4h because it aligns better with the 15min mean target.

## 7. Plotting: Initialization Time vs Target Time

### Correct Meaning

Canonical initialization time is the timestamp of the last observed input frame (`history_end_time`).

Target time is when the predicted value is valid.

For the existing 8-to-16 SimVP baseline, the 16 targets are:

```text
history_end_time + 15, +30, ..., +240 min
```

Legacy forecast files used `issue_time = first_target_time = history_end_time + 15 min` and saved `lead_minutes = 0,15,...,225`. These files are not missing the +240 min target. Convert them with:

```text
canonical_init = legacy_issue_time - 15 min
canonical_lead = legacy_lead + 15 min
target_time = canonical_init + canonical_lead
```

For 4h:

```text
target_time = init_time + 240min
```

### Plot Rule

If plotting by target time:

- 4h curve starts 4h after the first initialization time.

If plotting by initialization time:

- the y-value is the forecast for `init_time + lead`;
- the legend and axis label must explicitly say "forecast initialization time".

Do not mix these two meanings in one figure without clear labels.

## 8. Excel Reading Risks

Excel weather files may contain multiple sheets. Do not read only the first sheet.

For the current Hunan main experiments, station mapping must cover:

```text
Sili    -> 四里
Zhujia  -> 竺家
```

When reproducing Hubei experiments, station mapping must cover:

```text
Niushou  -> 牛首供电所
Wangzhai -> 王寨供电所
Mizhuang -> 米庄供电所
```

For scan-time matching, use nearest observation only after confirming the time system:

```text
scan UTC -> scan BJT -> nearest Excel timestamp
```

If nearest delta is too large, mask that station's scan label.

## 9. Forecast AGRI vs Real Historical AGRI

`forecast` data is future/forecast/extrapolated AGRI data. It is not the same as real historical AGRI.

Evaluation reports must state which input was used:

- real historical AGRI replay;
- forecast AGRI replay;
- single-day 0526 forecast-product inference.

Do not compare metrics across these modes as if they are the same experiment.

## 10. Model Selection Status

The current paper route uses Hunan as the main training/evaluation region. The existing 16-channel SimVP is a reproducible baseline, not the mandatory final backbone. New Hunan short-term models may be trained, but every model and channel change must be isolated, documented and compared on the same frozen temporal split and labels.

The former Hubei-to-Hunan zero-shot route is no longer the main paper question. Hubei results may be retained only as historical reproduction or optional external validation.

The legacy area-to-point v2 model is excluded from the current paper method and experiment matrix by explicit user decision on 2026-08-14.

Do not use the following invalid scan-time v3 variants as final/report models:

- old v3 with wrong `NOMObsTime` row;
- corrected-row v3 with BJT/UTC mismatch;
- v3 exact scan-time strong auxiliary models with high 4h bias.

This prohibition does not apply to the formally verified station Direct-GHI assets named `direct_ghi_v3_cached_<Station>` in the 2026-08-07 handoff document.

## 11. Minimum Checklist Before Any New Run

Before training or evaluating:

```text
[ ] Did I read AGENTS.md and this document?
[ ] Are matched filenames treated as BJT?
[ ] Is HDF lookup using BJT - 8h?
[ ] Is NOMObsTime using grid_static row_index, not R1+CROP_ROW?
[ ] For Hunan, are Hunan grid_static/crop/station coordinates used, with no Hubei-index reuse?
[ ] If reproducing Hubei, is the historical crop exactly full[:, 103:119, 118:134]?
[ ] Does the input exactly match the selected checkpoint contract?
[ ] If using the existing SimVP baseline, is it exactly 13 AGRI + cosSOZ + cosRAA + day_mask?
[ ] If training any new Hunan model/channel contract, is it isolated as a separately named experiment?
[ ] Is observed irradiance only label/evaluation, never input?
[ ] Is 4h plotted/evaluated at target_time = init_time + 240min?
[ ] Were legacy lead 0-225 fields converted to canonical lead 15-240 instead of deleting legacy lead 0?
[ ] Is the evaluation mode real AGRI or forecast AGRI clearly stated?
[ ] Is legacy area-to-point v2 excluded from the current paper matrix?
```
