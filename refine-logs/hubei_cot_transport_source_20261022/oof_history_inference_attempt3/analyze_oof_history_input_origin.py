from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

ROOT = Path(__file__).parent
EVID = ROOT / 'evidence' / 'remote_215541' / 'oof_train' / 'evidence' / 'independent_train_only_v3'
SUMMARY = json.loads((EVID / 'summary.json').read_text(encoding='utf-8'))
frames = pd.read_csv(EVID / 'per_frame.csv', dtype={'time_bjt': str, 'date_bjt': str})
assert len(frames) == 20206 and frames['time_bjt'].is_unique
assert set(frames['fold'].dropna().astype(int)) == {0, 1, 2, 3, 4}
assert int(frames['common_inrange_pixels'].sum()) == SUMMARY['common_daytime_inrange_pixels']

# Daily means and map disagreement use exactly the same common support:
# daytime pixels where both R outputs are finite and in [0,100].
rows = []
for date, g in frames.groupby('date_bjt', sort=True):
    common = g[g['common_inrange_pixels'] > 0]
    if len(common):
        w = common['common_inrange_pixels'].to_numpy(dtype=np.float64)
        fm = common['full_r_common_inrange_mean'].to_numpy(dtype=np.float64)
        om = common['oof_r_common_inrange_mean'].to_numpy(dtype=np.float64)
        full_mean = float(np.average(fm, weights=w))
        oof_mean = float(np.average(om, weights=w))
        mae = float(np.average(common['common_inrange_MAE'].to_numpy(dtype=np.float64), weights=w))
        rmse = float(np.sqrt(np.average(common['common_inrange_RMSE'].to_numpy(dtype=np.float64) ** 2, weights=w)))
        ncommon = int(w.sum())
    else:
        full_mean = oof_mean = mae = rmse = np.nan
        ncommon = 0
    rows.append({'date_bjt': date, 'frames': int(len(g)), 'day_pixels': int(g['day_pixels'].sum()),
                 'common_inrange_pixels': ncommon, 'full_r_common_inrange_mean': full_mean,
                 'oof_r_common_inrange_mean': oof_mean, 'oof_minus_full_mean': oof_mean-full_mean,
                 'common_map_MAE': mae, 'common_map_RMSE': rmse})
daily = pd.DataFrame(rows)
daily['date'] = pd.to_datetime(daily['date_bjt'], format='%Y%m%d')
daily = daily.sort_values('date')
assert len(daily) == 220 and int(daily['frames'].sum()) == 20206
assert np.isfinite(daily[['full_r_common_inrange_mean','oof_r_common_inrange_mean','common_map_MAE','common_map_RMSE']].to_numpy()).all()

fullfrac = SUMMARY['full_r_gt100_day_pixel_fraction']
ooffrac = SUMMARY['oof_r_gt100_day_pixel_fraction']
daypix = SUMMARY['daytime_pixels']
daily_stats = {
    'train_rows': 20206, 'train_dates': int(len(daily)), 'date_range_bjt': [str(daily['date_bjt'].min()), str(daily['date_bjt'].max())],
    'frame_fold_counts': {str(int(k)): int(v) for k,v in frames.groupby('fold').size().items()},
    'equal_date_full_r_mean_on_common_support': float(daily['full_r_common_inrange_mean'].mean()),
    'equal_date_oof_r_mean_on_common_support': float(daily['oof_r_common_inrange_mean'].mean()),
    'equal_date_oof_minus_full_mean_on_common_support': float(daily['oof_minus_full_mean'].mean()),
    'median_date_oof_minus_full_mean_on_common_support': float(daily['oof_minus_full_mean'].median()),
    'equal_date_common_map_MAE': float(daily['common_map_MAE'].mean()),
    'equal_date_common_map_RMSE': float(daily['common_map_RMSE'].mean()),
    'median_date_common_map_RMSE': float(daily['common_map_RMSE'].median()),
    'p90_date_common_map_RMSE': float(daily['common_map_RMSE'].quantile(.9)),
    'correlation_of_daily_means_on_common_support': float(np.corrcoef(daily['full_r_common_inrange_mean'],daily['oof_r_common_inrange_mean'])[0,1]),
    'common_daytime_inrange_pixels': SUMMARY['common_daytime_inrange_pixels'],
    'overall_pixel_weighted_map_MAE': SUMMARY['OOF_minus_full_R_common_inrange']['MAE'],
    'overall_pixel_weighted_map_RMSE': SUMMARY['OOF_minus_full_R_common_inrange']['RMSE'],
    'overall_median_abs_map_difference': SUMMARY['OOF_minus_full_R_common_inrange']['median_abs'],
    'overall_p95_abs_map_difference': SUMMARY['OOF_minus_full_R_common_inrange']['p95_abs'],
    'overall_pixel_correlation': SUMMARY['OOF_minus_full_R_common_inrange']['pearson_r'],
    'full_r_common_mean': SUMMARY['common_inrange_full_r_distribution']['mean'],
    'oof_r_common_mean': SUMMARY['common_inrange_oof_r_distribution']['mean'],
    'full_r_gt100_day_pixels': int(round(fullfrac*daypix)), 'full_r_gt100_day_fraction': fullfrac,
    'oof_r_gt100_day_pixels': int(round(ooffrac*daypix)), 'oof_r_gt100_day_fraction': ooffrac,
    'full_r_below0_day_fraction': SUMMARY['full_r_below0_day_pixel_fraction'],
    'oof_r_below0_day_fraction': SUMMARY['oof_r_below0_day_pixel_fraction'],
    'full_r_day_raw_max': SUMMARY['full_r_day_raw_distribution']['max'],
    'oof_r_day_raw_max': SUMMARY['oof_r_day_raw_distribution']['max'],
    'full_r_day_raw_p99': SUMMARY['full_r_day_raw_distribution']['p99'],
    'oof_r_day_raw_p99': SUMMARY['oof_r_day_raw_distribution']['p99'],
    'test_payloads_read': 0, 'buffer_payloads_read': 0, 'cpp_target_or_quality_payloads_read': 0, 'ghi_labels_read': 0,
    'interpretation': 'R-output/input-source disagreement only; not COT truth error, forecast skill, or GHI gain.'
}
(ROOT/'evidence'/'oof_history_input_origin_analysis.json').write_text(json.dumps(daily_stats,indent=2)+'\n',encoding='utf-8')
daily.drop(columns='date').to_csv(ROOT/'evidence'/'oof_history_input_origin_per_date.csv',index=False)

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'figure.titlesize':14})
fig, axes = plt.subplots(1,2,figsize=(12.2,4.8),layout='constrained')
ax=axes[0]
sc=ax.scatter(daily['full_r_common_inrange_mean'],daily['oof_r_common_inrange_mean'],c=np.arange(len(daily)),cmap='viridis',s=24,alpha=.85,edgecolors='none')
lo=float(min(daily['full_r_common_inrange_mean'].min(),daily['oof_r_common_inrange_mean'].min()))
hi=float(max(daily['full_r_common_inrange_mean'].max(),daily['oof_r_common_inrange_mean'].max()))
ax.plot([lo,hi],[lo,hi],color='#555555',linestyle='--',linewidth=1.2,label='1:1')
ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Daily mean COT from full-R in-sample bank',ylabel='Daily mean COT from date-held-out R')
ax.set_title('Daily means on shared in-range pixels')
ax.legend(loc='upper left',frameon=True)
cb=fig.colorbar(sc,ax=ax,pad=.02);cb.set_label('Train-date order')
ax=axes[1]
ax.scatter(daily['date'],daily['common_map_RMSE'],color='#0072B2',s=20,alpha=.8,edgecolors='none',label='Per-date map disagreement')
roll=daily.set_index('date')['common_map_RMSE'].rolling(15,min_periods=7,center=True).median()
ax.plot(roll.index,roll.values,color='#D55E00',linewidth=1.8,label='15-date rolling median')
ax.axhline(SUMMARY['OOF_minus_full_R_common_inrange']['RMSE'],color='#333333',linestyle='--',linewidth=1.1,label=f"All-pixel RMSE = {SUMMARY['OOF_minus_full_R_common_inrange']['RMSE']:.2f}")
ax.set(xlabel='Train date (BJT)',ylabel='COT-map disagreement RMSE',title='Common daytime, both outputs in [0,100]')
ax.xaxis.set_major_locator(MonthLocator(interval=2));ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.legend(loc='upper left',frameon=True)
fig.suptitle('Full-R in-sample bank vs date-held-out fold-R inputs')
fig.text(.5,-.015,'220 train dates; 2,545,664 common pixels. These are model-output differences, not error against COT truth.',ha='center',va='top',fontsize=9)
for ext in ('png','pdf'):
    fig.savefig(ROOT/'evidence'/f'oof_history_input_origin.{ext}',dpi=240 if ext=='png' else None,facecolor='white')
plt.close(fig)

s=SUMMARY
full=s['common_inrange_full_r_distribution']; oof=s['common_inrange_oof_r_distribution']; delta=s['OOF_minus_full_R_common_inrange']
report=f'''# Train-only historical COT input-origin diagnostic\n\n## Result\n\nAcross **20,206 training AGRI frames** on 220 BJT dates, the existing full-R in-sample bank and date-held-out fold-R inference retain a similar broad spatial pattern but differ in local amplitude. On the common daytime support where both outputs lie in [0,100], there are **2,545,664 pixel pairs**. The pixel-weighted map disagreement is **MAE {delta['MAE']:.3f} COT** and **RMSE {delta['RMSE']:.3f} COT**, with pixel correlation {delta['pearson_r']:.3f}. Median absolute difference is {delta['median_abs']:.3f}; the 95th percentile is {delta['p95_abs']:.3f}.\n\nOn that same common support, full-R versus date-held-out R output means are **{full['mean']:.3f} vs {oof['mean']:.3f} COT** (OOF minus full = {oof['mean']-full['mean']:.3f}, or {100*(oof['mean']/full['mean']-1):.1f}% of the full-R mean). Their p90 values are {full['p90']:.2f} vs {oof['p90']:.2f}; p99 values are {full['p99']:.2f} vs {oof['p99']:.2f}. The outputs are correlated, yet the held-out fold bank has a lower mean and lighter upper tail on this restricted support.\n\n## Important out-of-range behavior\n\nThe common-support scores exclude pixels outside [0,100] by design and do not clip them. On daytime pixels, the full-R bank has {daily_stats['full_r_gt100_day_pixels']:,} outputs above 100 ({100*fullfrac:.3f}%), with maximum **{SUMMARY['full_r_day_raw_distribution']['max']:.1f}**; the fold-OOF bank has {daily_stats['oof_r_gt100_day_pixels']:,} above 100 ({100*ooffrac:.3f}%), with maximum **{SUMMARY['oof_r_day_raw_distribution']['max']:.1f}**. Both have no below-zero daytime pixels. The sparse extreme tail materially inflates raw means and standard deviations, so the figure's daily means and discrepancy RMSE use only the shared [0,100] support. The out-of-range tail remains reported separately and is not treated as physically valid COT.\n\n## Interpretation\n\nThis establishes a plausible input-source mismatch: full-R train inputs were generated by a model that had seen those dates; the comparison uses an R checkpoint whose training excluded each date. The fold-OOF output is on average lower and less dispersed on common in-range pixels, while spatial patterns remain strongly correlated. A downstream GHI head could therefore see different COT feature distributions between fitting and held-out use. This experiment does **not** tell us which output is closer to physical COT, and it does **not** establish COT forecast skill or explain a GHI gain/loss. CPP/COT reference targets and GHI labels were not used.\n\n## Scope and audit correction\n\nAll five frozen R checkpoints inferred only their assigned training-date frames. The five fold shards contain {daily_stats['frame_fold_counts']}; their unique train keys sum to 20,206. No validation AGRI frames were inferred. Test, buffer, CPP/QC and GHI label payload reads are recorded as zero.\n\nThe first automated post-inference auditor performed a global finite-value scan and whole-file hash on the existing train/validation COT bank. That touched validation **model-output values** during a technical check, though it did not infer validation AGRI, read validation labels, use validation values in metrics, or select a model. For the accepted result, `audit_oof_history_bank_train_only_v3.py` re-ran the comparisons without numerically scanning or hashing validation predictions. It reads full-bank row metadata to map keys but accesses prediction values only for the 20,206 train keys; the reported bank-value hash covers those selected train rows. This deviation and correction are retained explicitly.\n\n## Compute and reproducibility\n\nPBS **215541.tc6000** completed on ZJNU node22 using A800 GPUs **0, 1, 3, 4**. Four workers processed folds concurrently; fold 4 reused GPU 0 after fold 0 finished. The receipt reports 56.5 seconds for inference/analysis (PBS wall time about one minute). This was inference only; no model was trained. Attempts 215539 and 215540 failed before valid results and are preserved.\n\nPrimary train-only summary: `evidence/remote_215541/oof_train/evidence/independent_train_only_v3/summary.json`. Per-date table: `evidence/oof_history_input_origin_per_date.csv`; numeric summary: `evidence/oof_history_input_origin_analysis.json`; reproducible analysis and figure source: `analyze_oof_history_input_origin.py`.\n\n## Next step\n\nDo not train or claim GHI benefit from this diagnostic alone. The full-region CPP source gate remains blocked, and this comparison does not identify which COT bank is more physically accurate. Any GHI follow-up requires a separately specified paired training-input comparison and an unexposed evaluation cohort.\n\nThe figure workflow follows general scientific-visualization principles described by Kassis et al. (2026), *Scientific Agent Skills*, arXiv:2609.00065.\n'''
(ROOT/'evidence'/'OOF_HISTORY_INPUT_ORIGIN_REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(daily_stats,indent=2));print('REPORT',ROOT/'evidence'/'OOF_HISTORY_INPUT_ORIGIN_REPORT.md')
