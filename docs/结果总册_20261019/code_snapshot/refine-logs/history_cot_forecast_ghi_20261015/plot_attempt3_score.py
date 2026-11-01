from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
j = json.loads((root / 'independent_score_attempt3.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
fig, axs = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
colors = {'transport': '#0072B2', 'hybrid': '#D55E00'}
names = {'transport': 'History-transport COT', 'hybrid': 'History + hybrid COT'}
leads = list(range(15, 241, 15))
for ax, arm in zip(axs, ('transport', 'hybrid')):
    d = j['results'][arm]
    xs, lo, hi = [], [], []
    for lead in leads:
        r = d[str(lead)]
        xs.append(r['delta_rmse'])
        lo.append(xs[-1] - r['date_bootstrap_95ci_delta'][0])
        hi.append(r['date_bootstrap_95ci_delta'][1] - xs[-1])
    y = list(range(len(leads)))
    ax.errorbar(xs, y, xerr=[lo, hi], fmt='o', color=colors[arm], ecolor=colors[arm], capsize=2, lw=1.4, ms=4)
    ax.axvline(0, color='#555', lw=1, ls='--')
    ax.set_title(names[arm])
    ax.set_xlabel('GHI RMSE difference (COT - no COT), W m$^{-2}$\nPositive = COT is worse')
    ax.grid(axis='x', alpha=.22)
    ax.set_yticks(y, [f'+{lead} min' for lead in leads], fontsize=7)
    ax.invert_yaxis()
    overall = d['0']
    ax.text(.02, .02,
        f"All leads: delta={overall['delta_rmse']:+.2f} W m$^{{-2}}$ ({-overall['rmse_reduction_pct']:+.2f}% RMSE increase)\n"
        f"Date-block 95% CI [{overall['date_bootstrap_95ci_delta'][0]:+.2f}, {overall['date_bootstrap_95ci_delta'][1]:+.2f}]",
        transform=ax.transAxes, va='bottom', ha='left', fontsize=9,
        bbox=dict(facecolor='white', alpha=.88, edgecolor='#cccccc'))
fig.suptitle('Paired GHI forecast with and without forecast COT: held-out initialization dates')
fig.text(.5, .005,
    'Exploratory Sili pilot under the old split; model selection and scoring use disjoint initialization-date groups. '
    'Error bars: paired target-day bootstrap 95% CI; leads +15 to +240 min.',
    ha='center', fontsize=9)
fig.tight_layout(rect=[0, .045, 1, .94])
fig.savefig(root / 'forecast_ghi_attempt3_dateheldout.png', dpi=180, bbox_inches='tight')
fig.savefig(root / 'forecast_ghi_attempt3_dateheldout.pdf', bbox_inches='tight')

lines = [
    '# attempt3: dense validation and held-out initialization-date scoring', '',
    f"Status: {j['state']}. Scope: old-contract Sili validation pilot; {j['scoring_rows']} scored rows over {len(j['scoring_target_dates'])} target dates. Model-selection and scoring initialization dates are disjoint. Metrics were independently recomputed in FP64 from saved per-row predictions. No new test payload was read.", '',
    '## Overall result (all 16 leads pooled)', '',
    '| Arm | No-COT RMSE | COT RMSE | Delta (COT - A), W/m^2 | RMSE change (positive = worse) | Date-block 95% CI for delta |',
    '|---|---:|---:|---:|---:|---:|'
]
for arm in ('transport', 'hybrid'):
    r = j['results'][arm]['0']
    lines.append(f"| {names[arm]} | {r['A_rmse']:.2f} | {r['COT_rmse']:.2f} | {r['delta_rmse']:+.2f} | {-r['rmse_reduction_pct']:+.2f}% | [{r['date_bootstrap_95ci_delta'][0]:+.2f}, {r['date_bootstrap_95ci_delta'][1]:+.2f}] |")
lines += [
    '', '## Selected leads', '',
    '| Lead | Transport COT delta | Transport RMSE change | Hybrid COT delta | Hybrid RMSE change |',
    '|---|---:|---:|---:|---:|'
]
for lead in (60, 120, 180, 240):
    t, h = j['results']['transport'][str(lead)], j['results']['hybrid'][str(lead)]
    lines.append(f"| +{lead//60} h | {t['delta_rmse']:+.2f} | {-t['rmse_reduction_pct']:+.2f}% | {h['delta_rmse']:+.2f} | {-h['rmse_reduction_pct']:+.2f}% |")
lines += [
    '', '## Protocol and interpretation', '',
    '- Nine runs: three seeds and three arms; each run used 1200 updates, with data order, initialization, optimizer, and all architecture inputs paired. Validation was checked every 10 updates.',
    '- Sixty initialization dates were deterministically split by the frozen SHA256 rule into 12 checkpoint-selection dates (1383 rows) and 48 scoring dates (4472 rows). Selection used only the first group; scoring was performed once on the second.',
    '- Overall pooled RMSE increased 5.33% for transport COT and 5.06% for hybrid COT; both date-block confidence intervals for the RMSE difference are above zero. The +1 h point estimates are slightly favorable but uncertain; later leads are generally worse.',
    '- This improves protection against checkpoint-selection overfitting relative to the earlier sparse-checkpoint run, but remains an old-split, single-station exploratory pilot, not the paper-wide two-station result. The hybrid arm also uses frozen SimVP auxiliary forecasts and is not a pure COT-only head.',
    '- The result says these forecast-COT inputs did not improve GHI on this held-out subset. It does not show that COT itself lacks value in same-time irradiance retrieval. Forecast-COT evolution error and GHI-head generalization are candidate explanations, not isolated causes.',
    '', 'Recompute script: `independent_score_attempt3.py`; figure: `forecast_ghi_attempt3_dateheldout.png`.'
]
(root / 'attempt3_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('saved English figure labels and updated report')
