"""Render saved reference diagnostics; no model inference or training."""
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/teacher_forecast_error_decomposition_v1_20261021'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    assert summary['state'] == 'COMPLETE_TEACHER_FORECAST_ERROR_DECOMPOSITION'
    cases = read_csv(OUT / 'posthoc_case_selection.csv')
    maps = np.load(OUT / 'posthoc_case_maps.npz', allow_pickle=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    cot_cmap = plt.get_cmap('magma').copy()
    cot_cmap.set_bad('#bdbdbd')
    cot_cmap.set_over('#00c5d5')
    gray = plt.get_cmap('gray').copy()
    gray.set_bad('#bdbdbd')
    valid_c03 = maps['c03'][maps['mask'] & np.isfinite(maps['c03'])]
    # Whole-field shared native-value range, no adaptive per-panel stretch.
    c03max = max(1.0, float(valid_c03.max()))
    columns = [('c03', 'Observed future AGRI\narchived C03'),
               ('cpp', 'CPP COT\nretrieval reference'),
               ('observed_r', 'Observed future AGRI\n→ frozen R'),
               ('p_xc', 'Historical AGRI + COT\n→ future COT'),
               ('simvp_r', 'SimVP future AGRI\n→ frozen R'),
               ('last', 'Last historical COT\npersistence')]
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    saturation = []
    for i, case in enumerate(cases):
        target = datetime.fromisoformat(case['target_time_utc']) + timedelta(hours=8)
        kind = 'Forecast vs R error' if case['selection'] == 'forecast_vs_teacher' else 'R vs CPP error'
        label = ('Sili' if case['station'] == 'sili' else 'Zhujia')
        for j, (key, title) in enumerate(columns):
            ax = axes[i, j]
            mask = maps['mask'][i] & np.isfinite(maps[key][i])
            data = np.ma.array(maps[key][i], mask=~mask)
            im = ax.imshow(data, cmap=gray if j == 0 else cot_cmap,
                           vmin=0, vmax=c03max if j == 0 else 100,
                           interpolation='nearest', origin='upper')
            if j == 0:
                c03im = im
            else:
                cotim = im
            ax.plot(8, 8, marker='*', color='white', markeredgecolor='black', markersize=10)
            ax.add_patch(Rectangle((6.5, 6.5), 3, 3, fill=False, edgecolor='white',
                                   linewidth=1, linestyle='--'))
            ax.set_xticks([0, 8, 15]); ax.set_yticks([0, 8, 15])
            if i == 0:
                ax.set_title(title, fontsize=11, pad=10)
            if j == 0:
                ax.set_ylabel(f'{label} · {kind}\nTarget {target:%Y-%m-%d %H:%M} BJT\nLead +{case["lead_minutes"]} min', fontsize=10)
            if i == 3:
                ax.set_xlabel('column (native 16 × 16)')
            saturation.append({'case': i, 'field': key, 'valid_pixels': int(mask.sum()),
                               'above_display_max': int(((maps[key][i] > (c03max if j == 0 else 100)) & mask).sum())})
    fig.suptitle('Actual cloud maps separate retrieval errors from future-forecast errors', fontsize=18, y=.985)
    fig.subplots_adjust(left=.115, right=.94, top=.91, bottom=.16, wspace=.22, hspace=.35)
    cax = fig.add_axes([.95, .20, .011, .60])
    fig.colorbar(cotim, cax=cax, extend='max', label='COT (dimensionless); cyan: >100')
    gax = fig.add_axes([.12, .065, .11, .012])
    fig.colorbar(c03im, cax=gax, orientation='horizontal', label='Archived C03 native value')
    fig.text(.32, .04, 'Post-hoc largest-error examples, two per station; not representative frequencies.\n'
             'All maps use the same valid pixels per case; gray = excluded/missing. Star = station (8, 8); dashed = 3 × 3.\n'
             'Observed future images appear only in this offline reference audit, never in deployable inference.', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('actual_cloud_error_cases.' + ext))
    plt.close(fig)

    rows = read_csv(OUT / 'metrics_by_support_scale_station_lead.csv')
    rows = [r for r in rows if r['support'] == 'corrected_common' and r['scale'] == 'station3'
            and r['reference'] == 'CPP' and int(r['lead_minutes']) > 0]
    arms = [('observed_agri_r_u', 'Observed future AGRI → R (offline)', '#009e73', 'o', '--'),
            ('cot_simvp_r', 'SimVP → R', '#0072b2', 's', '-'),
            ('cot_p_c', 'Historical COT → future COT', '#cc79a7', '^', '-'),
            ('cot_p_xc', 'Historical AGRI + COT → future COT', '#d55e00', 'D', '-')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey='row')
    for j, station in enumerate(('sili', 'zhujia')):
        for arm, label, color, marker, linestyle in arms:
            rr = sorted([r for r in rows if r['station'] == station and r['arm'] == arm], key=lambda r: int(r['lead_minutes']))
            assert len(rr) == 16
            x = [int(r['lead_minutes']) for r in rr]
            for i, metric in enumerate(('rmse', 'thick_csi')):
                axes[i, j].plot(x, [float(r[metric]) for r in rr], color=color,
                                marker=marker, markersize=3, linestyle=linestyle, label=label)
        axes[0, j].set_title('Sili' if station == 'sili' else 'Zhujia')
        for i in range(2):
            axes[i, j].grid(alpha=.2)
            axes[i, j].set_xlim(15, 240)
            axes[i, j].set_xticks([15, 60, 120, 180, 240])
        axes[1, j].set_xlabel('Forecast lead from last input frame (min)')
    axes[0, 0].set_ylabel('COT RMSE vs CPP (dimensionless)')
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel('Thick-cloud CSI (COT ≥30)')
    axes[1, 0].set_ylim(0, 1)
    fig.suptitle('Future prediction loses COT skill retained by observed-image retrieval', fontsize=15, y=.99)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=2, frameon=False)
    fig.subplots_adjust(top=.81, bottom=.14, hspace=.22, wspace=.12)
    fig.text(.08, .035, 'Seen-development V only; shared 3 × 3 pixel support. CPP is a retrieval reference, not independent truth.\n'
             '20,500 matched forecast items; 1,362 station-target keys; 23/26 fixed date blocks. Curves are descriptive, not GHI gains.\n'
             'Offline retrieval is evaluated on each lead’s matched samples, so its curve changes with sample composition.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('leadwise_retrieval_forecast_comparison.' + ext))
    plt.close(fig)
    (OUT / 'figure_display_audit.json').write_text(json.dumps({
        'state': 'RENDERED_PENDING_VISUAL_INSPECTION', 'cot_display_range': [0, 100],
        'cot_above_range_color': 'cyan', 'c03_shared_display_range': [0, c03max],
        'interpolation': 'nearest', 'saturation': saturation,
        'source_summary_sha256': sha(OUT / 'summary.json'), 'script_sha256': sha(Path(__file__)),
        'figures': {name: sha(OUT / name) for name in ('actual_cloud_error_cases.png', 'actual_cloud_error_cases.pdf',
                     'leadwise_retrieval_forecast_comparison.png', 'leadwise_retrieval_forecast_comparison.pdf')}
    }, indent=2) + '\n', encoding='utf-8')
    print('Rendered actual-map and lead-wise figures from saved audit outputs.')


if __name__ == '__main__':
    main()
