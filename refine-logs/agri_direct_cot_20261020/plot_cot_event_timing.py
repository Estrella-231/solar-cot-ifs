"""Render actual saved station trajectories without interpolation or GHI claims."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/cot_event_timing_diagnostic_v1_20261021'


def main():
    with (OUT / 'selected_actual_timelines.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    ids = list(dict.fromkeys(r['window_id'] for r in rows))
    styles = [('CPP', 'CPP reference', '#000000', 'o', '-'),
              ('observed_R', 'Observed future AGRI → R (offline)', '#009e73', 's', '--'),
              ('cot_p_xc', 'Historical AGRI + COT → future COT', '#d55e00', 'D', '-'),
              ('cot_p_c', 'Historical COT → future COT', '#cc79a7', '^', '-'),
              ('cot_simvp_r', 'SimVP → R', '#0072b2', 'v', '-'),
              ('cot_last', 'Last historical COT', '#888888', '', ':')]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(ids), figsize=(7 * len(ids), 5.5), sharey=True, squeeze=False)
    ymax = max(40, 1.12 * max(float(r[key]) for r in rows for key, *_ in styles))
    for ax, wid in zip(axes[0], ids):
        rr = sorted([r for r in rows if r['window_id'] == wid], key=lambda r: int(r['lead_minutes']))
        assert len(rr) == 16
        tt = [datetime.fromisoformat(r['target_time_bjt']) for r in rr]
        for key, label, color, marker, linestyle in styles:
            ax.plot(tt, [float(r[key]) for r in rr], label=label, color=color, marker=marker,
                    markersize=3.5, linestyle=linestyle, linewidth=1.5)
        ax.axhline(30, color='#777777', linewidth=.8, linestyle='--')
        ax.set_ylim(0, ymax); ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        station = 'Sili' if rr[0]['station'] == 'sili' else 'Zhujia'
        ax.set_title(f'{station} · targets {tt[0]:%Y-%m-%d} BJT\nInit {rr[0]["init_time_bjt"]} BJT')
        ax.set_xlabel('Target time BJT (+15 to +240 min)')
        ax.tick_params(axis='x', rotation=30)
    axes[0, 0].set_ylabel('Station-center COT (dimensionless)')
    fig.suptitle('Actual trajectories show both thick-cloud underprediction and non-thick COT overprediction', fontsize=14, y=.98)
    h, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, labels, loc='upper center', bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.72, bottom=.24, left=.065, right=.98, wspace=.10)
    fig.text(.065, .055, 'Post-hoc largest P-XC peak-error daily window per station/reference; duplicate selections merged.\n'
             'Only 6 complete-reference dates per station qualify. These examples do not represent the full validation period.\n'
             'CPP and observed-image retrieval are separate offline references. No future observations enter deployable inference; no GHI is scored.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('actual_daily_cot_trajectories.' + ext), dpi=180)
    plt.close(fig)
    audit = {'state': 'RENDERED_PENDING_VISUAL_INSPECTION', 'native_temporal_resolution_min': 15,
             'interpolation': 'none', 'selected_windows': ids, 'y_range': [0, ymax],
             'source_csv_sha256': hashlib.sha256((OUT/'selected_actual_timelines.csv').read_bytes()).hexdigest(),
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'files': {f'actual_daily_cot_trajectories.{ext}': hashlib.sha256((OUT/f'actual_daily_cot_trajectories.{ext}').read_bytes()).hexdigest() for ext in ('png', 'pdf')}}
    (OUT / 'figure_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(f'Rendered {len(ids)} unique actual trajectory windows.')


if __name__ == '__main__':
    main()
