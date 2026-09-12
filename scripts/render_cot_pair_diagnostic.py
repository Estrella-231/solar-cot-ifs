"""Render the frozen, independently checked validation diagnostic; no inference."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'audits/cot_pair_diag_20260908_v2/report.json'
    audit_path = root / 'audits/cot_pair_results_audit_20260908.json'
    report = json.loads(source.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['state'] == 'PASS_INDEPENDENT_SAVED_COT_RESULTS_AUDIT'
    assert audit['max_metric_absolute_difference'] == 0
    rows = report['metrics']['by_station_lead']
    assert len(rows) == 32
    out = root / 'figures/cot_pair_diag_20260908_v2'
    out.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        item = {k: row[k] for k in ('station', 'lead_minutes', 'n_pair', 'n_unique_target', 'selected_targets', 'unavailable_targets')}
        for branch in ('real', 'forecast'):
            item.update({branch + '_' + k: v for k, v in row[branch].items()})
        flat.append(item)
    with (out / 'all_station_leads.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
    for col, station in enumerate(('sili', 'zhujia')):
        part = sorted((r for r in rows if r['station'] == station), key=lambda r: r['lead_minutes'])
        leads = [r['lead_minutes'] for r in part]
        assert leads == list(range(15, 241, 15))
        for branch, label, color, marker in [('real', 'R(real AGRI)', '#2673b8', 'o'), ('forecast', 'R(SimVP AGRI)', '#d36925', 's')]:
            axes[0, col].plot(leads, [r[branch]['cot_rmse'] for r in part], label=label, color=color, marker=marker, markersize=3.5, linewidth=1.6)
        axes[0, col].set_title(station.capitalize())
        axes[0, col].set_ylim(bottom=0)
        axes[0, col].grid(axis='y', alpha=.2)
        axes[1, col].bar(leads, [r['n_pair'] for r in part], width=9, color='#8699a8')
        axes[1, col].axhline(32, color='#55616a', linestyle='--', linewidth=1)
        axes[1, col].set_ylim(0, 35)
        axes[1, col].set_yticks([0, 16, 32])
        axes[1, col].set_xticks([15, 60, 120, 180, 240])
        axes[1, col].set_xlabel('Forecast lead from history end (min)')
    axes[0, 0].set_ylabel('COT RMSE against CPP reference')
    axes[1, 0].set_ylabel('Matched targets')
    axes[0, 0].legend(frameon=False, loc='upper left')
    fig.suptitle('Frozen-R input diagnostic on a fixed validation subset', fontsize=13)
    fig.supxlabel('Within each lead: identical targets and CPP mask. Target availability varies across leads.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('cot_reference_rmse_by_lead.' + ext), dpi=180)
    plt.close(fig)
    table = ['|站点|lead (min)|目标数 / 32|有效像元|真实输入 COT RMSE|外推输入 COT RMSE|', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"|{row['station']}|{row['lead_minutes']}|{row['n_pair']}|{row['real']['valid_pixels']}|{row['real']['cot_rmse']:.6f}|{row['forecast']['cot_rmse']:.6f}|")
    (out / 'all_station_leads.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    manifest = {'report_sha256': sha(source), 'independent_audit_sha256': sha(audit_path), 'render_source_sha256': sha(Path(__file__)), 'scope': 'validation-only descriptive CPP-reference errors, all 32 groups, no inference or new filtering', 'files': {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
