"""Publication figure from immutable, paired validation predictions (no training).

Rebuild: F:/program/anaconda/envs/pytorch/python.exe plot_forecast_ghi.py
All numerical marks are recomputed from saved NPZ predictions. Date bootstrap
draws reproduce the experiment's registered scoring implementation exactly.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
SOURCE = REPO / 'refine-logs/hubei_forecast_cot_increment_20261022/evidence_attempt1'
STATIONS = ['Niushou', 'Wangzhai', 'Mizhuang']
SEEDS = [42, 43, 44]
ARMS = ['G0', 'G1', 'G2', 'G3']
ARM_LABELS = ['No COT', 'Historical COT', 'History + matched control', 'History + forecast COT']
LEADS = np.arange(15, 241, 15)
COLORS = ['#52616B', '#007F71', '#4574A0', '#B85B00']
BOOT_SEED = 20261022
NBOOT = 5000

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    score_path = SOURCE / 'scored_attempt1/score.json'
    score = json.loads(score_path.read_text())
    dates = score['target_dates']
    assert len(dates) == len(set(dates))
    # date x station x seed x arm x lead
    sums = np.zeros((len(dates), 3, 3, 4, 16), dtype=np.float64)
    counts = np.zeros_like(sums)
    date_indices = {v: i for i, v in enumerate(dates)}
    source_files = [score_path, SOURCE / 'INDEPENDENT_AUDIT.json', SOURCE / 'training_attempt1/ALL_COMPLETE.json']
    metrics = []
    supports = []
    for si, station in enumerate(STATIONS):
        site_reference = None
        for ki, seed in enumerate(SEEDS):
            folder = SOURCE / 'training_attempt1/pairs' / f'{station}_seed{seed}'
            complete_path = folder / 'COMPLETE.json'
            complete = json.loads(complete_path.read_text())
            assert complete['base_unchanged']
            source_files.append(complete_path)
            for a, b in [('G0', 'G1'), ('G2', 'G3')]:
                for field in ['initialization_sha256', 'order_sha256', 'updates', 'parameters']:
                    assert complete['arms'][a][field] == complete['arms'][b][field]
            reference = None
            for ai, arm in enumerate(ARMS):
                path = folder / f'{arm}_validation_predictions.npz'
                source_files.append(path)
                with np.load(path, allow_pickle=False) as z:
                    y = z['truth_ghi_wm2'].astype(np.float64)
                    p = z['pred_ghi_wm2'].astype(np.float64)
                    clear = z['clear_sky_wm2'].copy()
                    keys = tuple(zip(z['station'].tolist(), z['init_time_bjt'].tolist(),
                                     z['target_time_bjt'].tolist(), z['lead_minutes'].tolist()))
                    assert len(keys) == len(set(keys))
                    assert set(z['station'].tolist()) == {station}
                    assert np.isfinite(y).all() and np.isfinite(p).all()
                    if reference is None:
                        reference = keys, y, clear
                    else:
                        assert keys == reference[0]
                        assert np.array_equal(y, reference[1]) and np.array_equal(clear, reference[2])
                    if site_reference is None:
                        site_reference = reference
                    else:
                        assert keys == site_reference[0] and np.array_equal(y, site_reference[1])
                    e = p-y
                    d = np.array([str(t)[:8] for t in z['target_time_bjt']])
                    lead = z['lead_minutes'].astype(int)
                    assert set(d) <= set(dates) and set(lead) == set(LEADS)
                    for date in np.unique(d):
                        for li, l in enumerate(LEADS):
                            m = (d == date) & (lead == l)
                            sums[date_indices[date], si, ki, ai, li] = np.sum(e[m]**2)
                            counts[date_indices[date], si, ki, ai, li] = m.sum()
                    metrics.append(dict(station=station, seed=seed, arm=arm,
                        input=ARM_LABELS[ai], n=len(y), n_dates=len(np.unique(d)),
                        rmse_wm2=float(np.sqrt(np.mean(e**2))), mae_wm2=float(np.mean(abs(e))),
                        bias_wm2=float(np.mean(e))))
                    if seed == SEEDS[0] and arm == ARMS[0]:
                        for l in LEADS:
                            supports.append(dict(station=station, lead_minutes=int(l),
                                unique_rows=int((lead == l).sum()),
                                target_dates=int(len(np.unique(d[lead == l])))))
    assert sum(r['n'] for r in metrics if r['seed'] == SEEDS[0] and r['arm'] == ARMS[0]) == score['unique_station_rows']
    total_sse = sums.sum((0, 4))
    total_n = counts.sum((0, 4))
    cell_rmse = np.sqrt(total_sse/total_n)  # site x seed x arm
    overall = cell_rmse.mean((0, 1))
    assert np.allclose(overall, [score['equal_station_equal_seed_rmse'][a] for a in ARMS], rtol=0, atol=1e-10)
    weights = np.random.default_rng(BOOT_SEED).multinomial(len(dates), np.ones(len(dates))/len(dates), size=NBOOT)
    # Preserve paired date draws across stations, seeds and arms. Repeated
    # forecasts for one target date remain in the same resampling cluster.
    daily_sse = sums.sum(-1)
    daily_n = counts.sum(-1)
    boot_sse = np.tensordot(weights, daily_sse, axes=(1, 0))
    boot_n = np.tensordot(weights, daily_n, axes=(1, 0))
    assert (boot_n > 0).all()
    boot_cells = np.sqrt(boot_sse/boot_n)
    boot_overall = boot_cells.mean((1, 2))
    for name, receipt in score['comparisons'].items():
        candidate, control = name.split('_minus_')
        delta = boot_overall[:, ARMS.index(candidate)] - boot_overall[:, ARMS.index(control)]
        assert np.allclose(np.percentile(delta, [2.5, 97.5]), receipt['ci95_date_block'], rtol=0, atol=1e-10)
    lead_cells = np.sqrt(sums.sum(0)/counts.sum(0))
    lead_mean = lead_cells.mean((0, 1))
    lead_bs_sse = np.tensordot(weights, sums, axes=(1, 0))
    lead_bs_n = np.tensordot(weights, counts, axes=(1, 0))
    assert (lead_bs_n > 0).all()
    lead_boot = np.sqrt(lead_bs_sse/lead_bs_n).mean((1, 2))

    comparison_rows = []
    for comparison, c, b in [('Historical COT - no COT', 1, 0), ('Forecast COT head - control head', 3, 2)]:
        for si, station in enumerate(STATIONS + ['Equal-station mean']):
            points = cell_rmse[si] if si < 3 else cell_rmse.mean(0)
            boots = boot_cells[:, si].mean(1) if si < 3 else boot_overall
            delta = points[:, c] - points[:, b]
            lo, hi = np.percentile(boots[:, c]-boots[:, b], [2.5, 97.5])
            comparison_rows.append(dict(comparison=comparison, station=station,
                reference_rmse_wm2=float(points[:, b].mean()), candidate_rmse_wm2=float(points[:, c].mean()),
                delta_rmse_wm2=float(delta.mean()), ci95_low=float(lo), ci95_high=float(hi),
                seed42_delta=float(delta[0]), seed43_delta=float(delta[1]), seed44_delta=float(delta[2])))
    lead_rows = []
    for li, l in enumerate(LEADS):
        lo, hi = np.percentile(lead_boot[:, 3, li]-lead_boot[:, 2, li], [2.5, 97.5])
        lead_rows.append(dict(lead_minutes=int(l), no_cot_rmse=float(lead_mean[0, li]),
            history_cot_rmse=float(lead_mean[1, li]), history_control_rmse=float(lead_mean[2, li]),
            history_forecast_cot_rmse=float(lead_mean[3, li]),
            forecast_increment_delta=float(lead_mean[3, li]-lead_mean[2, li]),
            forecast_increment_ci95_low=float(lo), forecast_increment_ci95_high=float(hi),
            unique_station_rows=sum(r['unique_rows'] for r in supports if r['lead_minutes'] == l)))
    dump_csv('station_seed_arm_metrics.csv', metrics)
    dump_csv('paired_comparisons.csv', comparison_rows)
    dump_csv('lead_metrics.csv', lead_rows)
    dump_csv('sample_support.csv', supports)

    style = {'font.family':'serif', 'font.serif':['Times New Roman', 'DejaVu Serif'],
        'font.size':8.5, 'axes.labelsize':8.5, 'xtick.labelsize':8, 'ytick.labelsize':8,
        'legend.fontsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':0.7, 'xtick.major.width':0.7, 'ytick.major.width':0.7,
        'mathtext.fontset':'stix', 'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'none', 'savefig.facecolor':'white'}
    with mpl.rc_context(style):
        fig, ax = plt.subplots(2, 2, figsize=(177/25.4, 132/25.4), layout='constrained',
                               gridspec_kw={'height_ratios':[1.12, 1]})
        fig.get_layout_engine().set(h_pad=0.065, w_pad=0.05, hspace=.10, wspace=.07)
        def panel(a, letter, title):
            a.text(0, 1.045, f'({letter})  {title}', transform=a.transAxes, va='bottom', weight='bold', fontsize=9)
            a.grid(axis='y', color='#E2E4E6', linewidth=.6, zorder=0)
        panel(ax[0,0], 'a', 'GHI forecast error')
        marks=['o','s','^','x']; styles=['-','-','--',':']
        for ai in range(4):
            ax[0,0].plot(LEADS, lead_mean[ai], color=COLORS[ai], marker=marks[ai],
                markersize=3, linewidth=1.25, linestyle=styles[ai], markevery=2, label=ARM_LABELS[ai])
        ax[0,0].set(xlabel='Lead from last observed frame (min)', ylabel=r'GHI RMSE (W m$^{-2}$)',
                    xlim=(10,245), xticks=[15,60,120,180,240])
        handles, labels=ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc='outside upper center', ncols=2,
                   handlelength=2.4, columnspacing=2.0, labelspacing=.4, borderaxespad=.4)

        panel(ax[0,1], 'b', 'Forecast increment by lead')
        d=np.array([r['forecast_increment_delta'] for r in lead_rows])
        lo=np.array([r['forecast_increment_ci95_low'] for r in lead_rows])
        hi=np.array([r['forecast_increment_ci95_high'] for r in lead_rows])
        ax[0,1].fill_between(LEADS, lo, hi, color=COLORS[3], alpha=.15, linewidth=0,
                            label='Pointwise 95% date interval')
        ax[0,1].plot(LEADS,d, color=COLORS[3], marker='s', markersize=3, linewidth=1.3)
        ax[0,1].axhline(0,color='#333333',lw=.8,ls='--')
        ax[0,1].set(xlabel='Lead from last observed frame (min)',
            ylabel=r'$\Delta$RMSE (W m$^{-2}$)', xlim=(10,245), xticks=[15,60,120,180,240])
        ax[0,1].text(.02,.94,'Forecast COT − matched control',transform=ax[0,1].transAxes,va='top',fontsize=8)

        for coli, (offset, color, title) in enumerate([(0,COLORS[1],'History increment by station'),(4,COLORS[3],'Forecast increment by station')]):
            a=ax[1,coli]; panel(a,'cd'[coli],title)
            a.grid(False); a.grid(axis='x',color='#E2E4E6',lw=.6)
            a.axvline(0,color='#333333',lw=.8,ls='--')
            for yi, row in enumerate(comparison_rows[offset:offset+4]):
                y=3-yi; pt=row['delta_rmse_wm2']; low=row['ci95_low']; high=row['ci95_high']
                a.hlines(y,low,high,color=color,lw=1.3)
                a.plot([low,high],[y,y],'|',color=color,markersize=6)
                for ki, seed in enumerate(SEEDS):
                    a.plot(row[f'seed{seed}_delta'],y+(ki-1)*.10,'o',ms=3,mfc='white',mec=color,mew=.7)
                a.plot(pt,y,'D',ms=4.5,color=color)
            a.set_yticks([3,2,1,0], STATIONS+['Three-station mean'])
            a.set(ylim=(-.45,3.55),xlabel=r'$\Delta$RMSE (W m$^{-2}$)')
        for suffix in ['pdf','svg','png']:
            fig.savefig(OUT/f'forecast_ghi.{suffix}',dpi=350)
        plt.close(fig)

    manifest = dict(figure='forecast_ghi', status='revision v1; development-validation evidence; no manuscript edited by this script',
        scientific_question='Does forecast COT add GHI skill beyond historical COT and matched correction capacity?',
        region='Hubei', stations=STATIONS, seeds=SEEDS, target_dates=dates,
        unique_station_rows=score['unique_station_rows'], lead_minutes=LEADS.tolist(),
        statistic='Per-station, per-seed RMSE, then arithmetic mean; no ensemble averaging of predictions',
        bootstrap=dict(replication_unit='BJT target date', paired=True, seed=BOOT_SEED,resamples=NBOOT,
                       interval='percentile 95%; by-lead intervals are pointwise, not simultaneous'),
        selection='GHI heads selected on designated train-internal dates; upstream and development validation reused; not independent test',
        agreement_max_abs_rmse=float(max(abs(overall[ai]-score['equal_station_equal_seed_rmse'][ar]) for ai,ar in enumerate(ARMS))),
        future_observed_agri_or_cpp_as_input=False,test_payloads_read=0,buffer_payloads_read=0,
        output_width_mm=177, output_height_mm=132,
        semantic_palette=dict(zip(ARM_LABELS,COLORS)),
        publisher_requirement='provisional; journal-specific sizing not asserted',
        sources=[dict(path=str(p.relative_to(REPO)).replace('\\','/'),sha256=sha(p)) for p in source_files],
        plotting_script_sha256=sha(Path(__file__)),software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=mpl.__version__))
    (OUT/'forecast_ghi_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'rmse':dict(zip(ARMS,overall.tolist())), 'paired_rows':score['unique_station_rows'],
                      'target_dates':len(dates),'csv_rows':len(metrics),'outputs':str(OUT)}))

if __name__=='__main__':
    main()
