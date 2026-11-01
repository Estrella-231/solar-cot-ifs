"""Rebuild a fixed calendar-selected case, without fitting or changing support.

Run with the repository's plotting Python. All paths are resolved from this file.
The satellite images are native 16x16 samples, not generated illustrations.
"""
from pathlib import Path
from datetime import datetime, timedelta
import csv
import hashlib
import json
import platform

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
SRC = REPO / 'refine-logs/hubei_cot_conditional_information_20261023'
SITES = ['Niushou', 'Wangzhai', 'Mizhuang']
CENTERS = [(8, 6), (9, 8), (7, 10)]  # (row, column), audited source plot
SEEDS = [42, 43, 44]  # identities of the three frozen GHI baseline sources
ARM_KEYS = ['G1', 'observed_train_forecast_infer',
            'observed_train_observed_infer_OFFLINE']
COLORS = {'actual': '#80549A', 'forecast': '#D2691E',
          'baseline': '#727980', 'truth': '#181818'}
MM = 1 / 25.4
STYLE = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': .65, 'lines.linewidth': 1.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'stix', 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case():
    with np.load(SRC / 'CASE_MATERIAL.npz', allow_pickle=False) as z:
        case = {k: z[k] for k in z.files}
    scope = json.loads((SRC / 'CASE_SCOPE.json').read_text(encoding='utf-8'))
    init = str(case['init'])
    assert init == scope['init']
    assert case['observed_AGRI'].shape == (17, 16, 16, 16)
    assert case['forecast_AGRI'].shape == (16, 13, 16, 16)
    expected = [(datetime.strptime(init, '%Y%m%d%H%M') + timedelta(minutes=15*k))
                .strftime('%Y%m%d%H%M') for k in range(17)]
    assert list(case['times']) == expected
    records, data, input_paths = [], {}, []
    for station, (row, col) in zip(SITES, CENTERS):
        predictions = np.full((3, 3, 16), np.nan)
        truth = np.full(16, np.nan)
        availability = np.zeros(16, dtype=bool)
        for s, seed in enumerate(SEEDS):
            p = SRC / 'cross_source_attempt1' / station / f'seed{seed}_OOF.npz'
            input_paths.append(p)
            with np.load(p, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            arms = list(d['arms'])
            for li, lead in enumerate(range(15, 241, 15)):
                ix = np.flatnonzero((d['init_time'] == init) & (d['lead'] == lead))
                assert len(ix) <= 1
                if len(ix):
                    i = ix[0]
                    assert d['target_time'][i] == expected[li+1]
                    predictions[s, :, li] = d['prediction'][[arms.index(a) for a in ARM_KEYS], i]
                    if np.isfinite(truth[li]):
                        assert truth[li] == d['truth'][i]
                    truth[li] = d['truth'][i]
                    availability[li] = d['available'][i]
        assert np.array_equal(np.isfinite(predictions[0]), np.isfinite(predictions[1]))
        assert np.array_equal(np.isfinite(predictions[0]), np.isfinite(predictions[2]))
        count = np.isfinite(predictions).sum(0)
        mean = np.divide(np.nansum(predictions, axis=0), count,
                         out=np.full((3, 16), np.nan), where=count > 0)
        actual = case['actual_R_COT'][1:, 0, row, col].astype(np.float64)
        actual[~case['actual_valid'][1:, 0, row, col]] = np.nan
        forecast = case['forecast_R_COT'][:, 0, row, col].astype(np.float64)
        data[station] = dict(actual=actual, forecast=forecast, truth=truth,
                             prediction=mean, seeds=predictions, available=availability)
        for li, lead in enumerate(range(15, 241, 15)):
            records.append(dict(station=station, init_time_bjt=init,
                target_time_bjt=expected[li+1], lead_minutes=lead,
                observed_R_COT=actual[li], forecast_R_COT=forecast[li],
                observed_15min_GHI=truth[li], baseline_GHI=mean[0, li],
                same_head_forecast_COT_GHI=mean[1, li],
                same_head_observed_COT_GHI_OFFLINE=mean[2, li],
                scored=bool(np.isfinite(truth[li])), cot_available=bool(availability[li])))
    return case, scope, data, records, input_paths


def rgb(image):
    # One physical reflectance range for both sources. The old 0..0.7 stretch
    # clipped 35.46% of observed channel values, hiding thick-cloud texture.
    return np.sqrt(np.clip(np.moveaxis(image[[2, 1, 0]], 0, -1), 0, 1))


def format_time(ax):
    ax.set_xlim(12.15, 16.08)
    ax.set_xticks([12.25, 13, 14, 15, 16], ['12:15', '13:00', '14:00', '15:00', '16:00'])
    ax.set_xlabel('Target time (BJT)')
    ax.grid(axis='y', color='#DDE1E5', linewidth=.5, zorder=0)
    ax.tick_params(length=2.5, width=.6)


def time_series(ax, values, variable):
    hours = 12 + np.arange(15, 241, 15) / 60
    if variable == 'cot':
        ax.plot(hours, values['actual'], '--o', color=COLORS['actual'], ms=2.5,
                label='Observed AGRI → R (offline)')
        ax.plot(hours, values['forecast'], '-s', color=COLORS['forecast'], ms=2.5,
                label='SimVP AGRI → R')
        ax.set_ylabel('Station-pixel COT')
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 25, 50, 75, 100])
    else:
        ax.plot(hours, values['truth'], '-o', color=COLORS['truth'], ms=2.5,
                label='Observed 15-min GHI', zorder=5)
        ax.plot(hours, values['prediction'][0], '-o', color=COLORS['baseline'], ms=1.8,
                label='AGRI + historical COT')
        ax.plot(hours, values['prediction'][1], '-s', color=COLORS['forecast'], ms=2.3,
                label='Same head: forecast COT')
        ax.plot(hours, values['prediction'][2], '--o', color=COLORS['actual'], ms=1.8,
                label='Same head: observed COT (offline)')
        ax.set_ylabel('GHI (W m$^{-2}$)')
        ax.set_ylim(0, 800)
        ax.set_yticks([0, 200, 400, 600, 800])
    format_time(ax)


def save(fig, stem):
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{stem}.{suffix}')
    # Inspection preview at actual intended dimensions, 150 pixels per inch.
    fig.savefig(OUT / f'{stem}_print_preview.png', dpi=150)
    plt.close(fig)


def main_figure(case, data):
    # The first, central and last forecast frames: no error-conditioned selection.
    indices = [1, 8, 16]
    fig = plt.figure(figsize=(177*MM, 183*MM))
    gs = fig.add_gridspec(3, 4, left=.105, right=.925, bottom=.385, top=.925,
                          wspace=.10, hspace=.09)
    labels = ['Observed AGRI\n(offline)', 'SimVP AGRI',
              'Observed AGRI → R\n(offline)', 'SimVP AGRI → R']
    names = ['12:15\n+15 min', '14:00\n+120 min', '16:00\n+240 min']
    cm = plt.get_cmap('magma').copy()
    cm.set_bad('#B9BEC3')
    image_axes = []
    for ri, k in enumerate(indices):
        for ci in range(4):
            ax = fig.add_subplot(gs[ri, ci])
            if ci == 0: field = rgb(case['observed_AGRI'][k])
            elif ci == 1: field = rgb(case['forecast_AGRI'][k-1])
            elif ci == 2:
                field = np.where(case['actual_valid'][k, 0], case['actual_R_COT'][k, 0], np.nan)
            else: field = case['forecast_R_COT'][k-1, 0]
            kw = {} if ci < 2 else dict(cmap=cm, vmin=0, vmax=140)
            im = ax.imshow(field, interpolation='nearest', origin='upper', **kw)
            # The white box is the actual 3x3 GHI readout at Niushou, not 16x16 input.
            ax.add_patch(Rectangle((4.5, 6.5), 3, 3, fill=False,
                                   edgecolor='#242424', linewidth=1.45, linestyle='--'))
            ax.add_patch(Rectangle((4.5, 6.5), 3, 3, fill=False,
                                   edgecolor='white', linewidth=.8, linestyle='--'))
            ax.plot(6, 8, '+', color='white', ms=4, mew=.8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color('#62676B'); sp.set_linewidth(.35)
            if ri == 0: ax.set_title(labels[ci], pad=6)
            if ci == 0:
                ax.text(-.16, .5, names[ri], ha='right', va='center', transform=ax.transAxes,
                        fontsize=8)
            image_axes.append(ax)
    # Shared COT scale. Images and curves both retain values >100.
    cax = fig.add_axes([.945, .418, .012, .46])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 40, 80, 120, 140])
    cb.ax.tick_params(labelsize=8, length=2, width=.5)
    cb.outline.set_linewidth(.5)
    cb.ax.set_title('COT', fontsize=8, pad=6)
    fig.text(.025, .958, 'a', fontweight='bold', fontsize=10)
    fig.text(.105, .373, 'Niushou: + = station pixel; dashed box = 3 × 3 GHI input; images = 16 × 16 pixels',
             fontsize=8, va='top')
    lower = fig.add_gridspec(1, 2, left=.105, right=.925, bottom=.135, top=.31, wspace=.33)
    cot = fig.add_subplot(lower[0, 0]); ghi = fig.add_subplot(lower[0, 1])
    time_series(cot, data['Niushou'], 'cot')
    time_series(ghi, data['Niushou'], 'ghi')
    for ax, label in [(cot, 'b'), (ghi, 'c')]:
        ax.text(-.19, 1.04, label, transform=ax.transAxes, fontweight='bold', fontsize=10)
        for hour in [12.25, 14, 16]:
            ax.axvline(hour, color='#C1C5C8', lw=.55, ls=':', zorder=-1)
    # One compact four-item legend, matching the GHI readout. COT colors have same source meanings.
    handles, labels = ghi.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .032),
               ncol=2, frameon=False, columnspacing=1.4, handlelength=2.5)
    fig.text(.105, .016, '2025-08-06 • initialization 12:00 BJT • calendar-selected OOF diagnostic',
             fontsize=8, color='#42484D')
    save(fig, 'fig_cloud_source_mechanism')


def all_station_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(177*MM, 116*MM),
                             sharex=True, sharey='row', layout='constrained')
    fig.set_constrained_layout_pads(w_pad=2*MM, h_pad=2*MM, wspace=.04, hspace=.03)
    for col, station in enumerate(SITES):
        for row, variable in enumerate(['cot', 'ghi']):
            ax=axes[row, col]
            time_series(ax, data[station], variable)
            ax.text(.02, .96, chr(97+3*row+col), transform=ax.transAxes,
                    fontweight='bold', va='top', fontsize=10)
            if col:
                ax.set_ylabel('')
            if row == 0:
                ax.set_title(station, fontsize=9)
                ax.set_xlabel('')
            ax.set_xticks([13, 14, 15, 16], ['13:00', '14:00', '15:00', '16:00'])
    handles, labels = axes[1,0].get_legend_handles_labels()
    # Reserve outside axes space via the figure legend's constrained layout location.
    fig.legend(handles, labels, loc='outside lower center', ncol=2,
               frameon=False, handlelength=2.6, columnspacing=1.3)
    save(fig, 'fig_three_station_source_response')


def run():
    case, scope, data, records, paths = load_case()
    with (OUT/'plotted_timeseries.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with plt.rc_context(STYLE):
        main_figure(case, data)
        all_station_figure(data)
    summaries={}
    for station in SITES:
        d=data[station]
        summaries[station]=dict(scored_GHI_times=int(np.isfinite(d['truth']).sum()),
            observed_R_station_COT_max=float(np.nanmax(d['actual'])),
            forecast_R_station_COT_max=float(np.nanmax(d['forecast'])),
            observed_GHI_min=float(np.nanmin(d['truth'])), observed_GHI_max=float(np.nanmax(d['truth'])))
    common=[SRC/'CASE_MATERIAL.npz',SRC/'CASE_SCOPE.json',SRC/'plot_case.py',
            SRC/'RESULT_REPORT.md',SRC/'cross_source.py', SRC/'PLAN.md',
            SRC.parent/'hubei_forecast_cot_increment_20261022/EXPERIMENT_PLAN.md']
    provenance=dict(selection=scope,source_sha256={str(p.relative_to(REPO)):sha(p) for p in common+paths},
        dimensions_mm={'fig_cloud_source_mechanism':[177,183],'fig_three_station_source_response':[177,116]},
        snapshot_indices=[1,8,16],snapshot_rule='first, central and last forecast frames; independent of errors',
        primary_station_rule='first station in the original fixed station order, not selected by improvement',
        RGB='C03/C02/C01; sqrt(clip(reflectance,0,1)); same 0..1 reflectance range for both sources',
        RGB_display_audit={key:dict(old_fraction_above_0_7=float(np.mean(case[key][:,:3]>.7)),
                                  new_fraction_outside_0_1=float(np.mean((case[key][:,:3]<0)|(case[key][:,:3]>1))))
                           for key in ['observed_AGRI','forecast_AGRI']},
        spatial='native 16x16 pixels, nearest rendering; no geographic scale inferred; Niushou row8 col6 (zero-based), 3x3 box',
        COT='R retrieval proxy, not CPP or truth; linear shared 0..140 scale; raw values >100 retained and disclosed',
        semantic_colors=COLORS,
        GHI='right-ending 15-minute average; gaps kept; means of 3 frozen baseline-source variants for visualization, not ensemble scores',
        causal_boundary='actual future AGRI and derived R-COT shown only as offline diagnostic; same fixed reference-trained ridge head compared under two COT sources',
        test_read=False,training_performed=False,summary=summaries,
        software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        outputs={p.name:sha(p) for p in OUT.glob('fig_*.*')},
        script_sha256=sha(Path(__file__)))
    (OUT/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': run()
