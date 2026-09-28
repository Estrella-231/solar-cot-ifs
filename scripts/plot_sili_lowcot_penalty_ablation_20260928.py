from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1]/'results'/'regional_station_cot_20260927'/'sili_lowcot_penalty_ablation_seed42_20260928_v1'
arms={'Penalty on':'penalty_on_seed42','Penalty off':'penalty_off_seed42'}
colors={'Penalty on':'#6f4ab8','Penalty off':'#008f83'}
series={}
for name,folder in arms.items():
    rows=[json.loads(s) for s in (base/folder/'epochs.jsonl').read_text().splitlines()]
    step0=json.loads((base/folder/'step0.json').read_text())
    series[name]=(rows,step0['transport']['thick_rmse'],step0['transport']['rmse'])
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for name,(rows,step_thick,step_all) in series.items():
    x=[r['epoch'] for r in rows]
    axes[0].plot(x,[r['val']['learned']['thick_rmse'] for r in rows],marker='o',label=name,color=colors[name])
    axes[1].plot(x,[r['val']['learned']['rmse'] for r in rows],marker='o',label=name,color=colors[name])
axes[0].axhline(series['Penalty on'][1],color='#333333',linestyle='--',label='Step-0 transport')
axes[1].axhline(series['Penalty on'][2],color='#333333',linestyle='--',label='Step-0 transport')
axes[0].set(title='Thick-cloud RMSE (CPP COT ≥ 30)',xlabel='Epoch',ylabel='COT RMSE')
axes[1].set(title='All-valid COT RMSE',xlabel='Epoch',ylabel='COT RMSE')
for ax in axes: ax.grid(alpha=.25); ax.legend(frameon=False)
fig.suptitle('Sili fixed validation cohort — penalty ablation; step-0 is selectable',fontsize=12)
fig.savefig(base/'ablation_validation_curves.png',dpi=180)
fig.savefig(base/'ablation_validation_curves.pdf')
plt.close(fig)
print(base/'ablation_validation_curves.png')
