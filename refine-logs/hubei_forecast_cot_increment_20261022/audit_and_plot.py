"""Second implementation: saved predictions -> pooled errors -> date-cluster CI."""
import csv,json,hashlib,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('evidence',type=Path);a=ap.parse_args();root=a.evidence
score=json.loads((root/'scored_attempt1/score.json').read_text());arms=['G0','G1','G2','G3'];records=[];store={};all_dates=set();updates={}
for trial in sorted((root/'training_attempt1/pairs').iterdir()):
    if not trial.is_dir():continue
    completed=json.loads((trial/'COMPLETE.json').read_text());assert completed['base_unchanged']
    canonical=None
    for arm in arms:
        z=np.load(trial/(arm+'_validation_predictions.npz'));y=z['truth_ghi_wm2'].astype(np.float64);p=z['pred_ghi_wm2'].astype(np.float64)
        key=list(zip(z['station'],z['init_time_bjt'],z['target_time_bjt'],z['lead_minutes']));assert len(set(key))==len(key)
        if canonical is None:canonical=(key,y,z['clear_sky_wm2'])
        else:assert canonical[0]==key and np.array_equal(canonical[1],y) and np.array_equal(canonical[2],z['clear_sky_wm2'])
        err=p-y;dates=np.array([v[:8] for v in z['target_time_bjt']]);all_dates.update(dates)
        store[trial.name,arm]=(err,dates)
        records.append({'trial':trial.name,'arm':arm,'n':len(err),'rmse':float(np.linalg.norm(err)/np.sqrt(len(err)))})
        meta=completed['arms'][arm];logs=[json.loads(s) for s in (trial/(arm+'_epochs.jsonl')).read_text().splitlines()]
        assert len(logs)==30 and logs[-1]['updates']==meta['updates'];assert all('val' not in x['inner_role'] for x in logs)
        assert meta['best_epoch']==min(logs,key=lambda x:x['inner_rmse_wm2'])['epoch'];updates[trial.name+'/'+arm]=meta['updates']
    for x,y in [('G0','G1'),('G2','G3')]:
        for k in ['initialization_sha256','order_sha256','updates','parameters']:assert completed['arms'][x][k]==completed['arms'][y][k]
ind={arm:float(np.mean([r['rmse'] for r in records if r['arm']==arm])) for arm in arms}
max_delta=max(abs(ind[k]-score['equal_station_equal_seed_rmse'][k]) for k in arms);assert max_delta<1e-10
dates=sorted(all_dates);trials=sorted({k[0] for k in store});daily_square=np.zeros((len(dates),len(trials),4));daily_count=np.zeros_like(daily_square)
for ti,t in enumerate(trials):
    for ai,arm in enumerate(arms):
        e,d=store[t,arm]
        for di,date in enumerate(dates):daily_square[di,ti,ai]=np.dot(e[d==date],e[d==date]);daily_count[di,ti,ai]=(d==date).sum()
# Exactly the published date draws, independently reduce each station-seed cell.
w=np.random.default_rng(20261022).multinomial(len(dates),np.ones(len(dates))/len(dates),size=5000)
boot=np.zeros((5000,4))
for ti in range(len(trials)):
    for ai in range(4):boot[:,ai]+=np.sqrt((w@daily_square[:,ti,ai])/(w@daily_count[:,ti,ai]))/len(trials)
ci={}
for label,old in score['comparisons'].items():
    c,b=label.split('_minus_');interval=np.percentile(boot[:,arms.index(c)]-boot[:,arms.index(b)],[2.5,97.5]);assert np.allclose(interval,old['ci95_date_block'],atol=1e-10,rtol=0);ci[label]=interval.tolist()
audit={'state':'PASS_SECOND_IMPLEMENTATION_SAVED_NPZ_KEYS_LABELS_METRICS_CI_AND_BUDGET','trials':len(trials),'heads':len(records),'max_rmse_difference':max_delta,'rmse':ind,'ci':ci,'target_dates':dates,'updates':updates,'test_payloads_read':0,'buffer_payloads_read':0}
(root/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
labels=['No COT','Historical\nCOT','Historical COT\n+ correction\n(no future COT)','Historical COT\n+ correction\n(with future COT)'];colors=['#778899','#009e73','#b8b8b8','#0072b2']
vals=[ind[k] for k in arms];bars=ax[0].bar(range(4),vals,color=colors,width=.65)
ax[0].set_xticks(range(4),labels,fontsize=9);ax[0].set(ylabel='GHI RMSE (W m$^{-2}$)',ylim=(0,120),title='Three stations × three paired seeds')
for b,v in zip(bars,vals):ax[0].text(b.get_x()+b.get_width()/2,v+.8,f'{v:.2f}',ha='center')
names=['G1_minus_G0','G3_minus_G2'];ys=[1,0]
for name,y in zip(names,ys):
    m=score['comparisons'][name];point=m['rmse_difference'];lo,hi=ci[name]
    ax[1].errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color='#0072b2',capsize=5)
ax[1].axvline(0,color='#555555',ls='--',lw=1);ax[1].set_yticks(ys,['Historical COT − no COT','Forecast COT − capacity control']);ax[1].set(xlabel='Paired RMSE difference (W m$^{-2}$)',title='95% intervals resampled by target date',ylim=(-.5,1.5));ax[1].grid(axis='x',alpha=.2)
fig.suptitle('Hubei GHI forecast: historical and future COT contributions',fontsize=14)
fig.savefig(root/'four_arm_ghi_forecast.png',dpi=200);fig.savefig(root/'four_arm_ghi_forecast.pdf');plt.close(fig)
print(json.dumps({'audit':audit['state'],'rmse':ind,'max_delta':max_delta}),flush=True)
