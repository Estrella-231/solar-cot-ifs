"""Frozen-head source replacement. Recompute on pre-existing common support."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SRC=ROOT/'refine-logs/hubei_agri_cot_coherence_20261023'
STATIONS=['Niushou','Wangzhai','Mizhuang']
SEEDS=[42,43,44]
ARMS=['forecast_G1','forecast_no_COT','forecast_forecast_COT',
      'forecast_observed_COT_OFFLINE','observed_G1_OFFLINE',
      'observed_no_COT_OFFLINE','observed_forecast_COT_OFFLINE',
      'observed_observed_COT_OFFLINE']
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,
 'axes.labelsize':8.3,'xtick.labelsize':7.7,'ytick.labelsize':7.7,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def main():
    source_hashes={}; daily={}; counts={}; rows=[]
    for si,station in enumerate(STATIONS):
        common_keys=None
        for sj,seed in enumerate(SEEDS):
            path=SRC/'replay'/station/f'seed{seed}_factorial.npz'
            source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
            assert d['arms'].tolist()==ARMS
            mask=d['common']>0
            keys=list(zip(d['init_time'][mask].tolist(),d['target_time'][mask].tolist(),d['lead'][mask].tolist()))
            assert len(set(keys))==len(keys)
            if common_keys is None: common_keys=keys
            else: assert common_keys==keys
            err=d['prediction'].astype('float64')-d['truth'].astype('float64')[None,:]
            assert np.isfinite(err[:,mask]).all()
            counts[station]=int(mask.sum())
            for ai,arm in enumerate(ARMS):
                e=err[ai,mask]
                rows.append(dict(station=station,seed=seed,arm=arm,n=len(e),
                    rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),bias=np.mean(e)))
            for date in np.unique(d['date'][mask]):
                m=mask & (d['date']==date)
                s=daily.setdefault(str(date),np.zeros((3,3,8,2)))
                s[si,sj,:,0]=np.sum(err[:,m]**2,axis=1)
                s[si,sj,:,1]=int(m.sum())
    dates=sorted(daily); stats=np.stack([daily[d] for d in dates]); total=stats.sum(axis=0)
    mse=total[...,0]/total[...,1]; point=np.sqrt(mse).mean(axis=(0,1))
    expected=json.loads((SRC/'RESULTS.json').read_text())['summary']['common']['all']
    delta=max(abs(point[j]-expected['rmse'][a]) for j,a in enumerate(ARMS))
    assert delta<1e-9,delta
    weights=np.random.default_rng(20261023).multinomial(len(dates),np.ones(len(dates))/len(dates),5000)
    boot_mse=((weights@stats[...,0].reshape(len(dates),-1))/(weights@stats[...,1].reshape(len(dates),-1))).reshape(-1,3,3,8)
    boot=np.sqrt(boot_mse).mean((1,2))
    comparisons={}
    for name,j,k in [('replace_AGRI_no_future_COT',5,1),('forecast_COT_increment',2,1),('observed_COT_increment_OFFLINE',7,5)]:
        comparisons[name]={'delta':float(point[j]-point[k]),'ci95':np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist()}
    # Eq. E_FF-E_OO = (E_FO-E_OO)+(E_OF-E_OO)+interaction.
    m=mse.mean((0,1)); bm=boot_mse.mean((1,2))
    terms=np.array([m[3]-m[7],m[6]-m[7],m[2]-m[3]-m[6]+m[7],m[2]-m[7]])
    bt=np.stack([bm[:,3]-bm[:,7],bm[:,6]-bm[:,7],bm[:,2]-bm[:,3]-bm[:,6]+bm[:,7],bm[:,2]-bm[:,7]],axis=1)
    assert abs(terms[:3].sum()-terms[3])<1e-8
    summaries={'scope':'common pre-existing replacement support; development-only offline diagnosis',
       'dates':dates,'n_dates':len(dates),'n_rows':sum(counts.values()),'by_station_n':counts,
       'rmse':dict(zip(ARMS,point.tolist())),'comparisons':comparisons,
       'MSE_decomposition':{name:{'point':float(terms[i]),'ci95':np.quantile(bt[:,i],[.025,.975]).tolist()} for i,name in enumerate(['D_AGRI','D_COT','interaction','total'])},
       'aggregation':'equal mean of per-station per-seed RMSE; MSE decomposition averages per-station per-seed MSE',
       'bootstrap':'5000 paired target-date block draws, seed 20261023', 'source_score_max_abs_difference':delta}
    (HERE/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    with (HERE/'source_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig=plt.figure(figsize=(177/25.4,128/25.4))
    ax=fig.add_axes([.13,.59,.82,.28])
    grid=point[[1,2,3,5,6,7]].reshape(2,3)
    ax.imshow(grid,cmap='cividis_r',norm=Normalize(80,110),aspect='auto')
    ax.set_xticks([0,1,2],['No future COT','Forecast-derived COT','Observed-derived COT\n(offline)'])
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=7)
    ax.set_yticks([0,1],['Forecast\nAGRI','Observed\nAGRI\n(offline)'])
    for i in range(2):
        for j in range(3):
            ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',fontsize=12,
                color='white' if grid[i,j]>97 else '#17252A',fontweight='bold')
    ax.set_xticks(np.arange(-.5,3),minor=True);ax.set_yticks(np.arange(-.5,2),minor=True)
    ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    ax.spines[:].set_visible(False)
    fig.text(.025,.965,'a',fontsize=11,fontweight='bold')
    fig.text(.13,.965,'GHI RMSE (W m$^{-2}$)',fontsize=9)
    fig.text(.13,.55,f'{sum(counts.values()):,} matched rows  |  {len(dates)} target dates  |  3 stations × 3 seeds',fontsize=8)
    # Distinct scales are deliberate: absolute image-source sensitivity vs COT increment.
    a=fig.add_axes([.30,.375,.65,.105]);a.axvline(0,color='#737D86',lw=.7)
    c=comparisons['replace_AGRI_no_future_COT']; x=c['delta'];lo,hi=c['ci95']
    a.errorbar(x,0,xerr=[[x-lo],[hi-x]],fmt='D',ms=6,color='#64508E',capsize=3,lw=1.4)
    a.set_xlim(-25,2);a.set_ylim(-.7,.7);a.set_yticks([0],['Replace forecast AGRI\nwith observed AGRI']);a.tick_params(axis='y',length=0)
    a.set_xticks([-20,-10,0]);a.grid(axis='x',color='#E5E8EC',lw=.6)
    fig.text(.025,.45,'b',fontsize=11,fontweight='bold')
    fig.text(.30,.49,'Image-source change: $\\Delta$RMSE (W m$^{-2}$)',fontsize=8)
    a.text(.99,.9,f'{x:+.2f} [{lo:+.2f}, {hi:+.2f}]',transform=a.transAxes,ha='right',fontsize=7.6)
    b=fig.add_axes([.30,.12,.65,.16]);b.axvline(0,color='#737D86',lw=.7)
    for y,name,label,color,mark in [(1,'forecast_COT_increment','Forecast AGRI +\nforecast-derived COT','#B95716','s'),(0,'observed_COT_increment_OFFLINE','Observed AGRI +\nobserved-derived COT','#64508E','D')]:
        c=comparisons[name];x=c['delta'];lo,hi=c['ci95']
        b.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=mark,ms=5,color=color,capsize=3,lw=1.3)
    b.set_yticks([1,0],['Forecast AGRI +\nforecast-derived COT','Observed AGRI +\nobserved-derived COT']);b.tick_params(axis='y',length=0)
    b.set_xlim(-.8,.8);b.set_ylim(-.6,1.6);b.set_xticks([-.5,0,.5]);b.grid(axis='x',color='#E5E8EC',lw=.6)
    b.set_xlabel('$\Delta$RMSE (W m$^{-2}$); negative values improve')
    fig.text(.025,.27,'c',fontsize=11,fontweight='bold')
    fig.text(.30,.307,'COT addition relative to no future COT (same AGRI source)',fontsize=8)
    for fmt in ['pdf','svg','png']:fig.savefig(HERE/f'source_replacement.{fmt}',dpi=350)
    plt.close(fig)
    (HERE/'MANIFEST.json').write_text(json.dumps({'sources':source_hashes,
       'size_mm':[177,128],'transformations':['pre-existing common mask only','FP64 residuals','equal station-seed scores','paired target-date bootstrap'],
       'scope':'development diagnosis; observed future inputs are not deployable forecasts',
       'data_audit':summaries['source_score_max_abs_difference'],'source_images_used':False},indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
