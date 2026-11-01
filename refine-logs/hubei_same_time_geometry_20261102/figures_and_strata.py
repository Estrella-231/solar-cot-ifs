from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
P=Path(__file__).parent;F=P.parents[1]/'paper/figures/hubei_sync_corrected_20261102';F.mkdir(parents=True,exist_ok=True)
R=json.loads((P/'INDEPENDENT_FP64_RESULTS.json').read_text());B=P/'recovered';rows=list(csv.DictReader((B/'pack/rows.csv').open()))
rv=P/'r_validation';pred=np.load(rv/'val_prediction_cot.npy').astype(float);truth=np.load(rv/'val_reference_cpp_cot.npy').astype(float);mask=np.load(rv/'val_valid_mask.npy').astype(bool)
keys=list(csv.DictReader((rv/'val_keys.csv').open()));lut={r['time_bjt']:i for i,r in enumerate(keys)}
centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)};edges=[0,.1,5,30,100.000001];names=['Near-zero','Thin','Moderate','Thick'];cells=[];coverage=[]
for st,(y,x) in centers.items():
 for seed in [42,43,44]:
  q=B/'training'/f'{st}_seed{seed}';a=np.load(q/'A_validation.npz');b=np.load(q/'AC_validation.npz');ii=a['global_indices'];rr=[rows[i] for i in ii];ds=np.array([r['time_bjt'][:8] for r in rr]);cc=np.full(len(ii),np.nan)
  for j,r in enumerate(rr):
   ix=lut.get(r['time_bjt'])
   if ix is not None and mask[ix,0,y,x]:cc[j]=truth[ix,0,y,x]
  err=np.stack([a['prediction'],b['prediction']]).astype(float)-a['truth'].astype(float);cells.append((st,seed,ds,cc,err))
  if seed==42:coverage.append({'station':st,'all':len(ii),'CPP_valid':int(np.isfinite(cc).sum())})
strata=[]
for k in range(4):
 subset=[];dsall=set();n=0
 for st,seed,ds,cc,err in cells:
  keep=(cc>=edges[k])&(cc<edges[k+1])
  if not keep.any():continue
  subset.append((st,seed,ds[keep],err[:,keep]));dsall.update(ds[keep])
  if seed==42:n+=int(keep.sum())
 dates=sorted(dsall);points=np.array([np.sqrt(np.mean(e**2,axis=1)) for _,_,_,e in subset]);point=points.mean(0)
 sums=np.array([[[np.sum(e[a,ds==d]**2) for d in dates] for a in [0,1]] for _,_,ds,e in subset]);ns=np.array([[np.sum(ds==d) for d in dates] for _,_,ds,e in subset])
 rng=np.random.default_rng(20261102+k);draws=rng.integers(0,len(dates),size=(10000,len(dates)));w=np.stack([np.sum(draws==i,axis=1) for i in range(len(dates))],axis=1);den=w@ns.T;valid=(den>0).all(1);w=w[valid][:5000];den=den[valid][:5000]
 assert len(w)==5000
 bb=np.stack([w@sums[:,a].T for a in [0,1]],axis=2);boot=np.sqrt(bb/den[:,:,None]).mean(1);gain=100*(point[0]-point[1])/point[0]
 strata.append({'name':names[k],'n':n,'dates':len(dates),'station_seed_cells':len(subset),'A_rmse':point[0],'AC_rmse':point[1],'gain_pct':gain,'gain_ci95':np.quantile(100*(boot[:,0]-boot[:,1])/boot[:,0],[.025,.975]).tolist(),'delta_ci95':np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist(),'A_bias':float(np.mean([e[0].mean() for _,_,_,e in subset])),'AC_bias':float(np.mean([e[1].mean() for _,_,_,e in subset]))})
rawerror=(pred-truth)[mask];quality={'reference':'CPP retrieval, not independent truth','grid':'all valid 16x16 pixels','frames':len(keys),'dates':len({x['time_bjt'][:8] for x in keys}),'pixels':int(mask.sum()),'rmse':float(np.sqrt(np.mean(rawerror**2))),'mae':float(np.mean(abs(rawerror))),'bias':float(np.mean(rawerror))}
audit={'quality':quality,'coverage':coverage,'strata':strata,'scope':'same corrected-R and paired GHI prediction lineage; valid station-centre CPP only for strata; missing CPP retained in total GHI score','intervals':'5000 paired target-date draws per stratum; retain fixed available station-seed cells, reject draws with zero records in any cell'}
(P/'COT_QUALITY_AND_GHI_STRATA.json').write_text(json.dumps(audit,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
blue='#0072B2';orange='#D55E00'
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),dpi=240)
 plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.1,3),layout='constrained');names2=['Niushou','Wangzhai','Mizhuang','Equal mean'];vals=[R['by_station'][s] for s in centers]+[{'A_rmse':R['A_rmse'],'AC_rmse':R['AC_rmse'],'delta_ci95':R['delta_ci95']}]
for a,key,col,mark in [('Without COT','A_rmse',blue,'o'),('With COT','AC_rmse',orange,'s')]:axs[0].plot(range(4),[v[key] for v in vals],mark+'-',color=col,label=a)
for j,v in enumerate(vals):
 d=v['AC_rmse']-v['A_rmse'];lo,hi=v['delta_ci95'];axs[1].errorbar(j,d,yerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color=blue)
for ax in axs:ax.set_xticks(range(4),names2,rotation=12);ax.grid(axis='y',alpha=.18);ax.set_xlim(-.5,3.5)
axs[0].set_ylabel('GHI RMSE (W m$^{-2}$)');axs[0].set_ylim(0,105);axs[0].legend(frameon=False,fontsize=8);axs[1].set_ylabel('With COT $-$ without COT\nRMSE (W m$^{-2}$)');axs[1].axhline(0,color='.4',lw=.8);save(fig,'ghi_gain')
fig,axs=plt.subplots(1,2,figsize=(7.1,3),layout='constrained');x=truth[mask];y=pred[mask];axs[0].hist2d(x,y,bins=[np.linspace(0,100,81),np.linspace(0,100,81)],norm=LogNorm(),cmap='viridis');axs[0].plot([0,100],[0,100],'--',color='.4',lw=1);axs[0].set(xlabel='CPP reference COT',ylabel='Retrieved COT',xlim=(0,100),ylim=(0,100));axs[0].text(.04,.96,f"RMSE={quality['rmse']:.2f}\nMAE={quality['mae']:.2f}\nBias={quality['bias']:+.2f}\nN={quality['pixels']:,}",transform=axs[0].transAxes,va='top',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'},fontsize=8)
fig.colorbar(axs[0].collections[0],ax=axs[0],label='Pixel count')
rbias=[]
for k in range(4):
 m=(x>=edges[k])&(x<edges[k+1]);rbias.append(float(np.mean((y-x)[m])))
axs[1].plot(range(4),rbias,'o-',color=blue);axs[1].axhline(0,color='.4',lw=.8);axs[1].set_xticks(range(4),names,rotation=12);axs[1].set_ylabel('Retrieved $-$ reference COT');axs[1].set_xlabel('CPP reference COT class');axs[1].grid(alpha=.18);save(fig,'cot_quality')
fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
for key,col,m,label in [('A_rmse',blue,'o','Without COT'),('AC_rmse',orange,'s','With COT')]:axs[0].plot(range(4),[s[key] for s in strata],m+'-',color=col,label=label)
v=np.array([s['gain_pct'] for s in strata]);ci=np.array([s['gain_ci95'] for s in strata]);axs[1].errorbar(range(4),v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='o',capsize=4,color=blue);axs[1].axhline(0,color='.4',lw=.8)
for j,s in enumerate(strata):axs[1].annotate(f"n={s['n']}\ndays={s['dates']}",(j,ci[j,1]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
axs[0].set_ylabel('GHI RMSE (W m$^{-2}$)');axs[0].set_ylim(bottom=0);axs[0].legend(frameon=False,fontsize=8);axs[1].set_ylabel('RMSE reduction (%)');axs[1].set_ylim(min(-10,ci[:,0].min()-5),ci[:,1].max()+18)
for ax in axs:ax.set_xticks(range(4),names,rotation=12);ax.set_xlim(-.5,3.5);ax.grid(axis='y',alpha=.18)
save(fig,'ghi_thickness')
print(json.dumps(audit,indent=2))
