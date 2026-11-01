from __future__ import annotations
import argparse, csv, importlib.util, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_scorer(path):
    spec=importlib.util.spec_from_file_location('fullpatch_scorer',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def collect_absolute(data, stations, seeds, arms, short_only=False):
    out=[]
    for st in stations:
        for arm in arms:
            seed_rmse=[]; nrows=0
            for seed in seeds:
                p=[]; y=[]
                for fold in range(5):
                    d=data[(st,fold,arm,seed)]
                    use=np.asarray([int(k['lead_minutes'])<=60 for k in d['keys']],bool) if short_only else np.ones(len(d['keys']),bool)
                    p.extend(d['pred'][use]); y.extend(d['truth'][use]); nrows+=int(use.sum())
                seed_rmse.append(float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2))))
            out.append({'station':st,'arm':arm,'scope':'plus15_60' if short_only else 'all_leads',
                        'mean_seed_rmse_wm2':float(np.mean(seed_rmse)),'sd_across_seeds_wm2':float(np.std(seed_rmse,ddof=1)),
                        'seed_rmse_wm2':json.dumps(seed_rmse),'rows_per_seed':nrows//len(seeds)})
    return out

def leadwise(data, stations, seeds, arms):
    out=[]
    for st in stations:
        for lead in range(15,241,15):
            per={a:[] for a in arms}
            n=0
            for seed in seeds:
                for arm in arms:
                    p=[]; y=[]
                    for fold in range(5):
                        d=data[(st,fold,arm,seed)]
                        use=np.asarray([int(k['lead_minutes'])==lead for k in d['keys']],bool)
                        p.extend(d['pred'][use]); y.extend(d['truth'][use]); n+=int(use.sum()) if arm==arms[0] else 0
                    per[arm].append(float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2))))
            out.append({'station':st,'lead_minutes':lead,'delta_pxc16_minus_pxc3_wm2':float(np.mean(per['pxc16'])-np.mean(per['pxc3'])),
                        'delta_pxc16_minus_none_wm2':float(np.mean(per['pxc16'])-np.mean(per['none'])),
                        'rmse_pxc16_mean_seed_wm2':float(np.mean(per['pxc16'])),'rmse_pxc3_mean_seed_wm2':float(np.mean(per['pxc3'])),
                        'rmse_none_mean_seed_wm2':float(np.mean(per['none'])),'n_rows_per_seed':n//len(seeds)})
    return out

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def plot(metrics,absolute,leadrows,png,pdf):
    colors={'none':'#66727D','pxc3':'#0072B2','pxc16':'#D55E00'}
    labels={'none':'No COT','pxc3':'Predicted COT 3×3','pxc16':'Predicted COT 16×16'}
    fig,axs=plt.subplots(2,2,figsize=(12.4,8.2))
    for ax,scope,title in [(axs[0,0],'all_leads','All leads: +15 to +240 min'),(axs[0,1],'plus15_60','Short leads: +15 to +60 min')]:
        x=np.arange(2); width=.23
        for j,arm in enumerate(('none','pxc3','pxc16')):
            vals=[next(r['mean_seed_rmse_wm2'] for r in absolute if r['station']==st and r['arm']==arm and r['scope']==scope) for st in ('sili','zhujia')]
            err=[next(r['sd_across_seeds_wm2'] for r in absolute if r['station']==st and r['arm']==arm and r['scope']==scope) for st in ('sili','zhujia')]
            ax.bar(x+(j-1)*width,vals,width,label=labels[arm],color=colors[arm],edgecolor='#222222',linewidth=.45,
                   yerr=err,capsize=2,error_kw={'elinewidth':.8})
        ax.set_xticks(x,['Sili','Zhujia']); ax.set_ylabel('GHI RMSE (W m$^{-2}$)'); ax.set_title(title); ax.grid(axis='y',alpha=.22); ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    handles,legend_labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,.955))
    ax=axs[1,0]
    for st,color,marker in [('sili','#0072B2','o'),('zhujia','#D55E00','s')]:
        rs=[r for r in leadrows if r['station']==st]
        ax.plot([r['lead_minutes'] for r in rs],[r['delta_pxc16_minus_pxc3_wm2'] for r in rs],color=color,marker=marker,markersize=4,linewidth=1.5,label=st.capitalize())
    ax.axhline(0,color='#222222',linewidth=.9); ax.set(xlabel='Forecast lead (min)',ylabel='ΔRMSE (W m$^{-2}$)',title='16×16 − 3×3; positive means worse')
    ax.grid(alpha=.22); ax.legend(frameon=False)
    ax=axs[1,1]; sc=metrics['results']; items=[('16×16 − 3×3','all_leads','pxc16_minus_pxc3'),('16×16 − 3×3','plus15_60','pxc16_minus_pxc3'),('16×16 − none','all_leads','pxc16_minus_none'),('16×16 − none','plus15_60','pxc16_minus_none')]
    yy=np.arange(len(items))[::-1]
    for y,(label,scope,key) in zip(yy,items):
        q=sc[key][scope]; ax.errorbar(q['delta_rmse_wm2'],y,xerr=[[q['delta_rmse_wm2']-q['ci95_low']],[q['ci95_high']-q['delta_rmse_wm2']]],fmt='o',color='#0072B2' if '3×3' in label else '#D55E00',capsize=3,markersize=5,linewidth=1.4)
    ax.axvline(0,color='#222222',linewidth=.9); ax.set_yticks(yy,[f'{"16−3" if "3×3" in a else "16−none"}, {"+15–60" if b=="plus15_60" else "all"}' for a,b,_ in items]); ax.set_xlabel('Paired ΔRMSE (W m$^{-2}$); positive means worse'); ax.set_title('Equal-station/seed estimate ± 95% block-bootstrap CI'); ax.grid(axis='x',alpha=.22); ax.set_axisbelow(True)
    fig.suptitle('Full predicted COT patch does not improve H-CV GHI',fontsize=15,fontweight='semibold',y=.99)
    fig.subplots_adjust(left=.085,right=.985,bottom=.11,top=.86,wspace=.42,hspace=.38)
    fig.text(.5,.025,'Exploratory H-internal date-block CV only; 112 fixed BJT two-day blocks, 5,000 paired bootstrap resamples. Error bars in top panels are SD across 3 seeds.',ha='center',fontsize=8.5)
    fig.savefig(png,dpi=220,facecolor='white'); fig.savefig(pdf,facecolor='white'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--score',type=Path,required=True); ap.add_argument('--scorer',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    m=json.loads((a.run/'run_manifest.json').read_text(encoding='utf-8')); s=json.loads(a.score.read_text(encoding='utf-8'))
    if m.get('state')!='COMPLETE_DIRECT_COT_FULLPATCH_H_CV' or s.get('state')!='COMPLETE_DIRECT_COT_FULLPATCH_H_CV_SCORE': raise RuntimeError('complete run and score receipts required')
    if s.get('run_manifest_sha256')!=__import__('hashlib').sha256((a.run/'run_manifest.json').read_bytes()).hexdigest(): raise RuntimeError('score/run manifest mismatch')
    sc=load_scorer(a.scorer); data,seen=sc.load_all(a.run)
    absolute=collect_absolute(data,sc.STATIONS,sc.SEEDS,sc.ARMS,False)+collect_absolute(data,sc.STATIONS,sc.SEEDS,sc.ARMS,True)
    leadrows=leadwise(data,sc.STATIONS,sc.SEEDS,sc.ARMS)
    a.output.mkdir(parents=True,exist_ok=False)
    write_csv(a.output/'absolute_rmse_by_station_arm_scope.csv',absolute); write_csv(a.output/'leadwise_delta_rmse.csv',leadrows)
    plot(s,absolute,leadrows,a.output/'fullpatch_hcv_diagnostic.png',a.output/'fullpatch_hcv_diagnostic.pdf')
    summary={'state':'COMPLETE_DIRECT_COT_FULLPATCH_HCV_ANALYSIS','evidence_tier':s['evidence_tier'],'station_rows':{k:len(v) for k,v in seen.items()},'absolute_rmse':absolute,'leadwise_file':'leadwise_delta_rmse.csv','interpretation':'positive candidate-minus-reference delta means higher error; H-internal exploratory screening only; no independent confirmation','test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    (a.output/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    import matplotlib as mpl
    (a.output/'figure_manifest.json').write_text(json.dumps({'figure':'fullpatch_hcv_diagnostic','audience':'internal experiment diagnosis','formats':['PNG','PDF'],'png_dpi':220,'figure_inches':[12.4,8.2],'matplotlib_version':mpl.__version__,'source_score':str(a.score),'raw_predictions':'saved per-run OOF prediction files in H only','uncertainty':'fixed calendar two-day BJT block bootstrap, 5000 resamples; top-panel bars show SD across 3 seeds','sign':'positive delta means worse GHI RMSE','test_and_buffer_payloads_opened':False},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':summary['state'],'out':str(a.output),'absolute_rmse':absolute},indent=2),flush=True)
if __name__=='__main__': main()
