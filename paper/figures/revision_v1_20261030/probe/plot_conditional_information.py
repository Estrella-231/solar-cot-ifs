"""Source-locked conditional-information figure, including CI recomputation."""
from pathlib import Path
import csv,hashlib,json,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
SRC=ROOT/'refine-logs/hubei_cot_conditional_information_20261023'
files=[SRC/'RESULTS.json',SRC/'DATE_SUMS.npz',SRC/'cross_source_attempt1/RESULTS.json',
       SRC/'cross_source_attempt1/DATE_SUMS.npz',SRC/'RESULT_REPORT.md',SRC/'score.py',SRC/'cross_source.py']
J=json.loads(files[0].read_text());C=json.loads(files[2].read_text())
assert J['state']=='SCORED' and C['state']=='COLD_CROSS_SOURCE_COMPLETE'
assert J['test_payloads_read']==C['test_payloads_read']==0
assert J['summary']['all']['rows']==C['summary']['all']['rows']==32838
assert J['summary']['all']['dates']==C['summary']['all']['dates']
verification={}
for tag,path,result in [('main',files[1],J),('cross',files[3],C)]:
    with np.load(path) as z:
        arms=z['arms'].tolist()
        for horizon in [k for k in z.files if k!='arms']:
            stats=z[horizon];total=stats.sum(0)
            points=np.sqrt(total[...,0]/total[...,1]).mean((0,1))
            reference=np.array([result['summary'][horizon]['rmse'][a] for a in arms])
            point_error=float(np.max(abs(points-reference)));assert point_error<1e-10
            w=np.random.default_rng(20261023).multinomial(len(stats),np.ones(len(stats))/len(stats),5000)
            den=w@stats[...,1].reshape(len(stats),-1);assert (den>0).all()
            boot=np.sqrt((w@stats[...,0].reshape(len(stats),-1))/den).reshape(-1,3,3,len(arms)).mean((1,2))
            ci_error=0.
            for key,item in result['summary'][horizon]['comparisons'].items():
                lhs,rhs=key.split('_minus_')
                ci=np.quantile(boot[:,arms.index(lhs)]-boot[:,arms.index(rhs)],[.025,.975])
                ci_error=max(ci_error,float(np.max(abs(ci-item['ci95']))))
            assert ci_error<1e-10
            verification[f'{tag}:{horizon}']={'point_max_abs':point_error,'ci_max_abs':ci_error}

GRAY='#586976';ORANGE='#C06100';PURPLE='#80549A';INK='#202930'
STYLE={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
       'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.8,'xtick.labelsize':8,
       'ytick.labelsize':8,'legend.fontsize':7.8,'axes.linewidth':.65,
       'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
       'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix',
       'savefig.dpi':300,'figure.facecolor':'white','savefig.facecolor':'white'}
rows=[];all_=J['summary']['all'];cross=C['summary']['all']
def record(panel,key,point,lo='',hi=''):
    rows.append({'panel':panel,'comparison':key,'point':point,'ci_low':lo,'ci_high':hi,'unit':'W m-2'})

with plt.rc_context(STYLE):
    fig=plt.figure(figsize=(177/25.4,135/25.4))
    # One shared row, explicit numeric score, and one paired-difference axis.
    ax=fig.add_axes([.53,.64,.43,.25]);ax.set_xlim(-35,15);ax.set_ylim(-.5,3.6)
    ax.axvline(0,color='#687078',lw=.75);ax.grid(axis='x',color='#E1E5E8',lw=.5)
    ax.set_xticks([-30,-20,-10,0,10]);ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'$\Delta$RMSE vs. frozen baseline (W m$^{-2}$)',labelpad=4)
    label_ax=fig.add_axes([.035,.64,.465,.25],sharey=ax);label_ax.set_xlim(0,1);label_ax.axis('off')
    fig.text(.035,.963,'a  Overall error and paired difference',fontweight='bold',fontsize=9.2)
    label_ax.text(0,3.42,'Model / probe',fontsize=8.5,color=INK)
    label_ax.text(.99,3.42,'RMSE (W m$^{-2}$)',ha='right',fontsize=8.5,color=INK)
    groups=[('G1','Frozen baseline',GRAY,'o'),('context','Context probe',GRAY,'D'),
            ('forecast_COT','Context + predicted COT',ORANGE,'s'),
            ('observed_R_OFFLINE','Context + observed-image COT*',PURPLE,'^')]
    for y,(arm,name,color,marker) in zip([2.8,1.9,1.0,.1],groups):
        label_ax.text(0,y,name,va='center',fontsize=8.3)
        label_ax.text(.99,y,f"{all_['rmse'][arm]:.2f}",ha='right',va='center',color=color,fontsize=8.8)
        record('a:RMSE',arm,all_['rmse'][arm])
        if arm=='G1':
            ax.plot(0,y,marker,color=color,ms=4.6);ax.text(2,y,'reference',va='center',fontsize=7.7,color=color)
        else:
            item=all_['comparisons'][f'{arm}_minus_G1'];p=item['delta_rmse'];lo,hi=item['ci95']
            ax.errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt=marker,color=color,ms=4.6,capsize=2.3,lw=1.05)
            record('a:difference',f'{arm}_minus_G1',p,lo,hi)
    # Source matrix; row is fitting source, column is inference source.
    matrix_ax=fig.add_axes([.15,.165,.34,.315])
    keys=[['forecast_train_forecast_infer','forecast_train_observed_infer_OFFLINE'],
          ['observed_train_forecast_infer','observed_train_observed_infer_OFFLINE']]
    values=np.array([[cross['rmse'][k] for k in row] for row in keys])
    im=matrix_ax.imshow(values,cmap='Blues',norm=Normalize(95,130),interpolation='nearest',aspect='auto')
    for i in range(2):
        for j in range(2):
            value=values[i,j];delta=value-cross['rmse']['G1'];color='white' if value>115 else '#172530'
            matrix_ax.text(j,i-.10,f'{value:.2f}',ha='center',va='center',fontsize=11.5,fontweight='bold',color=color)
            matrix_ax.text(j,i+.20,f'({delta:+.2f})',ha='center',va='center',fontsize=8.3,color=color)
            record('b',keys[i][j],value)
    matrix_ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Predicted','Observed*'],
                  yticklabels=['Predicted','Observed'],xlabel='Inference COT source',ylabel='Fitting COT source')
    matrix_ax.tick_params(which='both',length=0)
    matrix_ax.spines['left'].set_visible(False);matrix_ax.spines['bottom'].set_visible(False)
    matrix_ax.set_xticks([.5],minor=True);matrix_ax.set_yticks([.5],minor=True)
    matrix_ax.grid(which='minor',color='white',lw=1.2)
    matrix_ax.add_patch(Rectangle((.5,-.5),1,2,fill=False,ec='#343D44',lw=1.1,ls=(0,(4,2)),clip_on=False))
    fig.text(.035,.537,'b  COT-source substitution',fontweight='bold',fontsize=9.2)
    cb=fig.colorbar(im,cax=fig.add_axes([.15,.068,.34,.013]),orientation='horizontal',ticks=[100,110,120,130])
    cb.set_label(r'GHI RMSE (W m$^{-2}$)',fontsize=8,labelpad=2);cb.ax.tick_params(length=2,labelsize=7.6);cb.outline.set_linewidth(.5)
    # Relative information value changes with the forecast horizon.
    horizon_ax=fig.add_axes([.635,.165,.325,.315])
    horizon_ax.axhline(0,color='#687078',lw=.75)
    for arm,color,marker,offset,label in [('forecast_COT',ORANGE,'s',-.04,'Predicted COT'),
                                       ('observed_R_OFFLINE',PURPLE,'^',.04,'Observed-image COT*')]:
        p=[];low=[];high=[]
        for h in range(1,5):
            item=J['summary'][f'{h}h']['comparisons'][f'{arm}_minus_context']
            p.append(item['delta_rmse']);low.append(item['ci95'][0]);high.append(item['ci95'][1])
            record('c',f'{h}h:{arm}_minus_context',p[-1],low[-1],high[-1])
        p,low,high=map(np.array,(p,low,high))
        horizon_ax.errorbar(np.arange(1,5)+offset,p,yerr=[p-low,high-p],color=color,
                            marker=marker,ms=4,capsize=2.1,lw=1.05,
                            ls='-' if arm=='forecast_COT' else '--',label=label)
    horizon_ax.set(xlim=(.7,4.3),ylim=(-56,5),xticks=[1,2,3,4],xticklabels=['0–1','1–2','2–3','3–4'],
                   yticks=[-50,-40,-30,-20,-10,0],xlabel='Forecast lead interval (h)',
                   ylabel=r'$\Delta$RMSE vs. context (W m$^{-2}$)')
    horizon_ax.grid(axis='y',color='#E1E5E8',lw=.5)
    horizon_ax.legend(loc='lower left',frameon=False,handlelength=1.5,borderpad=.15,fontsize=7.3)
    fig.text(.585,.537,'c  Increment by forecast horizon',fontweight='bold',fontsize=9.2)
    fig.text(.635,.068,'*Future observed imagery: offline only',fontsize=7.6,color=PURPLE)
    for ext in ['pdf','svg','png']: fig.savefig(OUT/f'cot_conditional_information.{ext}',dpi=300)
    fig.savefig(OUT/'cot_conditional_information_print_preview.png',dpi=150)
    plt.close(fig)

with (OUT/'plotted_values.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'figure':'cot_conditional_information','physical_size_mm':[177,135],
          'scope':'Development-date three-fold OOF probe diagnosis; not independent validation/test',
          'rows':32838,'target_dates':50,'date_range':[min(all_['dates']),max(all_['dates'])],
          'aggregation':'RMSE per station x frozen-baseline source, then equal-weight mean of nine values',
          'uncertainty':'Paired target-date percentile bootstrap,5000 replicates,seed20261023',
          'transformations':'Same saved scores; all displayed point estimates and intervals verified from date-level SSE/counts. No refitting or filters.',
          'source_substitution':'Posthoc cold reuse of already fitted probes, same rows; observed inference column offline',
          'reference':'Observed-image COT is frozen-R proxy, not CPP independent truth',
          'verification':verification,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
          'test_read':False,'training_performed':False,
          'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
          'code_sha256':sha(Path(__file__)),'outputs':{p.name:sha(p) for p in OUT.glob('cot_conditional_information.*')}}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'verification':verification,'outputs':list(manifest['outputs'])},indent=2))
