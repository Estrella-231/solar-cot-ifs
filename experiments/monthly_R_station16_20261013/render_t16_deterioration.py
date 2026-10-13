#!/usr/bin/env python3
"""Render saved validation diagnostics only; no model/data/test payload reads."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'experiments/monthly_R_station16_20261013_v1/results/t16_deterioration_forensics_20261014'


def main():
    a=json.loads((OUT/'forensic_audit.json').read_text(encoding='utf-8'))
    rows=json.loads((OUT/'per_target_diagnostics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(4,3,figsize=(17,14),layout='constrained')
    for j,c in enumerate(a['case_summary']):
        r=[t for t in rows if t['case']==c['case']]
        x=np.array([t['lead_min'] for t in r])
        left,mid,right=ax[j]
        low=[min(t['raw_5min']) for t in r];high=[max(t['raw_5min']) for t in r]
        left.fill_between(x,low,high,color='0.7',alpha=.22,label='range of 3 observed 5-min records')
        for k,col,style in [('ghi','black','-'),('A','#3977b7','-'),('AC','#d77b13','-'),('CPP_reference','#07936a','--')]:
            left.plot(x,[t[k] for t in r],style,color=col,lw=1.8,
                      label={'ghi':'observed 15-min GHI','CPP_reference':'AC with CPP COT (frozen head)'}.get(k,k))
        left.set_ylim(0,1350);left.set_ylabel('GHI (W m$^{-2}$)')
        left.set_title(f"Case {c['case']}: {c['station'].title()} | {c['init_time_BJT']} BJT\n"
                       f"AC minus A case RMSE: {c['mean_seed_rmse']['AC_minus_A']:+.2f} W m$^{{-2}}$")
        if j==0:left.legend(fontsize=8,loc='upper right')
        desired=np.array([t['desired_y_minus_A'] for t in r]);actual=np.array([t['AC_minus_A'] for t in r])
        mid.plot(x,desired,'o-',color='#3977b7',ms=3,label='needed change: observed minus A')
        mid.plot(x,actual,'s-',color='#d77b13',ms=3,label='actual model difference: AC minus A')
        wrong=desired*actual<0
        mid.scatter(x[wrong],actual[wrong],facecolors='none',edgecolors='crimson',s=65,label='opposite direction')
        mid.axhline(0,color='0.4',lw=.8);mid.set_ylim(-450,450);mid.set_ylabel('GHI difference (W m$^{-2}$)')
        mid.set_title('AC often lowers GHI where A already underestimates')
        if j==0:mid.legend(fontsize=8)
        err=np.array([t['mean_seed_delta_squared_error'] for t in r])/1000
        right.bar(x,err,width=10,color=np.where(err>0,'#b83d3d','#07936a'))
        right.axhline(0,color='0.4',lw=.8);right.set_ylim(-60,75)
        right.set_ylabel('Added squared error ($10^3$ W$^2$ m$^{-4}$)')
        right.set_title(f"Red = worse; green = better\nTop 3 harmful targets: {100*c['positive_SSE_top3_fraction']:.1f}% of positive added SSE")
        for panel in (left,mid,right):
            panel.set_xticks([15,60,120,180,240]);panel.grid(alpha=.16);panel.set_axisbelow(True)
            if j==3:panel.set_xlabel('Elapsed from case anchor (min); these are same-time retrievals')
    fig.suptitle('Saved validation outputs: why these COT cases deteriorate\n'
                 'Curves = three-seed mean predictions; RMSE/SSE = mean of seed-wise scores. '
                 'Zhujia cases share 13/16 targets.',fontsize=14)
    save(fig,'01_wrong_direction_and_error_concentration')

    fig,ax=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for col,station in enumerate(('sili','zhujia')):
        groups=a['all_validation'][station]['posthoc_target_based_kt_groups']
        xx=np.arange(3)
        for shift,key,color,label in [(-.18,'posthoc_training_kt_groups','#3977b7','train, frozen selected head'),
                                       (.18,'posthoc_target_based_kt_groups','#d77b13','validation')]:
            gs=a['all_validation'][station][key];vals=[g['AC_minus_A'] for g in gs]
            ax[0,col].bar(xx+shift,vals,color=color,width=.35,label=label)
            for x,v,g in zip(xx+shift,vals,gs):
                ax[0,col].text(x,v+(1 if v>=0 else -1),f'{v:+.2f}\nn={g["n"]}',ha='center',fontsize=8,
                               va='bottom' if v>=0 else 'top')
        ax[0,col].axhline(0,color='0.4',lw=.8);ax[0,col].set_ylim(-24,25)
        ax[0,col].legend(fontsize=8,loc='upper left')
        ax[0,col].set_xticks(xx,['observed kt < 0.5','0.5 <= kt < 0.9','observed kt >= 0.9'])
        ax[0,col].set_ylabel('AC minus A RMSE (W m$^{-2}$)')
        ax[0,col].set_title(f'{station.title()}: cloud-present but bright intervals fail\n'
                            'R centre COT >= 3; clear-sky GHI >= 100 W m$^{-2}$')
    points=[next(t for t in rows if t['case']==ci and t['lead_min']==lead)
            for ci,lead in [(1,30),(1,60),(3,105),(3,165)]]
    xx=np.arange(4);b=ax[1,0]
    b.plot(xx,[p['training_condition_median_kt'] for p in points],'s--',color='0.4',label='train-bin median kt')
    b.plot(xx,[p['training_condition_p90_kt'] for p in points],'v--',color='#3977b7',label='train-bin p90 kt (diagnostic only)')
    b.scatter(xx,[p['kt'] for p in points],color='black',marker='*',s=100,label='observed validation kt')
    for x,p in zip(xx,points):
        b.text(x,p['kt']+.06,f"train percentile {100*p['observed_kt_training_condition_percentile']:.1f}%",ha='center',fontsize=8)
    b.set_xticks(xx,['Sili +30','Sili +60','Zhujia +105','Zhujia +165']);b.set_ylim(0,1.32)
    b.set_ylabel('15-min kt = GHI / clear-sky reference');b.legend(fontsize=8,loc='lower left')
    b.set_title('Harmful targets lie in the bright tail of coarse train conditions\n'
                'Same station, centre-COT bin and solar-cosine bin; not all-input conditioning')
    b=ax[1,1];labels=[];perm=[];mag=[]
    for c in a['case_summary']:
        rr=[t for t in rows if t['case']==c['case']]
        labels.append(f"Case {c['case']}");perm.append(c['permutation_32_mean_abs_GHI_change'])
        mag.append(np.mean([abs(t['COT_x0.8_minus_AC_replay']) for t in rr]))
    b.bar(np.arange(4)-.18,perm,width=.36,color='#3977b7',label='shuffle 8 neighbours; centre fixed')
    b.bar(np.arange(4)+.18,mag,width=.36,color='#d77b13',label='scale physical COT by 0.8')
    b.set_xticks(np.arange(4),labels);b.set_ylabel('Mean absolute GHI change (W m$^{-2}$)')
    b.legend(fontsize=8);b.set_title('Frozen AC: much more response to COT level than arrangement\n'
                                      'Sensitivity probe only; 0.8 is not a calibration recommendation')
    for panel in ax.flat:panel.grid(axis='y',alpha=.16);panel.set_axisbelow(True)
    fig.suptitle('Mechanism evidence across train and validation\n'
                 'Post-hoc observed-kt groups use the target and cannot serve as a deployable gate. '
                 'No test data; no subgroup confidence intervals.',fontsize=13)
    save(fig,'02_conditional_tail_and_cot_response')
    names=[p.name for p in OUT.glob('0[12]_*.png')]+[p.name for p in OUT.glob('0[12]_*.pdf')]
    hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in sorted(names)}
    (OUT/'figure_audit.json').write_bytes((json.dumps({'source_diagnostic_sha256':hashlib.sha256((OUT/'forensic_audit.json').read_bytes()).hexdigest(),
                                                 'source_targets_sha256':hashlib.sha256((OUT/'per_target_diagnostics.json').read_bytes()).hexdigest(),
                                                 'render_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                 'output_sha256':hashes,'test_payload_opened':False},indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'output_files':sorted(names)},indent=2))


def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=150)
    fig.savefig(OUT/(stem+'.pdf'),metadata={'Title':stem,'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__=='__main__':main()
