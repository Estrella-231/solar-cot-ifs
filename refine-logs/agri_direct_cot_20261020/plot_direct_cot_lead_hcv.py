"""Honest diagnostic plot for the H-only lead/COT factorial OOF score."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--score',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 report=json.loads(a.score.read_text(encoding='utf-8'));r=report['results']
 a.output_dir.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
 # Panel A: lead-wise point estimates are descriptive and have no per-lead CI.
 x=np.asarray([z['lead_minutes'] for z in r['leadwise_pxc_lead_minus_none_lead']])
 y=np.asarray([z['delta_rmse_wm2'] for z in r['leadwise_pxc_lead_minus_none_lead']])
 ax=axs[0];ax.axvspan(15,60,color='#e9f2f8',zorder=0,label='+15–60 min')
 ax.axhline(0,color='#555555',lw=1)
 ax.plot(x,y,color='#0072B2',marker='o',lw=1.7,ms=5)
 ax.set(xlabel='Forecast lead (min)',ylabel='ΔRMSE: P-XC − no COT (W m$^{-2}$)',title='By lead (point estimates)')
 ax.grid(axis='y',alpha=.25);ax.set_xlim(10,245)
 # Panel B: date-block confidence intervals.
 ax=axs[1]; labels=[];vals=[];los=[];his=[];colors=[]
 entries=[('COT, no lead','cot_effect_without_lead'),('COT, with lead','cot_effect_with_lead'),('Lead×COT','lead_cot_interaction')]
 for lab,key in entries:
  for horizon,disp in (('all_leads','0–240 min'),('plus15_60','+15–60 min')):
   q=r[key][horizon];labels.append(f'{lab}: {disp}');vals.append(q['delta_rmse_wm2']);los.append(q['ci95_low']);his.append(q['ci95_high'])
   colors.append('#D55E00' if 'interaction' in lab else ('#0072B2' if 'with lead' in lab else '#009E73'))
 yy=np.arange(len(labels))[::-1]
 for yi,v,lo,hi,c in zip(yy,vals,los,his,colors):
  ax.errorbar(v,yi,xerr=[[v-lo],[hi-v]],fmt='o',color=c,ecolor=c,capsize=3,lw=1.5,ms=5)
 ax.axvline(0,color='#555555',lw=1);ax.set_yticks(yy,labels);ax.set_xlabel('ΔRMSE (W m$^{-2}$)');ax.set_title('Paired date-block 95% CI')
 ax.grid(axis='x',alpha=.25)
 # Panel C: station/seed spread for the primary comparison.
 ax=axs[2];sd=r['station_delta_all_leads'];ypos={'sili':1,'zhujia':0};col={'sili':'#0072B2','zhujia':'#D55E00'}
 ax.axvline(0,color='#555555',lw=1)
 offsets={42:-.12,43:0,44:.12}
 for st,rec in sd.items():
  for seed,val in zip((42,43,44),rec['seed_deltas']):
   ax.scatter(val,ypos[st]+offsets[seed],color=col[st],marker={42:'o',43:'s',44:'^'}[seed],s=42,label=f'{st}, seed {seed}')
  ax.plot(rec['mean'],ypos[st],marker='D',color=col[st],ms=7)
 ax.set_yticks([0,1],['Zhujia','Sili']);ax.set_xlabel('ΔRMSE (W m$^{-2}$)');ax.set_title('Primary contrast by station/seed')
 ax.grid(axis='x',alpha=.25)
 handles,labels0=ax.get_legend_handles_labels();by_label=dict(zip(labels0,handles))
 ax.legend(by_label.values(),by_label.keys(),fontsize=7,loc='lower right',frameon=False)
 fig.suptitle('Forecast-COT value with explicit lead conditioning\nH-only internal date-block cross-validation; not independent test',fontsize=12)
 png=a.output_dir/'lead_conditioned_cot_hcv.png';pdf=a.output_dir/'lead_conditioned_cot_hcv.pdf'
 fig.savefig(png,dpi=300,facecolor='white');fig.savefig(pdf,facecolor='white');plt.close(fig)
 (a.output_dir/'figure_manifest.json').write_text(json.dumps({'state':'COMPLETE_LEAD_HCV_DIAGNOSTIC_FIGURE',
  'score_json':str(a.score),'score_sha256':__import__('hashlib').sha256(a.score.read_bytes()).hexdigest(),
  'leadwise_curves':'descriptive point estimates; no per-lead CI','intervals':'5,000 paired calendar two-day block bootstrap draws',
  'independent_test':False,'figure_files':[png.name,pdf.name]},indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'png':str(png),'pdf':str(pdf)}))
if __name__=='__main__':main()
