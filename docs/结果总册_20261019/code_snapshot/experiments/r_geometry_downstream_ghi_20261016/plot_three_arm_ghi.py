#!/usr/bin/env python3
"""Publication-ready validation summary for paired three-arm GHI experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ARMS=('A_history','AC_history_old_R','AC_history_corrected_R')
LABELS=('AGRI only','AGRI + old-formula R-COT','AGRI + corrected-geometry R-COT')
PAIRS=(('AC_history_old_R','A_history'),('AC_history_corrected_R','A_history'),('AC_history_corrected_R','AC_history_old_R'))
PLABELS=('old R − no COT','corrected R − no COT','corrected R − old R')

def main():
 p=argparse.ArgumentParser();p.add_argument('--score',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 z=json.loads(a.score.read_text(encoding='utf-8'));v=z['groups']['all_validation'];st=z['groups']['all_validation']['per_station']
 a.output.parent.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.unicode_minus':False})
 fig,axs=plt.subplots(1,2,figsize=(13,5.3),gridspec_kw={'width_ratios':[1.1,1]},constrained_layout=True)
 colors=('#557A95','#E69F00','#009E73');x=np.arange(len(ARMS));w=.24
 for j,station in enumerate(('sili','zhujia')):
  vals=[st[station][arm]['RMSE'] for arm in ARMS]
  axs[0].bar(x+(j-.5)*w,vals,width=w,label='Sili' if station=='sili' else 'Zhujia',alpha=.92)
 for i,arm in enumerate(ARMS):
  vals=[st[s][arm]['RMSE'] for s in ('sili','zhujia')]
  axs[0].scatter([i-w/2,i+w/2],vals,c=colors[i],edgecolor='black',linewidth=.45,s=34,zorder=3)
 axs[0].set_xticks(x,LABELS,rotation=18,ha='right');axs[0].set_ylabel('Validation RMSE (W m$^{-2}$)');axs[0].set_title('Station-specific GHI error');axs[0].grid(axis='y',alpha=.22);axs[0].legend(frameon=False)
 pairs=[v['contrasts'][f'{ca} - {ct}'] for ca,ct in PAIRS]
 means=np.asarray([q['delta_RMSE'] for q in pairs]);lo=np.asarray([q['date_cluster_CI95'][0] for q in pairs]);hi=np.asarray([q['date_cluster_CI95'][1] for q in pairs]);y=np.arange(3)
 axs[1].axvline(0,color='black',lw=1);axs[1].errorbar(means,y,xerr=np.vstack((means-lo,hi-means)),fmt='o',color='#305F72',ecolor='#305F72',capsize=4,lw=1.7)
 axs[1].set_yticks(y,PLABELS);axs[1].invert_yaxis();axs[1].set_xlabel('Paired ΔRMSE (W m$^{-2}$); negative favors first arm');axs[1].set_title('Paired BJT-date bootstrap 95% intervals');axs[1].grid(axis='x',alpha=.22)
 fig.suptitle('Three-arm frozen-R COT comparison · validation only',fontsize=14)
 fig.text(.5,-.02,'Observed AGRI at T−30/T−15/T; target is the same 15-min interval. This is offline synchronous reconstruction, not a causal forecast. 4,263 common validation rows; all R-COT inputs are frozen.',ha='center',fontsize=9)
 fig.savefig(a.output,dpi=180,bbox_inches='tight');fig.savefig(a.output.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
 print(json.dumps({'state':'FIGURE_COMPLETE','output':str(a.output),'overall_rmse':v['rmse_station_seed_mean'],'contrasts':v['contrasts']},indent=2))
if __name__=='__main__':main()
