"""Actual maps and saved curves exposing confirmed eight-hour time-key mismatch."""
from pathlib import Path
from datetime import datetime,timedelta
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R=Path(__file__).parent;D=R/'source_probe';F=R/'source_figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200})
ST={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
def rgb(a):return np.clip(np.moveaxis(a[[2,1,0]],0,-1)/.8,0,1)**.6
def main():
 report=json.loads((D/'report.json').read_text());captions=[]
 for e in report['events']:
  init=e['init'];z=np.load(D/(init+'_with_cot_ghi.npz'));actual=e['actual_targets_bjt'];labels=e['declared_targets'];sel=[3,7,11,15]
  for st,(y,x) in ST.items():
   fig=plt.figure(figsize=(11,14));gs=fig.add_gridspec(8,4,height_ratios=[1,1,1,1,1,1,1.3,1.3],hspace=.35,wspace=.12)
   rownames=['Correct-time AGRI\nC03/C02/C01','Time-misbound AGRI\nactual BJT +8 h','Time-misbound SimVP\nactual BJT +8 h','Correct-time AGRI\nR-derived COT','Time-misbound AGRI\nR-derived COT','Time-misbound SimVP\nR-derived COT']
   for r in range(6):
    for c,l in enumerate(sel):
     ax=fig.add_subplot(gs[r,c]);ax.set_xticks([]);ax.set_yticks([])
     name=['canonical_agri','saved_truth','saved_pred','canonical_cot','saved_truth_cot','saved_pred_cot'][r]
     if r<3:ax.imshow(rgb(z[name][l]))
     else:
      ge=z['canonical_geometry' if r==3 else 'saved_geometry'][l];ok=(ge[2]>.5)&(ge[0]>np.cos(np.deg2rad(80)))
      field=np.where(ok,z[name][l,0],np.nan);cmap=plt.colormaps['magma'].copy();cmap.set_bad('#dddddd');im=ax.imshow(np.log1p(field),vmin=0,vmax=np.log1p(100),cmap=cmap)
      if not np.isfinite(field).any():ax.text(.5,.08,'Night / low sun',transform=ax.transAxes,ha='center',va='center',fontsize=8)
     t=labels[l] if r in (0,3) else actual[l];ax.set_xlabel(datetime.strptime(t,'%Y%m%d%H%M').strftime('%H:%M BJT'),fontsize=9,labelpad=2)
     ax.scatter(x,y,marker='*',s=55,c='white',edgecolors='black',linewidths=.7);ax.add_patch(Rectangle((x-1.5,y-1.5),3,3,fill=False,edgecolor='#56b4e9',lw=1))
     if c==0:ax.set_ylabel(rownames[r],fontsize=10)
   cax=fig.add_axes([.92,.32,.012,.20]);cb=fig.colorbar(im,cax=cax,ticks=np.log1p([0,1,3,10,30,100]));cb.ax.set_yticklabels(['0','1','3','10','30','100']);cb.set_label('Physical COT (log colour spacing)')
   clock=[datetime.strptime(t,'%Y%m%d%H%M') for t in labels];minutes=np.arange(1,17)*15
   def curve(name,ge):
    ok=(z[ge][:,2,y,x]>.5)&(z[ge][:,0,y,x]>np.cos(np.deg2rad(80)));return np.where(ok,z[name][:,0,y,x],np.nan)
   ax=fig.add_subplot(gs[6,:]);ax.plot(minutes,curve('canonical_cot','canonical_geometry'),'-ok',ms=3,label='Correct-time real AGRI → R');ax.plot(minutes,curve('saved_truth_cot','saved_geometry'),'-s',color='#0072b2',ms=3,label='Saved real AGRI → R (actually 8 h later)');ax.plot(minutes,curve('saved_pred_cot','saved_geometry'),'-^',color='#d55e00',ms=3,label='SimVP → R (time key is wrong)');ax.set_ylabel('Station-pixel COT');ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2)
   ax=fig.add_subplot(gs[7,:]);ax.plot(minutes,z[st+'_truth'],'-ok',ms=3,label='Observed 15-min GHI at declared target');ax.plot(minutes,z[st+'_FA'].mean(0),'-s',ms=3,color='#0072b2',label='Saved head output without COT');ax.plot(minutes,z[st+'_FAC'].mean(0),'--^',ms=3,color='#d55e00',label='Saved head output with COT');ax.set_ylabel('GHI (W m$^{-2}$)');ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2)
   for ax in fig.axes[-2:]:
    # Colourbar is earlier in axes; the last two axes are the two time curves.
    ax.set_xlim(0,245);ax.set_xticks([15,60,120,180,240],[clock[j].strftime('%H:%M') for j in (0,3,7,11,15)])
   fig.axes[-1].set_xlabel('Declared target time (BJT); saved imagery is actually 8 h later')
   fig.subplots_adjust(left=.15,right=.9,bottom=.05,top=.985)
   name=f'{st}_{init}_time_mismatch';fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
   captions.append(dict(file=name,station=st,declared_init=init,actual_imagery_init=e['actual_source_init_bjt'],description='Illustrative events from the complete 407-sequence time-key audit. Rows: correct-time, time-misbound saved real and time-misbound SimVP AGRI, then their corresponding R-COT fields. Same fixed reflectance stretch; COT uses log colour spacing. White star: station; blue square: 3x3 GHI input. Gray: night/low sun. Curves: same corrected frozen R and mean GHI predictions across three fixed seeds. Invalid time-paired diagnostic outputs, not forecast-performance evidence. Cloud maps are instantaneous; GHI is the preceding 15-min interval mean. Correct-time missing source frames are gaps, never interpolated.'))
   # Separate curve panels retain readable fonts when inserted at ordinary page width.
   small,axs=plt.subplots(2,1,figsize=(10,5.8),sharex=True)
   for cotname,ge,col,style,label in [('canonical_cot','canonical_geometry','black','-o','Correct-time real AGRI → R'),('saved_truth_cot','saved_geometry','#0072b2','-s','Time-misbound real AGRI → R (+8 h)'),('saved_pred_cot','saved_geometry','#d55e00','--^','Time-misbound SimVP → R (+8 h)')]:axs[0].plot(minutes,curve(cotname,ge),style,c=col,ms=3,label=label)
   axs[0].set_ylabel('Station COT');axs[0].legend(fontsize=9,ncol=2);axs[0].grid(alpha=.18)
   axs[1].plot(minutes,z[st+'_truth'],'-ok',ms=3,label='Observed 15-min GHI');axs[1].plot(minutes,z[st+'_FA'].mean(0),'-s',ms=3,c='#0072b2',label='Saved without-COT head');axs[1].plot(minutes,z[st+'_FAC'].mean(0),'--^',ms=3,c='#d55e00',label='Saved with-COT head');axs[1].set_ylabel('GHI (W m$^{-2}$)');axs[1].legend(fontsize=9,ncol=2);axs[1].grid(alpha=.18);axs[1].set_xlabel('Declared target time (BJT); cloud inputs are actually 8 h later');axs[1].set_xticks([15,60,120,180,240],[clock[j].strftime('%H:%M') for j in (0,3,7,11,15)]);small.tight_layout()
   small.savefig(F/(name+'_curves.png'),bbox_inches='tight');small.savefig(F/(name+'_curves.pdf'),bbox_inches='tight');plt.close(small)
 (F/'captions.json').write_text(json.dumps(captions,indent=2))
 (F/'latex_includes.tex').write_text('\n'.join('\\begin{figure}[p]\n\\centering\n\\includegraphics[width=\\textwidth]{'+c['file']+'_curves.pdf}\n\\caption{'+c['station']+'; declared initialization '+c['declared_init']+', actual imagery initialization '+c['actual_imagery_init']+'. Illustrative event from all 407 audited validation sequences. Upper panel: frozen-R COT trajectories of correct-time AGRI, time-misbound saved real AGRI, and time-misbound SimVP AGRI. Lower panel: observed preceding 15-minute mean GHI and mean saved GHI predictions across three fixed seeds. Cloud inputs of saved predictions are eight hours later than the declared target. Night/low-sun retrievals and missing source frames are gaps. Diagnostic only, not valid forecast scores.}\n\\end{figure}' for c in captions))
 print('RENDERED_ACTUAL_SOURCE_EVENTS',len(captions))
if __name__=='__main__':main()
