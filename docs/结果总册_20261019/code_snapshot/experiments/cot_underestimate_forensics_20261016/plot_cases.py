"""Native-resolution true-cloud cases, matching controls, and frozen sensitivities."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cases=json.loads((a.results/'selected_cases.json').read_text(encoding='utf-8'));cloud=np.load(a.payload/'real_cloud_cases_and_gradients.npz');probe=np.load(a.payload/'selected_frozen_probes.npz');receipt=json.loads((a.payload/'extraction_receipt.json').read_text());frame={(r['case_index'],r['frame']):r for r in receipt['frames']};cpp=np.load(a.payload/'cpp_reference_cases.npz')
 run=a.repo/'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1';rows=list(csv.DictReader((run/'validation_rows.csv').open(encoding='utf-8',newline='')));z=np.load(run/'analysis/paired_validation_scoring.npz');pred=z['prediction'];y=z['truth'];ids={c['case_index']:j for j,c in enumerate(cases['cases'])}
 rgb=np.stack([np.moveaxis(cloud[f'c{c["case_index"]}_t{j}_rgb'],0,-1) for c in cases['cases'] for j in range(3)])
 lo=np.nanpercentile(rgb,[2],axis=(0,1,2))[0];hi=np.nanpercentile(rgb,[98],axis=(0,1,2))[0];cv=max(float(np.nanmax(cloud[f'c{c["case_index"]}_cot_corrected'])) for c in cases['cases']);cv=float(np.ceil(cv/10)*10)
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.unicode_minus':False})
 with PdfPages(a.output/'paired_real_cloud_cases.pdf') as pdf:
  for pi,pair in enumerate(cases['pairs']):
   fig=plt.figure(figsize=(15,11.5),layout='constrained');gs=fig.add_gridspec(4,4,width_ratios=[1,1,1,2.3],hspace=.1)
   for block,key in enumerate(('bad_case','control_case')):
    ci=pair[key];c=cases['cases'][ci];i=c['validation_index'];t=datetime.fromisoformat(c['bjt']);cot=cloud[f'c{ci}_cot_corrected'];role='Failure: A low, AC lower in all seeds' if key=='bad_case' else 'Matched control: A low, AC improves in all seeds'
    for j,minutes in enumerate((-30,-15,0)):
     ax=fig.add_subplot(gs[2*block,j]);im=np.moveaxis(cloud[f'c{ci}_t{j}_rgb'],0,-1);im=np.clip((im-lo)/(hi-lo),0,1);ax.imshow(im,interpolation='nearest');ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,edgecolor='cyan',lw=1.2));ax.plot(8,8,'+',c='cyan',ms=9,mew=1.4);ax.set_xticks([]);ax.set_yticks([]);fr=frame[(ci,j)];scan=datetime.fromisoformat(fr['actual_station_scan_utc'])+timedelta(hours=8);ax.set_title(f'{c["station"].title()}  {minutes:+d} min\nscan {scan:%H:%M:%S} BJT',fontsize=9)
     ax=fig.add_subplot(gs[2*block+1,j]);cm=ax.imshow(cot[j],vmin=0,vmax=cv,cmap='magma',interpolation='nearest');ax.set_xticks([0,1,2]);ax.set_yticks([0,1,2]);ax.plot(1,1,'+',c='cyan',ms=10,mew=1.3)
     for rr in range(3):
      for cc in range(3):ax.text(cc,rr,f'{cot[j,rr,cc]:.1f}',ha='center',va='center',fontsize=8,color='white' if cot[j,rr,cc]<.6*cv else 'black')
     ref=f'{cpp["reference_COT"][ci,j,1,1]:.2f}' if cpp['valid'][ci,j,1,1] else 'unavailable'
     ax.set_title(f'R-COT center {cot[j,1,1]:.2f}; CPP {ref}',fontsize=9)
    ax=fig.add_subplot(gs[2*block:2*block+2,3]);pick=[k for k,r in enumerate(rows) if r['station']==c['station'] and abs((datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)-t).total_seconds())<=3600 and (datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date()==t.date()]
    xx=np.array([(datetime.fromisoformat(rows[k]['target_time_utc'])+timedelta(hours=8)-t).total_seconds()/60 for k in pick]);order=np.argsort(xx);xx=xx[order];pick=np.array(pick)[order]
    for r in c['native_5min']:
     if r['ghi'] is not None:
      x0=(datetime.fromisoformat(r['bjt'])-t).total_seconds()/60;ax.hlines(r['ghi'],x0-5,x0,color='.65',lw=2,label='Native 5-min interval' if x0==-60 else None)
    ax.plot(xx,y[pick],'o-',c='black',lw=1.3,ms=3,label='Observed 15-min mean')
    for ai,color,label in [(0,'#0072B2','A: AGRI only'),(2,'#D55E00','AC: corrected R-COT')]:
     pp=pred[:,ai,pick];ax.plot(xx,pp.mean(0),marker='s' if ai==0 else '^',c=color,ms=3,lw=1,label=label);ax.fill_between(xx,pp.min(0),pp.max(0),color=color,alpha=.12)
    ax.axvspan(-15,0,color='#009E73',alpha=.10,label='Target label interval');ax.axvline(0,color='.3',lw=.8)
    for j,minutes in enumerate((-30,-15,0)):
     scanrel=minutes+frame[(ci,j)]['scan_offset_s']/60;ax.axvline(scanrel,color='#CC79A7',ls=':',lw=1,label='Actual input scans' if j==0 else None)
    ax.set_xlim(-45,45);ax.set_ylabel('GHI (W m$^{-2}$)');ax.set_xlabel('Minutes from target interval end; separate interval estimates');ax.grid(alpha=.18)
    cirow=ids[ci];gz=probe['zero_standardized_cot'][:,cirow].mean();gactual=probe['original'][:,cirow].mean();grad=probe['gradient_dGHI_dlog1pCOT'][:,cirow].sum((2,3)).mean(0)
    ax.set_title(f'{role}\n{c["bjt"]} BJT; GHI {c["ghi"]:.1f}, A {c["A_mean"]:.1f}, AC {c["AC_mean"]:.1f}\nkt={c["kt"]:.2f}; local COT sensitivity by frame: {grad[0]:+.1f}, {grad[1]:+.1f}, {grad[2]:+.1f}',fontsize=9)
    if block==0:ax.legend(fontsize=7,loc='best',ncol=2)
   fig.suptitle(f'Paired true-cloud forensics {pi+1}/6 · fixed 3×3, three-frame GHI models',fontsize=14)
   fig.supxlabel('RGB: real C03/C02/C01, shared linear 2–98% stretch. RGB 16×16 is existing R input context only; cyan square is the unchanged GHI 3×3.\nCOT: exact saved 3×3 input, shared scale. Shaded bands: three-seed range. Five-minute records diagnose timing; metrics remain 15-minute. Local derivatives are model sensitivities.',fontsize=8)
   fig.colorbar(cm,ax=fig.axes,location='right',shrink=.5,pad=.015,label='COT (dimensionless)');pdf.savefig(fig);fig.savefig(a.output/f'pair_{pi+1}_{cases["cases"][pair["bad_case"]]["station"]}.png',dpi=150);plt.close(fig)
  # Exact current-model CTH anchors are listed separately; do not reuse old GHI values.
  fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');cth=list(csv.DictReader((a.results/'existing_CTH_current_model_join.csv').open(encoding='utf-8',newline='')))
  for ax,r in zip(axes.flat,cth):
   ci=int(r['case_index']);c=cases['cases'][ci];im=np.moveaxis(cloud[f'c{ci}_t2_rgb'],0,-1);ax.imshow(np.clip((im-lo)/(hi-lo),0,1),interpolation='nearest');ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,edgecolor='cyan'));ax.plot(8,8,'+',c='cyan',ms=12);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{c["station"]} {c["bjt"]} BJT\nGHI {c["ghi"]:.0f}; current A {c["A_mean"]:.0f}, AC {c["AC_mean"]:.0f}\nNearest projected shadow: {float(r["shadow_nearest_km"]):.2f} km; helpful seeds {r["current_helpful_seed_count"]}/3',fontsize=10)
  fig.suptitle('Existing target-time CTH geometry joined to the current three-frame models',fontsize=13);fig.supxlabel('CTH geometry is reused only at identical validation station/time/grid keys. RGB is qualitative context; 4 km is a product-scale geometric screen, not measured shadow truth.',fontsize=8);pdf.savefig(fig);fig.savefig(a.output/'existing_CTH_current_model_context.png',dpi=150);plt.close(fig)
 bad=[c for c in cases['cases'] if c['role']=='severe_under_drop'];vals=np.concatenate([cloud[f'c{c["case_index"]}_gradient'].mean(0).reshape(-1) for c in bad]);lim=float(max(abs(vals.min()),abs(vals.max())));norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
 fig,axs=plt.subplots(len(bad),3,figsize=(10,2.25*len(bad)),layout='constrained')
 for row,c in enumerate(bad):
  g=cloud[f'c{c["case_index"]}_gradient'];m=g.mean(0)
  for j in range(3):
   ax=axs[row,j];im=ax.imshow(m[j],cmap='RdBu_r',norm=norm,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.plot(1,1,'+',c='black',ms=8)
   for rr in range(3):
    for cc in range(3):ax.text(cc,rr,f'{m[j,rr,cc]:+.1f}',ha='center',va='center',fontsize=8)
   ax.set_title(f'{c["station"]} {c["bjt"][:16]}\nT{(-30,-15,0)[j]:+d}; total {m[j].sum():+.1f}',fontsize=9)
 fig.colorbar(im,ax=axs,label='Local dGHI/dlog1p(COT), seed mean (W m$^{-2}$)',shrink=.65);fig.suptitle('Exact-input frozen-model COT sensitivity · six prespecified underestimation failures',fontsize=12);fig.supxlabel('Negative: locally increasing COT lowers this AC model output. This is a model derivative, not a physical causal effect or AC−A attribution.',fontsize=9);fig.savefig(a.output/'local_COT_sensitivity.png',dpi=150);fig.savefig(a.output/'local_COT_sensitivity.pdf');plt.close(fig)
 fig,axs=plt.subplots(len(bad),4,figsize=(12,2.15*len(bad)),layout='constrained');dif=[]
 for c in bad:
  ci=c['case_index'];m=cpp['valid'][ci,-1];dif.extend((cloud[f'c{ci}_cot_corrected'][-1]-cpp['reference_COT'][ci,-1])[m].tolist())
 dl=max(max(abs(x) for x in dif),1.);dc=plt.get_cmap('RdBu_r').with_extremes(bad='.75');mc=plt.get_cmap('magma').with_extremes(bad='.75')
 for k,c in enumerate(bad):
  ci=c['case_index'];im=np.moveaxis(cloud[f'c{ci}_t2_rgb'],0,-1);axs[k,0].imshow(np.clip((im-lo)/(hi-lo),0,1),interpolation='nearest');axs[k,0].add_patch(Rectangle((6.5,6.5),3,3,fill=False,edgecolor='cyan'));axs[k,0].plot(8,8,'+',c='cyan');axs[k,0].set_title(f'{c["station"]} {c["bjt"][:16]} BJT\nReal AGRI context',fontsize=9)
  rv=cloud[f'c{ci}_cot_corrected'][-1];ref=cpp['reference_COT'][ci,-1];m=cpp['valid'][ci,-1];axs[k,1].imshow(rv,vmin=0,vmax=cv,cmap=mc,interpolation='nearest');rf=axs[k,2].imshow(ref,vmin=0,vmax=cv,cmap=mc,interpolation='nearest');df=axs[k,3].imshow(rv-ref,vmin=-dl,vmax=dl,cmap=dc,interpolation='nearest')
  axs[k,1].set_title(f'Exact saved R-COT 3×3\ncenter {rv[1,1]:.3f}',fontsize=9);axs[k,2].set_title('CPP retrieval reference\n'+(f'center {ref[1,1]:.3f}' if m[1,1] else 'No exact supported reference'),fontsize=9);axs[k,3].set_title('R − CPP, common valid pixels',fontsize=9)
  for ax in axs[k]:ax.set_xticks([]);ax.set_yticks([])
 fig.colorbar(rf,ax=axs[:,1:3],label='COT (dimensionless)',shrink=.5);fig.colorbar(df,ax=axs[:,3],label='R − CPP (COT)',shrink=.5);fig.suptitle('Current R inputs versus exact-key CPP reference · six GHI failure cases',fontsize=13);fig.supxlabel('CPP is a retrieval reference, not independent truth. Grey means no exact eligible CPP sample, never clear sky. The unchanged GHI input is the 3×3 marked in cyan.',fontsize=9);fig.savefig(a.output/'R_CPP_failure_comparison.png',dpi=150);fig.savefig(a.output/'R_CPP_failure_comparison.pdf');plt.close(fig)
 (a.output/'figure_transformations.json').write_text(json.dumps(dict(RGB_bands=['C03','C02','C01'],RGB_low=lo.tolist(),RGB_high=hi.tolist(),stretch='shared across all selected contexts and times; clipped only for display',interpolation='nearest; original 16x16 R context and exact 3x3 GHI COT inputs',COT_limits=[0,cv],gradient_limits=[-lim,lim],test_payloads_read=0),indent=2)+'\n')
 print('FIGURES_COMPLETE')
if __name__=='__main__':main()
